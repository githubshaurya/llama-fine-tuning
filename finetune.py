"""RLHF Pipeline for GSM8K — Stage 1: SFT | Stage 2: PRM | Stage 3: PPO/GRPO

``create_prm_dataset`` derives step-level labels *heuristically* from GSM8K reference solutions — it does NOT use human annotation.  Labels are assigned as follows:

    correctness: 1.0 for steps from a gold solution; 0.0 if an inline <<expr=result>> annotation is numerically inconsistent.
    math_validity: Same sanity check; 1.0 when no annotation exists.
    clarity: Heuristic proxy based on word count.
    progress: (step_index + 1) / total_steps.

Because these labels are synthetic, the PRM trains on a proxy signal rather than true step-level correctness. Real PRM training requires human or verified-incorrect-trace labels (e.g., PRM800K).
"""

import argparse
import logging
import os
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    get_linear_schedule_with_warmup,
)
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

IGNORE_INDEX = -100   # standard label mask value for CrossEntropyLoss

# Restricted eval environment arithmetic operators only, no builtins.
_SAFE_EVAL_GLOBALS: Dict = {"__builtins__": {}}

@dataclass
class Config:
    """Master configuration for the entire pipeline."""

    model_name: str = "meta-llama/Llama-3.2-3B"
    use_4bit: bool = True

    max_seq_length: int = 512
    train_batch_size: int = 2
    eval_batch_size: int = 4

    #1 SFT
    sft_learning_rate: float = 2e-4
    sft_epochs: int = 3
    sft_warmup_steps: int = 100

    #2 PRM
    prm_learning_rate: float = 1e-4
    prm_epochs: int = 5
    prm_hidden_size: int = 768   # projection dimension for the reward heads

    #3 PPO
    ppo_learning_rate: float = 1e-5
    ppo_num_epochs: int = 4 # inner gradient epochs per rollout batch
    ppo_steps: int = 500 # outer rollout steps
    ppo_clip_ratio: float = 0.2 # probability-ratio clipping range
    ppo_kl_coef: float = 0.1 # KL divergence penalty coefficient
    ppo_vf_coef: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95

    #3 GRPO
    grpo_learning_rate: float = 1e-5
    grpo_num_epochs: int = 3 # inner gradient epochs per problem batch
    grpo_steps: int = 500
    grpo_group_size: int = 4 # G: candidate responses sampled per question
    grpo_kl_coef: float = 0.04  # KL coefficient against reference policy

    output_dir: str = "./outputs"
    model_save_dir: str = "./models"

    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    seed: int = 42

#  Utilities

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_4bit_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )


def apply_lora(model, use_4bit: bool = True):
    if use_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def load_model_and_tokenizer(
    model_name: str,
    use_4bit: bool = True,
    apply_lora_adapters: bool = False,
    hf_token: Optional[str] = None,
):
    logger.info(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if use_4bit:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=get_4bit_config(),
            device_map="auto",
            trust_remote_code=True,
            token=hf_token,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            token=hf_token,
        )

    if apply_lora_adapters:
        model = apply_lora(model, use_4bit=use_4bit)
        model.config.use_cache = False

    return model, tokenizer

#  Data Loading

class GSM8KDataset:
    def __init__(self, tokenizer, max_seq_length: int = 512):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        logger.info("Loading GSM8K dataset …")
        try:
            self.dataset = load_dataset("gsm8k", "main", trust_remote_code=True)
        except Exception as e:
            logger.warning(f"Could not load from HuggingFace: {e}. Using mock dataset.")
            self.dataset = self._create_mock_dataset()

    def _create_mock_dataset(self) -> Dict[str, "Dataset"]:
        # Mirrors the real GSM8K format: <<expr=result>> annotations + #### answer.
        mock_problems = [
            {
                "question": "If James has 60 apples and gives 10 to Mia, how many does he have?",
                "answer": (
                    "James starts with 60 apples.\n"
                    "He gives 10 to Mia.\n"
                    "So he has 60-10=<<60-10=50>>50 apples.\n"
                    "#### 50"
                ),
            }
            for _ in range(100)
        ]
        return {
            "train": Dataset.from_dict({
                "question": [p["question"] for p in mock_problems],
                "answer": [p["answer"] for p in mock_problems],
            })
        }

    def get_train_dataset(self, max_samples: Optional[int] = None):
        dataset = self.dataset["train"]
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        return dataset.map(self._preprocess_sft, remove_columns=dataset.column_names)

    def get_test_dataset(self, max_samples: Optional[int] = None):
        if "test" not in self.dataset:
            dataset = self.dataset["train"]
            split_idx = int(0.9 * len(dataset))
            dataset = dataset.select(range(split_idx, len(dataset)))
        else:
            dataset = self.dataset["test"]
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        return dataset.map(self._preprocess_sft, remove_columns=dataset.column_names)

    def get_raw_test_dataset(self, max_samples: Optional[int] = None):
        """Return un-tokenized examples with 'question' and 'answer' fields."""
        if "test" not in self.dataset:
            dataset = self.dataset["train"]
            split_idx = int(0.9 * len(dataset))
            dataset = dataset.select(range(split_idx, len(dataset)))
        else:
            dataset = self.dataset["test"]
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        return dataset

    def _preprocess_sft(self, example):
        """
        Tokenise a question/answer pair for SFT.
        Labels are IGNORE_INDEX (-100) for every prompt token so the loss only applies to answer tokens.  Padding positions are also masked.
        """
        question = example.get("question", example.get("problem", ""))
        answer = example.get("answer", "")

        prompt = f"Question: {question}\n\nAnswer: "
        full_text = prompt + answer

        # Tokenise the prompt separately to learn its exact token length.
        # add_special_tokens=True matches the behaviour of the full-text call so that the BOS token (where present) is included in both counts.
        prompt_len = len(
            self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_seq_length,
                add_special_tokens=True,
            )["input_ids"]
        )

        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
            return_tensors=None,
        )
        input_ids = encoding["input_ids"]
        pad_id = self.tokenizer.pad_token_id

        # Mask prompt tokens and padding; copy answer token ids as labels.
        labels = [IGNORE_INDEX] * len(input_ids)
        for i in range(prompt_len, len(input_ids)):
            if input_ids[i] != pad_id:
                labels[i] = input_ids[i]

        encoding["labels"] = labels
        return encoding

    def _parse_solution_steps(self, solution: str) -> List[str]:
        """
        Parse a GSM8K reference solution into individual reasoning steps.

        GSM8K solutions separate steps with newlines and embed inline arithmetic in <<expr=result>> markers (e.g. "60-10=<<60-10=50>>50").  The final line is always "#### N".
        Heuristic strategy (not human-validated):
          1. Split on newlines.
          2. Strip <<…>> annotations, keeping surrounding prose.
          3. Drop empty lines; include "#### N" as the terminal step.
        """
        steps = []
        for line in solution.split("\n"):
            line = line.strip()
            if not line:
                continue
            clean = re.sub(r"<<[^>]*>>", "", line).strip()
            if clean:
                steps.append(clean)
        return steps


# PRM dataset helpers

def _check_annotation_consistency(step: str) -> bool:
    """
    Return True when every <<expr=result>> annotation in the step is numerically consistent.  Returns True when no annotation is present.
    Uses a restricted eval limited to arithmetic characters to avoid arbitrary code execution.
    """
    for m in re.finditer(r"<<([^>]+)>>", step):
        annotation = m.group(1)
        if "=" not in annotation:
            continue
        expr_str, expected_str = annotation.rsplit("=", 1)
        expr_str = expr_str.replace(",", "").strip()
        expected_str = expected_str.replace(",", "").strip()
        # Only evaluate expressions made of digits and arithmetic operators.
        if not re.match(r"^[\d\s\+\-\*/\.\(\)]+$", expr_str):
            continue
        try:
            computed = float(eval(expr_str, _SAFE_EVAL_GLOBALS))  # noqa: S307
            expected = float(expected_str)
            if abs(computed - expected) > 0.5:
                return False
        except Exception:
            pass   # cannot evaluate; do not penalise
    return True


def _clarity_score(step: str) -> float:
    """
    Heuristic clarity proxy in [0,1] based on word count.
    Very short (< 3 words) or very long (> 50 words) steps score lower. This is not validated annotation - it is a rough signal only.
    """
    n = len(step.split())
    if n < 3:
        return 0.3
    if n <= 20:
        return 0.9
    if n <= 40:
        return 0.75
    return 0.6


def create_prm_dataset(gsm8k_dataset: GSM8KDataset, tokenizer, config: Config):
    """
    Build a step-level PRM training dataset from GSM8K reference solutions.
    Steps come exclusively from gold (correct) solutions, so they receive correctness = 1.0 by default; this drops to 0.0 only when an inline <<expr=result>> annotation fails an arithmetic sanity check.
    """
    raw_dataset = gsm8k_dataset.dataset["train"]
    max_samples = min(1000, len(raw_dataset))
    raw_dataset = raw_dataset.select(range(max_samples))

    prm_data: Dict[str, list] = {
        "step_text": [],
        "problem_context": [],
        "correctness_label": [],
        "math_validity_label": [],
        "clarity_label": [],
        "progress_label": [],
    }

    logger.info("Creating PRM dataset from reference solutions (heuristic labels) …")
    for example in tqdm(raw_dataset, total=len(raw_dataset)):
        question = example.get("question", example.get("problem", ""))
        answer = example.get("answer", "")
        steps = gsm8k_dataset._parse_solution_steps(answer)
        n = len(steps)
        if n == 0:
            continue

        for i, step in enumerate(steps):
            consistent = _check_annotation_consistency(step)
            prm_data["step_text"].append(step)
            prm_data["problem_context"].append(question)
            prm_data["correctness_label"].append(1.0 if consistent else 0.0)
            prm_data["math_validity_label"].append(1.0 if consistent else 0.0)
            prm_data["clarity_label"].append(_clarity_score(step))
            prm_data["progress_label"].append(float(i + 1) / n)

    return Dataset.from_dict(prm_data)

#  PROCESS REWARD MODEL
class MultiAspectPRM(nn.Module):
    """
    Process Reward Model with four reward heads: correctness, math_validity,
    clarity, progress.  The base model is kept frozen; only the projection
    layer and heads are trained.
    """

    def __init__(self, base_model, hidden_size: int = 768, dropout: float = 0.1):
        super().__init__()
        self.base_model = base_model

        model_hidden_size = getattr(base_model.config, "hidden_size", hidden_size)
        self.projection = (
            nn.Linear(model_hidden_size, hidden_size)
            if model_hidden_size != hidden_size
            else nn.Identity()
        )
        self.dropout = nn.Dropout(dropout)

        def _head(d: int, dp: float) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(d, d // 2), nn.ReLU(),
                nn.Dropout(dp),
                nn.Linear(d // 2, 1), nn.Sigmoid(),
            )

        self.correctness_head = _head(hidden_size, dropout)
        self.math_validity_head = _head(hidden_size, dropout)
        self.clarity_head = _head(hidden_size, dropout)
        self.progress_head = _head(hidden_size, dropout)
        self.reward_weights = nn.Parameter(torch.tensor([0.4, 0.3, 0.15, 0.15]))

    def to(self, *args, **kwargs):
        # Only move the trainable heads; the 4-bit base model is managed by bitsandbytes.
        for m in (self.projection, self.dropout,
                  self.correctness_head, self.math_validity_head,
                  self.clarity_head, self.progress_head):
            m.to(*args, **kwargs)
        self.reward_weights = nn.Parameter(self.reward_weights.to(*args, **kwargs))
        return self

    def forward(self, input_ids, attention_mask=None):
        with torch.no_grad():
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = outputs.hidden_states[-1]

        if attention_mask is not None:
            last_token_idx = attention_mask.sum(dim=1) - 1
            step_emb = last_hidden[
                torch.arange(last_hidden.shape[0], device=last_hidden.device),
                last_token_idx,
            ]
        else:
            step_emb = last_hidden[:, -1, :]

        step_emb = self.dropout(self.projection(step_emb))

        correctness = self.correctness_head(step_emb)
        math_validity = self.math_validity_head(step_emb)
        clarity = self.clarity_head(step_emb)
        progress = self.progress_head(step_emb)

        weights = F.softmax(self.reward_weights, dim=0)
        final_reward = (
            weights[0] * correctness +
            weights[1] * math_validity +
            weights[2] * clarity +
            weights[3] * progress
        )

        return {
            "final_reward": final_reward,
            "correctness": correctness,
            "math_validity": math_validity,
            "clarity": clarity,
            "progress": progress,
        }

# SFT TRAINER

class SFTTrainer:
    """Supervised Fine-Tuning on GSM8K using HuggingFace Trainer with LoRA adapters."""

    def __init__(self, config: Config, hf_token: Optional[str] = None):
        self.config = config
        self.hf_token = hf_token
        set_seed(config.seed)
        self.model, self.tokenizer = load_model_and_tokenizer(
            config.model_name,
            use_4bit=config.use_4bit,
            apply_lora_adapters=True,
            hf_token=hf_token,
        )
        self.gsm8k = GSM8KDataset(self.tokenizer, max_seq_length=config.max_seq_length)

    def train(self):
        logger.info("Starting SFT training …")
        train_dataset = self.gsm8k.get_train_dataset(max_samples=5000)
        eval_dataset = self.gsm8k.get_test_dataset(max_samples=500)

        sft_output = os.path.join(self.config.output_dir, "sft")
        os.makedirs(sft_output, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=sft_output,
            num_train_epochs=self.config.sft_epochs,
            per_device_train_batch_size=self.config.train_batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            learning_rate=self.config.sft_learning_rate,
            warmup_steps=self.config.sft_warmup_steps,
            weight_decay=0.01,
            logging_steps=50,
            save_steps=500,
            eval_steps=100,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=False,
            gradient_accumulation_steps=4,
            max_grad_norm=1.0,
            optim="paged_adamw_32bit",
            fp16=True,
            remove_unused_columns=False,
            report_to="none",
            dataloader_num_workers=0,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=default_data_collator,
        )

        trainer.train()

        model_path = os.path.join(self.config.model_save_dir, "sft_model")
        os.makedirs(model_path, exist_ok=True)
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        logger.info(f"SFT LoRA adapters saved to {model_path}")
        return self.model, self.tokenizer

# PRM TRAINER

class PRMTrainerClass:
    """Train Process Reward Model heads on top of the frozen SFT model."""

    def __init__(self, sft_model, tokenizer, config: Config):
        self.config = config
        self.tokenizer = tokenizer
        self.device = config.device
        self.prm = MultiAspectPRM(
            base_model=sft_model,
            hidden_size=config.prm_hidden_size,
            dropout=0.1,
        ).to(self.device)
        trainable_params = [p for p in self.prm.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable_params, lr=config.prm_learning_rate)
        self.gsm8k = GSM8KDataset(tokenizer, max_seq_length=config.max_seq_length)

    def train(self):
        logger.info("Starting PRM training …")
        prm_dataset = create_prm_dataset(self.gsm8k, self.tokenizer, self.config)

        numeric_cols = ["correctness_label", "math_validity_label", "clarity_label", "progress_label"]
        prm_dataset.set_format(type="torch", columns=numeric_cols, output_all_columns=True)

        train_loader = DataLoader(
            prm_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            num_workers=0,
        )

        num_training_steps = len(train_loader) * self.config.prm_epochs
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=min(500, num_training_steps // 10),
            num_training_steps=num_training_steps,
        )

        self.prm.train()

        for epoch in range(self.config.prm_epochs):
            total_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"PRM Epoch {epoch + 1}")
            for batch in progress_bar:
                raw = batch["step_text"]
                if isinstance(raw, list):
                    enc = self.tokenizer(
                        raw, padding=True, truncation=True,
                        max_length=self.config.max_seq_length, return_tensors="pt",
                    )
                    input_ids = enc["input_ids"].to(self.device)
                    attention_mask = enc["attention_mask"].to(self.device)
                else:
                    input_ids = raw.to(self.device)
                    attention_mask = batch.get("attention_mask", torch.ones_like(input_ids)).to(self.device)

                correctness_t = batch["correctness_label"].to(self.device).float().view(-1, 1)
                math_t = batch["math_validity_label"].to(self.device).float().view(-1, 1)
                clarity_t = batch["clarity_label"].to(self.device).float().view(-1, 1)
                progress_t = batch["progress_label"].to(self.device).float().view(-1, 1)

                outputs = self.prm(input_ids, attention_mask=attention_mask)
                loss = (
                    F.mse_loss(outputs["correctness"], correctness_t) +
                    F.mse_loss(outputs["math_validity"], math_t) +
                    F.mse_loss(outputs["clarity"], clarity_t) +
                    F.mse_loss(outputs["progress"], progress_t)
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.prm.parameters(), 1.0)
                self.optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_loss = total_loss / len(train_loader)
            logger.info(f"PRM Epoch {epoch + 1} — Average Loss: {avg_loss:.4f}")

        model_path = os.path.join(self.config.model_save_dir, "prm_model")
        os.makedirs(model_path, exist_ok=True)
        torch.save(self.prm.state_dict(), os.path.join(model_path, "prm.pt"))
        logger.info(f"PRM model saved to {model_path}")
        return self.prm

# PPO TRAINER

class ValueHead(nn.Module):
    """
    Scalar value head for PPO baseline estimation.
    Takes a per-token hidden state of shape [T, H] and returns [T] value estimates.
    """

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_state).squeeze(-1)


class PPOTrainer:
    """
    PPO training loop for a LoRA-adapted language model.

    policy_model  LoRA-adapted model being optimised.
    ref_model     Frozen snapshot of the base model — provides the KL baseline.
    prm_model     Process Reward Model; scores each completed response.
    value_head    Small MLP trained alongside the policy to estimate per-token returns.

    Objective (per token t in the response)
      r_t  = π(a_t|s_t) / π_old(a_t|s_t)            probability ratio
      L_CLIP = E[min(r_t A_t, clip(r_t, 1-self.config.ppo_clip_ratio, 1+self.config.ppo_clip_ratio) A_t)]
      L_VF   = 0.5 * E[(V(s_t) − R_t)²]
      L_KL   = β * (log π_cur − log π_ref)            sequence-level KL
      L_total = −L_CLIP + c_vf · L_VF + L_KL

    Rewards
    The PRM score is placed on the final response token.  All other token rewards are 0.  GAE then propagates this terminal signal backwards using per-token value estimates.
    """

    def __init__(
        self,
        policy_model,
        prm_model,
        tokenizer,
        config: Config,
        hf_token: Optional[str] = None,
    ):
        self.config = config
        self.device = config.device
        self.tokenizer = tokenizer
        self.policy_model = policy_model
        self.prm_model = prm_model

        # Frozen reference policy — loaded with the same quantisation as the policy.
        logger.info("Loading frozen reference model for PPO KL penalty …")
        if config.use_4bit:
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                config.model_name, quantization_config=get_4bit_config(),
                device_map="auto", token=hf_token,
            )
        else:
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                config.model_name, torch_dtype=torch.float16,
                device_map="auto", token=hf_token,
            )
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        self.ref_model.eval()

        # Value head (trained, not frozen).
        hidden_size = getattr(policy_model.config, "hidden_size", config.prm_hidden_size)
        self.value_head = ValueHead(hidden_size).to(self.device)

        self.optimizer = AdamW(
            [p for p in self.policy_model.parameters() if p.requires_grad]
            + list(self.value_head.parameters()),
            lr=config.ppo_learning_rate,
        )

    # Public API 

    def train(self, num_steps: int = 500):
        logger.info(f"Starting PPO training for {num_steps} rollout steps …")
        gsm8k = GSM8KDataset(self.tokenizer, max_seq_length=self.config.max_seq_length)
        raw_train = gsm8k.dataset["train"]

        for step in range(num_steps):
            indices = np.random.choice(
                len(raw_train), size=self.config.train_batch_size, replace=False
            )
            batch_problems = [raw_train[int(i)] for i in indices]

            trajectories = self._generate_trajectories(batch_problems)
            rewards = self._compute_rewards(trajectories)
            advantages, returns = self._compute_gae(trajectories, rewards)

            total_loss = 0.0
            for _ in range(self.config.ppo_num_epochs):
                total_loss = self._ppo_update(trajectories, advantages, returns)

            if (step + 1) % 50 == 0:
                logger.info(f"PPO step {step + 1}/{num_steps} — loss: {total_loss:.4f}")

        model_path = os.path.join(self.config.model_save_dir, "ppo_model")
        os.makedirs(model_path, exist_ok=True)
        self.policy_model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        logger.info(f"PPO model saved to {model_path}")
        return self.policy_model

    # Rollout generation

    def _generate_trajectories(self, batch_problems: List[Dict]) -> List[Dict]:
        """
        Generate one response per problem and record per-token log-probs under
        both the current policy (used as π_old during the update) and the frozen
        reference model (used for KL penalty).  Also records per-token value
        estimates for GAE computation.
        """
        self.policy_model.eval()
        device = next(self.policy_model.parameters()).device
        trajectories = []

        with torch.no_grad():
            for problem in batch_problems:
                question = problem.get("question", problem.get("problem", ""))
                prompt = f"Question: {question}\n\nAnswer:"

                prompt_enc = self.tokenizer(
                    prompt, return_tensors="pt", truncation=True,
                    max_length=self.config.max_seq_length // 2,
                ).to(device)
                prompt_len = prompt_enc["input_ids"].shape[1]

                output_ids = self.policy_model.generate(
                    **prompt_enc,
                    max_new_tokens=min(128, self.config.max_seq_length - prompt_len),
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

                response_ids = output_ids[0, prompt_len:]
                if response_ids.numel() == 0:
                    continue

                response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

                # One forward pass for policy log-probs AND value estimates.
                out = self.policy_model(input_ids=output_ids, output_hidden_states=True)
                policy_lp = self._token_log_probs(out.logits[0], response_ids, prompt_len)
                hidden = out.hidden_states[-1][0, prompt_len:].to(self.device)
                values = self.value_head(hidden).detach()   # [resp_len]

                # Reference log-probs (frozen).
                ref_logits = self.ref_model(input_ids=output_ids).logits[0]
                ref_lp = self._token_log_probs(ref_logits, response_ids, prompt_len)

                trajectories.append({
                    "question": question,
                    "response_text": response_text,
                    "response_ids": response_ids.cpu(),
                    "output_ids": output_ids.cpu(),
                    "prompt_len": prompt_len,
                    "policy_log_probs": policy_lp.cpu(),   # π_old
                    "ref_log_probs": ref_lp.cpu(),
                    "values": values.cpu(),                # [resp_len]
                })

        self.policy_model.train()
        return trajectories

    # Reward computation

    def _compute_rewards(self, trajectories: List[Dict]) -> List[torch.Tensor]:
        """
        Assign per-token rewards.
        The PRM terminal score is placed on the final response token; all
        other positions receive 0.  The GAE then propagates this signal
        backwards through the per-token value estimates.
        """
        device = next(self.policy_model.parameters()).device
        all_rewards = []
        self.prm_model.eval()

        with torch.no_grad():
            for traj in trajectories:
                resp_len = traj["response_ids"].shape[0]
                token_rewards = torch.zeros(resp_len)

                if resp_len > 0:
                    enc = self.tokenizer(
                        traj["response_text"], return_tensors="pt",
                        truncation=True, max_length=self.config.max_seq_length,
                    ).to(device)
                    prm_out = self.prm_model(
                        enc["input_ids"], attention_mask=enc.get("attention_mask")
                    )
                    token_rewards[-1] = prm_out["final_reward"].item()

                all_rewards.append(token_rewards)

        return all_rewards

    # GAE

    def _compute_gae(
        self,
        trajectories: List[Dict],
        rewards: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        all_advantages, all_returns = [], []

        for traj, reward in zip(trajectories, rewards):
            values = traj["values"]   # [T]
            T = len(reward)
            advantages = torch.zeros(T)
            gae = 0.0
            for t in reversed(range(T)):
                next_val = values[t + 1].item() if t + 1 < T else 0.0
                delta = reward[t].item() + self.config.gamma * next_val - values[t].item()
                gae = delta + self.config.gamma * self.config.gae_lambda * gae
                advantages[t] = gae
            all_advantages.append(advantages)
            all_returns.append(advantages + values)

        return all_advantages, all_returns

    # PPO gradient step

    def _ppo_update(
        self,
        trajectories: List[Dict],
        advantages: List[torch.Tensor],
        returns: List[torch.Tensor],
    ) -> float:
        """
        One inner epoch of clipped PPO updates across all trajectories.

        For each trajectory:
          1. Recompute current log-probs and value predictions via a fresh forward pass (so gradients flow through LoRA adapters).
          2. Compute the clipped surrogate loss, value MSE loss, and KL penalty.
          3. Back-propagate and clip gradients.
        """
        device = next(self.policy_model.parameters()).device
        self.config.ppo_clip_ratio = self.config.ppo_clip_ratio
        c_vf = self.config.ppo_vf_coef
        total_loss = 0.0

        self.policy_model.train()
        self.value_head.train()

        for traj, adv, ret in zip(trajectories, advantages, returns):
            if traj["response_ids"].numel() == 0:
                continue

            output_ids = traj["output_ids"].to(device)
            response_ids = traj["response_ids"].to(device)
            prompt_len = traj["prompt_len"]
            adv = adv.to(device)
            ret = ret.to(device)

            if adv.numel() > 1:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            # Fresh forward pass — gradients flow through LoRA adapters.
            out = self.policy_model(input_ids=output_ids, output_hidden_states=True)
            cur_lp = self._token_log_probs(out.logits[0], response_ids, prompt_len)
            hidden = out.hidden_states[-1][0, prompt_len:].to(device)
            values_pred = self.value_head(hidden)   # [resp_len]

            # Clipped surrogate loss.
            old_lp = traj["policy_log_probs"].to(device)
            ratio = torch.exp(cur_lp - old_lp)
            surr1 = ratio * adv
            surr2 = ratio.clamp(1.0 - self.config.ppo_clip_ratio, 1.0 + self.config.ppo_clip_ratio) * adv
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value function loss.
            value_loss = F.mse_loss(values_pred, ret)

            # KL penalty: current policy vs frozen reference (sequence-level mean).
            kl = (cur_lp - traj["ref_log_probs"].to(device)).mean()

            loss = policy_loss + c_vf * value_loss + self.config.ppo_kl_coef * kl

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.value_head.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / max(len(trajectories), 1)

    @staticmethod
    def _token_log_probs(
        logits: torch.Tensor,
        response_ids: torch.Tensor,
        prompt_len: int,
    ) -> torch.Tensor:
        """
        Extract per-token log-probs for the response tokens.

        logits [seq_len, vocab] full sequence logits (prompt + response)
        response_ids [resp_len]
        prompt_len: token count of the prompt (including BOS if present)

        Shifted indexing: logit at position t predicts token t+1, so the logit for the first response token is at index prompt_len − 1.
        """
        resp_len = response_ids.shape[0]
        resp_logits = logits[prompt_len - 1: prompt_len - 1 + resp_len]   # [resp_len, vocab]
        log_probs = F.log_softmax(resp_logits.float(), dim=-1)
        return log_probs.gather(1, response_ids.unsqueeze(1)).squeeze(1)


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 3B: GRPO TRAINER
# ══════════════════════════════════════════════════════════════════════════════

class GRPOTrainer:
    """
    GRPO (Group Relative Policy Optimization) training loop.
    """

    def __init__(
        self,
        policy_model,
        prm_model,
        tokenizer,
        config: Config,
        hf_token: Optional[str] = None,
    ):
        self.config = config
        self.device = config.device
        self.tokenizer = tokenizer
        self.policy_model = policy_model
        self.prm_model = prm_model

        logger.info("Loading frozen reference model for GRPO KL penalty …")
        if config.use_4bit:
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                config.model_name, quantization_config=get_4bit_config(),
                device_map="auto", token=hf_token,
            )
        else:
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                config.model_name, torch_dtype=torch.float16,
                device_map="auto", token=hf_token,
            )
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        self.ref_model.eval()

        self.optimizer = AdamW(
            [p for p in self.policy_model.parameters() if p.requires_grad],
            lr=config.grpo_learning_rate,
        )

    def train(self, num_steps: int = 500):
        logger.info(f"Starting GRPO training for {num_steps} steps …")
        gsm8k = GSM8KDataset(self.tokenizer, max_seq_length=self.config.max_seq_length)
        raw_train = gsm8k.dataset["train"]
        G = self.config.grpo_group_size

        for step in range(num_steps):
            indices = np.random.choice(
                len(raw_train), size=self.config.train_batch_size, replace=False
            )
            batch_problems = [raw_train[int(i)] for i in indices]

            groups = self._generate_group_samples(batch_problems, G)
            groups = self._score_group_samples(groups)
            groups = self._compute_group_advantages(groups)

            total_loss = 0.0
            for _ in range(self.config.grpo_num_epochs):
                total_loss = self._grpo_update(groups)

            if (step + 1) % 50 == 0:
                logger.info(f"GRPO step {step + 1}/{num_steps} — loss: {total_loss:.4f}")

        model_path = os.path.join(self.config.model_save_dir, "grpo_model")
        os.makedirs(model_path, exist_ok=True)
        self.policy_model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        logger.info(f"GRPO model saved to {model_path}")
        return self.policy_model

    def _generate_group_samples(
        self, batch_problems: List[Dict], G: int
    ) -> List[Dict]:
        """
        Independently sample G candidate responses per problem.
        Records per-token log-probs under both the current policy (π_old) and
        the frozen reference (π_ref).
        """
        self.policy_model.eval()
        device = next(self.policy_model.parameters()).device
        groups = []

        with torch.no_grad():
            for problem in batch_problems:
                question = problem.get("question", problem.get("problem", ""))
                prompt = f"Question: {question}\n\nAnswer:"
                prompt_enc = self.tokenizer(
                    prompt, return_tensors="pt", truncation=True,
                    max_length=self.config.max_seq_length // 2,
                ).to(device)
                prompt_len = prompt_enc["input_ids"].shape[1]

                samples = []
                for _ in range(G):
                    output_ids = self.policy_model.generate(
                        **prompt_enc,
                        max_new_tokens=min(128, self.config.max_seq_length - prompt_len),
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                    response_ids = output_ids[0, prompt_len:]
                    if response_ids.numel() == 0:
                        continue

                    response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

                    pol_logits = self.policy_model(input_ids=output_ids).logits[0]
                    pol_lp = PPOTrainer._token_log_probs(pol_logits, response_ids, prompt_len)

                    ref_logits = self.ref_model(input_ids=output_ids).logits[0]
                    ref_lp = PPOTrainer._token_log_probs(ref_logits, response_ids, prompt_len)

                    samples.append({
                        "response_text": response_text,
                        "response_ids": response_ids.cpu(),
                        "output_ids": output_ids.cpu(),
                        "prompt_len": prompt_len,
                        "policy_log_probs": pol_lp.cpu(),
                        "ref_log_probs": ref_lp.cpu(),
                    })

                groups.append({"question": question, "samples": samples})

        self.policy_model.train()
        return groups

    def _score_group_samples(self, groups: List[Dict]) -> List[Dict]:
        """Score each response with the PRM and store the scalar reward."""
        device = next(self.policy_model.parameters()).device
        self.prm_model.eval()
        with torch.no_grad():
            for group in groups:
                for sample in group["samples"]:
                    enc = self.tokenizer(
                        sample["response_text"], return_tensors="pt",
                        truncation=True, max_length=self.config.max_seq_length,
                    ).to(device)
                    prm_out = self.prm_model(
                        enc["input_ids"], attention_mask=enc.get("attention_mask")
                    )
                    sample["reward"] = prm_out["final_reward"].item()
        return groups

    def _compute_group_advantages(self, groups: List[Dict]) -> List[Dict]:
        """
        Normalise rewards within each group
        """
        for group in groups:
            if not group["samples"]:
                continue
            rewards = np.array([s["reward"] for s in group["samples"]], dtype=np.float32)
            mean = float(rewards.mean())
            std = float(rewards.std()) + 1e-8
            for s, r in zip(group["samples"], rewards):
                s["advantage"] = (r - mean) / std
        return groups

    def _grpo_update(self, groups: List[Dict]) -> float:
        """
        One inner epoch of GRPO updates across all samples in all groups.
        """
        device = next(self.policy_model.parameters()).device
        self.config.ppo_clip_ratio =    # same clip ratio as PPO
        total_loss = 0.0
        n_updates = 0

        self.policy_model.train()
        for group in groups:
            for sample in group["samples"]:
                if sample["response_ids"].numel() == 0:
                    continue

                output_ids = sample["output_ids"].to(device)
                response_ids = sample["response_ids"].to(device)
                prompt_len = sample["prompt_len"]
                adv = torch.tensor(sample["advantage"], dtype=torch.float32, device=device)
                old_lp = sample["policy_log_probs"].to(device)
                ref_lp = sample["ref_log_probs"].to(device)

                cur_logits = self.policy_model(input_ids=output_ids).logits[0]
                cur_lp = PPOTrainer._token_log_probs(cur_logits, response_ids, prompt_len)

                ratio = torch.exp(cur_lp - old_lp)
                surr = torch.min(ratio * adv, ratio.clamp(1.0 - self.config.ppo_clip_ratio, 1.0 + self.config.ppo_clip_ratio) * adv)
                policy_loss = -surr.mean()

                kl = (cur_lp - ref_lp).mean()
                loss = policy_loss + self.config.grpo_kl_coef * kl

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
                self.optimizer.step()

                total_loss += loss.item()
                n_updates += 1

        return total_loss / max(n_updates, 1)

#  EVALUATION

class Evaluator:
    """Evaluate model on GSM8K: cross-entropy loss and exact-match solve rate."""

    def __init__(self, model, tokenizer, config: Config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    def evaluate(self, test_dataset) -> float:
        """Average cross-entropy loss on a pre-tokenised dataset."""
        logger.info("Computing eval loss …")
        self.model.eval()
        total_loss = 0.0
        device = next(self.model.parameters()).device

        with torch.no_grad():
            for example in tqdm(test_dataset, desc="Eval loss"):
                input_ids = torch.tensor(example["input_ids"]).unsqueeze(0).to(device)
                labels = torch.tensor(example["labels"]).unsqueeze(0).to(device)
                outputs = self.model(input_ids=input_ids, labels=labels)
                total_loss += outputs.loss.item()

        avg_loss = total_loss / len(test_dataset)
        logger.info(f"  Eval Loss : {avg_loss:.4f}")
        return avg_loss

    def evaluate_accuracy(self, raw_dataset, max_samples: int = 200) -> float:
        """
        Solve rate: fraction of problems where the model's numeric answer matches the ground truth within a tolerance of 1e-6.

        GSM8K ground truth always contains a '#### <number>' marker.
        """
        logger.info(f"Computing solve rate on {min(max_samples, len(raw_dataset))} problems …")
        self.model.eval()
        device = next(self.model.parameters()).device

        samples = raw_dataset.select(range(min(max_samples, len(raw_dataset))))
        correct = 0

        for example in tqdm(samples, desc="Eval accuracy"):
            question = example.get("question", example.get("problem", ""))
            ground_truth = self._extract_answer(example.get("answer", ""))

            prompt = f"Question: {question}\n\nAnswer:"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            generated = self.tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            predicted = self._extract_answer(generated)

            if ground_truth is not None and predicted is not None:
                correct += int(abs(predicted - ground_truth) < 1e-6)

        solve_rate = correct / len(samples)
        logger.info(f"  Solve Rate: {solve_rate:.2%}  ({correct}/{len(samples)})")
        return solve_rate

    @staticmethod
    def _extract_answer(text: str) -> Optional[float]:
        """
        Extract the final numeric answer from a GSM8K-style response.
        Strategy:
          1. Look for the canonical '#### <number>' marker and parse it.
          2. Fall back to the last number found anywhere in the text.
        Commas are stripped from all candidates.
        Returns None if no number can be parsed.
        """
        # Canonical marker.
        m = re.search(r"####\s*([\d,.\-]+)", text)
        if m:
            try:
                return float(m.group(1).replace(",", ""))
            except ValueError:
                pass
        # Fallback: last number in the text (handles plain numeric answers).
        numbers = re.findall(r"-?\d[\d,]*\.?\d*", text)
        for candidate in reversed(numbers):
            try:
                return float(candidate.replace(",", ""))
            except ValueError:
                continue
        return None

    def generate_sample(self, prompt: str, max_new_tokens: int = 256) -> str:
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)


def _log_metrics(stage: str, eval_loss: float, solve_rate: float):
    logger.info(
        f"\n  {'Metric':<14} {'Value':>8}\n"
        f"  {'─' * 24}\n"
        f"  {'Eval loss':<14} {eval_loss:>8.4f}\n"
        f"  {'Solve rate':<14} {solve_rate:>8.2%}"
    )

def _run_evaluation(stage: str, model, tokenizer, config: Config):
    logger.info("=" * 50 + f"\nEVALUATION — {stage.upper()}\n" + "=" * 50)
    gsm8k = GSM8KDataset(tokenizer, max_seq_length=config.max_seq_length)
    evaluator = Evaluator(model, tokenizer, config)
    eval_loss = evaluator.evaluate(gsm8k.get_test_dataset(max_samples=200))
    solve_rate = evaluator.evaluate_accuracy(gsm8k.get_raw_test_dataset(max_samples=200))
    _log_metrics(stage, eval_loss, solve_rate)
    return eval_loss, solve_rate


def run_pipeline(
    stage: str = "all",
    model_name: str = "meta-llama/Llama-3.2-3B",
    output_dir: str = "./outputs",
    use_4bit: bool = True,
    hf_token: Optional[str] = None,
):
    config = Config(model_name=model_name, output_dir=output_dir, use_4bit=use_4bit)
    set_seed(config.seed)
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.model_save_dir, exist_ok=True)

    if stage in ("sft", "all"):
        logger.info("=" * 50 + "\nSTAGE 1: SUPERVISED FINE-TUNING\n" + "=" * 50)
        sft_trainer = SFTTrainer(config, hf_token=hf_token)
        sft_model, tokenizer = sft_trainer.train()
        _run_evaluation("sft", sft_model, tokenizer, config)
    else:
        logger.info("Loading base model for downstream stages …")
        sft_model, tokenizer = load_model_and_tokenizer(
            config.model_name, use_4bit=config.use_4bit,
            apply_lora_adapters=False, hf_token=hf_token,
        )

    if stage in ("prm", "all"):
        logger.info("=" * 50 + "\nSTAGE 2: PROCESS REWARD MODEL\n" + "=" * 50)
        prm_trainer = PRMTrainerClass(sft_model, tokenizer, config)
        prm_model = prm_trainer.train()
    else:
        logger.info("Initialising PRM without training …")
        prm_model = MultiAspectPRM(sft_model, hidden_size=config.prm_hidden_size)

    if stage in ("ppo", "all"):
        logger.info("=" * 50 + "\nSTAGE 3A: PPO TRAINING\n" + "=" * 50)
        ppo_trainer = PPOTrainer(sft_model, prm_model, tokenizer, config, hf_token=hf_token)
        ppo_model = ppo_trainer.train(num_steps=config.ppo_steps)
        _run_evaluation("ppo", ppo_model, tokenizer, config)

    if stage == "grpo":
        logger.info("=" * 50 + "\nSTAGE 3B: GRPO TRAINING\n" + "=" * 50)
        grpo_trainer = GRPOTrainer(sft_model, prm_model, tokenizer, config, hf_token=hf_token)
        grpo_model = grpo_trainer.train(num_steps=config.grpo_steps)
        _run_evaluation("grpo", grpo_model, tokenizer, config)

    logger.info("=" * 50 + "\nPIPELINE COMPLETE!\n" + "=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RLHF Pipeline for GSM8K")
    parser.add_argument(
        "--stage", default="all",
        choices=["sft", "prm", "ppo", "grpo", "all"],
        help="Pipeline stage to run (default: all)",
    )
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B")
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.warning("No HF_TOKEN found in environment. Gated models like Llama 3.2 require it.")

    run_pipeline(
        stage=args.stage,
        model_name=args.model,
        output_dir=args.output_dir,
        use_4bit=not args.no_4bit,
        hf_token=hf_token,
    )
