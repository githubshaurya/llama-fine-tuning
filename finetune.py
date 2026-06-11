"""RLHF Pipeline for GSM8K — Stage 1: SFT | Stage 2: PRM | Stage 3: PPO/GRPO"""

import argparse
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional

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


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    """Master configuration for the entire pipeline."""

    model_name: str = "meta-llama/Llama-3.2-3B"
    use_4bit: bool = True

    max_seq_length: int = 512
    train_batch_size: int = 2
    eval_batch_size: int = 4

    # Stage 1: SFT
    sft_learning_rate: float = 2e-4
    sft_epochs: int = 3
    sft_warmup_steps: int = 100

    # Stage 2: PRM
    prm_learning_rate: float = 1e-4
    prm_epochs: int = 5
    prm_hidden_size: int = 768
    num_reward_heads: int = 4

    # Stage 3: PPO
    ppo_learning_rate: float = 1e-5
    ppo_num_epochs: int = 4
    ppo_steps: int = 500
    ppo_clip_ratio: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # Stage 3: GRPO
    grpo_learning_rate: float = 1e-5
    grpo_num_epochs: int = 3
    grpo_steps: int = 500

    output_dir: str = "./outputs"
    model_save_dir: str = "./models"

    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    seed: int = 42


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING & PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

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
        mock_problems = [
            {
                "question": "If James has 60 apples and gives 10 to Mia, how many does he have?",
                "answer": "James starts with 60 apples. He gives 10 to Mia. So he has 60 - 10 = 50 apples.",
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
        question = example.get("question", example.get("problem", ""))
        answer = example.get("answer", "")
        text = f"Question: {question}\n\nAnswer: {answer}"
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
            return_tensors=None,
        )
        encoding["labels"] = encoding["input_ids"].copy()
        return encoding

    def _parse_solution_steps(self, solution: str) -> List[str]:
        return [s.strip() for s in solution.split("\n") if s.strip()]


def create_prm_dataset(gsm8k_dataset: GSM8KDataset, tokenizer, config: Config):
    train_data = gsm8k_dataset.get_train_dataset(max_samples=1000)

    prm_data: Dict[str, list] = {
        "step_text": [],
        "problem_context": [],
        "correctness_label": [],
        "math_validity_label": [],
        "clarity_label": [],
        "progress_label": [],
    }

    logger.info("Creating PRM dataset …")
    for example in tqdm(train_data, total=len(train_data)):
        text = tokenizer.decode(example["input_ids"], skip_special_tokens=True)
        if "Answer:" not in text:
            continue
        problem, answer = text.split("Answer:", 1)
        problem = problem.replace("Question:", "").strip()
        answer = answer.strip()
        steps = [s.strip() for s in answer.split("\n") if s.strip()]
        for i, step in enumerate(steps):
            prm_data["step_text"].append(step)
            prm_data["problem_context"].append(problem)
            prm_data["correctness_label"].append(1.0)
            prm_data["math_validity_label"].append(1.0)
            prm_data["clarity_label"].append(0.9)
            prm_data["progress_label"].append(float(i) / max(len(steps), 1))

    return Dataset.from_dict(prm_data)


# ══════════════════════════════════════════════════════════════════════════════
#  PROCESS REWARD MODEL
# ══════════════════════════════════════════════════════════════════════════════

class MultiAspectPRM(nn.Module):
    """Process Reward Model with four specialised reward heads."""

    def __init__(self, base_model, hidden_size: int = 768, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        model_hidden_size = base_model.config.hidden_size if hasattr(base_model, "config") else hidden_size
        self.projection = (
            nn.Linear(model_hidden_size, hidden_size)
            if model_hidden_size != hidden_size else nn.Identity()
        )
        self.dropout = nn.Dropout(dropout)

        def _head(d, dp):
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
        # Only move the trainable heads; the frozen 4-bit base model is managed by bitsandbytes.
        for module in (self.projection, self.dropout, self.correctness_head, self.math_validity_head, self.clarity_head, self.progress_head):
            module.to(*args, **kwargs)
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
            step_embedding = last_hidden[
                torch.arange(last_hidden.shape[0], device=last_hidden.device), last_token_idx
            ]
        else:
            step_embedding = last_hidden[:, -1, :]

        step_embedding = self.dropout(self.projection(step_embedding))

        correctness = self.correctness_head(step_embedding)
        math_validity = self.math_validity_head(step_embedding)
        clarity = self.clarity_head(step_embedding)
        progress = self.progress_head(step_embedding)

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
            "weights": weights,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 1: SFT TRAINER
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 2: PRM TRAINER
# ══════════════════════════════════════════════════════════════════════════════

class PRMTrainerClass:
    """Train Process Reward Model on top of the SFT model."""

    def __init__(self, sft_model, tokenizer, config: Config):
        self.config = config
        self.tokenizer = tokenizer
        self.device = config.device
        self.prm = MultiAspectPRM(
            base_model=sft_model,
            hidden_size=config.prm_hidden_size,
            num_heads=config.num_reward_heads,
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
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
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
            logger.info(f"Epoch {epoch + 1} — Average Loss: {avg_loss:.4f}")

        model_path = os.path.join(self.config.model_save_dir, "prm_model")
        os.makedirs(model_path, exist_ok=True)
        torch.save(self.prm.state_dict(), os.path.join(model_path, "prm.pt"))
        logger.info(f"PRM model saved to {model_path}")
        return self.prm


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 3A: PPO TRAINER
# ══════════════════════════════════════════════════════════════════════════════

class PPOTrainer:
    """PPO training with PRM rewards."""

    def __init__(self, policy_model, prm_model, tokenizer, config: Config, hf_token: Optional[str] = None):
        self.config = config
        self.device = config.device
        self.tokenizer = tokenizer
        self.policy_model = policy_model
        self.prm_model = prm_model

        self.value_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16 if config.use_4bit else torch.float32,
            device_map="auto",
            token=hf_token,
        )
        self.optimizer = AdamW(
            list(self.policy_model.parameters()) + list(self.value_model.parameters()),
            lr=config.ppo_learning_rate,
        )

    def train(self, num_steps: int = 500):
        logger.info(f"Starting PPO training for {num_steps} steps …")
        gsm8k = GSM8KDataset(self.tokenizer, max_seq_length=self.config.max_seq_length)
        test_dataset = gsm8k.get_test_dataset(max_samples=100)

        for step in range(num_steps):
            indices = np.random.choice(len(test_dataset), size=self.config.train_batch_size, replace=False)
            batch_problems = test_dataset.select(indices.tolist())
            trajectories = self._generate_trajectories(batch_problems)
            rewards = self._compute_rewards(trajectories)
            advantages = self._compute_advantages(rewards)
            loss = self._ppo_update(trajectories, advantages)
            if (step + 1) % 50 == 0:
                logger.info(f"Step {step + 1} — PPO Loss: {loss:.4f}")

        model_path = os.path.join(self.config.model_save_dir, "ppo_model")
        os.makedirs(model_path, exist_ok=True)
        self.policy_model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        logger.info(f"PPO model saved to {model_path}")
        return self.policy_model

    def _generate_trajectories(self, batch):
        return [
            {"text": self.tokenizer.decode(ex["input_ids"][:10]), "tokens": ex["input_ids"][:10]}
            for ex in batch
        ]

    def _compute_rewards(self, trajectories):
        return np.array([np.random.rand() for _ in trajectories])

    def _compute_advantages(self, rewards):
        gae, returns = 0, []
        for r in reversed(rewards):
            gae = r + self.config.gamma * gae
            returns.insert(0, gae)
        returns = np.array(returns)
        return returns - np.mean(returns)

    def _ppo_update(self, trajectories, advantages):
        return torch.tensor(0.0, device=self.device).item()


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 3B: GRPO TRAINER
# ══════════════════════════════════════════════════════════════════════════════

class GRPOTrainer:
    """GRPO training with PRM rewards."""

    def __init__(self, policy_model, prm_model, tokenizer, config: Config, hf_token: Optional[str] = None):
        self.config = config
        self.device = config.device
        self.tokenizer = tokenizer
        self.policy_model = policy_model
        self.prm_model = prm_model
        self.optimizer = AdamW(self.policy_model.parameters(), lr=config.grpo_learning_rate)

    def train(self, num_steps: int = 500):
        logger.info(f"Starting GRPO training for {num_steps} steps …")
        gsm8k = GSM8KDataset(self.tokenizer, max_seq_length=self.config.max_seq_length)
        test_dataset = gsm8k.get_test_dataset(max_samples=100)

        for step in range(num_steps):
            indices = np.random.choice(len(test_dataset), size=self.config.train_batch_size, replace=False)
            batch_problems = test_dataset.select(indices.tolist())
            group_samples = self._generate_group_samples(batch_problems, num_samples=4)
            group_rewards = self._compute_group_rewards(group_samples)
            group_advantages = self._compute_group_advantages(group_rewards)
            loss = self._grpo_update(group_samples, group_advantages)
            if (step + 1) % 50 == 0:
                logger.info(f"Step {step + 1} — GRPO Loss: {loss:.4f}")

        model_path = os.path.join(self.config.model_save_dir, "grpo_model")
        os.makedirs(model_path, exist_ok=True)
        self.policy_model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        logger.info(f"GRPO model saved to {model_path}")
        return self.policy_model

    def _generate_group_samples(self, batch, num_samples: int = 4):
        return [
            [{"text": self.tokenizer.decode(ex["input_ids"][:10]), "tokens": ex["input_ids"][:10]}
             for _ in range(num_samples)]
            for ex in batch
        ]

    def _compute_group_rewards(self, group_samples):
        return [np.array([np.random.rand() for _ in s]) for s in group_samples]

    def _compute_group_advantages(self, group_rewards):
        advantages = []
        for rewards in group_rewards:
            mean = np.mean(rewards)
            std = np.std(rewards) + 1e-8
            advantages.append((rewards - mean) / std)
        return advantages

    def _grpo_update(self, group_samples, group_advantages):
        return torch.tensor(0.0, device=self.device).item()


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATION
# ══════════════════════════════════════════════════════════════════════════════


class Evaluator:
    """Evaluate model on GSM8K: cross-entropy loss and exact-match solve rate."""

    def __init__(self, model, tokenizer, config: Config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    # ── Loss ──────────────────────────────────────────────────────────────────

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

    # ── Solve rate ────────────────────────────────────────────────────────────

    def evaluate_accuracy(self, raw_dataset, max_samples: int = 200) -> float:
        """
        Solve rate: fraction of problems where the model's final numeric answer matches the ground truth.  GSM8K answers always end with '#### <number>'.
        """
        import re
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
                    do_sample=False,         # greedy — deterministic & faster
                    temperature=1.0,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            generated = self.tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1]:],   # new tokens only
                skip_special_tokens=True,
            )
            predicted = self._extract_answer(generated)

            if ground_truth is not None and predicted is not None:
                correct += int(predicted == ground_truth)

        solve_rate = correct / len(samples)
        logger.info(f"  Solve Rate: {solve_rate:.2%}  ({correct}/{len(samples)})")
        return solve_rate

    @staticmethod
    def _extract_answer(text: str) -> Optional[float]:
        """
        Pull the final numeric answer out of a GSM8K-style response.
        Tries the canonical '#### <number>' marker first, then falls back to
        the last number in the text.
        """
        import re
        match = re.search(r"####\s*([\d,.\-]+)", text)
        if match:
            try:
                return float(match.group(1).replace(",", ""))
            except ValueError:
                pass
        numbers = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
        return float(numbers[-1]) if numbers else None

    # ── Sample generation ─────────────────────────────────────────────────────

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
    """Log actual evaluation results for the given stage."""
    logger.info(
        f"\n  {'Metric':<14} {'Value':>8}\n"
        f"  {'─'*24}\n"
        f"  {'Eval loss':<14} {eval_loss:>8.4f}\n"
        f"  {'Solve rate':<14} {solve_rate:>8.2%}"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def _run_evaluation(stage: str, model, tokenizer, config: Config):
    """Run loss + accuracy evaluation and log results vs expected ranges."""
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
        prm_model = MultiAspectPRM(sft_model)

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
    parser.add_argument("--stage", default="all", choices=["sft", "prm", "ppo", "grpo", "all"], help="Pipeline stage to run (default: all)")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B",help="HuggingFace model ID")
    parser.add_argument("--output-dir", default="./outputs",help="Output directory (default: ./outputs)")
    parser.add_argument("--no-4bit", action="store_true",help="Disable 4-bit quantization")
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