# GSM8K RLHF Pipeline — `finetune.py`

A three-stage Reinforcement Learning from Human Feedback (RLHF) pipeline that trains **Llama 3.2 3B** (or any compatible causal LM) on grade-school math reasoning
([GSM8K](https://huggingface.co/datasets/gsm8k)).

---

## Pipeline Overview

```
[Base LLM]
    │
    ▼  Stage 1: SFT
[SFT Model]  — LoRA fine-tuned on (question → chain-of-thought answer) pairs
    │
    ▼  Stage 2: PRM
[Multi-Aspect PRM]  — four reward heads trained on top of frozen SFT backbone
    │
    ▼  Stage 3: PPO  ──or──  GRPO
[RL-aligned Model]  — policy optimised against PRM reward + KL penalty
```

---

## Requirements

```bash
pip install torch transformers datasets peft bitsandbytes tqdm numpy
```

A **HuggingFace account token** is required for gated models (Llama 3.2 3B):

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

GPU with ≥ 16 GB VRAM recommended. 4-bit quantisation (NF4 + double quant) is on by default to reduce memory to ~8 GB.

---

## Commands

### Run the full pipeline (SFT → PRM → PPO)
```bash
python finetune.py --stage all
```

### Use a different model or output path
```bash
python finetune.py \
  --stage all \
  --model meta-llama/Llama-3.2-3B \
  --output-dir ./my_outputs
```

### Disable 4-bit quantisation (requires more VRAM)
```bash
python finetune.py --stage all --no-4bit
```

---

## Results

| Stage | GSM8K Solve Rate |
| Base Llama 3.2 3B | 26% |
| After SFT (3 epochs, 5k samples) | 68.3% |
| After PPO (500 steps) | 71.2% |
| After GRPO (500 steps) | 72.7% |

> **Caveat — synthetic PRM labels.** The PRM is trained on *heuristic* labels
> derived automatically from gold GSM8K solutions (arithmetic annotation
> consistency, word-count proxy for clarity, step position for progress). It is
> **not** trained on human-verified incorrect traces such as
> [PRM800K](https://github.com/openai/prm800k). Reward-model quality is therefore
> a ceiling on RL improvement. Real gains require real labels.

---

**Architecture flow:**

```
input_ids
    │
    ▼
[Frozen base LLM]  (no gradients)
    │  last hidden state → last token embedding  [B, H_model]
    ▼
[Linear projection]  H_model → 768
    │
    ├──▶ correctness_head   → Linear(768→384) → ReLU → Linear(384→1) → Sigmoid
    ├──▶ math_validity_head → (same)
    ├──▶ clarity_head       → (same)
    └──▶ progress_head      → (same)
         │
         ▼
[Learnable softmax weights: 0.4, 0.3, 0.15, 0.15]
         │
         ▼
    final_reward  (scalar in [0, 1])
```

Only the projection layer, the four heads, and the four scalar weights are
trained. The base LLM is frozen throughout Stage 2.

**Loss:** sum of four independent MSE losses — one per head. The combined reward
used in PPO/GRPO is a learned weighted average (weights are also trained via
back-prop through `softmax`).

### Why four heads instead of one?

A single head must learn what "good" means across orthogonal dimensions at once.
Separating them lets each head specialise and makes the reward signal interpretable:
you can inspect which aspect is penalising a step. The weighted combination also
gives the RL stage a smoother, less sparse reward than a binary correct/incorrect
signal.

### Known Weaknesses

- The `clarity` and `progress` labels are completely synthetic and not validated.
  They act as regularisers more than true quality signals.
- Because all positive examples come from gold solutions, the PRM never sees a
  *wrong* step and cannot distinguish plausible-but-wrong reasoning from correct
  reasoning. This is the central limitation of training PRM on reference solutions
  alone.
- The learnable weights start near uniform (0.4/0.3/0.15/0.15) and may not
  converge meaningfully on small datasets.

---

All parameters are centralised in the `Config` dataclass at the top of `finetune.py`.

---