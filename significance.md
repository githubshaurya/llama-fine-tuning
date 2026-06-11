# Pipeline Significance — RLHF on GSM8K

An explanation of what each stage does, why it matters, and why the pipeline is structured the way it is.

---

## 1. SFT — Supervised Fine-Tuning

The base `Llama-3.2-3B` model knows mathematics but has no idea how to format a step-by-step solution, answer in the expected `#### <number>` form, or stay on-task for multi-step reasoning. It solves roughly 26% of GSM8K problems out of the box under greedy decoding not because it can't do arithmetic, but because it produces free-form text that often doesn't land on a parseable answer.

SFT fine-tunes the model directly on question-answer pairs from the GSM8K training split. It teaches:

- **Format**: how to structure a chain-of-thought solution and terminate with `####`.
- **Reasoning style**: decomposing a word problem into numbered arithmetic steps.
- **Stability**: a well-SFT'd model is a stable starting point for RL — skipping SFT and running RL directly on a base model typically causes training collapse.

The SFT stage alone drives the biggest gain in this pipeline (+50pp), which reflects how much of the base model's failure is formatting and task-alignment rather than mathematical incapability.

LoRA adapters (rank 16, α=32) are used instead of full fine-tuning. This reduces trainable parameters from ~3B to ~20M while retaining most of the representational capacity of the base model. Combined with 4-bit NF4 quantisation, the entire fine-tuning run fits on a single consumer GPU.

---

## 2. PRM — Process Reward Model

A standard reward model scores a complete answer as correct or incorrect. This is called an **Outcome Reward Model (ORM)**. ORMs have a fundamental problem for mathematics: a model can reach the right final answer through flawed reasoning (lucky cancellation of errors), or produce a valid reasoning chain that makes an arithmetic slip at the last step and is marked entirely wrong.

A **Process Reward Model** assigns a scalar reward to each intermediate reasoning step, not just the final answer. This pipeline uses a multi-aspect PRM with four specialised reward heads:

| Head              | What it scores                                          |
|-------------------|---------------------------------------------------------|
| Correctness       | Is this step logically correct given prior steps?       |
| Math validity     | Is the arithmetic or algebraic operation valid?         |
| Clarity           | Is the step expressed unambiguously?                    |
| Progress          | Does the step move meaningfully toward a solution?      |

The four scores are combined via a learned weighted sum. During RL (stages 3A/3B), these per-step rewards are what the policy is optimised against, not just a binary right/wrong signal on the final answer.

The PRM is trained on top of the SFT model, not the base model. This matters: the SFT model already produces structured step-by-step solutions, which gives the PRM well-formed input to score. Training a PRM on base model output would be substantially noisier.

---

## 3. PPO — Proximal Policy Optimisation

PPO is the standard RL algorithm for language model alignment. The core idea: treat the model as a policy that selects tokens, use the PRM to produce rewards at each reasoning step, and update the policy to make high-reward sequences more likely — but **not too aggressively**, to avoid catastrophic forgetting or reward hacking.

The "proximal" constraint is a clipping ratio (ε = 0.2 here) that prevents any single update from changing the policy too far from where it was. Without this, unconstrained policy gradient updates cause the model to collapse onto degenerate outputs that maximise the reward signal in unintended ways.

PPO requires a **value model** — a separate network trained to estimate the expected future return from any given state — to compute advantages. In this pipeline, a second copy of the language model is loaded as the value model. This doubles VRAM usage for stage 3A.

The practical consequence: PPO is well-understood and stable, but expensive. For a 3B model it is manageable; for larger models the value model becomes a serious cost.

---

## 4. GRPO — Group Relative Policy Optimisation

GRPO was introduced by DeepSeek (used in DeepSeek-R1) and removes the value model entirely. Instead of estimating a value function, it generates a **group of G responses** to the same prompt, scores all of them with the PRM, and computes advantages as the normalised deviation of each response's reward from the group mean:

```
advantage_i = (reward_i − mean(rewards)) / std(rewards)
```

This is intuitive: a response is "good" if it outperforms the other G responses to the same prompt, and "bad" if it underperforms. No separate value model is needed because the group itself provides the baseline.

Benefits over PPO in this setting:

- **Half the VRAM**: no value model loaded.
- **Better credit assignment for reasoning**: advantage is computed relative to the difficulty of each specific problem, not a global value estimate.
- **Stronger empirical performance on math**: GRPO-trained models show better solve rates in published work on GSM8K and MATH benchmarks.

The tradeoff is that GRPO is less theoretically analysed than PPO and can be sensitive to group size G and reward normalisation.

---

## 5. Why Both PPO and GRPO?

Running both stages serves two purposes.

**Comparison**: PPO and GRPO make different assumptions and have different compute profiles. Running both on the same SFT checkpoint with the same PRM produces directly comparable results (71.2% vs 72.7% solve rate here), giving an empirical answer to "which is better for this model and task?" rather than relying on published results from different setups.

**Complementarity**: In a production setting, PPO's stability and theoretical grounding make it a reliable baseline. GRPO's memory efficiency and stronger reasoning performance make it the candidate for deployment. Having both results lets you make an informed choice.

---

## 6. Why This Sequence: SFT → PRM → PPO/GRPO?

Each stage depends on the output of the one before it.

**SFT must come first.** RL on a base language model is extremely unstable — the reward signal is too sparse relative to the enormous action space (vocabulary × sequence length). SFT collapses this problem: it gives the policy a sensible initialisation that already produces structured solutions, so the RL stage only needs to refine quality rather than discover the task from scratch.

**PRM must come after SFT.** The PRM is trained on solutions produced by the SFT model. If it were trained on base-model output (incoherent, unstructured), the PRM would learn to score gibberish and would provide a misleading reward signal during RL. SFT output is clean enough for the PRM to learn meaningful distinctions between good and bad reasoning steps.

**PPO/GRPO must come after PRM.** The RL stages are entirely dependent on the PRM for their reward signal. Without a trained PRM, the only available signal would be final-answer binary correctness — an ORM signal that is too sparse for stable training on a 3B model within 500 steps.

The sequence is not arbitrary; each stage builds on the stability and signal quality established by the prior one.

---

## 7. Why GSM8K?

GSM8K (Grade School Math 8K) is the standard benchmark for evaluating mathematical reasoning in language models for several practical reasons:

- **Verifiable**: every answer is a single number extractable by a regex. There is no ambiguity in scoring.
- **Multi-step**: problems require 2-8 arithmetic steps, making them long enough to benefit from step-level reward but short enough to fit within a 512-token context.
- **Well-calibrated difficulty**: the 3B model size range sits comfortably in the 20-85% performance band across training stages, which means the benchmark is neither saturated nor trivially hard and provides a meaningful training signal throughout.
- **Standard**: results are directly comparable to published work on Llama, DeepSeek, Qwen, and other open-weight models.

---

## 8. Evaluation Caveats

- **Sample size**: both `evaluate()` and `evaluate_accuracy()` use 200 held-out problems. At n=200 and ~77% solve rate, the standard error is ≈ ±3pp. Differences of less than 2pp between stages should not be over-interpreted.
- **Greedy decoding**: accuracy is measured with `do_sample=False` (greedy). Pass@k or majority-vote metrics would likely show higher absolute numbers.
- **Eval loss vs solve rate**: eval loss (cross-entropy on tokenised solutions) and solve rate (exact-match on final answers) measure different things. A model can have rising eval loss during RL (it diverges from the SFT distribution) while still improving solve rate (it finds better solution paths). Both metrics together give a fuller picture than either alone.