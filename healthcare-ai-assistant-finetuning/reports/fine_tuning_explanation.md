# Step 11 — LoRA, QLoRA, Non-Instruction FT, SFT, and DPO/ORPO Explained

A short, plain-language explanation of the techniques used in this project.

## Why full fine-tuning is expensive
Full fine-tuning updates **every weight** in the model. For even a small 0.5B–1B model that
means storing the weights, their gradients, and optimizer state (e.g. Adam keeps two extra
values per weight) all in GPU memory at once — roughly 4× the model size or more. It also
produces a full-size copy of the model for every task you train. On a free/limited GPU this is
slow, memory-hungry, and often simply won't fit.

## What LoRA does
**LoRA (Low-Rank Adaptation)** freezes the original model weights and inserts small pairs of
trainable matrices (A and B) into selected layers. Instead of updating a big weight matrix `W`,
it learns a low-rank update `ΔW = A·B`, where A and B are tiny (controlled by the **rank** `r`).
Only these adapters are trained — typically **<1% of the parameters** — so training is fast and
the result is a small adapter file you can swap in and out.

## What QLoRA does
**QLoRA** = **Quantized LoRA**. It first loads the frozen base model in **4-bit** precision
(instead of 16-bit), which cuts the model's memory footprint by about 4×, and then trains LoRA
adapters on top of that quantized model. The base stays quantized and frozen; only the small
LoRA adapters (kept in higher precision) are updated.

## Why QLoRA is useful on a limited GPU
Because the large base model sits in 4-bit and only the tiny adapters are trained, QLoRA lets
you fine-tune models that would never fit in full precision on a free T4 (16 GB) Colab GPU. You
get most of the quality of full fine-tuning at a fraction of the memory and time — which is
exactly why this project uses it for all three stages.

## What is non-instruction fine-tuning?
Also called **continued pre-training**. You train the model on **raw domain text** (no
questions or answers) using the same next-token objective it was originally trained with. The
goal is for the model to absorb the **vocabulary, terminology, facts, and writing style** of the
domain — here, healthcare. It does **not** teach the model to follow instructions; it builds
domain knowledge that later stages build on. (See `notebooks/non_instruction_finetuning.ipynb`.)

## What is instruction fine-tuning (SFT)?
**Supervised Fine-Tuning** trains the model on **instruction → response** pairs so it learns to
*answer questions* in the desired format and tone. Each example shows the model a user question
and the ideal answer, formatted with the chat template. After SFT the model behaves like an
assistant for the domain. (See `notebooks/instruction_finetuning.ipynb`.)

## What is DPO?
**Direct Preference Optimization** aligns the model using **preference data**: for each prompt
it sees a **chosen** (better) answer and a **rejected** (worse) answer. DPO directly increases
the relative likelihood of the chosen answer over the rejected one, without needing a separate
reward model or reinforcement-learning loop (the way classic RLHF does). It nudges the model
toward safer, more helpful, more professional responses. (See `notebooks/dpo_alignment.ipynb`.)

**ORPO** (Odds Ratio Preference Optimization) is a related alternative that combines the SFT and
preference objectives into a **single** training step (no separate SFT stage and no reference
model needed). This project uses **DPO**, but ORPO would be a valid substitute for Stage 3.

## Difference between SFT and DPO
| | SFT | DPO |
|---|-----|-----|
| Data | instruction → response (one good answer) | prompt + chosen + rejected (a comparison) |
| Teaches | *how to answer* | *which answer is better* |
| Signal | imitate the reference answer | prefer chosen over rejected |
| Typical LR | higher (e.g. 2e-4) | much lower (e.g. 5e-6) |
| Role here | builds the assistant | aligns it to be safer/more professional |

In short: **SFT teaches the model to answer; DPO teaches it to answer *better*.**

## Hyperparameters used in this project
These are set in the notebooks; tune them for your model/GPU.

| Hyperparameter | Stage 1 (Non-Instruction) | Stage 2 (SFT) | Stage 3 (DPO) |
|----------------|---------------------------|---------------|---------------|
| Rank (`r`) | 16 | 16 | 16 |
| Alpha (`lora_alpha`) | 16 | 16 | 16 |
| Dropout (`lora_dropout`) | 0 | 0 | 0 |
| Learning rate | 5e-5 (embeddings 5e-6) | 2e-4 | 5e-6 |
| Batch size (per device) | 2 | 2 | 2 |
| Grad accumulation | 4 (effective 8) | 4 (effective 8) | 4 (effective 8) |
| Epochs | 2 | 3 | 1 |
| Quantization | 4-bit (QLoRA) | 4-bit (QLoRA) | 4-bit (QLoRA) |
| Optimizer | adamw_8bit | adamw_8bit | adamw_8bit |
| DPO beta | — | — | 0.1 |

## Measured results from the actual Colab runs

The values above are the *settings*; these are what the runs actually produced (base model
`unsloth/Qwen2.5-0.5B`, 4-bit QLoRA):

| Stage | Examples | Steps × Epochs | Trainable params | Final train loss | Runtime | GPU |
|-------|----------|----------------|------------------|------------------|---------|-----|
| 1 — Non-Instruction FT | 56 paragraphs | 21 × 3 | 147,378,176 / 777,545,600 (18.95%) | **1.9265** | 82 s | Tesla T4 |
| 2 — SFT | 112 Q&A | 42 × 3 | 8,798,208 / 502,830,976 (1.75%) | **1.3746** | 109 s | Tesla T4 |
| 3 — DPO | 56 triples | 7 × 1 | 8,798,208 / 502,830,976 (1.75%) | **0.6701** | 31 s | Tesla T4 |

- The **18.95%** trainable share in Stage 1 (vs 1.75% in Stages 2–3) is because continued
  pre-training also adapts `embed_tokens` and `lm_head`, not just the attention/MLP LoRA — this
  is what lets the model absorb domain vocabulary.
- **Loss falls monotonically** (1.93 → 1.37 → 0.67). Note the DPO loss (~0.67) is **not**
  comparable to the SFT loss: DPO optimises a preference (logistic) objective, not next-token
  cross-entropy, so its scale is different by construction.
- Each stage saved a LoRA adapter + a merged 16-bit model; only Stage 1 was pushed to the Hugging
  Face Hub (`mannyiyer/healthcare-faq-qwen2.5-0.5b-stage1[-merged]`).

**Why these values**
- **rank = alpha = 16** — a common, balanced setting; alpha ≈ rank keeps the effective LoRA
  scaling (`alpha/r`) at 1.0, a stable default for small models.
- **dropout = 0** — Unsloth's optimized path is fastest with no LoRA dropout, and our datasets
  are clean, so heavy regularization isn't needed.
- **LR 2e-4 for SFT** — standard LoRA SFT learning rate; high enough to learn quickly on a small
  adapter.
- **LR 5e-6 for DPO** — preference alignment is sensitive, so a much smaller LR prevents the
  model from drifting away from its SFT behaviour.
- **beta = 0.1** — controls how strongly DPO trusts the preferences; 0.1 is the common default.
- **batch size 2 × grad-accum 4** — fits comfortably on a free T4 GPU while giving an effective
  batch of 8.
