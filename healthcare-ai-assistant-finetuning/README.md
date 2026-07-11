# Healthcare FAQ Assistant — Domain-Specific LLM Fine-Tuning with Unsloth

A practical, end-to-end fine-tuning project that turns a small open-source LLM into a
**Healthcare FAQ Assistant** using [Unsloth](https://github.com/unslothai/unsloth).
The model is adapted through three training stages and then evaluated against the base
model at every stage.

```
Base Model
   ↓  Stage 1: Non-Instruction Fine-Tuning   (learn healthcare language & terminology)
   ↓  Stage 2: Instruction Fine-Tuning (SFT)  (learn to answer health questions)
   ↓  Stage 3: DPO Preference Alignment       (prefer safe, helpful, professional answers)
Final Healthcare FAQ Assistant
```

> ⚠️ **Educational project — not medical advice.** Every answer is general health
> information only and consistently directs users to consult a qualified healthcare
> professional. The datasets are written with this safety framing on purpose.

---

## 1. Project title
**Healthcare FAQ Assistant** — a domain-specific AI assistant fine-tuned with Unsloth.

## 2. Domain selected
**Healthcare FAQ.** The assistant answers general, patient-style questions about topics
such as fever, cold and flu, hydration, blood pressure, diabetes, cholesterol,
vaccinations, sleep, nutrition, exercise, mental health, common over-the-counter
medications, allergies, asthma, headaches, first aid, preventive screenings, and
"when should I see a doctor".

## 3. Business problem
A healthcare provider wants an internal assistant that answers common patient questions
clearly, uses correct medical terminology, gives **safe** general guidance, and always
recommends professional care when appropriate — instead of the generic, sometimes unsafe
answers a base model produces.

## 4. Dataset details
All three datasets share one **coherent general-health knowledge base** so that SFT and
DPO have a consistent target and the before/after evaluation is meaningful.

| File | Purpose | Size |
|------|---------|------|
| [`data/non_instruction_data.txt`](data/non_instruction_data.txt) | Raw domain text for non-instruction FT | 56 paragraphs |
| [`data/instruction_dataset.jsonl`](data/instruction_dataset.jsonl) | Instruction → response pairs for SFT | 112 examples |
| [`data/preference_dataset.jsonl`](data/preference_dataset.jsonl) | prompt / chosen / rejected for DPO | 56 examples |

Each dataset was AI-assisted, then cleaned and verified for safety and consistency.

## 5. Base model used
Selectable at the top of every notebook via a single `MODEL_NAME` variable. Default:
**`unsloth/Qwen2.5-0.5B`**. Other supported options: Llama-3.2-1B, Qwen2.5-1.5B,
TinyLlama-1.1B, Gemma-2-2B.

```python
MODEL_OPTIONS = {
    "qwen2.5-0.5b":  "unsloth/Qwen2.5-0.5B",
    "llama-3.2-1b":  "unsloth/Llama-3.2-1B",
    "qwen2.5-1.5b":  "unsloth/Qwen2.5-1.5B",
    "tinyllama-1.1b":"unsloth/tinyllama",
    "gemma-2-2b":    "unsloth/gemma-2-2b",
}
MODEL_NAME = "qwen2.5-0.5b"   # ← change this one line to switch models
```

## 6. Non-instruction fine-tuning approach
[`notebooks/non_instruction_finetuning.ipynb`](notebooks/non_instruction_finetuning.ipynb) —
load raw domain text, clean & chunk it, load the base model with Unsloth, apply QLoRA, and
continue-pretrain on raw healthcare text so the model absorbs domain language before SFT.

## 7. Instruction fine-tuning approach
[`notebooks/instruction_finetuning.ipynb`](notebooks/instruction_finetuning.ipynb) —
start from the Stage-1 adapter, format the instruction dataset into a chat template, apply
QLoRA, and train with TRL's `SFTTrainer` so the model learns to answer questions.

## 8. DPO alignment approach
[`notebooks/dpo_alignment.ipynb`](notebooks/dpo_alignment.ipynb) —
load the SFT model, feed prompt/chosen/rejected triples to TRL's `DPOTrainer` so the model
learns to prefer safe, helpful, professional, domain-specific answers.

## 9. LoRA / QLoRA configuration
| Hyperparameter | Value |
|----------------|-------|
| Rank (`r`) | 16 |
| Alpha (`lora_alpha`) | 16 |
| Dropout | 0 |
| Quantization | 4-bit (QLoRA) |
| Learning rate (SFT) | 2e-4 |
| Learning rate (DPO) | 5e-6 |
| Batch size | 2 (×4 grad-accum = effective 8) |
| Target modules | q,k,v,o,gate,up,down proj |

See [`reports/fine_tuning_explanation.md`](reports/fine_tuning_explanation.md) for the full write-up.

## 10. Training screenshots / logs
All three stages were run end-to-end on Google Colab (base model `unsloth/Qwen2.5-0.5B`, 4-bit
QLoRA). The run notebooks with full output are in
[`notebooks/From-colab/`](notebooks/From-colab/). Measured training metrics:

| Stage | Data | Steps × Epochs | Effective batch | Trainable params | Final train loss | Runtime | GPU |
|-------|------|----------------|-----------------|------------------|------------------|---------|-----|
| 1 — Non-Instruction FT | 56 paragraphs | 21 × 3 | 8 | 147,378,176 / 777,545,600 (**18.95%**)¹ | **1.9583** | 66 s | NVIDIA L4 |
| 2 — Instruction FT (SFT) | 112 Q&A pairs | 42 × 3 | 8 | 8,798,208 / 502,830,976 (**1.75%**) | **1.3212** | 85 s | Tesla T4 |
| 3 — DPO Alignment | 56 preference triples | 7 × 1 | 8 | 8,798,208 / 502,830,976 (**1.75%**) | **0.6952** | 32 s | NVIDIA L4 |

¹ Stage 1 also trains `embed_tokens` and `lm_head` (the Unsloth continued-pre-training recipe),
which is why its trainable-parameter share is much higher than the pure-attention/MLP LoRA used
in Stages 2–3.

The training loss falls monotonically across the pipeline (1.96 → 1.32 → 0.70), consistent with
the model first learning domain language, then learning to answer, then sharpening its
preferences. Each stage saved a LoRA adapter plus a merged 16-bit model to
`outputs/stageN_*` on Google Drive. The **Stage-1** model was also pushed to the Hugging Face
Hub (`mannyiyer/healthcare-faq-qwen2.5-0.5b-stage1` and `…-stage1-merged`); Stages 2–3 were kept
local/Drive only (`PUSH_TO_HUB = False` in those runs).

> Note on scale: this is a **0.5B demonstration** model trained on small (50–110 example)
> datasets for a few dozen steps — enough to clearly show the Base → SFT → DPO progression, but
> not a production-grade medical model. See *Challenges* and *Future improvements* below.

## 11. Before vs after output comparison
The evaluation reports below contain the **actual, verbatim** answers captured from the Colab
runs (greedy decoding) for the questions that were exercised, clearly marked apart from the
illustrative rows that were not re-run in this pass:
- [`reports/base_model_evaluation.md`](reports/base_model_evaluation.md) — base model weaknesses
- [`reports/sft_model_comparison.md`](reports/sft_model_comparison.md) — base vs SFT
- [`reports/final_evaluation.md`](reports/final_evaluation.md) — base vs SFT vs DPO
- [`reports/fine_tuning_explanation.md`](reports/fine_tuning_explanation.md) — LoRA/QLoRA/SFT/DPO write-up + measured metrics

## 12. Final observations
- **Non-instruction FT (Stage 1)** taught domain *style* but, as expected, not Q&A behaviour:
  completions still drifted into the base model's exam-style / multiple-choice patterns. Its job
  was only to warm up the domain vocabulary before SFT — which it did (loss 1.96).
- **SFT (Stage 2)** is where the model became an *assistant*. With greedy decoding it now answers
  the healthcare question directly, uses correct thresholds (e.g. fever ≈ 38 °C / 100.4 °F), and
  no longer rambles the way the base model does. This is the single biggest quality jump.
- **DPO (Stage 3)** sharpened *safety and tone* on high-risk questions. The clearest win is
  blood-pressure medication: the DPO model firmly says *don't stop without consulting a
  clinician*, whereas the base model suggested deciding "by feel."
- **Honest limitation:** at 0.5B the aligned model is still imperfect. On the 2-month-old-fever
  question the DPO answer was self-contradictory ("a fever is a sign that your baby is
  healthy"), which is exactly the kind of failure a larger model and a bigger preference set
  would fix. We report it as-is rather than hide it.
- **Overall:** training loss and qualitative behaviour both improve monotonically
  Base → SFT → DPO, and the DPO model is the one shipped as the final assistant
  (`outputs/stage3_merged`, used by `src/inference.py`).

## Challenges faced
- **Library/version churn.** Newer `transformers` removed the `tokenizer=` argument to
  `Trainer.__init__()`; pinning an old `trl` broke the run. Fix: let Unsloth install its own
  compatible stack and use `processing_class=` instead of `tokenizer=`.
- **Tokenizer/padding warnings.** Qwen2.5 base ships without a pad token and (after merging) with
  a mismatched tokenizer regex, producing repeated warnings; harmless here but noisy.
- **Model capacity.** A 0.5B model on tiny datasets is prone to hallucinated specifics and
  self-contradiction (see the infant-fever answer). More data / a larger base is the real fix.
- **Persisting outputs across Colab sessions.** Because each stage feeds the next, we mounted
  Google Drive so `outputs/stage1_merged` → `stage2_merged` → `stage3_merged` survive reconnects.
- **Non-deterministic spot checks.** The intermediate comparison cells used sampling
  (`temperature=0.7`), so their answers vary run to run; the canonical evaluation answers use
  greedy decoding (`do_sample=False`).

## Future improvements
- Scale up: use Qwen2.5-1.5B or Llama-3.2-1B and 500+ instruction / 200+ preference examples.
- Run **all 10** evaluation questions through **all three** models in one deterministic pass and
  auto-generate the comparison tables, instead of the current subset.
- Add an automatic **LLM-as-judge** or rubric scorer (correctness/safety/clarity) for objective
  win counts rather than manual judgement.
- Hold out a validation split to track eval loss and catch over-fitting on the small datasets.
- Expand the preference set toward the residual failure modes (infant fever, dosing, emergencies)
  and consider ORPO as a single-stage alternative to SFT+DPO.
- Push the final Stage-2/Stage-3 models to the Hub for easy reload and add a small Gradio demo.

---

## Repository structure
```
healthcare-ai-assistant-finetuning/
├── data/
│   ├── non_instruction_data.txt
│   ├── instruction_dataset.jsonl
│   └── preference_dataset.jsonl
├── notebooks/
│   ├── non_instruction_finetuning.ipynb
│   ├── instruction_finetuning.ipynb
│   └── dpo_alignment.ipynb
├── reports/
│   ├── base_model_evaluation.md
│   ├── sft_model_comparison.md
│   ├── final_evaluation.md
│   └── fine_tuning_explanation.md
├── src/
│   └── inference.py
├── README.md
└── requirements.txt
```

## How to run (Google Colab)
1. Upload this folder to Google Drive (or `git clone` it in Colab).
2. Open `notebooks/non_instruction_finetuning.ipynb`, set Runtime → GPU (T4).
3. Run all cells. It saves a LoRA adapter to `outputs/stage1_non_instruction` and a merged
   model to `outputs/stage1_merged`.
4. Run `instruction_finetuning.ipynb` (set `RESUME_FROM_STAGE1 = True`) → `outputs/stage2_*`.
5. Run `dpo_alignment.ipynb` → `outputs/stage3_*` (the final model).
6. Use `src/inference.py` to chat with the final model.

## Pushing / loading models on Hugging Face Hub
Each notebook has an **optional Hugging Face** section. To enable it, set these in the
`HF_CONFIG` cell near the top:
```python
PUSH_TO_HUB = True
HF_USERNAME = "your-hf-username"
HF_TOKEN    = "hf_..."        # a WRITE token from https://huggingface.co/settings/tokens
```
After training, each notebook pushes both the **LoRA adapter** and a **merged 16-bit** model,
named like:
```
your-hf-username/healthcare-faq-qwen2.5-0.5b-stage1
your-hf-username/healthcare-faq-qwen2.5-0.5b-stage2-sft-merged
your-hf-username/healthcare-faq-qwen2.5-0.5b-stage3-dpo-merged   # final model
```
Reload any of them later without retraining (see the "Reload from the Hub" cell in each
notebook), e.g. with Unsloth or plain Transformers.

## Running inference
```bash
# Single question against the local final (DPO) model:
python src/inference.py -q "How can I apply for reimbursement?"

# Against a Hugging Face repo:
python src/inference.py -m your-hf-username/healthcare-faq-qwen2.5-0.5b-stage3-dpo-merged \
                        -q "What temperature is considered a fever?"

# Interactive chat:
python src/inference.py
```
`inference.py` uses plain Transformers, so it runs on CPU or GPU and does not require Unsloth.
