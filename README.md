# Generic LLM Fine‑Tuning Pipeline (Colab‑First)

This repository is a **generic, reusable fine‑tuning framework** for small/medium LLMs (Gemma / Llama / Mistral, etc.).
It is designed to be **edited in VS Code** and **executed in Google Colab**. The pipeline supports **LoRA / QLoRA / full fine‑tuning**
and expects a **message‑style JSONL dataset** that generalizes across use cases.

---

## Why Colab‑first?

- Colab provides a **free/low‑cost GPU** (T4) suited to LoRA/QLoRA.
- We fetch open models **from Kaggle Models** via `kagglehub`, which is officially supported for Gemma and avoids local GPU/runtime issues.

> Tip: If your organization blocks Colab on your work account, sign in with a **personal Gmail** OR run the notebook on **Kaggle Notebooks**.

---

## Folder Layout

```
PROJECT_ROOT/
├─ configs/
│   ├─ model_config.yaml         # model family & source (kaggle / hf), quantization
│   ├─ training_config.yaml      # lora/qlora/full; epochs, batch size, LR, LoRA ranks etc.
│   ├─ dataset_config.yaml       # train/eval paths relative to /content; schema hints
│   └─ usecase_config.yaml       # 20% domain specifics (usecase name, output format, notes)
│
├─ data/
│   ├─ train.jsonl               # message-based SFT format
│   ├─ eval.jsonl
│   └─ schema.json               # optional JSON schema for assistant outputs (e.g., strict JSON)
│
├─ docs/
│   ├─ UseCase_Template.md
│   ├─ QuickSetup.md
│   └─ Readme_GenericPipeline.md
│
├─ notebooks/
│   └─ FineTune_Generic_Pipeline.ipynb  # Colab execution entry point
│
├─ scripts/
│   ├─ prepare_dataset.py        # validates / light transforms; enforces message schema
│   ├─ synth_data_template.py    # optional synthetic data bootstrapping
│   ├─ export_adapters.py        # saves adapters to /adapters/<usecase>/
│   └─ load_model_generic.py     # loads any Kaggle/HF model with chosen quantization
│
├─ adapters/                     # output adapters (gitignored)
├─ requirements.txt              # dev-only (VS Code lint/format)
├─ .gitignore
└─ README.md (this file)
```

---

## Quick Run (Colab)

1. **Zip this project** on your laptop (from the project root):
   - Windows: select all → right‑click → Send to → Compressed (zipped) folder.
   - macOS/Linux: `zip -r project.zip .`

2. Open **Colab** with a personal Gmail (recommended) → https://colab.research.google.com

3. Click **File → Upload notebook** → upload `notebooks/FineTune_Generic_Pipeline.ipynb`.

4. In the first cell of the notebook, **upload your `project.zip`** when prompted. The notebook will unzip to `/content/project`.

5. If you are using **Kaggle Models** (recommended for Gemma), create a Kaggle API key (Kaggle → Profile → Settings → _Create API Token_)
   and add two Colab **secrets**:
   - `KAGGLE_USERNAME`
   - `KAGGLE_KEY`

6. Set **Runtime → Change runtime type → GPU (T4)** and **Run all**.

> The notebook installs: `kagglehub`, `transformers`, `peft`, `bitsandbytes`, `accelerate`, `datasets`, `pyyaml`.

---

## Configuration

- `configs/model_config.yaml` : select the base model and source.
- `configs/training_config.yaml` : choose `method: lora|qlora|full`, LoRA ranks, learning rate, etc.
- `configs/dataset_config.yaml` : set `/content`-relative paths to `train.jsonl` and `eval.jsonl`.
- `configs/usecase_config.yaml` : set `usecase_name`, `domain`, `output_format` (e.g., `json`), and optional notes.

---

## Dataset Format (Generic, Reusable)

Each line in `train.jsonl`/`eval.jsonl` is a JSON record:

```json
{
  "messages": [
    { "role": "user", "content": "<user text>" },
    { "role": "assistant", "content": "<assistant text or JSON string>" }
  ]
}
```

This format works for classification, generation, or structured JSON outputs. The trainer will format messages into a single prompt+target.

---

## Exported Artifacts

- LoRA/QLoRA adapters are exported to `/content/adapters/<usecase_name>/` with a `metadata.json` for traceability.

---

## VS Code Notes

- This repo is **edited locally** in VS Code but **executed on Colab**.
- If Pylance flags missing imports for `torch`, `transformers`, or `kagglehub`, that’s expected locally. You can either:
  - ignore diagnostics (recommended, no local install), or
  - `pip install -r requirements.txt` (heavier), **not required** for execution.

---

## Security / Compliance

- Do not upload company‑confidential data into Colab.
- Use **synthetic or anonymized** datasets for POC.

---

## Troubleshooting

- If Colab "Upload notebook" is blocked by your organization, use a **personal Gmail** or run on **Kaggle Notebooks** instead.
- If `kagglehub` fails because no secrets are set, re‑add `KAGGLE_USERNAME` and `KAGGLE_KEY` as Colab secrets.
