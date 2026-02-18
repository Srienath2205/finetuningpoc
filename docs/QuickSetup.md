# Quick Setup (Generic Pipeline)

## 0) Local authoring (VS Code)

- Keep this repository structure intact.
- Edit YAML configs in `/configs`.
- Place `train.jsonl`, `eval.jsonl`, `schema.json` in `/data`.

## 1) Open Colab (or Kaggle Notebooks)

> If your organization blocks Colab Drive APIs, use a **personal Gmail** on Colab or switch to **Kaggle Notebooks**.

### Colab

- Go to https://colab.research.google.com
- Create a new notebook
- Drag–drop folders `/configs`, `/data`, `/scripts` and `notebooks/FineTune_Generic_Pipeline.ipynb` into the left **Files** pane.
- Open `notebooks/FineTune_Generic_Pipeline.ipynb` and run top → bottom.

### Kaggle Notebooks (Alternative)

- Go to https://www.kaggle.com/code
- New Notebook → **Upload Notebook** → select `FineTune_Generic_Pipeline.ipynb`
- Upload `/configs`, `/data`, `/scripts`
- **Settings → Accelerator → GPU**

## 2) Add Kaggle credentials (only if model_source: kaggle)

In Colab:

- Left pane → 🔐 **Secrets**:
  - `KAGGLE_USERNAME`
  - `KAGGLE_KEY`

## 3) Run the Notebook

- Installs deps
- Loads configs
- Validates data
- Downloads model (Kaggle or HF)
- Applies LoRA/QLoRA
- Trains & evaluates
- Exports adapters to `/adapters/<usecase_name>/`

## 4) Artifacts

- `/adapters/<usecase_name>/`
- `/output/metrics.json` (if produced)
