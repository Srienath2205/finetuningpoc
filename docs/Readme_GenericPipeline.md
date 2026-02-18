# Generic Fine-Tuning Pipeline (Reusable 80%)

This repository contains a **generic, model-agnostic** instruction fine-tuning setup that runs on **Google Colab** (or Kaggle Notebooks). It standardizes:

- Config-driven model selection (Kaggle or Hugging Face)
- LoRA/QLoRA vs Full fine-tuning via `training_config.yaml`
- Chat-style SFT format with a universal `messages[]` schema
- Simple metrics with optional JSON schema fidelity checks
- Standardized adapter export layout

## What changes per use-case (20%)

- `/configs/usecase_config.yaml`
- The dataset in `/data/` (train/eval)
- Optional evaluation add-ons

## Quick steps

1. Edit YAMLs in `/configs`
2. Put JSONL datasets in `/data`
3. Run `notebooks/FineTune_Generic_Pipeline.ipynb` in Colab

## Output

- Saved LoRA/QLoRA adapters in `/adapters/<usecase_name>/`
- Optional `metrics.json`
