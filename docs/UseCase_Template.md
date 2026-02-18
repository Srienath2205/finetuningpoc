# Use-Case Template

**Use-case name:** <replace_me>
**Domain:** <e.g., telecom, legal, healthcare, hr>
**Owner:** <your_name>

## 1) Problem Statement

- What short text transformation/classification/generation is needed?
- Who are the end-users?
- Where will the model output go (UI, API, ticketing tool)?

## 2) Data

- Source (synthetic/curated)
- Volume (train/eval)
- Redaction / privacy steps

## 3) Output & Metrics

- Output format: free-text or JSON?
- Metrics: accuracy / schema_fidelity / BLEU / ROUGE / EM
- Success criteria (POC bar)

## 4) Model Choice

- Baseline size (2B–7B)
- Quantization mode (4-bit QLoRA on Colab)
- Why this size for this dataset?

## 5) Risks & Mitigations

- GPU availability (Colab free)
- Data quality
- Overfit on tiny datasets

## 6) Handoff Artifacts

- `adapters/` (LoRA/QLoRA)
- `metrics.json`
- `inference_demo.ipynb` (optional)
