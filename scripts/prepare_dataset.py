"""
Generic dataset preparation & validation.

- Validates each JSONL record against the provided JSON schema.
- (Optional) Limits # of records for quick training spins.
- Returns file paths for train/eval (may write temp filtered copies).
"""

import json
from jsonschema import Draft202012Validator
from pathlib import Path
from typing import Optional, Tuple


def _load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _write_jsonl(records, path: Path):
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def validate_dataset(
    train_path: str,
    eval_path: str,
    schema_path: str,
    max_train: Optional[int] = None,
    max_eval: Optional[int] = None,
) -> Tuple[str, str]:
    train_p = Path(train_path)
    eval_p = Path(eval_path)
    schema_p = Path(schema_path)

    schema = json.loads(schema_p.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)

    def _validate_and_limit(p: Path, max_n: Optional[int]):
        filtered = []
        for i, rec in enumerate(_load_jsonl(p)):
            errors = list(validator.iter_errors(rec))
            if errors:
                # Skip invalid records but print the first error for visibility
                print(
                    f"[WARN] Skipping invalid record at {p.name}:{i}: {errors[0].message}"
                )
                continue
            filtered.append(rec)
            if max_n and len(filtered) >= max_n:
                break
        return filtered

    train_valid = _validate_and_limit(train_p, max_train)
    eval_valid = _validate_and_limit(eval_p, max_eval)

    out_train = train_p.with_name("train.valid.jsonl")
    out_eval = eval_p.with_name("eval.valid.jsonl")

    _write_jsonl(train_valid, out_train)
    _write_jsonl(eval_valid, out_eval)

    print(f"[OK] Validated: {len(train_valid)} train, {len(eval_valid)} eval")
    return str(out_train), str(out_eval)
