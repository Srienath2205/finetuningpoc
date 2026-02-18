"""
Synthetic data template.

Use this to bootstrap small SFT datasets quickly.
Customize `SYN_SEEDS` and generation logic as needed.
"""

import json
import random
from pathlib import Path

SYN_SEEDS = [
    ("example user prompt 1", "example assistant answer 1"),
    ("example user prompt 2", "example assistant answer 2"),
]


def make_pair(user_txt: str, assistant_txt: str):
    return {
        "messages": [
            {"role": "user", "content": user_txt},
            {"role": "assistant", "content": assistant_txt},
        ]
    }


def build_synth_dataset(n: int = 100, out_path: str = "data/train.synth.jsonl"):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for _ in range(n):
            u, a = random.choice(SYN_SEEDS)
            rec = make_pair(u, a)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[OK] Wrote synthetic dataset: {out_path}")
