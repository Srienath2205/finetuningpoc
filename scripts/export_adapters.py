"""
Adapter export helper.

- Saves PEFT adapters to /adapters/<usecase_name>/
- Optionally writes a small metadata.json for traceability.
"""

from typing import Optional, Dict, Any
import json
from pathlib import Path
from datetime import datetime


def export_adapters(
    model,
    usecase_name: str,
    extra_meta: Optional[Dict[str, Any]] = None,
    base_dir: str = "/content/adapters",
) -> None:
    out_dir = Path(base_dir) / usecase_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save PEFT adapters (works for LoRA/QLoRA)
    model.save_pretrained(str(out_dir))

    meta: Dict[str, Any] = {
        "usecase_name": usecase_name,
        "exported_at_utc": datetime.utcnow().isoformat() + "Z",
    }
    if extra_meta:
        meta.update(extra_meta)

    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] Adapters exported to: {out_dir}")
