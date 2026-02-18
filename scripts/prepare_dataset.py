import sys, os, textwrap, importlib.util, inspect

SCRIPTS_DIR = "/content/project/scripts"
MODULE_PATH = os.path.join(SCRIPTS_DIR, "prepare_dataset.py")
assert os.path.isdir(SCRIPTS_DIR), f"Scripts folder not found: {SCRIPTS_DIR}"

# 1) Make sure scripts dir is importable
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

# 2) Good, clean source for prepare_dataset.py (no HTML entities, correct typing)
GOOD_SRC = textwrap.dedent("""
    \"\"\"
    prepare_dataset.py
    Generic dataset validator / light normalizer for message-based SFT.

    Each JSONL line must be:
    {
      "messages": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
      ]
    }
    \"\"\"

    from typing import Iterable, Dict, Any, Tuple
    from pathlib import Path
    import json

    REQUIRED_ROLES: Tuple[str, str] = ("user", "assistant")

    def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield i, json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"[{path}] JSON parse error on line {i}: {e}") from e

    def _validate_messages_struct(rec: Dict[str, Any], line_no: int, path: Path) -> None:
        if "messages" not in rec or not isinstance(rec["messages"], list):
            raise ValueError(f"[{path}] line {line_no}: missing or invalid 'messages' list")

        roles_present = set()
        for m in rec["messages"]:
            if not isinstance(m, dict):
                raise ValueError(f"[{path}] line {line_no}: each message must be an object")
            if "role" not in m or "content" not in m:
                raise ValueError(f"[{path}] line {line_no}: each message must have 'role' and 'content'")
            if not isinstance(m["role"], str) or not isinstance(m["content"], str):
                raise ValueError(f"[{path}] line {line_no}: 'role' and 'content' must be strings")
            roles_present.add(m["role"])

        # Require at least one user and one assistant message
        for r in REQUIRED_ROLES:
            if r not in roles_present:
                raise ValueError(f"[{path}] line {line_no}: required role '{r}' not found in 'messages'")

    def validate_or_raise(path_str: str) -> None:
        \"\"\"Validates the JSONL dataset; raises on first error.\"\"\"
        p = Path(path_str)
        if not p.exists():
            raise FileNotFoundError(f"Dataset file not found: {p}")

        # Allow .json as well as .jsonl
        if p.suffix.lower() not in (".jsonl", ".json"):
            pass

        count = 0
        for line_no, rec in _iter_jsonl(p):
            _validate_messages_struct(rec, line_no, p)
            count += 1

        if count == 0:
            raise ValueError(f"[{p}] contains 0 valid records")

        print(f"[OK] {p} validated with {count} records")
""")

# 3) Overwrite the file with the known-good version
with open(MODULE_PATH, "w", encoding="utf-8") as f:
    f.write(GOOD_SRC)

# 4) Load the module from its absolute path (avoid name shadowing)
spec = importlib.util.spec_from_file_location("prepare_dataset", MODULE_PATH)
pd_mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(pd_mod)

# Expose the function in the notebook scope
validate_or_raise = pd_mod.validate_or_raise

print("[OK] Loaded prepare_dataset directly from file")
print(" - has validate_or_raise:", callable(validate_or_raise))
print(" - preview:\n", inspect.getsource(validate_or_raise))
