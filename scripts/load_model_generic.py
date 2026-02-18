"""
Generic model loader.

Supports:
- Kaggle Models via `kagglehub`
- Hugging Face model ids

Handles 4-bit/8-bit/none quantization for Transformers models with PEFT.
"""

from typing import Tuple, Optional
import os

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
except Exception:
    # VS Code local environment may not have transformers installed.
    # Colab installs this in the first cell; locally, this is editor-only.
    AutoTokenizer = AutoModelForCausalLM = BitsAndBytesConfig = object  # type: ignore[misc]


def _maybe_quant_config(quantization: str) -> Optional["BitsAndBytesConfig"]:
    try:
        if quantization == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype="bfloat16",  # handled by transformers
            )
        if quantization == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True)
    except Exception:
        # In local/editor-only context without transformers, just skip.
        return None
    return None


def load_model_and_tokenizer(
    model_source: str,
    model_name: str,
    quantization: str = "4bit",
    device_map: str = "auto",
    torch_dtype: str = "bfloat16",
) -> Tuple["AutoTokenizer", "AutoModelForCausalLM", str]:
    """
    Returns (tokenizer, model, model_path)
    """
    if model_source == "kaggle":
        try:
            import kagglehub  # installed in Colab first cell
        except Exception as e:
            raise RuntimeError(
                "kagglehub is not available in this environment. "
                "Run in Colab after installing deps in the first cell."
            ) from e
        model_path = kagglehub.model_download(model_name)
    elif model_source == "hf":
        model_path = model_name
    else:
        raise ValueError(f"Unknown model_source: {model_source}")

    quant_cfg = _maybe_quant_config(quantization)

    tok = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    if quant_cfg is not None:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            quantization_config=quant_cfg,
        )
    else:
        try:
            import torch  # installed in Colab

            dtype = getattr(torch, torch_dtype, None)
        except Exception:
            dtype = None  # fallback: let transformers decide
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=dtype,
        )

    try:
        model.config.use_cache = False
    except Exception:
        pass

    return tok, model, model_path
