"""
Model loader and deterministic generation utilities.
"""

import copy
import random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def set_seed(seed: int = 42) -> None:
    """Set deterministic seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_model_revision(model: torch.nn.Module, model_id: str = "Qwen/Qwen2.5-0.5B-Instruct") -> str:
    """Capture exact model revision / commit SHA from model config or Hugging Face Hub."""
    commit_hash = getattr(model.config, "_commit_hash", None)
    if commit_hash:
        return str(commit_hash)
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(model_id)
        if info and getattr(info, "sha", None):
            return str(info.sha)
    except Exception:
        pass
    return "unknown"


def load_model_and_tokenizer(
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
):
    """Load pretrained model and tokenizer with reproducible defaults."""
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def get_weights_snapshot(model: torch.nn.Module) -> dict:
    """Create a deepcopy CPU state dict snapshot for exact restoration."""
    return {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}


def restore_weights_snapshot(model: torch.nn.Module, snapshot: dict) -> None:
    """Restore model weights exactly from an in-memory snapshot."""
    model.load_state_dict(snapshot)
    model.eval()


def generate_response(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    system_prompt: str | None = None,
    max_new_tokens: int = 80,
    device: str = "cpu",
) -> str:
    """Generate response deterministically using greedy decoding (do_sample=False)."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(chat_text, return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = output_ids[0][prompt_len:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return response
