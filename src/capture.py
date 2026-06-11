"""Residual-stream activation capture (assistant-prefix position).

Matches Paper 3 METHOD_DESIGN ``last_assistant_prefix`` capture point so probe
geometry is comparable across the two papers. Uses raw forward hooks (no
TransformerLens dependency) so it runs on the exact HF checkpoints, including
multimodal ones (Gemma-3) where the text decoder is nested under
``language_model``.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Optional, Sequence

import torch

__all__ = [
    "n_layers", "last_prefix_index", "chat_tokenize",
    "capture_block_outputs", "capture_assistant_prefix",
]


def _decoder_blocks(model):
    """Return the list of transformer decoder blocks, robust to architecture.

    Qwen-2.5 / Llama-3.2 / Gemma-2 expose ``model.model.layers``. Gemma-3 4B is
    multimodal, so the text decoder is nested under a ``language_model``
    submodule (the exact path varies by transformers version).
    """
    candidates = [
        lambda m: m.model.layers,                  # Qwen2.5, Llama3.2, Gemma2
        lambda m: m.model.language_model.layers,   # Gemma3 multimodal (newer tf)
        lambda m: m.language_model.model.layers,   # Gemma3 multimodal (older tf)
        lambda m: m.language_model.layers,         # Gemma3 text submodule
    ]
    for get in candidates:
        try:
            layers = get(model)
        except AttributeError:
            continue
        if layers is not None and len(layers) > 0:
            return layers
    raise AttributeError(
        "Could not locate decoder blocks; checked model.model.layers and "
        "language_model.* variants (Qwen/Llama/Gemma2/Gemma3)."
    )


def n_layers(model) -> int:
    """Number of transformer decoder blocks. Robust to nested (multimodal)
    configs where ``config.num_hidden_layers`` is absent (e.g. Gemma3Config,
    where it lives under ``config.text_config``)."""
    return len(_decoder_blocks(model))


def last_prefix_index(attention_mask: torch.Tensor) -> torch.Tensor:
    """Index of the last non-pad token per row (the assistant-prefix position
    when the chat template ends at the generation prompt).

    Robust to padding side: finds the *last* position where ``mask == 1``. Under
    left-padding (which we use) the real tokens are right-aligned, so this is
    ``seq_len-1``; under right-padding it is ``length-1``.
    """
    seq_len = attention_mask.shape[1]
    last_from_end = attention_mask.flip(1).long().argmax(dim=1)
    return (seq_len - 1 - last_from_end).long()


def chat_tokenize(tokenizer, prompts: Sequence[str], *, device: str = "cuda",
                  max_len: int = 1024):
    """Apply the chat template (+ generation prompt) and tokenize a batch.

    Single source of truth for turning user prompts into model inputs, shared by
    capture / generation / patching. Enforces the two settings that matter for
    a causal chat decoder:
      - ``padding_side='left'``  → correct batched generation + a shared prompt
        length per batch.
      - ``truncation_side='left'`` → if a prompt exceeds ``max_len`` we drop the
        *oldest* tokens, never the assistant-generation-prompt suffix (which is
        exactly the position we read activations from / generate at).
    """
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True,
        )
        for p in prompts
    ]
    return tokenizer(texts, return_tensors="pt", padding=True,
                     truncation=True, max_length=max_len).to(device)


@contextmanager
def capture_block_outputs(model, store: dict, gather_idx: Optional[torch.Tensor] = None):
    """Record each decoder block's output hidden state via forward hooks.

    If ``gather_idx`` (one position per row) is given, only the activation at
    that position is kept — ``store['hidden'][ℓ]`` becomes a ``(batch, d_model)``
    bf16 CPU tensor (cheap). Otherwise the full ``(batch, seq, d_model)`` tensor
    is kept on-device (callers slice it themselves).
    """
    blocks = _decoder_blocks(model)
    handles = []
    store["hidden"] = [None] * len(blocks)

    def make_hook(idx):
        def hook(_module, _inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            if gather_idx is None:
                store["hidden"][idx] = hs.detach()
            else:
                rows = torch.arange(hs.size(0), device=hs.device)
                gi = gather_idx.to(hs.device)
                store["hidden"][idx] = hs[rows, gi].detach().to(torch.bfloat16).cpu()
        return hook

    try:
        for i, blk in enumerate(blocks):
            handles.append(blk.register_forward_hook(make_hook(i)))
        yield store
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def capture_assistant_prefix(model, tokenizer, prompts: Sequence[str],
                             device: str = "cuda", batch_size: int = 8,
                             max_len: int = 1024) -> torch.Tensor:
    """Return a ``(n_prompts, n_blocks, d_model)`` bf16 tensor of block-output
    residuals at the last assistant-prefix position.

    Memory-light: each block's target-position vector is gathered inside the
    forward hook, so we never materialise full ``(b, seq, d)`` activations for
    every block at once.
    """
    if len(prompts) == 0:
        raise ValueError("capture_assistant_prefix: empty prompt list.")
    model.eval()
    all_rows = []
    for start in range(0, len(prompts), batch_size):
        batch = list(prompts[start:start + batch_size])
        enc = chat_tokenize(tokenizer, batch, device=device, max_len=max_len)
        idx = last_prefix_index(enc["attention_mask"])               # (b,) on enc device
        store: dict = {}
        with capture_block_outputs(model, store, gather_idx=idx):
            model(**enc)
        # store['hidden'] is a list (n_blocks) of (b, d) bf16 CPU tensors.
        all_rows.append(torch.stack(store["hidden"], dim=1))          # (b, n_blocks, d)
        del store
    return torch.cat(all_rows, dim=0)
