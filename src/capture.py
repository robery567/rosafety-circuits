"""Residual-stream activation capture (assistant-prefix position).

Matches Paper 3 METHOD_DESIGN ``last_assistant_prefix`` capture point so probe
geometry is comparable across the two papers. Uses raw forward hooks (no
TransformerLens dependency for capture) so it runs on the exact HF checkpoints.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Sequence

import torch


def _decoder_blocks(model):
    """Return the list of transformer decoder blocks for the supported anchors.

    Gemma-2 / Qwen-2.5 / Llama-3.2 all expose ``model.model.layers``.
    """
    base = getattr(model, "model", model)
    layers = getattr(base, "layers", None)
    if layers is None:
        raise AttributeError(
            "Could not locate decoder blocks; expected model.model.layers for "
            "Gemma-2 / Qwen-2.5 / Llama-3.2."
        )
    return layers


@contextmanager
def capture_block_outputs(model, store: dict):
    """Context manager that records each block's output hidden state.

    On exit, ``store['hidden']`` is a list (len = n_blocks) of the *full*
    last-forward block-output tensors. Callers slice the assistant-prefix
    position themselves (it depends on the batch's attention mask).
    """
    blocks = _decoder_blocks(model)
    handles = []
    store["hidden"] = [None] * len(blocks)

    def make_hook(idx):
        def hook(_module, _inp, out):
            # Decoder blocks return a tuple; hidden state is element 0.
            hs = out[0] if isinstance(out, tuple) else out
            store["hidden"][idx] = hs.detach()
        return hook

    try:
        for i, blk in enumerate(blocks):
            handles.append(blk.register_forward_hook(make_hook(i)))
        yield store
    finally:
        for h in handles:
            h.remove()


def last_prefix_index(attention_mask: torch.Tensor) -> torch.Tensor:
    """Index of the last non-pad token per row (the assistant-prefix position
    when the chat template ends at the generation prompt).

    Robust to padding side: finds the *last* position where mask == 1. Under
    left-padding (which we use for generation) the real tokens are right-
    aligned, so this is seq_len-1; under right-padding it is length-1.
    """
    seq_len = attention_mask.shape[1]
    # position of the last 1 in each row = (seq_len-1) - argmax of the reversed mask
    last_from_end = attention_mask.flip(1).float().argmax(dim=1)
    return (seq_len - 1 - last_from_end).long()


@torch.no_grad()
def capture_assistant_prefix(model, tokenizer, prompts: Sequence[str],
                             device: str = "cuda", batch_size: int = 8,
                             max_len: int = 1024) -> torch.Tensor:
    """Return a (n_prompts, n_blocks, d_model) bf16 tensor of block-output
    residuals at the last assistant-prefix position.

    Prompts are wrapped with the model's chat template + generation prompt, so
    the last token is exactly where the first generated token would attend from.
    """
    model.eval()
    all_rows = []
    for start in range(0, len(prompts), batch_size):
        batch = list(prompts[start:start + batch_size])
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False, add_generation_prompt=True,
            )
            for p in batch
        ]
        enc = tokenizer(texts, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_len).to(device)
        store: dict = {}
        with capture_block_outputs(model, store):
            model(**enc)
        idx = last_prefix_index(enc["attention_mask"]).to(device)  # (b,)
        # Stack per-block last-position vectors -> (n_blocks, b, d_model)
        per_block = []
        for hs in store["hidden"]:
            gathered = hs[torch.arange(hs.size(0), device=device), idx]  # (b, d)
            per_block.append(gathered.to(torch.bfloat16).cpu())
        # -> (b, n_blocks, d_model)
        all_rows.append(torch.stack(per_block, dim=1))
        del store
    return torch.cat(all_rows, dim=0)
