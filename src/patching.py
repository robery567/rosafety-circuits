"""RO<-EN activation patching harness + controls (H1c — the causal experiment).

For a matched parallel pair (x_en, x_ro) where the model complies on RO and
refuses on EN, replace the RO residual at a target block with the cached EN
residual and continue the forward pass. "Restoration" = fraction of
gap-exhibiting pairs flipped comply->refuse by the patch, swept over layers.

Controls (EXPERIMENT_DESIGN sec 5.2):
  C1 benign->benign     : must NOT induce refusal.
  C2 mismatched harmful : different harmful prompt's EN residual.
  C3 random direction   : matched-norm random vector.
  C4 reverse (EN<-RO)   : noising; does removing EN signal break EN refusal?
"""
from __future__ import annotations

from contextlib import contextmanager

import torch

from .capture import _decoder_blocks, last_prefix_index


@contextmanager
def patch_block_at_position(model, layer: int, replacement: torch.Tensor,
                            position_idx: torch.Tensor):
    """Replace block ``layer`` output at ``position_idx`` with ``replacement``.

    ``replacement`` : (batch, d_model) tensor to write at the assistant-prefix
    position of each row. The patch propagates through all downstream layers.
    """
    blocks = _decoder_blocks(model)
    handle = None

    def hook(_module, _inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        rows = torch.arange(hs.size(0), device=hs.device)
        hs[rows, position_idx] = replacement.to(hs.dtype).to(hs.device)
        if isinstance(out, tuple):
            return (hs,) + tuple(out[1:])
        return hs

    try:
        handle = blocks[layer].register_forward_hook(hook)
        yield
    finally:
        if handle is not None:
            handle.remove()


@torch.no_grad()
def generate_with_patch(model, tokenizer, ro_prompt: str,
                        layer: int, replacement_vec: torch.Tensor,
                        device: str = "cuda", max_new_tokens: int = 256) -> str:
    """Greedy-generate on the RO prompt while patching ``layer`` at the
    assistant-prefix position with ``replacement_vec`` (1, d_model)."""
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": ro_prompt}],
        tokenize=False, add_generation_prompt=True,
    )
    enc = tokenizer(text, return_tensors="pt").to(device)
    pos = last_prefix_index(enc["attention_mask"]).to(device)
    repl = replacement_vec.view(1, -1)
    with patch_block_at_position(model, layer, repl, pos):
        out = model.generate(**enc, max_new_tokens=max_new_tokens,
                             do_sample=False, temperature=None, top_p=None)
    gen = out[0, enc["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True)


def restoration_rate(judged_after: list[str], gap_mask: list[bool]) -> float:
    """Fraction of gap-exhibiting pairs (mask True) judged 'refuse' after patch.

    ``judged_after`` : per-pair judge label in {'refuse','comply'} post-patch.
    ``gap_mask``     : per-pair True if the pair exhibited the gap pre-patch
                       (RO comply + EN refuse).
    """
    idx = [i for i, m in enumerate(gap_mask) if m]
    if not idx:
        return float("nan")
    flipped = sum(1 for i in idx if judged_after[i] == "refuse")
    return flipped / len(idx)


def bootstrap_ci(values: list[float], n_resamples: int = 2000,
                 alpha: float = 0.05, seed: int = 17) -> tuple[float, float]:
    """Percentile bootstrap CI for a mean over per-pair indicators."""
    rng = torch.Generator().manual_seed(seed)
    arr = torch.tensor(values, dtype=torch.float32)
    n = arr.numel()
    if n == 0:
        return (float("nan"), float("nan"))
    means = torch.empty(n_resamples)
    for b in range(n_resamples):
        idx = torch.randint(0, n, (n,), generator=rng)
        means[b] = arr[idx].mean()
    lo = torch.quantile(means, alpha / 2).item()
    hi = torch.quantile(means, 1 - alpha / 2).item()
    return (lo, hi)
