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

from capture import _decoder_blocks, chat_tokenize, last_prefix_index


@contextmanager
def patch_block_at_position(model, layer: int, replacement: torch.Tensor,
                            position_idx: torch.Tensor):
    """Replace block ``layer`` output at ``position_idx`` with ``replacement``.

    ``replacement`` : ``(batch, d_model)`` (or ``(1, d_model)``) tensor written
    at the assistant-prefix position of each row. The patch propagates through
    all downstream layers.
    """
    blocks = _decoder_blocks(model)
    handle = None
    pos_max = int(position_idx.max())

    def hook(_module, _inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        # Only patch the prefill pass. During cached decoding the sequence
        # collapses to the new token(s), so the prefix position no longer
        # exists; patching there is wrong (and would be out of bounds).
        if hs.shape[1] <= pos_max:
            return out
        rows = torch.arange(hs.size(0), device=hs.device)
        hs[rows, position_idx.to(hs.device)] = replacement.to(hs.dtype).to(hs.device)
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
    assistant-prefix position with ``replacement_vec`` (``(d_model,)`` or
    ``(1, d_model)``)."""
    enc = chat_tokenize(tokenizer, [ro_prompt], device=device)
    pos = last_prefix_index(enc["attention_mask"]).to(device)
    repl = replacement_vec.reshape(1, -1)
    with patch_block_at_position(model, layer, repl, pos):
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
    gen = out[0, enc["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True)


def bootstrap_ci(values: list[float], n_resamples: int = 2000,
                 alpha: float = 0.05, seed: int = 17) -> tuple[float, float]:
    """Percentile bootstrap CI for a mean over per-pair 0/1 indicators."""
    arr = torch.tensor(values, dtype=torch.float32)
    n = arr.numel()
    if n == 0:
        return (float("nan"), float("nan"))
    rng = torch.Generator().manual_seed(seed)
    means = torch.empty(n_resamples)
    for b in range(n_resamples):
        means[b] = arr[torch.randint(0, n, (n,), generator=rng)].mean()
    return (torch.quantile(means, alpha / 2).item(),
            torch.quantile(means, 1 - alpha / 2).item())


def run_patch_sweep(model, tokenizer, source_acts, target_prompts, ids, layers,
                    label_fn, *, device: str = "cuda", random_norm_match: bool = False,
                    max_new_tokens: int = 256) -> dict:
    """Generic layer-sweep patcher (drives both the main experiment and controls).

    For each layer in ``layers``, patch ``source_acts[:, layer]`` (or, if
    ``random_norm_match``, a random vector of matched norm) at the
    assistant-prefix position of each ``target_prompts[i]``, greedily generate,
    and label refuse/comply via ``label_fn``.

    Args:
      source_acts : (n, B, d) tensor — the residuals to inject.
      target_prompts : list[str] (len n) — the forward passes to patch.
      ids : list[str] (len n) — stable per-row ids (for the judge cache key).
      label_fn : callable(list[(id, prompt, response)]) -> {id: 'refuse'|'comply'}.

    Returns ``{layer: {'refusal_rate': float, 'n': int,
                       'per_pair': [(id, label), ...]}}``.
    """
    out: dict = {}
    for layer in layers:
        triples = []
        for i, prompt in enumerate(target_prompts):
            vec = source_acts[i, layer].to(device).float()
            if random_norm_match:
                r = torch.randn_like(vec)
                vec = r / (r.norm() + 1e-8) * vec.norm()
            resp = generate_with_patch(model, tokenizer, prompt, layer, vec,
                                       device=device, max_new_tokens=max_new_tokens)
            triples.append((f"{ids[i]}::L{layer}", prompt, resp))
        labels = label_fn(triples)
        # label_fn keys are the "<id>::L<layer>" strings; map back to bare id.
        per_pair = [(ids[i], labels.get(f"{ids[i]}::L{layer}", "comply"))
                    for i in range(len(ids))]
        n = len(per_pair)
        rr = sum(1 for _, l in per_pair if l == "refuse") / n if n else float("nan")
        out[layer] = {"refusal_rate": rr, "n": n, "per_pair": per_pair}
    return out


def refusal_indicator_list(sweep_layer_result: dict) -> list[float]:
    """0/1 refusal indicators for one layer's sweep result (for bootstrap_ci)."""
    return [1.0 if l == "refuse" else 0.0 for _, l in sweep_layer_result["per_pair"]]
