"""Gemma Scope SAE loading + cross-lingual feature firing comparison (H1e).

Gemma-only corroboration. Uses pretrained Gemma Scope 2 JumpReLU residual SAEs
(Gemma 3 family) via ``sae_lens`` — no SAE training. We identify
detection features (separate harm_en vs benign_en) and refusal features
(separate refused vs complied prompts), then compare their firing on EN vs RO
harmful prompts across the detection/execution bands.
"""
from __future__ import annotations

import numpy as np

__all__ = ["load_gemma_scope_sae", "encode_acts", "difference_in_means_features",
           "firing_rate", "en_ro_firing_gap"]


def load_gemma_scope_sae(layer: int, *, width: str = "16k", l0: str = "medium",
                         device: str = "cuda",
                         release: str = "gemma-scope-2-4b-it-resid_post_all"):
    """Load one Gemma Scope 2 residual SAE for a Gemma-3 block. Returns the SAE.

    Gemma Scope 2 (Gemma 3 family) API differs from v1:
      - release e.g. ``gemma-scope-2-4b-it-resid_post_all`` (the ``_all`` folder
        has an SAE for every layer; the plain ``resid_post`` folder only covers
        4 depths but with more widths).
      - sae_id ``layer_{L}_width_{W}_l0_{small|medium|large}`` (underscores).
      - ``from_pretrained`` returns ``(sae, cfg_dict, sparsity)``.

    Verify the available (width, l0) combos for the ``_all`` release at run time
    (PAPER4_PLAN §13.1); 16k/medium is the conservative default.
    """
    from sae_lens import SAE  # lazy: Colab only
    sae_id = f"layer_{layer}_width_{width}_l0_{l0}"
    r = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
    sae = r[0] if isinstance(r, (tuple, list)) else r
    return sae.eval()


def encode_acts(sae, acts) -> np.ndarray:
    """Encode (n, d_model) residual activations into (n, n_features) SAE latents."""
    import torch
    with torch.no_grad():
        x = acts.to(next(sae.parameters()).dtype).to(next(sae.parameters()).device) \
            if hasattr(sae, "parameters") else acts
        z = sae.encode(x)
    return z.detach().float().cpu().numpy()


def difference_in_means_features(acts_pos: np.ndarray, acts_neg: np.ndarray,
                                 top_m: int = 50) -> np.ndarray:
    """Return indices of the top-m SAE latents by mean-activation separation.

    acts_pos : (n_pos, n_features) SAE activations on the positive class
               (e.g. harm_en for detection features).
    acts_neg : (n_neg, n_features) SAE activations on the negative class.
    """
    sep = acts_pos.mean(axis=0) - acts_neg.mean(axis=0)
    return np.argsort(-sep)[:top_m]


def firing_rate(acts: np.ndarray, feature_idx: np.ndarray,
                threshold: float = 0.0) -> np.ndarray:
    """Per-feature fraction of examples where the latent fires above threshold.

    JumpReLU SAEs are already thresholded, so the default threshold is 0.
    Returns a (len(feature_idx),) array of firing rates.
    """
    sub = acts[:, feature_idx]
    return (sub > threshold).mean(axis=0)


def en_ro_firing_gap(acts_en: np.ndarray, acts_ro: np.ndarray,
                     feature_idx: np.ndarray, threshold: float = 0.0) -> dict:
    """Compare firing of a feature class on EN vs RO harmful prompts.

    H1e predicts detection features under-fire on RO (positive en_minus_ro in
    the detection band) and refusal features fire comparably (~0).
    """
    fr_en = firing_rate(acts_en, feature_idx, threshold)
    fr_ro = firing_rate(acts_ro, feature_idx, threshold)
    return {
        "mean_firing_en": float(fr_en.mean()),
        "mean_firing_ro": float(fr_ro.mean()),
        "en_minus_ro": float((fr_en - fr_ro).mean()),
        "per_feature_en": fr_en.tolist(),
        "per_feature_ro": fr_ro.tolist(),
    }
