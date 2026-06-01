"""Gemma Scope SAE loading + cross-lingual feature firing comparison (H1e).

Gemma-only corroboration. Uses pretrained Gemma Scope JumpReLU residual SAEs
(Lieberum et al. 2024) via ``sae_lens`` — no SAE training. We identify
detection features (separate harm_en vs benign_en) and refusal features
(separate refusal vs compliance generations), then compare their firing on
EN vs RO harmful prompts across the detection/execution bands.
"""
from __future__ import annotations

import numpy as np


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
