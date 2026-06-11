"""Per-layer detection / execution probes + EN->RO transfer (H1a / H1b).

Detection probe : harmful vs benign (construction label).
Execution probe : refuse vs comply (behavioral label, gpt-5-mini judged).

Train on English activations, evaluate in-language (EN ceiling) and zero-shot
on Romanian (transfer). The per-layer EN->RO accuracy drop is the H1a/H1b
signal. Bands are read off the EN in-language curve only (EXPERIMENT_DESIGN
sec 4.3), before any RO number is computed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

__all__ = ["LayerProbeResult", "ProbeSuite", "define_bands", "fit_all_layers",
           "compose_bands", "wilson_ci"]


def wilson_ci(acc: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% score interval for a binomial proportion (Paper 2 standard)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p, z2 = acc, z * z
    denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass
class LayerProbeResult:
    layer: int
    acc_en_cv: float          # 5-fold CV accuracy on EN train (ceiling estimate)
    acc_en_held: float        # accuracy on EN held-out split
    acc_ro: float             # zero-shot accuracy on RO (transfer)
    n_en_held: int
    n_ro: int
    best_C: float
    acc_en_held_wilson: tuple = (float("nan"), float("nan"))
    acc_ro_wilson: tuple = (float("nan"), float("nan"))

    @property
    def drop(self) -> float:
        return self.acc_en_held - self.acc_ro


@dataclass
class ProbeSuite:
    family: str = "logreg"
    cv_folds: int = 5
    Cs: tuple = (0.01, 0.1, 1.0, 10.0)
    max_iter: int = 2000
    per_layer: list = field(default_factory=list)

    def fit_layer(self, X_en_train, y_en_train, X_en_held, y_en_held,
                  X_ro, y_ro, layer: int) -> LayerProbeResult:
        y_en_train = np.asarray(y_en_train)
        if len(np.unique(y_en_train)) < 2:
            raise ValueError(
                f"layer {layer}: EN train split has a single class "
                f"({np.unique(y_en_train).tolist()}). Check that both contrastive "
                "cells are non-empty and the split keeps both labels."
            )
        n_train = len(y_en_train)
        folds = max(2, min(self.cv_folds, np.min(np.bincount(y_en_train.astype(int)))))

        scaler = StandardScaler().fit(X_en_train)
        Xtr = scaler.transform(X_en_train)
        best_C, best_cv = self.Cs[0], -1.0
        for C in self.Cs:
            cv = cross_val_score(LogisticRegression(C=C, max_iter=self.max_iter),
                                 Xtr, y_en_train, cv=folds).mean()
            if cv > best_cv:
                best_cv, best_C = cv, C
        clf = LogisticRegression(C=best_C, max_iter=self.max_iter).fit(Xtr, y_en_train)

        acc_en_held = float(clf.score(scaler.transform(X_en_held), y_en_held))
        acc_ro = float(clf.score(scaler.transform(X_ro), y_ro))
        res = LayerProbeResult(
            layer=layer, acc_en_cv=float(best_cv), acc_en_held=acc_en_held,
            acc_ro=acc_ro, n_en_held=len(y_en_held), n_ro=len(y_ro), best_C=float(best_C),
            acc_en_held_wilson=wilson_ci(acc_en_held, len(y_en_held)),
            acc_ro_wilson=wilson_ci(acc_ro, len(y_ro)),
        )
        self.per_layer.append(res)
        return res


def define_bands(acc_en_by_layer: list[float], saturation_fraction: float = 0.95
                 ) -> tuple[int, int]:
    """Return (band_start, band_peak) read off an EN in-language accuracy curve.

    band_start = first layer reaching ``saturation_fraction`` of the max acc.
    band_peak  = argmax layer. Caller composes detection/execution bands from
    the detection-probe and execution-probe curves respectively
    (EXPERIMENT_DESIGN sec 4.3). RO numbers must NOT be passed in here.
    """
    arr = np.asarray(acc_en_by_layer, dtype=float)
    thresh = saturation_fraction * arr.max()
    return int(np.argmax(arr >= thresh)), int(np.argmax(arr))


def fit_all_layers(acts_en, y_en, train_mask, acts_ro, y_ro, *,
                   family: str = "logreg") -> ProbeSuite:
    """Fit a per-layer probe over every block.

    acts_en : (n_en, B, d) float array of EN activations.
    y_en    : (n_en,) integer labels.
    train_mask : (n_en,) bool — EN train rows; the rest are EN held-out.
    acts_ro : (n_ro, B, d) RO activations (transfer eval).
    y_ro    : (n_ro,) RO labels.
    """
    y_en = np.asarray(y_en)
    y_ro = np.asarray(y_ro)
    train_mask = np.asarray(train_mask, dtype=bool)
    held_mask = ~train_mask
    suite = ProbeSuite(family=family)
    for layer in range(acts_en.shape[1]):
        suite.fit_layer(
            acts_en[train_mask, layer], y_en[train_mask],
            acts_en[held_mask, layer], y_en[held_mask],
            acts_ro[:, layer], y_ro, layer=layer,
        )
    return suite


def compose_bands(det_acc_en: list[float], exe_acc_en: list[float], n_blocks: int,
                  saturation_fraction: float = 0.95) -> dict:
    """Compose detection / execution bands from EN in-language curves only
    (EXPERIMENT_DESIGN sec 4.3). Detection band runs from where the detection
    probe saturates up to where the execution probe saturates; execution band
    from there to the end. Overlap/degenerate cases are reported honestly via
    the raw saturation points so the claim degrades gracefully (PAPER4_PLAN
    risk register)."""
    det_start, det_peak = define_bands(det_acc_en, saturation_fraction)
    exe_start, exe_peak = define_bands(exe_acc_en, saturation_fraction)
    det_end = max(det_start + 1, min(exe_start, n_blocks))
    return {
        "detection": list(range(det_start, det_end)),
        "execution": list(range(exe_start, n_blocks)),
        "det_saturation_start": det_start, "det_peak": det_peak,
        "exe_saturation_start": exe_start, "exe_peak": exe_peak,
        "overlap_warning": exe_start <= det_start,
    }
