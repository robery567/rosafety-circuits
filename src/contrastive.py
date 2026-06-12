"""Build + freeze the Paper 4 contrastive sets (EXPERIMENT_DESIGN sec 2).

Five cells per anchor:
  harm_ro   : RO harmful   <- RoSafetyBench toxicity + jailbreak (culturally native)
  benign_ro : RO benign    <- RoSafetyBench overrefusal (benign-but-risky-looking)
  harm_en   : EN harmful   <- HarmBench standard (+ local core from crosslingual text_en)
  benign_en : EN benign    <- XSTest *safe* split (matches overrefusal semantics)
  parallel  : EN<->RO harmful matched <- RoSafetyBench crosslingual pairs (patching-only)

Design note (benign semantics). The detection probe must learn *harmfulness*,
not surface topic. So both benign cells are "looks risky, is benign":
RoSafetyBench overrefusal on the RO side, XSTest-safe on the EN side. This keeps
the EN->RO transfer measurement (H1a) from being confounded by a benign-class
distribution shift.

Determinism. RO cells + parallel + the harm_en local core derive only from
committed Paper 2 files, so their bytes (and SHA-256) are stable across machines
and pre-registrable now. The EN cells that need HF datasets (HarmBench, XSTest)
are built on Colab; their SHA-256 freezes at that point. The MANIFEST records
per-cell status (`final` vs `pending`).
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

# Unified output row schema (superset of Paper 2's row).
#   id, cell, lang, label, text            (single-language cells)
#   id, cell, label, text_en, text_ro      (parallel cell)
# plus provenance: source_dim, category, expected_behavior, source.

SEED = 17


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _stable_sample(rows: list[dict], n: int, seed: int = SEED) -> list[dict]:
    """Deterministic sample: sort by id, sample with a fixed RNG, re-sort by id
    so the written bytes are identical on every machine."""
    rows_sorted = sorted(rows, key=lambda r: r["id"])
    if n >= len(rows_sorted):
        chosen = rows_sorted
    else:
        rng = random.Random(seed)
        chosen = rng.sample(rows_sorted, n)
    return sorted(chosen, key=lambda r: r["id"])


# --- RO cells + parallel (fully local, deterministic) ----------------------

def build_harm_ro(expanded: Path, n: int = 100, seed: int = SEED) -> list[dict]:
    pool = _read_jsonl(expanded / "toxicity.jsonl") + _read_jsonl(expanded / "jailbreak.jsonl")
    pool = [r for r in pool if r.get("expected_behavior") == "refuse"]   # defensive
    rows = _stable_sample(pool, n, seed)
    return [{
        "id": r["id"], "cell": "harm_ro", "lang": "ro", "label": "harmful",
        "text": r["text_ro"], "source_dim": r["dimension"], "category": r.get("category"),
        "expected_behavior": r.get("expected_behavior"), "source": r.get("source"),
    } for r in rows]


def build_benign_ro(expanded: Path, n: int = 100, seed: int = SEED) -> list[dict]:
    pool = _read_jsonl(expanded / "overrefusal.jsonl")
    pool = [r for r in pool if r.get("expected_behavior") == "answer"]   # defensive
    rows = _stable_sample(pool, n, seed)
    return [{
        "id": r["id"], "cell": "benign_ro", "lang": "ro", "label": "benign",
        "text": r["text_ro"], "source_dim": r["dimension"], "category": r.get("category"),
        "expected_behavior": r.get("expected_behavior"), "source": r.get("source"),
    } for r in rows]


def build_parallel(expanded: Path, include_bias: bool = True) -> list[dict]:
    """Matched EN<->RO safety-relevant pairs (patching workhorse). The
    crosslingual set mixes harmful / bias / benign by ``category`` (== the id
    prefix; ``expected_behavior`` is unreliable — see below). We keep the
    safety-relevant categories (harmful, and by default bias, where EN should
    refuse / reject the premise) so the patch pool is large enough to find
    enough gap-exhibiting pairs. ``harm_type`` records the original category so
    analysis can split harmful-only vs harmful+bias.

    NB: ``expected_behavior`` in Paper 2's crosslingual file is corrupted for
    benign rows (22/25 benign prompts such as "What is the capital of Romania?"
    are mislabeled 'refuse'); we never rely on it — ``category`` is authoritative.
    """
    keep = {"harmful", "bias"} if include_bias else {"harmful"}
    rows = [r for r in _read_jsonl(expanded / "crosslingual.jsonl")
            if r.get("category") in keep]
    rows = sorted(rows, key=lambda r: r["id"])
    return [{
        "id": r["id"], "cell": "parallel", "label": "harmful",
        "text_en": r["text_en"], "text_ro": r["text_ro"],
        "source_dim": r["dimension"], "harm_type": r.get("category"),
        "expected_behavior": "refuse", "source": r.get("source"),
    } for r in rows]


def build_harm_en_core(expanded: Path) -> list[dict]:
    """Local EN-harmful core = matched text_en from the *harmful* crosslingual
    pairs (category == 'harmful'; see build_parallel for why not
    expected_behavior). Topped up to target with HarmBench on Colab."""
    rows = [r for r in _read_jsonl(expanded / "crosslingual.jsonl")
            if r.get("category") == "harmful"]
    rows = sorted(rows, key=lambda r: r["id"])
    return [{
        "id": r["id"].replace("cro_", "harmen_"), "cell": "harm_en", "lang": "en",
        "label": "harmful", "text": r["text_en"], "source_dim": "crosslingual_en",
        "category": r.get("category"), "expected_behavior": "refuse", "source": "rosafetybench_parallel_en",
    } for r in rows]


# --- EN cells that need HF datasets (built on Colab) ------------------------

def build_harm_en_topup(target_n: int = 250, seed: int = SEED) -> list[dict]:
    """HarmBench standard behaviors (EN harmful). Requires `datasets`."""
    from datasets import load_dataset  # lazy: only on Colab / with network
    ds = load_dataset("walledai/HarmBench", "standard", split="train")
    rows = [{"id": f"hb_{i}", "cell": "harm_en", "lang": "en", "label": "harmful",
             "text": ex["prompt"], "source_dim": "harmbench_standard",
             "category": ex.get("category"), "expected_behavior": "refuse",
             "source": "harmbench"} for i, ex in enumerate(ds)]
    return _stable_sample(rows, target_n, seed)


def build_benign_en(target_n: int = 250, seed: int = SEED) -> list[dict]:
    """XSTest *safe* prompts (EN benign-but-risky-looking). Requires `datasets`.

    Schema-robust: XSTest copies vary in their id/label columns, so we derive
    the text from ``prompt`` (fallback ``text``), classify safe-vs-unsafe from a
    ``label`` column if present else from the ``type`` prefix (XSTest unsafe
    types are ``contrast_*``), and use the row index for a stable id.
    """
    from datasets import load_dataset  # lazy
    ds = load_dataset("natolambert/xstest-v2-copy", split="prompts")
    cols = set(ds.column_names)
    text_col = "prompt" if "prompt" in cols else ("text" if "text" in cols else None)
    if text_col is None:
        raise KeyError(f"XSTest: no prompt/text column in {sorted(cols)}")

    def is_safe(ex) -> bool:
        # XSTest's canonical signal is `type`: unsafe prompts are `contrast_*`,
        # the 250 safe prompts are not. Fall back to a `label` column only if
        # `type` is missing in this copy.
        if "type" in cols and ex.get("type") is not None:
            return not str(ex["type"]).startswith("contrast")
        if "label" in cols and ex.get("label") is not None:
            return str(ex["label"]).lower() == "safe"
        return True

    rows = []
    for i, ex in enumerate(ds):
        if not is_safe(ex):
            continue
        rows.append({"id": f"xst_{i}", "cell": "benign_en", "lang": "en",
                     "label": "benign", "text": ex[text_col], "source_dim": "xstest_safe",
                     "category": ex.get("type"), "expected_behavior": "answer", "source": "xstest"})
    return _stable_sample(rows, target_n, seed)


# --- Orchestration ----------------------------------------------------------

def build_all(expanded: Path, out_dir: Path, *, with_en: bool = False,
              n_ro: int = 100, n_en: int = 250, seed: int = SEED) -> dict:
    """Build every cell that is currently buildable, write JSONL + MANIFEST.

    ``with_en=False`` (default, offline): builds RO cells + parallel + harm_en
    local core; marks benign_en / harm_en-topup as pending.
    ``with_en=True`` (Colab/network): also pulls HarmBench + XSTest.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"seed": seed, "cells": {}}

    def emit(name: str, rows: list[dict], status: str, source: str):
        path = out_dir / f"{name}.jsonl"
        _write_jsonl(rows, path)
        manifest["cells"][name] = {"n": len(rows), "status": status,
                                   "source": source, "sha256": sha256_of(path)}

    emit("harm_ro", build_harm_ro(expanded, n_ro, seed), "final", "rosafetybench_tox_jb")
    emit("benign_ro", build_benign_ro(expanded, n_ro, seed), "final", "rosafetybench_overrefusal")
    emit("parallel", build_parallel(expanded), "final", "rosafetybench_crosslingual")

    harm_en_core = build_harm_en_core(expanded)
    if with_en:
        harm_en = harm_en_core + build_harm_en_topup(n_en - len(harm_en_core), seed)
        emit("harm_en", _stable_sample(harm_en, n_en, seed), "final", "crosslingual_en+harmbench")
        emit("benign_en", build_benign_en(n_en, seed), "final", "xstest_safe")
    else:
        emit("harm_en", harm_en_core, "core_local_topup_pending", "crosslingual_en (HarmBench top-up pending)")
        manifest["cells"]["benign_en"] = {"n": 0, "status": "pending_colab",
                                          "source": "xstest_safe (needs datasets+network)", "sha256": None}

    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    _validate(out_dir)
    return manifest


def make_probe_split(out_dir: Path, train_frac: float = 0.7, seed: int = SEED) -> dict:
    """Freeze the probe split (EXPERIMENT_DESIGN sec 2.3) and record SHA-256s.

    EN train = ``train_frac`` of (harm_en + benign_en), class-stratified.
    EN eval  = the remaining EN (in-language ceiling).
    RO eval  = all of (harm_ro + benign_ro) (transfer target).
    parallel = patching-only (not in any probe split).

    Deterministic: stable sort by id, fixed RNG. Writes
    ``data/splits/probe_split.json`` under the anchor dir's parent / splits.
    """
    def _ids(cell, label):
        path = out_dir / f"{cell}.jsonl"
        rows = _read_jsonl(path) if path.exists() else []
        return sorted(r["id"] for r in rows if r["label"] == label)

    rng = random.Random(seed)
    split = {"train_frac": train_frac, "seed": seed, "train_en": [], "eval_en": [], "eval_ro": []}
    for cell, label in [("harm_en", "harmful"), ("benign_en", "benign")]:
        ids = _ids(cell, label)
        k = int(round(train_frac * len(ids)))
        train = set(rng.sample(ids, k)) if ids else set()
        split["train_en"] += sorted(train)
        split["eval_en"] += sorted(i for i in ids if i not in train)
    split["eval_ro"] = _ids("harm_ro", "harmful") + _ids("benign_ro", "benign")
    split["train_en"].sort(); split["eval_en"].sort()

    splits_dir = out_dir.parent.parent / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    short = out_dir.name
    path = splits_dir / f"probe_split_{short}.json"
    path.write_text(json.dumps(split, indent=2))
    split["_sha256"] = sha256_of(path)
    split["_path"] = str(path)
    return split


# Known-benign EN strings that must never appear in a harmful cell (regression
# guard against the crosslingual expected_behavior corruption).
_BENIGN_CANARIES = ("Sziget Festival", "capital of Romania", "make polenta", "ciorba")


def _validate(out_dir: Path) -> None:
    """Cheap structural checks. Raises AssertionError on contamination so a bad
    build can never be silently committed / pre-registered."""
    for cell, label in [("harm_ro", "harmful"), ("harm_en", "harmful"),
                        ("benign_ro", "benign"), ("parallel", "harmful")]:
        path = out_dir / f"{cell}.jsonl"
        if not path.exists():
            continue
        rows = _read_jsonl(path)
        assert all(r["label"] == label for r in rows), f"{cell}: wrong label present"
        assert len({r["id"] for r in rows}) == len(rows), f"{cell}: duplicate ids"
        texts = " ".join((r.get("text") or r.get("text_en") or "") for r in rows)
        if label == "harmful":
            for canary in _BENIGN_CANARIES:
                assert canary not in texts, f"{cell}: benign canary leaked: {canary!r}"
