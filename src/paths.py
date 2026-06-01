"""Path + anchor-name resolution shared across Paper 4 notebooks.

Keeps Drive layout in one place so notebooks don't drift. Mirrors the
``DRIVE_ROOT`` convention used in Paper 3 notebooks.
"""
from __future__ import annotations

from pathlib import Path

# --- Drive layout (Colab). Notebooks override DRIVE_ROOT if running elsewhere.
DRIVE_ROOT = Path("/content/drive/MyDrive/PhD/paper4-interpretability")
PAPER2_ROOT = Path("/content/drive/MyDrive/PhD/paper2-benchmark")
PAPER3_ROOT = Path("/content/drive/MyDrive/PhD/paper3-alignment")


def data_dirs(root: Path = DRIVE_ROOT) -> dict[str, Path]:
    """Return (and create) the standard Paper 4 data subdirectories."""
    dirs = {
        "contrastive": root / "data" / "contrastive",
        "activations": root / "data" / "activations",
        "probes": root / "data" / "probes",
        "splits": root / "data" / "splits",
        "results": root / "results",
        "figures": root / "figures",
        "logs": root / "logs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


# --- Anchor short-name registry (kept in sync with configs/models.yaml).
_SHORT = {
    "google/gemma-2-2b-it": "gemma-2-2b",
    "Qwen/Qwen2.5-3B-Instruct": "qwen2.5-3b",
    "meta-llama/Llama-3.2-3B-Instruct": "llama-3.2-3b",
}
_FAMILY = {
    "google/gemma-2-2b-it": "gemma2",
    "Qwen/Qwen2.5-3B-Instruct": "qwen2.5",
    "meta-llama/Llama-3.2-3B-Instruct": "llama",
}


def short_of(model_id: str) -> str:
    if model_id in _SHORT:
        return _SHORT[model_id]
    # Fallback: last path segment, lowercased, instruct/-it stripped.
    tail = model_id.split("/")[-1].lower()
    for suffix in ("-instruct", "-it"):
        if tail.endswith(suffix):
            tail = tail[: -len(suffix)]
    return tail


def family_of(model_id: str) -> str:
    return _FAMILY.get(model_id, model_id.split("/")[-1].split("-")[0].lower())
