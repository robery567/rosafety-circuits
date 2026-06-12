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
# All three are the exact Paper 3 anchors; gemma-3-4b-it doubles as the SAE
# anchor via Gemma Scope 2 (Gemma 3 family SAEs).
_SHORT = {
    "google/gemma-3-4b-it": "gemma-3-4b",
    "Qwen/Qwen2.5-3B-Instruct": "qwen2.5-3b",
    "meta-llama/Llama-3.2-3B-Instruct": "llama-3.2-3b",
}
_FAMILY = {
    "google/gemma-3-4b-it": "gemma3",
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


def savefig(fig, path) -> str | None:
    """Save a matplotlib figure robustly for camera-ready use.

    Tries, in order: PDF (default backend), PDF via cairo, SVG, then high-dpi
    PNG. Never raises - the analysis JSON is always written before plotting, so
    a matplotlib/backend hiccup (e.g. a mid-session matplotlib upgrade breaking
    the PDF backend: "cannot import name FontPath") must not abort a run.
    Returns the path actually written, or None.
    """
    base = Path(str(path)).with_suffix("")
    last = None
    for ext, backend in [("pdf", None), ("pdf", "cairo"), ("svg", None), ("png", None)]:
        out = base.with_suffix(f".{ext}")
        try:
            kw = {"bbox_inches": "tight"}
            if backend:
                kw["backend"] = backend
            if ext == "png":
                kw["dpi"] = 600
            fig.savefig(out, **kw)
            if (ext, backend) != ("pdf", None):
                tag = f" ({backend})" if backend else ""
                print(f"[savefig] fell back to {ext}{tag}: {out.name}")
            return str(out)
        except Exception as e:  # noqa: BLE001
            last = e
    print(f"[savefig] all formats failed: {last}")
    return None
