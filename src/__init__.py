"""Paper 4 — shared helpers for mechanistic interpretability of the cross-lingual safety gap.

Modules here are imported by the notebooks under ``experiments/`` via
``sys.path.insert(0, str(SRC_DIR))``. Notebooks remain the runnable surface
(Paper 2/3 discipline); ``src/`` holds only helpers reused across more than one
notebook.

  - paths.py     : Drive/path resolution + anchor short-name helpers.
  - capture.py   : residual-stream activation capture hooks (assistant-prefix).
  - probes.py    : per-layer detection / execution probes + EN->RO transfer.
  - patching.py  : RO<-EN activation patching harness + controls.
  - sae_utils.py : Gemma Scope SAE loading + feature firing comparison.
"""
