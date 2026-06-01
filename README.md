# Paper 4 — Mechanistic Interpretability of the Cross-Lingual Safety Gap

> **Working title:** *"Detection, Not Execution: A Mechanistic Account of the
> Cross-Lingual Safety Gap in Small Language Models"*
>
> **Phase:** 4 / Months 17-24 of the PhD plan.
>
> **Target venue:** TACL (WoS, IF ~10.9, rolling, 2-3 month review).
> **Backup:** Pattern Recognition (WoS). Short version: BlackboxNLP (workshop).
>
> **Role:** First author.
>
> **Status:** Plan locked; implementation not started.

## What this is

A mechanistic explanation of *why* small language models are less safe in
Romanian than in English — and *why* the refusal-direction-guided alignment
recipe of [Paper 3 (RD-DPO)][p3] could not close that gap.

The chain across the thesis is the wedge:

1. [**Paper 2 (RoSafetyBench)**][p2] measured a **universal** cross-lingual
   safety gap on 953 culturally-native Romanian prompts: every one of 20
   evaluatees refuses far less in Romanian than on the same intent in English.
2. [**Paper 3 (RD-DPO)**][p3] tried to close it by restricting DPO to the
   residual-stream blocks where the base model's *refusal direction* is most
   expressed. It **failed** — a consistent −10 pp jailbreak regression across
   three anchors, with no gap closure.
3. **Paper 4 (this paper)** asks the mechanistic question Paper 3 deferred
   (Paper 3 plan §3, §14): *where, inside the network, does the cross-lingual
   gap actually live?*

## Central hypothesis

> The cross-lingual safety gap is localized in **harmfulness detection**
> (early-to-mid layers, "is this request harmful?"), not in **refusal
> execution** (mid-to-late layers, "convert a harm judgment into a refusal").
> On Romanian harmful prompts the detection signal is attenuated relative to
> English; the refusal-execution direction, once triggered, is largely
> language-agnostic.

If true, this explains Paper 3 directly: RD-DPO selected and trained
**execution-band** layers, so it could not repair an **upstream detection**
deficit — and over-tuning execution plausibly produced the jailbreak
regression.

The hypothesis is grounded in two 2025 results we extend to the small-model,
low-resource, Romanian setting:

- *Refusal Direction is Universal Across Safety-Aligned Languages*
  (Wang et al. 2025, arXiv:2505.17306) — supports execution universality.
- *LLMs Encode Harmfulness and Refusal Separately* (Zhao et al. 2025,
  arXiv:2507.11878) — supports the detection/execution split as separable axes.

## Models

| Role | Model | Why |
|------|-------|-----|
| **SAE anchor** | `google/gemma-2-2b-it` | Pretrained **Gemma Scope** SAEs on every layer — no SAE training cost. |
| **Cross-arch anchor** | `Qwen/Qwen2.5-3B-Instruct` | Shared with Paper 3: has `selected_blocks.json` + RD-DPO adapters to test H1d. |
| **Cross-arch anchor** | `meta-llama/Llama-3.2-3B-Instruct` | Shared with Paper 3: weakest Paper-2 baseline, biggest headroom. |

All three have published Paper 2 behavioral baselines, so every mechanistic
number can be correlated against a real behavioral gap.

## Quick start

The full study runs as a sequence of seven Colab notebooks under
`experiments/`. Each is configured for an A100 high-RAM runtime, mounts Drive,
and reuses Paper 2's judge harness (`gpt-5-mini` primary, `claude-opus-4.5`
second-rater) and Paper 3's probe + split artefacts so numbers are directly
comparable across the three papers.

| # | Notebook | What it does | Runtime (A100) | Cost |
|---|----------|--------------|----------------|------|
| 0 | `00_pilot_smoke_test.ipynb` | Load one anchor, capture activations, run one patch end-to-end | ~10 min | $0 |
| 1 | `01_build_contrastive_sets.ipynb` | Build (harmful/benign)×(EN/RO) sets + EN↔RO parallel pairs; freeze splits | ~15 min | <$1 |
| 2 | `02_linear_probes.ipynb` | Per-layer harmfulness-detection & refusal-execution probes; EN→RO transfer drop (H1a/H1b) | ~30-60 min | $0 |
| 3 | `03_activation_patching.ipynb` | Causal localization: patch RO←EN activations per layer; restore refusal (H1c) | ~1-2 h | <$1 |
| 4 | `04_sae_features.ipynb` | Gemma Scope SAE feature analysis: detection vs refusal features fire EN vs RO (H1e) | ~1 h | $0 |
| 5 | `05_paper3_crossref.ipynb` | Map Paper 3's `selected_blocks` onto the detection/execution bands; detection-targeted confirmatory probe (H1d) | ~1-2 h | <$1 |
| 6 | `06_aggregate_and_figures.ipynb` | Headline table, layer-band figures, correlation-with-behavioral-gap, LaTeX | ~5 min, no GPU | $0 |

Notebooks 2-5 are idempotent and cache to Drive. Notebook 6 picks up everything
from `results/` automatically.

Set `HF_TOKEN` (gated models: Gemma, Llama) and `OPENROUTER_API_KEY` in
**Colab → 🔑 → Secrets** with notebook access enabled before running.

`requirements.txt` lists everything the notebooks pip-install on their own.

## Project structure

```
paper4-interpretability/
├── configs/
│   ├── models.yaml          # Anchor registry + Gemma Scope SAE coordinates
│   └── experiments.yaml     # Probe / patching / SAE hyperparameters
├── data/                     # operational Drive-only state (gitignored)
│   ├── contrastive/<short>/      # per-anchor (harm/benign)×(EN/RO) sets
│   ├── activations/<short>/      # cached residual-stream captures
│   ├── probes/<short>/            # per-layer probe weights + scores
│   └── splits/                   # frozen train/dev/holdout assignment
├── src/                      # shared helpers (probe, patching, sae shims)
├── experiments/              # Notebooks per experimental block (the runnable surface)
├── results/                  # Per-run JSON outputs
├── figures/                  # Generated figures (released)
├── manuscript/               # Paper LaTeX source (private until submission)
├── logs/                     # Capture + run logs
├── PAPER4_PLAN.md           # The plan (contribution, hypotheses, risks)
├── EXPERIMENT_DESIGN.md     # The method spec (probes, patching, SAE protocol)
├── EXPERIMENT_LOG.md        # Private run journal (gitignored)
├── requirements.txt
└── README.md
```

## Reproducibility

Following Paper 2/3 discipline:

- Contrastive sets, probe sets, and analysis configs are **pre-registered**
  before the first causal run; commit hash quoted in the manuscript.
- Activation captures recorded at a fixed position (last assistant-prefix
  token) with fixed generation params matched to Paper 2/3.
- Linear probes reported with cross-validated accuracy + Wilson 95% CIs;
  patching effects with bootstrap CIs; ≥3 seeds where stochastic.
- Behavioral refusal judged by the same `gpt-5-mini` / `claude-opus-4.5`
  protocol as Paper 2 (κ = 0.78 reference) for cross-paper consistency.
- Public release: contrastive sets, probe weights, SAE feature indices,
  patching tensors, judge labels, figures. Zenodo + HF. Manuscript excluded.

## Citation

```bibtex
@article{colca2028interp,
  title={Detection, Not Execution: A Mechanistic Account of the Cross-Lingual
         Safety Gap in Small Language Models},
  author={Colca, Robert-Mihai},
  journal={TBD},
  year={2028}
}
```

[p2]: ../paper2-benchmark/README.md
[p3]: ../paper3-alignment/README.md
