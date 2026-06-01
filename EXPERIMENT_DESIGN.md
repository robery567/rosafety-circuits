# Experiment Design Spec — Detection/Execution Split

> Companion to `PAPER4_PLAN.md`. The plan defines the contribution; this file
> defines what we actually implement. Keep it precise — anything ambiguous
> here surfaces as a reviewer question or a re-run. Mirrors Paper 3's
> `METHOD_DESIGN.md` format.

---

## 1. Notation

- `M` — instruction-tuned anchor model (Gemma-2-2b-it, Qwen-2.5-3B-Instruct,
  Llama-3.2-3B-Instruct). Frozen for all headline analyses.
- `B` — number of transformer blocks in `M`.
- `h_ℓ(x)` — residual-stream activation at the **output of block `ℓ`**, read at
  the **last position of the assistant prefix** (immediately before the first
  generated token), given prompt `x`. Same capture point as Paper 3
  METHOD_DESIGN §1, so probe geometry is directly comparable.
- `x^{en}`, `x^{ro}` — an English / Romanian prompt. For the `parallel` set,
  `(x^{en}_i, x^{ro}_i)` is a matched harmful-intent pair.
- `harm`, `benign` — intent labels (set by construction).
- `refuse`, `comply` — *behavioral* labels (read off `M`'s own greedy
  generation, judged by `gpt-5-mini`).
- `p^det_ℓ` — detection probe at layer `ℓ` (harm vs benign).
- `p^exe_ℓ` — execution probe at layer `ℓ` (refuse vs comply).
- `r̂(x)` — judge-scored refusal indicator for `M`'s greedy generation on `x`.

## 2. Contrastive sets (pre-registered, frozen)

### 2.1 Construction

Five cells, built once per anchor, written to `data/contrastive/<short>/`:

| Cell | Lang | Intent | n target | Source |
|------|------|--------|----------|--------|
| `harm_en` | EN | harmful | 250 | HarmBench standard behaviors + Paper 2 EN translated-ablation set |
| `benign_en` | EN | benign | 250 | Paper 2 over-refusal EN + Alpaca-cleaned sample |
| `harm_ro` | RO | harmful | 250 | RoSafetyBench toxicity + jailbreak (culturally native) |
| `benign_ro` | RO | benign | 250 | RoSafetyBench over-refusal (benign-but-risky-looking) |
| `parallel` | EN↔RO | harmful | 86 | RoSafetyBench cross-lingual parallel pairs (Paper 2/3 eval-only set) |

Class balance is enforced 50/50 within each language for the probe sets.
The `parallel` set is **patching-only** and never trains a probe.

### 2.2 Behavioral labels (for the execution probe)

For every prompt in all cells:
1. Generate one greedy completion from `M` (max 256 new tokens, matched to
   Paper 2/3 generation config).
2. Judge with `gpt-5-mini` (Paper 2 `src/judges.py` refusal prompt). Record
   `r̂(x) ∈ {refuse, comply}` plus `finish_reason` and token usage (R10 lesson).
3. The execution probe's label is `r̂(x)`; the detection probe's label is the
   construction intent.

### 2.3 Splits

- Probe **train** = 70% of `{harm_en, benign_en}` (EN only — probes are
  trained in English and tested cross-lingually).
- Probe **eval-EN** = remaining 30% of EN (in-language ceiling).
- Probe **eval-RO** = all of `{harm_ro, benign_ro}` (transfer target).
- Split is frozen, SHA-256 recorded in `data/splits/probe_split.json`, quoted
  in the manuscript. Holdout IDs intersect Paper 3 `held_out_ids.json` so
  behavioral numbers match same-prompt/same-judge across papers.

## 3. Activation capture

### 3.1 Protocol
- Forward `M` on each prompt with the model's chat template applied; **no
  generation needed for capture** — we read `h_ℓ` at the assistant-prefix
  position in a single forward pass with hooks on every block output.
- Greedy / deterministic; `torch.no_grad()`; bf16.
- Capture **all** `B` blocks in one pass (one hook per block, store last-position
  hidden state only → `(B, d_model)` per prompt).
- Batch by language cell; write `(n_cell, B, d_model)` bf16 tensors to
  `data/activations/<short>/<cell>.pt` with an index JSON mapping row→prompt id.

### 3.2 Ablation captures (§8 of plan)
Also cache, behind flags (off by default to save Drive):
- attention-out residual and MLP-out residual (sub-layer granularity);
- mean-over-prompt and last-prompt-token positions.

## 4. Detection vs execution probes

### 4.1 Probe family (default)
Per layer `ℓ`, an L2-regularized logistic regression on `h_ℓ`:
- standardize features on the train split (store mean/std);
- `C` selected by 5-fold CV on the EN train split;
- report 5-fold CV accuracy on EN (ceiling), accuracy on eval-EN held-out,
  and zero-shot accuracy on eval-RO (transfer).

`p^det_ℓ`: labels = harm/benign. `p^exe_ℓ`: labels = refuse/comply.

### 4.2 Transfer drop (the H1a/H1b signal)
```
drop^det_ℓ = acc_EN(p^det_ℓ) − acc_RO(p^det_ℓ)
drop^exe_ℓ = acc_EN(p^exe_ℓ) − acc_RO(p^exe_ℓ)
```
H1a: `drop^det` is large in the detection band. H1b: `drop^exe` is small in
the execution band.

### 4.3 Band definition (pre-registered, computed before any RO number)
1. Compute `acc_EN(p^det_ℓ)` and `acc_EN(p^exe_ℓ)` for all `ℓ` (EN only).
2. **Detection band** = the contiguous run of layers from where `acc_EN(p^det)`
   first reaches 95% of its max to where it starts declining (or to mid-net,
   whichever first). Typically early-to-mid.
3. **Execution band** = the contiguous run from where `acc_EN(p^exe)` first
   reaches 95% of its max onward. Typically mid-to-late.
4. Bands are written to `results/<short>/bands.json` and frozen *before*
   `eval-RO` is touched. This ordering kills the circularity objection.

### 4.4 Probe-family ablation
Repeat with difference-in-means probe and a 1-hidden-layer MLP probe. The
transfer-drop *shape* (where the gap is, not its exact size) must survive
probe-family choice (Paper 3 §9 discipline).

## 5. Activation patching (causal — the load-bearing experiment)

### 5.1 Core operation (RO ← EN, per layer)
For each parallel pair `(x^{en}_i, x^{ro}_i)` where `M` complies on RO and
refuses on EN (the cases that *exhibit* the gap):
1. Run `M(x^{en}_i)`, cache `h_ℓ(x^{en}_i)` for the target layer `ℓ`.
2. Run `M(x^{ro}_i)`; at block `ℓ`, at the assistant-prefix position, **replace**
   the RO residual with the cached EN residual; let the forward pass continue
   through all downstream layers (the patch propagates).
3. Generate greedily from the patched run; score `r̂` (judge) and the
   refusal-direction projection (Arditi-style direction computed on this model).
4. **Restoration** at layer `ℓ` = fraction of gap-exhibiting pairs flipped
   comply→refuse by the patch.

Sweep `ℓ` over all blocks → the **layer-resolved restoration curve**. H1c
predicts the peak sits in the detection band.

### 5.2 Controls
- **C1 benign→benign:** patch a benign EN residual into a benign RO pass.
  Must *not* induce refusal (else the patch is just perturbing toward refusal).
- **C2 mismatched harmful:** patch EN residual from a *different* harmful prompt.
  Tests whether *any* harmful EN signal works or the *matched* one is needed.
- **C3 random direction:** add a random vector of matched norm at layer `ℓ`.
  Permutation baseline for the restoration metric.
- **C4 noising direction (reverse):** patch RO→EN (does removing the EN
  detection signal *break* EN refusal?). Symmetric evidence.

### 5.3 Patch granularity ablation
- whole-residual patch (default);
- refusal-direction-only patch (project EN residual onto the refusal direction,
  add only that component);
- detection-direction-only patch (the `p^det_ℓ` weight vector direction).
Distinguishes "the whole representation differs" from "only the
detection-relevant component differs".

### 5.4 Tooling
Default: raw forward hooks (Paper 3 `src/` already implements assistant-prefix
hooks; extend for mid-forward replacement). Cross-check a subset against
TransformerLens `run_with_hooks` if the exact checkpoints are supported.
Decided week 1 (plan §13.2).

## 6. SAE feature analysis (Gemma anchor only — H1e)

### 6.1 SAEs
Gemma Scope JumpReLU residual SAEs for `gemma-2-2b` (Lieberum et al. 2024),
per layer, loaded via `sae_lens`. Use the `-it` SAEs if they cover the needed
layers; else `-pt` with a documented caveat (plan §13.1).

### 6.2 Feature classes
- **Detection features:** SAE latents whose activation separates `harm_en` from
  `benign_en` (difference-in-means over SAE activations, top-m by separation).
- **Refusal features:** SAE latents whose activation separates refusal vs
  compliance generations.
- Auto-interp label each via the Neuronpedia/`sae_lens` description if available;
  otherwise report max-activating examples.

### 6.3 Cross-lingual firing comparison
For each feature class and band, compare mean firing on EN vs RO harmful
prompts. H1e: detection features under-fire on RO in the detection band;
refusal features fire comparably (conditional on detection firing).

### 6.4 SAE robustness ablation
Repeat with a second Gemma Scope width / L0 setting. Conclusion (direction of
the EN-vs-RO firing gap) must survive (plan §8).

## 7. Paper 3 cross-reference (H1d)

### 7.1 Band membership
Load `paper3-alignment/data/probes/<short>/selected_blocks.json` for `qwen2.5-3b`
and `llama-3.2-3b`. Overlay on this paper's `bands.json`. H1d predicts the
selected blocks fall in the **execution** band.

### 7.2 Confirmatory detection-targeted DPO run
One run per anchor, reusing Paper 3 wholesale:
1. Set Paper 3 `target_blocks` = this paper's detection band (top-k by detection
   probe accuracy, k matched to Paper 3's k=4).
2. Run Paper 3 `experiments/03_train_rd_dpo.ipynb` (one seed; if promising,
   three seeds).
3. Eval with Paper 3 `experiments/04_eval_safety.ipynb` on the RoSafetyBench
   holdout.
4. Report Δ gap-closure vs Paper 3's execution-band selection.

Outcomes both publishable:
- detection-band > execution-band → confirms H1d, "Paper 3 picked the wrong band";
- both fail → the deficit isn't LoRA-repairable at this budget, and the
  mechanistic localization stands independently of the fix.

## 8. Result schema (locked v1)

`results/<short>/<analysis>.json`. Example for the probe analysis:

```json
{
  "anchor_model": "google/gemma-2-2b-it",
  "short": "gemma-2-2b",
  "analysis": "linear_probes",
  "n_blocks": 26,
  "capture_position": "last_assistant_prefix",
  "probe_family": "logreg",
  "bands": {"detection": [3, 4, 5, 6, 7, 8], "execution": [14, 15, 16, 17, 18, 19, 20]},
  "per_layer": [
    {"layer": 0, "det_acc_en": 0.61, "det_acc_ro": 0.55, "det_drop": 0.06,
     "exe_acc_en": 0.58, "exe_acc_ro": 0.56, "exe_drop": 0.02,
     "det_acc_en_wilson95": [0.55, 0.67]}
  ],
  "behavioral_gap_ref": {"source": "paper2", "ro_refusal": 0.31, "en_refusal": 0.93},
  "mech_vs_behavioral_spearman": {"rho": 0.74, "p": 0.003},
  "contrastive_sets_sha256": "…",
  "probe_split_sha256": "…"
}
```

Patching result schema:
```json
{
  "anchor_model": "google/gemma-2-2b-it",
  "analysis": "activation_patching",
  "n_pairs_gap_exhibiting": 41,
  "per_layer": [
    {"layer": 5, "restoration": 0.34, "boot95": [0.21, 0.47],
     "ctrl_benign": 0.02, "ctrl_mismatch": 0.11, "ctrl_random": 0.03}
  ],
  "peak_layer": 6, "peak_band": "detection"
}
```

## 9. Public release

Following Paper 2/3 discipline:
- **Zenodo** — contrastive sets (CC BY 4.0), probe weights + standardizers,
  SAE feature indices, patching restoration tensors, judge labels, response
  JSONs, `bands.json`.
- **Hugging Face** — `rosafety-interp/contrastive-v1` dataset; probe + feature
  artefacts; reproducibility notebooks.
- **GitHub** — code, harness, configs, datasheet. `manuscript/` excluded.
- Anonymized at submission; public IDs at camera-ready.

## 10. Threats to validity (bake in early)

1. **Correlational probes ≠ causation.** Mitigation: H1c activation patching is
   the causal claim; probes are explicitly framed as setup. Lead causal.
2. **Band boundaries are fuzzy.** Mitigation: report continuous curves; the
   claim survives as "more upstream than Paper 3's band" even without a crisp
   cut. Pre-registered band rule (§4.3) prevents post-hoc gerrymandering.
3. **Tokenization confound (RO vs EN length).** Mitigation: last-token-only
   patching by default; alignment variant as ablation; documented (plan §13.4).
4. **SAE conclusions are Gemma-only.** Mitigation: H1e is corroboration; H1a-d
   carry cross-arch on all three anchors.
5. **Judge confounding.** Same `gpt-5-mini` labels behavior here and in Paper 2.
   Mitigation: κ vs `claude-opus-4.5` second-rater on a stratified 200-sample;
   cross-paper consistency.
6. **Base/it mismatch for Gemma (Scope is pt-trained).** Mitigation: prefer
   `-it` SAEs; if using `-pt`, run probes/patching on `-it` and SAEs on `-pt`
   with the caveat stated; the two need not be the same model for H1e to
   corroborate a direction-of-effect already shown causally on `-it`.

## 11. Pre-registration

Before the first patching run we commit, with git timestamps:
- `data/contrastive/<short>/*.jsonl` (the frozen cells) + SHA-256.
- `data/splits/probe_split.json` + SHA-256.
- `results/<short>/bands.json` (bands read off EN in-language curves only).
- This document with no further substantive edits to §2-§7.

Same lock-then-run discipline as Paper 2 (keyword-vs-judge) and Paper 3
(probe set before training).
