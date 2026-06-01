# Paper 4 — Plan

> **Working title:** *"Detection, Not Execution: A Mechanistic Account of the
> Cross-Lingual Safety Gap in Small Language Models"*
>
> **Internal short name:** **the detection/execution split**.
>
> **Phase:** 4 / Months 17-24 of the PhD plan.
> **Target venue:** TACL (WoS, IF ~10.9, rolling submission, 2-3 month review).
> **Backup:** Pattern Recognition (WoS, IF ~7.5). Short version: BlackboxNLP @ EMNLP.
> **Role:** First author.
> **Status:** Plan locked. Implementation not started.

---

## 1. The wedge

Two of my own published results set this paper up, and a third (Paper 3's
explicit non-goal) leaves the door open:

- **Paper 2 (RoSafetyBench, CIKM 2026)** measured a **universal cross-lingual
  safety gap**: on 953 culturally-native Romanian prompts, all 20 evaluatees
  (17 open-weight + 3 frontier) refuse far less in Romanian than on the same
  intent class in English. The translated-from-English ablation showed
  models look ≥92.5% safe on mechanically-translated prompts but refuse only
  47-56% on culturally-native Romanian — a ~+44 pp gap.
- **Paper 3 (RD-DPO, EACL/NAACL 2028 submission)** tried to *close* that gap
  by restricting DPO to the residual blocks where the base model's refusal
  direction is most expressed. It produced a **negative result**: a consistent
  −10 pp jailbreak regression across Qwen-2.5-3B, Llama-3.2-3B, and Gemma-3-4b,
  with no cross-lingual gap closure.
- **Paper 3 plan §14 (non-goals)** explicitly states: *"We will not make
  Paper 4's mechanistic-interpretability claims here. The probe is used as a
  parameter selector; we don't need to argue why the refusal direction works
  at the Anthropic-circuit level. That argument is Paper 4."* And §3:
  *"if RD-DPO works because the refusal direction is the right thing to push
  on, the same probes become Paper 4's primary lens. If RD-DPO works without
  correlating with probe quality, that's a Paper 4 finding too."*

Paper 3 came back negative. That is the most interesting possible input to
Paper 4: **we have a documented intervention failure and we get to explain it
mechanistically.** This is a far stronger TACL story than a generic "where is
refusal encoded" study, because the mechanistic claim is anchored to a real,
published, failed intervention on real Romanian data.

## 2. Hypothesis

> **H1 (master).** The cross-lingual safety gap in small instruction-tuned LMs
> is localized in the **harmfulness-detection** subsystem (early-to-mid layers:
> "is this request harmful?"), not in the **refusal-execution** subsystem
> (mid-to-late layers: "emit a refusal given a harm judgment"). On Romanian
> harmful prompts the *detection* signal is attenuated relative to matched
> English prompts; the *execution* direction, once triggered, is approximately
> language-agnostic.

This decomposes into five falsifiable sub-claims, ordered by importance:

1. **H1a — Detection probes transfer poorly EN→RO.** A linear probe trained to
   classify harmful-vs-benign on English residual activations, evaluated
   zero-shot on Romanian activations at the same layer, suffers a large
   accuracy drop in the detection band (early-to-mid layers). Target: ≥ 15 pp
   transfer drop in the detection band.
2. **H1b — Execution/refusal probes transfer well EN→RO.** A probe trained to
   classify will-refuse-vs-will-comply (read off the model's *own* behavior)
   on English activations transfers to Romanian with a small drop in the
   execution band (mid-to-late layers). Target: ≤ 5 pp transfer drop.
3. **H1c — Causal: patching detection-band activations restores RO refusal.**
   Activation-patching English harmful-prompt activations into the Romanian
   forward pass at **detection-band** layers flips Romanian compliance →
   refusal on a meaningful fraction of cases; patching at execution-band layers
   does not (or does much less). Target: detection-band patch restores ≥ 30 pp
   of the behavioral gap; execution-band patch restores ≤ 10 pp.
4. **H1d — Paper 3's failure is explained by band mismatch.** Paper 3's
   `selected_blocks.json` (the refusal-direction top-k it trained on) lands in
   the **execution** band for all three shared anchors. A confirmatory
   *detection-band-targeted* DPO probe (one small run, reusing Paper 3's
   pipeline) closes more of the gap than Paper 3's execution-band selection,
   or — if it also fails — shows the deficit is not repairable by LoRA at this
   data budget (still a clean mechanistic finding).
5. **H1e — SAE features corroborate the split.** Using Gemma Scope SAEs on
   `gemma-2-2b-it`, harmfulness-detection features fire markedly less on
   Romanian harmful prompts than on English ones in detection-band layers,
   while refusal-execution features fire comparably once detection fires.

**Viability gate.** If H1a and H1c both fail (detection probes transfer
*well* and detection-band patching does *not* restore refusal more than
execution-band), the master hypothesis is wrong and we pivot (see §10). H1a
+ H1c are the load-bearing pair; H1b/H1d/H1e are corroboration.

## 3. Why this is an interpretability paper, not an eval note

The contribution is a **mechanistic causal account** of a behavioral
phenomenon, validated three independent ways (correlational probes, causal
patching, SAE features) and tied to a failed alignment intervention. It sits
in a clean gap:

| Work | Localizes refusal? | Cross-lingual? | Detection vs execution split? | Tied to a failed intervention? |
|------|--------------------|----------------|-------------------------------|--------------------------------|
| Arditi et al. 2024 (single direction) | Yes (execution) | No | No | No |
| Wang et al. 2025 (universal refusal dir.) | Yes (execution) | Yes (EN-centric langs) | No | No |
| Zhao et al. 2025 (harmfulness ≠ refusal) | Yes (both) | No | **Yes** (English only) | No |
| Lee et al. 2024 (DPO mech. understanding) | Partial | No | No | No (un-alignment, not failure) |
| **This paper** | **Yes (both bands)** | **Yes (RO low-resource)** | **Yes, cross-lingually** | **Yes (Paper 3 RD-DPO)** |

The novel joint move: **take the detection/execution decomposition
cross-lingual, into a genuinely low-resource language, and use it to causally
explain why a published alignment intervention failed.** Zhao et al. give us
the English-only split; Wang et al. give us execution universality; nobody has
put them together to explain a *low-resource* gap, and nobody has the matching
failed-intervention artefact that Paper 3 hands us.

## 4. Models

| Role | Model | Params | Why |
|------|-------|--------|-----|
| **SAE anchor** | `google/gemma-2-2b-it` | 2B | **Gemma Scope** ships pretrained JumpReLU SAEs on every layer/sublayer of Gemma-2-2B (Lieberum et al. 2024). H1e needs SAEs; this gives them for free — no SAE training. |
| **Cross-arch anchor** | `Qwen/Qwen2.5-3B-Instruct` | 3B | Shared Paper 3 anchor. Has `selected_blocks.json` + RD-DPO adapters → directly tests H1d. |
| **Cross-arch anchor** | `meta-llama/Llama-3.2-3B-Instruct` | 3B | Shared Paper 3 anchor. Weakest Paper-2 baseline, biggest behavioral gap to localize. |

Notes:
- Gemma-2-2b-it (not Gemma-3-4b-it, Paper 3's anchor) is chosen as the SAE
  anchor *because* Gemma Scope exists for it. We accept a small base-model
  mismatch with Paper 3 here and flag it; the probe + patching analyses
  (H1a-d) still run on the two exact Paper-3 anchors (Qwen, Llama), so the
  cross-paper link is preserved on those two.
- All three are small enough to run full forward passes with hooks on a single
  A100-40G with room for the reference (English) pass cached.
- No model receives gradient training in the headline analyses. The only
  training is linear probes (logistic regression on frozen activations,
  seconds) and the single confirmatory H1d DPO run (reuses Paper 3 code).

## 5. Data: contrastive sets

Reuse Paper 2's RoSafetyBench prompts and Paper 3's frozen splits. **No new
prompt authoring for the core sets**; the novelty is mechanistic, not a new
benchmark.

Four contrastive cells, plus a parallel set, per anchor:

| Set | Lang | Label | Source | Approx n |
|-----|------|-------|--------|----------|
| `harm_en` | EN | harmful | HarmBench standard + Paper 2 EN ablation set | 250 |
| `benign_en` | EN | benign | Paper 2 over-refusal EN + Alpaca-clean sample | 250 |
| `harm_ro` | RO | harmful | RoSafetyBench toxicity + jailbreak (culturally native) | 250 |
| `benign_ro` | RO | benign | RoSafetyBench over-refusal (benign-looking) | 250 |
| `parallel` | EN↔RO | harmful | RoSafetyBench cross-lingual 86 parallel pairs (Paper 2/3) | 86 |

Roles:
- **Probes (H1a/H1b)** train on `*_en`, evaluate transfer on `*_ro`.
- **Patching (H1c)** needs *aligned* EN/RO pairs of the same harmful intent →
  the `parallel` set is the patching workhorse (same prompt, two languages,
  matched token structure where possible).
- **SAE (H1e)** uses all four cells on the Gemma anchor.

**Behavioral labels for execution probes (H1b).** The "will-refuse vs
will-comply" label is read off the model's own greedy generation, judged by
the Paper 2 `gpt-5-mini` judge. This is the same labeling pipeline as Paper 2/3
— cross-paper consistency, no new annotation.

**Split discipline (Paper 2/3 lesson).** Probe train/eval split is frozen and
pre-registered. The `parallel` set is patching-only, never used to train a
probe. Holdout IDs intersect with Paper 3's `held_out_ids.json` so any
behavioral number can be matched same-prompt/same-judge across the two papers
(the `anonymous_artifacts/per_id_intersection.py` trick from Paper 3).

## 6. Method (summary; full spec in EXPERIMENT_DESIGN.md)

### 6.1 Activation capture
- Hook the residual stream at the output of every block, at the **last
  assistant-prefix position** (matches Paper 3 METHOD_DESIGN §1 exactly, so
  probe geometry is comparable).
- Greedy, deterministic. Cache to Drive as `(n, B, d_model)` bf16 tensors.

### 6.2 Detection vs execution probes
- **Detection probe** (per layer): logistic regression on `harm_* vs benign_*`
  activations. Train on EN, evaluate on EN (in-language ceiling) and RO
  (transfer). The EN→RO accuracy drop, per layer, is the H1a signal.
- **Execution probe** (per layer): logistic regression on `will-refuse vs
  will-comply` (behavioral label). Same EN-train / RO-eval transfer protocol;
  the H1b signal.
- **Band definition.** "Detection band" = the contiguous early-to-mid layers
  where the EN detection probe first saturates (in-language accuracy plateau).
  "Execution band" = the mid-to-late layers where the execution probe
  saturates. Bands are read off the EN in-language curves *before* looking at
  any RO transfer number (pre-registered procedure, avoids circularity).

### 6.3 Activation patching (causal)
- Run the RO forward pass; at a target layer, replace the RO residual at the
  assistant-prefix position with the EN residual from the matched `parallel`
  partner; continue the RO forward pass; measure the change in refusal
  (judge-scored) and in the refusal-direction projection.
- Sweep the patch layer across all blocks. The layer-resolved "refusal
  restoration" curve is the central causal figure. H1c predicts a peak in the
  detection band.
- Controls: (a) patch benign→benign (should not induce refusal); (b) patch
  RO→RO from a *different* harmful prompt (controls for "any perturbation
  triggers refusal"); (c) random-direction patch of matched norm.

### 6.4 SAE features (Gemma anchor only)
- Load Gemma Scope `gemma-scope-2b-it`/`-pt-res` SAEs per layer.
- Identify candidate **detection features** (fire on harmful, not benign,
  in EN) and **refusal features** (fire on refusal generations) via
  difference-in-means over SAE activations, with auto-interp labels.
- Compare firing rates EN vs RO for each feature class across the bands.
  H1e predicts detection features under-fire on RO in the detection band.

### 6.5 Paper 3 cross-reference (H1d)
- Load Paper 3's `data/probes/<short>/selected_blocks.json` for Qwen and Llama.
- Plot those selected blocks against the detection/execution band map. H1d
  predicts they fall in the execution band.
- Confirmatory run: re-run Paper 3's `03_train_rd_dpo.ipynb` once per anchor
  with `target_blocks` forced to the *detection* band instead of the
  refusal-direction top-k, then eval with Paper 3's `04_eval_safety.ipynb`.
  Compare gap closure. This is the one place Paper 4 touches training, and it
  reuses Paper 3's code wholesale.

## 7. Evaluation / what we report

### 7.1 Mechanistic axis (headline)
- Per-layer detection-probe EN→RO transfer drop (H1a) — line plot + table.
- Per-layer execution-probe EN→RO transfer drop (H1b) — same plot, second line.
- Layer-resolved patching refusal-restoration curve (H1c) — the causal figure.
- SAE detection/refusal feature firing EN vs RO (H1e) — Gemma anchor.

### 7.2 Behavioral anchoring
- Every mechanistic number is correlated against the **behavioral** refusal
  gap from Paper 2 for the same prompts (same judge). The mechanistic story
  must track the behavior, or it isn't an explanation.

### 7.3 Paper 3 explanation (H1d)
- Band membership of Paper 3's selected blocks.
- Detection-band-targeted DPO vs Paper 3's execution-band selection: Δ gap
  closure on the RoSafetyBench holdout, three seeds, Wilson CIs.

### 7.4 Statistical reporting
- Probe accuracies: 5-fold CV mean ± Wilson 95% CI.
- Patching effects: bootstrap 95% CI over prompts; permutation test vs the
  random-direction control.
- ≥ 3 seeds for any stochastic step. Spearman (exact-p) for the
  mechanistic-vs-behavioral correlation. Same statistical discipline as
  Paper 2 (R1 lesson) and Paper 3.

## 8. Ablations

- **Capture position.** Last assistant-prefix token vs mean-over-prompt vs
  last prompt token — does the band map move?
- **Probe family.** Logistic regression vs difference-in-means vs MLP probe —
  robustness of the transfer-drop signal (Paper 3 §9 lesson: probe-metric
  choice must be shown not to matter).
- **Layer granularity.** Block output vs attention-out vs MLP-out residual.
- **Patch granularity.** Whole-residual patch vs refusal-direction-only patch
  vs detection-direction-only patch.
- **Prompt-set size.** 50 / 100 / 250 per cell — stability of the band map.
- **SAE width / L0** (Gemma Scope ships multiple) — does feature firing
  conclusion survive SAE choice?

## 9. Ablations → figures → claims map

| Claim | Primary evidence | Figure |
|-------|------------------|--------|
| H1a | Detection probe EN→RO drop, detection band | fig: per-layer transfer drop |
| H1b | Execution probe EN→RO drop, execution band | fig: per-layer transfer drop (2nd line) |
| H1c | Patching restoration peaks in detection band | fig: layer-resolved restoration curve |
| H1d | Paper 3 blocks ∈ execution band; detection-targeted DPO Δ | fig: band map + bar |
| H1e | SAE detection features under-fire on RO | fig: feature-firing EN vs RO |
| master | mechanistic-vs-behavioral Spearman | fig: scatter |

## 10. Risk register

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Detection and execution bands overlap / aren't cleanly separable | Medium | Report the continuous per-layer curves regardless; the claim degrades gracefully to "the gap is more upstream than Paper 3's selection" even without a crisp boundary. |
| Patching is noisy at this model scale (3B) | Medium | Gemma-2-2b SAEs + denoising/noising patch variants (Heimersheim & Nanda 2024); bootstrap CIs; the three controls in §6.3. |
| H1a fails: detection probes transfer fine | Low-Medium | Then the gap is in execution or in tokenization/representation upstream of detection — pivot to "the gap is pre-detection (embedding/early-layer)" which is *also* a clean finding and still explains Paper 3. |
| Gemma Scope SAE conclusions don't generalize to Qwen/Llama (no SAEs there) | Medium | H1e is explicitly Gemma-only and framed as corroboration; H1a-d (probes+patching) carry the cross-arch claim and run on all three. |
| Reviewer: "probes are correlational, not causal" | Certain | That is exactly why H1c (activation patching) is load-bearing and H1a/H1b are framed as setup. Lead with the causal result. |
| Reviewer: "n=3 models is thin for a mechanistic claim" | High | Three architectures (Gemma/Qwen/Llama) is standard for this sub-field (Arditi 2024, Zhao 2025 use comparable counts); the scaling is over *layers*, not models; behavioral anchoring strengthens external validity. |
| Reviewer: "Romanian is arbitrary; why not a standard MI language?" | Medium | The whole point: RO is where we *have a measured gap and a failed intervention* (Papers 2-3). Generic-language MI cannot make the Paper-3 explanation claim. Frame RO as the controlled probe, not a limitation. |
| Scoop risk (detection/execution split is hot in 2025-26) | Medium-High | Move fast on H1a/H1c; the cross-lingual + failed-intervention framing is the moat. Maintain the bibliography watch (§13). |

**Pivot trigger.** If the Gemma pilot (week 4) shows detection probes transfer
EN→RO with < 5 pp drop *and* detection-band patching restores < 10 pp, the
master hypothesis is dead. Pivot in week 5 to the most-supported alternative
the pilot reveals (likely "pre-detection / embedding-level" localization, or
"execution band is the locus after all and Paper 3 failed for a data-budget
reason, not a band reason") and re-frame around that. Either way there is a
mechanistic paper; only the headline noun changes.

## 11. Authoring discipline (Paper 2/3 lessons, carried forward)

- **Notebooks are the runnable surface.** Every experiment is a Colab/A100
  notebook under `experiments/`. `src/` holds only helpers reused across more
  than one notebook (capture hooks, probe trainer, patching harness, SAE shim).
  No CLI, no Makefile-as-orchestrator. Notebooks pip-install inline and
  read/write Drive; `requirements.txt` is the env spec source of truth.
- `manuscript/` **gitignored** at repo init. Public repo never sees draft text.
- `EXPERIMENT_LOG.md` is private (gitignored). Public journal is README +
  `data/RELEASE_NOTES.md`.
- Reuse, don't copy: `sys.path` into Paper 2 `src/` for judges/llm_judge and
  into Paper 3 `src/` for probe helpers + `configs/`. One source of truth per
  artefact.
- Every API call records `finish_reason`, `usage.completion_tokens`,
  `usage.reasoning_tokens` (Paper 2 R10 lesson).
- LLM-judge: `gpt-5-mini` primary, `claude-opus-4.5` second-rater on a
  stratified subsample, Cohen's κ reported. Same protocol as Paper 2/3.
- Pre-register contrastive sets + band-definition procedure + probe configs
  before the first causal (patching) run. SHA-256 quoted in the paper.
- Public release: contrastive sets (CC BY 4.0), code (Apache 2.0), probe
  weights, SAE feature indices, patching tensors, judge labels. Zenodo + HF.

## 12. Timeline (8 weeks of focused work inside Months 17-24, overlapping Paper 3 writing)

| Weeks | Block | Deliverables |
|-------|-------|--------------|
| 1 | Lit pass + scaffold | Updated MI bibliography (2026 vintage); contrastive sets built + frozen; capture hooks stood up; pilot smoke test green. |
| 2 | Probes (H1a/H1b) | Per-layer detection + execution probes on all three anchors; EN→RO transfer curves; band map. |
| 3-4 | Patching (H1c) + **pilot/pivot gate** | Layer-resolved restoration curve on Gemma + Qwen; controls; **master-hypothesis go/no-go**. |
| 5 | SAE (H1e) | Gemma Scope feature analysis; detection vs refusal firing EN vs RO. |
| 6 | Paper 3 cross-ref (H1d) | Band membership of Paper 3 blocks; one detection-targeted DPO run per anchor; gap-closure delta. |
| 7-8 | Writing + figures + supervisor review | Full TACL draft; figures regenerated; release v1.0 to Zenodo + HF; submit. |

## 13. Open questions (resolve in week 1)

1. **SAE anchor confirmation.** Verify Gemma Scope `-it` residual SAEs cover
   all layers we need on `gemma-2-2b-it` (vs only `-pt` base). If `-it` SAEs
   are sparse, run H1e on the base model and note the it/pt caveat.
2. **Patching tooling.** TransformerLens vs nnsight vs raw hooks. TransformerLens
   has the cleanest patching API but model coverage (Gemma-2, Qwen-2.5,
   Llama-3.2) must be confirmed for the exact checkpoints; fall back to raw
   forward hooks (Paper 3 already uses these) if coverage is thin. Decide week 1.
3. **Band-definition procedure pre-registration.** Lock the "read bands off EN
   in-language saturation" rule in writing before any RO number is computed.
4. **Tokenization confound.** RO and EN tokenize to different lengths; the
   assistant-prefix position is well-defined but the *prompt* token count
   differs. Decide whether patching uses last-token-only (clean) or needs
   alignment (messy). Last-token-only is the default; document.

## 14. Non-goals (so we don't drift)

- We will **not** propose a new alignment method or a new SAE. We *use* Gemma
  Scope and standard probing/patching. (Paper 5 is where a new method lives.)
- We will **not** train SAEs from scratch. If a model lacks pretrained SAEs,
  it doesn't get the SAE analysis — probes + patching carry it.
- We will **not** expand beyond Romanian to a multi-language panel. RO is the
  controlled setting where we have the Paper-2 gap and the Paper-3 failure.
  Multi-language generalization is Paper 6 framing.
- We will **not** re-litigate Paper 3's data budget. H1d is a *mechanistic*
  cross-reference (which band), not a re-run of Paper 3's full sweep.
- We will **not** make capability/alignment-tax claims. That's Paper 3/5.

## 15. Cost ledger

This paper is almost free — the expensive parts (training, large API spend)
are not here. API cost is only judge calls for behavioral labels.

| Item | Calls | Per-call | Subtotal |
|------|-------|----------|----------|
| Behavioral refusal labels (3 anchors × ~600 prompts, greedy, judged once) | ~1,800 | $0.001 | ~$2 |
| H1d detection-targeted DPO eval (2 anchors × ~200 holdout × judge) | ~400 | $0.001 | <$1 |
| Second-rater κ audit (one-time, claude-opus-4.5 on 200 stratified) | 200 | $0.025 | ~$5 |
| **Headline budget** | | | **~$8** |
| Safety pad | | | +$20 |
| **Recommended OpenRouter cap** | | | **$30** |

Compute: probes are CPU-seconds on cached activations; patching is the only
real GPU cost (~full forward pass per (prompt, patch-layer) — batched, a few
A100-hours per anchor); SAE inference is cheap. Total < 20 A100-hours, well
inside a month of Colab Pro+ quota.
