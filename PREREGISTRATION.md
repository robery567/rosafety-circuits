# Paper 4 — Pre-registration record

> Locks the analysis-defining artefacts before any causal (patching) run, with a
> git timestamp. Mirrors Paper 2 (keyword-vs-judge) and Paper 3 (probe set before
> training): the actual data lives in `data/` (gitignored, Drive/Zenodo), so the
> *hashes* are recorded here in a tracked file and quoted in the manuscript.
>
> Any change to a `final` artefact requires a new dated entry below, never an
> in-place edit.

---

## 1. Contrastive sets (EXPERIMENT_DESIGN §2)

Built by `src/contrastive.py::build_all(..., with_en=False)` from the committed
Paper 2 RoSafetyBench files in `paper2-benchmark/benchmark/expanded/`.
Deterministic (seed 17, stable sort by id); byte-identical across all three
anchor directories for the offline cells.

| Cell | n | Status | SHA-256 |
|------|---|--------|---------|
| `harm_ro` | 100 | **final** | `1ca385525293ed23a6aa9e35982897433731ab73206fc5f556be44116e2611da` |
| `benign_ro` | 100 | **final** | `1994b34a3fdb3b7c70f0cdb8e749b123e3dc9fb5df23f5a3747faf119389ced7` |
| `parallel` | 61 | **final** | `d0a8b8c0b6bcb9bdbbedca044c50693789932fd026af6c27e36e523c6f8e75a4` |
| `harm_en` | 36 | core (local); HarmBench top-up pending → re-freezes on Colab | `c6ac3bc68261c87f41be53f94219274f3749fd349a94c9bbc72f7fc8bc16bf6c` |
| `benign_en` | — | **pending** (XSTest-safe; needs `datasets` + network, built on Colab) | — |

**Provenance / decisions locked here:**
- `harm_ro` ← RoSafetyBench `toxicity` (261) + `jailbreak` (97), all
  `expected_behavior == refuse`; deterministic 100-sample.
- `benign_ro` ← RoSafetyBench `overrefusal` (100), all `expected_behavior == answer`
  (benign-but-risky-looking; the hard-negative class for harmfulness detection).
- `parallel` / `harm_en` ← RoSafetyBench `crosslingual`, filtered on
  **`category`** (NOT `expected_behavior`, which is corrupted for benign rows —
  22/25 benign prompts such as "What is the capital of Romania?" are mislabeled
  `refuse`). `harm_en` (probe) uses `category == "harmful"` (36, clean harmful);
  the `parallel` *patching* pool uses `category ∈ {harmful, bias}` (61), since
  bias probes are matched, safety-relevant EN/RO pairs and the harmful-only
  pool (36) yielded too few gap-exhibiting pairs on safe anchors (Gemma-3-4b:
  6/36). Each parallel row keeps `harm_type` so analysis can split harmful-only
  vs harmful+bias. id-prefix (`cro_harm_`/`cro_bias_`) and `category` agree.
- EN benign class = **XSTest-safe** (built on Colab) to match the
  benign-but-risky semantics of the RO `overrefusal` class, so the H1a EN→RO
  transfer measurement is not confounded by a benign-class distribution shift.

## 2. Band-definition procedure (EXPERIMENT_DESIGN §4.3) — locked

Bands are read off the **EN in-language** probe-accuracy curves only, *before*
any Romanian transfer number is computed:

- **Detection band** = contiguous layers from where the EN detection probe first
  reaches 95% of its max accuracy, up to where the execution band begins.
- **Execution band** = contiguous layers from where the EN execution probe first
  reaches 95% of its max accuracy, onward.
- Written to `results/<short>/bands.json` and frozen before `eval-RO` is touched.

This ordering is the pre-registered defence against the circularity objection
("you defined the bands to fit the result").

## 3. Pending (freeze on first Colab run, dated entry appended here)

- `harm_en` final (after HarmBench top-up to 250) + `benign_en` (XSTest-safe, 250).
- `data/splits/probe_split.json` — the 70/30 EN train/eval-EN split + RO eval set.
- `results/<short>/bands.json` — once EN probes are fit.

---

### Log

- **2026-06-01** — Offline cells frozen (`harm_ro`, `benign_ro`, `parallel`,
  `harm_en` core). Band-definition rule locked. EN cells + splits pending Colab.
- **2026-06-01 (rev)** — `parallel` widened from harmful-only (36) to
  harmful+bias (61) after the first Gemma-3-4b run yielded only 6 gap-exhibiting
  pairs; new SHA recorded above. `harm_en` (probe) unchanged (clean harmful).
