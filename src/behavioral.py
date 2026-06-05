"""Behavioral labeling for the execution probe + patching pair selection.

Greedy-generate one completion per contrastive-set prompt, then judge
refuse/comply with the Paper 2 harness (`gpt-5-mini` + REFUSAL_SYSTEM), so the
labels are produced by the exact same protocol as Paper 2/3 (cross-paper
consistency). Outputs:

  - the execution-probe target label per prompt (refuse vs comply);
  - the per-prompt refusal flag used to find gap-exhibiting parallel pairs
    (RO comply + EN refuse) for activation patching (H1c).

Every record keeps the raw completion + finish_reason (Paper 2 R10 lesson).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch


@torch.no_grad()
def generate_completions(model, tokenizer, prompts: Sequence[str], *,
                         device: str = "cuda", batch_size: int = 8,
                         max_new_tokens: int = 256) -> list[dict]:
    """Greedy one completion per prompt. Left-padding assumed (set
    tokenizer.padding_side='left') so the prompt length is shared per batch."""
    model.eval()
    rows: list[dict] = []
    for start in range(0, len(prompts), batch_size):
        batch = list(prompts[start:start + batch_size])
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False, add_generation_prompt=True,
            ) for p in batch
        ]
        enc = tokenizer(texts, return_tensors="pt", padding=True,
                        truncation=True, max_length=1024).to(device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        prompt_len = enc["input_ids"].shape[1]
        for i in range(len(batch)):
            new = gen[i, prompt_len:]
            completion = tokenizer.decode(new, skip_special_tokens=True).strip()
            finished = bool((new == tokenizer.eos_token_id).any().item())
            rows.append({"completion": completion,
                         "finish_reason": "stop" if finished else "length"})
    return rows


def label_refusals(judge, triples: Sequence[tuple[str, str, str]], *,
                   workers: int = 8) -> dict[str, dict]:
    """Judge refuse/comply for ``(id, prompt_text, response_text)`` triples.

    Returns ``{id: {"label": "refuse"|"comply", "justification": str}}`` using
    the Paper 2 REFUSAL judge prompt (label "refusal"/"compliance" -> mapped).
    """
    from judges import REFUSAL_SYSTEM, REFUSAL_USER_TEMPLATE  # Paper 2 src/
    items = [
        (pid, REFUSAL_SYSTEM,
         REFUSAL_USER_TEMPLATE.format(prompt=pt, response=(rt or "[empty response]")))
        for pid, pt, rt in triples
    ]
    raw = judge.classify_many(items, workers=workers)
    out: dict[str, dict] = {}
    for pid, _, _ in triples:
        lbl = raw.get(pid, {}).get("label", "compliance")
        out[pid] = {"label": "refuse" if lbl == "refusal" else "comply",
                    "justification": raw.get(pid, {}).get("justification", "")}
    return out


def behavioral_labels_for_cells(model, tokenizer, judge, contrastive_dir: Path, *,
                                cells=("harm_en", "benign_en", "harm_ro", "benign_ro"),
                                device: str = "cuda", batch_size: int = 8) -> Path:
    """Generate + judge every prompt in the single-language cells; also score
    both sides of the parallel set. Writes ``behavioral_labels.jsonl`` keyed by
    the prompt id (parallel rows get ``<id>::en`` / ``<id>::ro``). Idempotent:
    re-running reuses the judge's on-disk cache."""
    out_path = contrastive_dir / "behavioral_labels.jsonl"
    records: list[dict] = []

    def _read(cell):
        p = contrastive_dir / f"{cell}.jsonl"
        return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []

    # Single-language cells.
    for cell in cells:
        rows = _read(cell)
        if not rows:
            continue
        gens = generate_completions(model, tokenizer, [r["text"] for r in rows],
                                    device=device, batch_size=batch_size)
        triples = [(r["id"], r["text"], g["completion"]) for r, g in zip(rows, gens)]
        labels = label_refusals(judge, triples)
        for r, g in zip(rows, gens):
            records.append({"id": r["id"], "cell": cell, "lang": r["lang"],
                            "intent": r["label"], **g, **labels[r["id"]]})

    # Parallel set: score EN and RO sides separately (for gap-exhibiting pairs).
    par = _read("parallel")
    if par:
        for side in ("en", "ro"):
            texts = [r[f"text_{side}"] for r in par]
            gens = generate_completions(model, tokenizer, texts, device=device, batch_size=batch_size)
            triples = [(f"{r['id']}::{side}", r[f"text_{side}"], g["completion"])
                       for r, g in zip(par, gens)]
            labels = label_refusals(judge, triples)
            for r, g in zip(par, gens):
                key = f"{r['id']}::{side}"
                records.append({"id": key, "cell": "parallel", "lang": side,
                                "intent": "harmful", **g, **labels[key]})

    with open(out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
    return out_path


def gap_exhibiting_pairs(behavioral_labels_path: Path) -> list[str]:
    """Parallel ids where EN refuses and RO complies — the cases that exhibit
    the cross-lingual gap and drive the patching experiment (H1c)."""
    by_key: dict[str, str] = {}
    for line in Path(behavioral_labels_path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["cell"] == "parallel":
            by_key[r["id"]] = r["label"]
    pairs = []
    for key, lbl in by_key.items():
        if key.endswith("::en") and lbl == "refuse":
            base = key[:-4]
            if by_key.get(f"{base}::ro") == "comply":
                pairs.append(base)
    return sorted(pairs)
