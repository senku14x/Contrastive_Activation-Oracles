#!/usr/bin/env python3
"""
measure_target_logits.py — OFF no-cue target behavior for every candidate (GPU).

Step (4) of data_construction_v2 §21. For each candidate measures the §10 OFF readout (letters at the
first answer position, NO appended cue — labels the exact pre-output state) for all three conditions:
  - neutral  = the ablated/bare-givens condition (load-bearing + competence baseline)
  - correct  = context_correct (the H_A reference)
  - flawed   = context_flawed  (the H_B / label condition)
Fills neutral/correct/flawed distributions + derived margins. Optional --on adds a reasoning-ON
cross-check (K sampled CoTs per condition; spec forbids finalizing labels from K=8 — default 32).

Writes data/candidates_measured.jsonl. Labelling (Stage A–D) is the separate label_candidates.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, ".")
from cao import ao_runtime as rt   # noqa: E402
from cao import dataset as D       # noqa: E402
from cao import measure as M       # noqa: E402

LETTERS = ("A", "B", "C", "D")


def _r(p):
    return {k: round(float(v), 4) for k, v in p.items()}


def _margin_over_rest(p, key):
    rest = [v for k, v in p.items() if k != key]
    return float(p[key] - max(rest)) if rest else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="data/candidates_unfiltered.jsonl")
    ap.add_argument("--out", default="data/candidates_measured.jsonl")
    ap.add_argument("--on", action="store_true", help="add reasoning-ON cross-check (slow)")
    ap.add_argument("--k", type=int, default=32, help="ON samples per condition")
    a = ap.parse_args()

    recs = [json.loads(l) for l in open(a.candidates)]
    model, tok = rt.load_oracle()
    lids = M.letter_ids(tok)
    print(f"measuring {len(recs)} candidates (OFF no-cue{'+ON cross-check' if a.on else ''})")

    out = []
    for i, r in enumerate(recs):
        suffix = r["shared_suffix"]
        conds = {
            "neutral": D.user_content(r.get("context_ablated") or "", suffix),
            "correct": D.user_content(r["context_correct"], suffix),
            "flawed": D.user_content(r["context_flawed"], suffix),
        }
        dist = {name: M.answer_distribution_nocue(model, tok, c, lids) for name, c in conds.items()}
        gold, fc = r["gold_option"], r["flawed_consistent_option"]
        r["neutral_distribution"] = _r(dist["neutral"]["p"])
        r["correct_distribution"] = _r(dist["correct"]["p"])
        r["flawed_distribution"] = _r(dist["flawed"]["p"])
        r["derived"].update({
            "neutral_argmax": dist["neutral"]["argmax"],
            "correct_argmax": dist["correct"]["argmax"],
            "flawed_argmax": dist["flawed"]["argmax"],
            "neutral_gold_margin": round(_margin_over_rest(dist["neutral"]["p"], gold), 4),
            "delta_flawed_consistent": round(float(dist["flawed"]["p"][fc] - dist["correct"]["p"][fc]), 4)
            if fc else None,
        })

        if a.on:
            on = {}
            for name in ("correct", "flawed"):
                d = M.answer_distribution(model, tok, conds[name], reasoning=True, k=a.k, lids=lids)
                on[name] = d.get("counts", {})
            r["on_cross_check"].update({
                "n_correct": a.k, "n_flawed": a.k,
                "correct_letter_counts": on["correct"], "flawed_letter_counts": on["flawed"],
                "on_agrees_with_off": (max(on["flawed"], key=on["flawed"].get) == dist["flawed"]["argmax"])
                if on["flawed"] else None,
            })

        out.append(r)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(recs)}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {a.out}  -> next: scripts/label_candidates.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
