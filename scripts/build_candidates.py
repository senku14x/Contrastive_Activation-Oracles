#!/usr/bin/env python3
"""
build_candidates.py — emit data/candidates_unfiltered.jsonl in the §20 record schema.

Step (2) of data_construction_v2 §21. Pure Python (no model/tokenizer) — runs anywhere. For each
Family-L item and each Family-P (item, template) pair, write one record with the three condition
contexts (ablated / correct / flawed), the shared suffix, gold, and flawed-consistent option.
Measured fields are left null; they are filled by measure_target_logits.py on the GPU box.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, ".")
from cao import dataset as D       # noqa: E402
from cao import family_l as L      # noqa: E402
from cao import family_p as P      # noqa: E402


def _records():
    # Family L (primary): one record per item.
    for it in L.FAMILY_L:
        suffix = L.suffix_for(it)
        yield D.candidate_record(
            candidate_id=it["slug"], family="L", question=it["question"], options=it["options"],
            gold_option=it["gold_option"], flawed_consistent_option=it["flawed_consistent_option"],
            context_correct=it["context_correct"], context_flawed=it["context_flawed"],
            context_ablated=it["context_ablated"], shared_suffix=suffix,
            source_item_id=it["slug"], domain=it["domain"], flaw_subtype=it["flaw_subtype"],
        )
    # Family P (validation shakedown): one record per (item, template).
    for rid, it, tpl in P.enumerate_pairs():
        _, weak, strong = tpl
        w = it["wrong_option"]
        suffix = P.suffix_for(it)
        yield D.candidate_record(
            candidate_id=rid, family="P", question=it["question"], options=it["options"],
            gold_option=it["gold_option"], flawed_consistent_option=w,
            context_correct=weak.format(w=w), context_flawed=strong.format(w=w),
            context_ablated="", shared_suffix=suffix,
            source_item_id=it["slug"], domain="competent-fact-MCQ", flaw_subtype=None,
            extra={"template_id": tpl[0], "named_wrong_option": w},
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/candidates_unfiltered.jsonl")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    recs = list(_records())
    with open(a.out, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    nL = sum(r["family"] == "L" for r in recs)
    nP = sum(r["family"] == "P" for r in recs)
    print(f"wrote {len(recs)} candidate records to {a.out}  (L={nL}, P={nP})")
    print("next: scripts/verify_token_invariants.py (tokenizer) then scripts/measure_target_logits.py (GPU)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
