#!/usr/bin/env python3
"""
ablation_verify.py — STRUCTURAL load-bearing check (NO model; data_construction_v2 §4 / §21 step 3).

Sets `ablation_verified_load_bearing` by inspection — it's a semantic property, not a model call. A
Family-L item is load-bearing iff deleting the manipulated element (the worked step) leaves the problem
under-determined. By construction that holds when:
  - context_ablated (the shared givens) is contained in BOTH context_correct and context_flawed
    (so the ONLY thing ablation removes is the worked step),
  - context_correct != context_flawed (a real correct-vs-flawed contrast exists),
  - the gold option is an indeterminacy option ("cannot be determined" / "undetermined" / ...),
    i.e. with the worked step removed the honest answer is "cannot be determined" → under-determined,
  - gold != flawed_consistent.

Family P is not load-bearing in this sense (it varies commitment, not content) → field = null (N/A).
Rewrites the candidates file in place with the field set. Run AFTER build_candidates, BEFORE measuring.
"""
from __future__ import annotations

import argparse
import json

INDET = ("cannot be determined", "undetermined", "not enough", "cannot be inferred",
         "insufficient", "unsettled")


def structural_ok(r):
    reasons = []
    abl = r.get("context_ablated") or ""
    if not abl:
        reasons.append("empty context_ablated")
    if abl and abl not in r["context_correct"]:
        reasons.append("ablated givens not contained in context_correct")
    if abl and abl not in r["context_flawed"]:
        reasons.append("ablated givens not contained in context_flawed")
    if r["context_correct"] == r["context_flawed"]:
        reasons.append("correct == flawed (no contrast)")
    g = r["gold_option"]
    if not any(k in r["options"][g].lower() for k in INDET):
        reasons.append("gold is not an indeterminacy option (ablated would not be under-determined)")
    if g == r["flawed_consistent_option"]:
        reasons.append("gold == flawed_consistent")
    return (not reasons), reasons


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="data/candidates_unfiltered.jsonl")
    a = ap.parse_args()

    recs = [json.loads(l) for l in open(a.candidates)]
    npass = nfail = 0
    for r in recs:
        if r["family"] != "L":
            r["ablation_verified_load_bearing"] = None     # N/A for Family P
            continue
        ok, reasons = structural_ok(r)
        r["ablation_verified_load_bearing"] = bool(ok)
        if ok:
            npass += 1
        else:
            nfail += 1
            print(f"FAIL {r['candidate_id']}: {reasons}")

    with open(a.candidates, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    nL = sum(r["family"] == "L" for r in recs)
    print(f"\nFamily L: {npass}/{nL} ablation-verified load-bearing (field set true). "
          f"Family P: N/A (commitment, not content).")
    if nfail:
        print("** Fix or drop the failing items before measurement (§4: must ablate to under-determined).")
    else:
        print("All Family-L items ablate to under-determined. field is no longer null -> ready to measure.")
    return 1 if nfail else 0


if __name__ == "__main__":
    raise SystemExit(main())
