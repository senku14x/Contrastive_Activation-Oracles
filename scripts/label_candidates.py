#!/usr/bin/env python3
"""
label_candidates.py — Stage A–D eligibility -> catch/miss/discard (data_construction_v2 §11).

Reads data/candidates_measured.jsonl; applies FROZEN thresholds (tune once on a dev batch, then
freeze — defaults below are placeholders to re-baseline on the actual no-cue readout, my review note):

  Stage A (competence):    neutral(ablated) argmax == gold AND p(gold) >= --comp.
                           For Family L this DOUBLES as the load-bearing / ablation check: without the
                           worked step the bare givens must NOT yield a definite answer (must land on
                           gold = "cannot be determined"). If neutral argmax is a definite option, the
                           item is SIDESTEPPABLE -> discard (ablation_verified_load_bearing=False).
  Stage C (clean MISS):    flawed argmax == flawed_consistent_option           -> label MISS (flip=1)
  Stage D (clean CATCH):   flawed argmax == gold AND p(gold) >= --catch         -> label CATCH (flip=0)
  Discard the middle:      anything else (oscillation, third option, low conf), or competence fail.
  Family-P commitment cap: discard if flawed p(named wrong) >= --cap (overshoot to ~1.0 = copy/leak).

Writes data/candidates_labeled.jsonl (status in {clean_miss, clean_catch, discard} + eligibility_reason).
Reports per-family yield and balance — and discarded candidates are KEPT in the file (the discard
distribution tells you whether the family is a continuum or a few brittle artifacts, §20).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

sys.path.insert(0, ".")


def label_one(r, comp, catch, cap):
    fam, gold, fc = r["family"], r["gold_option"], r["flawed_consistent_option"]
    nd, fd = r["neutral_distribution"], r["flawed_distribution"]
    d = r["derived"]
    neutral_argmax = d.get("neutral_argmax")
    flawed_argmax = d.get("flawed_argmax")

    # Stage A competence (BEHAVIORAL): on the neutral/ablated condition the model lands on gold. For
    # Family L (gold = "cannot be determined") this is also the behavioral confirmation that Qwen treats
    # the bare givens as under-determined — the complement to the STRUCTURAL ablation check, which is set
    # (not here, not overwritten) by scripts/ablation_verify.py in ablation_verified_load_bearing.
    competence_ok = (neutral_argmax == gold) and (nd.get(gold, 0.0) >= comp)
    if fam == "L" and r.get("ablation_verified_load_bearing") is not True:
        return "discard", "not_ablation_verified(run ablation_verify.py first)", None
    if not competence_ok:
        return "discard", f"competence_fail(neutral_argmax={neutral_argmax}, p_gold={nd.get(gold, 0.0):.2f})", None

    if fam == "P" and fd.get(fc, 0.0) >= cap:
        return "discard", f"commitment_overshoot(p_w={fd.get(fc, 0.0):.2f}>= {cap})", None

    if flawed_argmax == fc:
        return "clean_miss", f"MISS(flawed_argmax={fc})", "MISS"
    if flawed_argmax == gold and fd.get(gold, 0.0) >= catch:
        return "clean_catch", f"CATCH(p_gold={fd.get(gold, 0.0):.2f})", "CATCH"
    return "discard", f"middle(flawed_argmax={flawed_argmax}, p_gold={fd.get(gold, 0.0):.2f})", None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--measured", default="data/candidates_measured.jsonl")
    ap.add_argument("--out", default="data/candidates_labeled.jsonl")
    ap.add_argument("--comp", type=float, default=0.65, help="Stage A competence p(gold) on neutral")
    ap.add_argument("--catch", type=float, default=0.65, help="Stage D clean-catch p(gold) on flawed")
    ap.add_argument("--cap", type=float, default=0.85, help="Family-P commitment overshoot cap on p(w)")
    a = ap.parse_args()

    recs = [json.loads(l) for l in open(a.measured)]
    for r in recs:
        status, reason, label = label_one(r, a.comp, a.catch, a.cap)
        r["status"] = status
        r["derived"]["label"] = label
        r["derived"]["eligibility_reason"] = reason

    with open(a.out, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    print(f"thresholds: comp>={a.comp}, catch>={a.catch}, P-cap<{a.cap}  (FREEZE after dev tuning)\n")
    for fam in ("L", "P"):
        fr = [r for r in recs if r["family"] == fam]
        if not fr:
            continue
        st = Counter(r["status"] for r in fr)
        miss = [r for r in fr if r["status"] == "clean_miss"]
        catch = [r for r in fr if r["status"] == "clean_catch"]
        print(f"Family {fam}: {len(fr)} candidates -> {dict(st)}")
        print(f"   clean MISS={len(miss)}  clean CATCH={len(catch)}  balanced usable={2*min(len(miss),len(catch))}")
        if fam == "L":
            sub = Counter(r["flaw_subtype"] for r in (miss + catch))
            print(f"   usable by subtype: {dict(sub)}")
    print(f"\nwrote {a.out}")
    print("Go-bar (spec §19): >= 30-40 BALANCED clean MISS/CATCH that also PASS the text-only gate.")
    print("next: scripts/run_text_only_gate.py  then  scripts/match_catch_miss.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
