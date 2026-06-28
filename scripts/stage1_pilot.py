#!/usr/bin/env python3
"""
stage1_pilot.py — 4-pair cue-calibration pilot (project_spec.md Stage 1 / §7).

Measures 2 follow-candidates + 2 resist-candidates under reasoning ON and OFF, K samples
each, and prints the cue-tuning decision plus parse health and a couple of raw generations
(so we can eyeball that letter-parsing is correct). Run BEFORE measuring all 16, to avoid
sinking GPU time into a mis-calibrated cue.

Decision rule (per reasoning mode):
  - both resist-candidates followed  -> cue TOO STRONG  -> weaken ("mentioned it might be")
  - both follow-candidates resisted  -> cue TOO WEAK    -> strengthen ("was confident it is")
  - a split appears                  -> cue OK          -> proceed to measure all 16

Usage:
  python scripts/stage1_pilot.py                  # both reasoning modes, K=8
  python scripts/stage1_pilot.py --reasoning off --k 8
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, ".")
from cao import ao_runtime as rt   # noqa: E402
from cao import measure as M       # noqa: E402
from cao import pairs as P         # noqa: E402

PILOT = ["F2", "F5", "R1", "R4"]   # knowledge-follow (robust) + decoupled-resist


def run_mode(model, tok, by_id, reasoning, k, temperature, show_raw):
    print(f"\n===== reasoning {'ON' if reasoning else 'OFF'}  (K={k}, T={temperature}) =====")
    rows = []
    for pid in PILOT:
        r = M.measure_pair(model, tok, by_id[pid], reasoning, k=k, temperature=temperature)
        rows.append(r)
        print(f"{pid} {r['type']:6} qt={r['query_target']} "
              f"domA={r['domA']} domB={r['domB']} fA={r['fA']:.2f} fB={r['fB']:.2f} "
              f"delta={r['delta']:+.2f} -> measured={r['measured_direction']:4} "
              f"(pred {r['predicted_direction']:4} agree={r['agree']} clean={r['clean']} "
              f"parse_fail={r['parse_fail']}/{2*k})")
    follows = [r for r in rows if r["type"] == "follow"]
    resists = [r for r in rows if r["type"] == "resist"]
    n_foll = sum(r["measured_direction"] != "SAME" for r in follows)
    n_res = sum(r["measured_direction"] == "SAME" for r in resists)
    print(f"  follows that followed: {n_foll}/{len(follows)}   resists that resisted: {n_res}/{len(resists)}")
    if n_res == 0:
        print("  -> CUE TOO STRONG: weaken to e.g. 'mentioned in passing it might be (X)'")
    elif n_foll == 0:
        print("  -> CUE TOO WEAK: strengthen to e.g. 'was confident the answer is (X)'")
    else:
        print("  -> SPLIT PRESENT: cue OK at this strength; proceed to measure all 16")
    if show_raw:
        ex = rows[0]
        for cond in ("A", "B"):
            t = ex["texts"][cond][0].replace("\n", " ")
            print(f"    [{ex['pair_id']} {cond}] {t[:240]}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--reasoning", choices=["on", "off", "both"], default="both")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--no-raw", action="store_true")
    a = ap.parse_args()

    model, tok = rt.load_oracle()
    by_id = {p["pair_id"]: p for p in P.PAIRS}
    modes = [True, False] if a.reasoning == "both" else [a.reasoning == "on"]
    for reasoning in modes:
        run_mode(model, tok, by_id, reasoning, a.k, a.temperature, show_raw=not a.no_raw)
    print("\nNext: if a split is present, run the full 16-pair measurement + attrition projection.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
