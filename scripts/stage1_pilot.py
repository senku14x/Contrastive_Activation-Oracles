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

PILOT = ["F3", "F6", "R5", "R1"]   # span correct answers B/D/C/A: validates the forced-choice
                                   # probe on non-A answers AND tests harder follows (CRT trap,
                                   # "all of the above", syllogism)


def probe_sanity(model, tok):
    """Rule out an A-bias in the forced-choice probe: place 'Paris' at A,B,C,D in turn on a
    fact the model knows cold, and check argmax tracks the position (not always 'A')."""
    print("\n===== forced-choice probe sanity (A-bias check) =====")
    lids = M.letter_ids(tok)
    layouts = [("Paris", "Berlin", "Rome", "Madrid"),
               ("Berlin", "Paris", "Rome", "Madrid"),
               ("Berlin", "Rome", "Paris", "Madrid"),
               ("Berlin", "Rome", "Madrid", "Paris")]
    ok = True
    for i, opts in enumerate(layouts):
        want = "ABCD"[i]
        content = P.shared_suffix("What is the capital of France?", opts)
        d = M.answer_distribution(model, tok, content, reasoning=False, lids=lids)
        good = d["argmax"] == want
        ok = ok and good
        print(f"  Paris@{want}: argmax={d['argmax']} p={ {k: round(v, 2) for k, v in d['p'].items()} }"
              f"  {'OK' if good else '** A-BIAS / probe broken **'}")
    print("  probe tracks content" if ok else "  PROBE SUSPECT — fix before trusting any label")
    return ok


def run_mode(model, tok, by_id, reasoning, k, temperature, show_raw):
    print(f"\n===== reasoning {'ON' if reasoning else 'OFF'}  (K={k}, T={temperature}) =====")
    rows = []
    for idx, pid in enumerate(PILOT):
        want = show_raw and reasoning and idx == 0
        r = M.measure_pair(model, tok, by_id[pid], reasoning, k=k, temperature=temperature, want_raw=want)
        rows.append(r)
        print(f"{pid} {r['type']:6} qt={r['query_target']} corr={by_id[pid]['correct_answer']} "
              f"pA={r['pA']} pB={r['pB']} delta={r['delta']:+.2f} -> {r['measured_direction']:4} "
              f"(pred {r['predicted_direction']:4} agree={r['agree']} clean={r['clean']})")
        if want and r.get("rawA"):
            print(f"    [{pid} A CoT] {r['rawA'][:240].replace(chr(10), ' ')}")
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
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--reasoning", choices=["on", "off", "both"], default="both")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--no-raw", action="store_true")
    a = ap.parse_args()

    model, tok = rt.load_oracle()
    probe_sanity(model, tok)
    by_id = {p["pair_id"]: p for p in P.PAIRS}
    modes = [True, False] if a.reasoning == "both" else [a.reasoning == "on"]
    for reasoning in modes:
        run_mode(model, tok, by_id, reasoning, a.k, a.temperature, show_raw=not a.no_raw)
    print("\nNext: if a split is present, run the full 16-pair measurement + attrition projection.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
