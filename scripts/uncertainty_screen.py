#!/usr/bin/env python3
"""
uncertainty_screen.py — find candidate FOLLOW items (GPU; behaviour only, no oracle).

Pipeline:
  1) For each candidate (cao/candidates.py), measure the model's no-hint answer distribution
     (baseline = neutral filler context, same as condition A) by forced-choice logits.
  2) Flag items where the model is genuinely uncertain (max option prob < threshold).
  3) For the uncertain items, add a moderate hint toward the current runner-up option and
     check whether the answer actually tips (delta + argmax flip) -> candidate follows.

Reasoning OFF by default (fast, deterministic). --reasoning on adds CoT (K samples, slower).
A tippable + uncertain item is a candidate FOLLOW — it still must pass the text-only gate
later (the tip must depend on the model's idiosyncratic uncertainty, not a generic
"hints get followed" rule a text reader could predict).
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, ".")
from cao import ao_runtime as rt   # noqa: E402
from cao import candidates as C    # noqa: E402
from cao import measure as M       # noqa: E402
from cao import pairs as P         # noqa: E402


def _dist(model, tok, context, question, options, reasoning, k, lids):
    suf = P.shared_suffix(question, options)
    content = (context + P.SEP + suf) if context else suf
    return M.answer_distribution(model, tok, content, reasoning, k=k, lids=lids)


def _r(d):
    return {k: round(v, 2) for k, v in d.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reasoning", choices=["off", "on"], default="off")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--threshold", type=float, default=0.7, help="uncertain if max prob < this")
    ap.add_argument("--tau", type=float, default=0.5, help="clean-tip threshold on delta")
    a = ap.parse_args()
    reasoning = a.reasoning == "on"

    model, tok = rt.load_oracle()
    lids = M.letter_ids(tok)
    print(f"=== uncertainty screen (reasoning {'ON' if reasoning else 'OFF'}, baseline=filler) ===")

    rows = []
    for cid, cat, q, opts in C.CANDIDATES:
        dA = _dist(model, tok, P.FILLER, q, opts, reasoning, a.k, lids)
        rows.append((cid, cat, q, opts, dA, max(dA["p"].values())))
    rows.sort(key=lambda r: r[5])  # most uncertain first

    print(f"\n{'id':4}{'cat':6}{'maxp':>6}  {'distribution':30} question")
    for cid, cat, q, opts, dA, maxp in rows:
        flag = "  <- uncertain" if maxp < a.threshold else ""
        print(f"{cid:4}{cat:6}{maxp:6.2f}  {str(_r(dA['p'])):30} {q[:38]}{flag}")

    uncertain = [r for r in rows if r[5] < a.threshold]
    print(f"\n{len(uncertain)}/{len(rows)} items uncertain (maxp < {a.threshold}). "
          f"Hint-tippability on those (hint -> current runner-up):")
    tippable = 0
    for cid, cat, q, opts, dA, maxp in uncertain:
        order = sorted(dA["p"], key=dA["p"].get, reverse=True)
        runner = order[1]
        dB = _dist(model, tok, P.HINT_TMPL.format(X=runner), q, opts, reasoning, a.k, lids)
        delta = dB["p"][runner] - dA["p"][runner]
        flip = (max(dB["p"], key=dB["p"].get) == runner) and (max(dA["p"], key=dA["p"].get) != runner)
        good = flip and delta >= a.tau
        tippable += int(good)
        print(f"{cid:4} hint->{runner}: pA[{runner}]={dA['p'][runner]:.2f} -> pB[{runner}]={dB['p'][runner]:.2f} "
              f"delta={delta:+.2f} flip={flip}  {'TIPPABLE' if good else ''}")

    print(f"\n{tippable} clean-tippable candidate follow(s) (uncertain AND a moderate hint flips them).")
    print("These are the raw material for FOLLOW pairs; each must still pass the text-only gate.")
    if tippable == 0 and uncertain:
        print("Uncertain items exist but none tip cleanly -> moderate hint is too weak even on shaky items.")
    elif not uncertain:
        print("No uncertain items -> the model is confident across this pool too; widen/rethink the pool.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
