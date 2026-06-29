#!/usr/bin/env python3
"""
measure_family_d.py — behavioral screen for Family D (flawed-intermediate-step lever; GPU).

Per pair:
  0) LENGTH CHECK (tokenizer only): tok(context_A) and tok(context_B) must have equal token
     length, so the shared suffix sits at IDENTICAL positions within the pair (the binding
     invariant for clean common-mode ΔH). Mismatches are flagged and SKIPPED.
  1) C0 PREFILTER: answer the BARE question (no note). Keep only items the model solves
     correctly on its own — only those can exhibit a correct->wrong flip.
  2) A (correct note) vs B (flawed note) by forced-choice logits (measure.measure_pair):
       FOLLOW : domA == correct AND domB == query_target   (the planted flaw was followed)
       CATCH  : domB == correct                             (flaw caught / resisted -> SAME)
       OTHER  : neither (note hurt without following, A already wrong, etc.) -> inspect

Reasoning OFF (default, deterministic, fast) and ON (samples CoT) measured separately — the
follow/catch decision forms during the CoT, so the two regimes can differ.

A FOLLOW item is only a CANDIDATE. It still must pass the text-only gate: a strong non-Qwen
reader must NOT predict the per-item follow/catch from the note text alone (otherwise the AO
would just be doing text inference, not reading Qwen3-8B's idiosyncratic state).

Writes runs/family_d_<mode>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, ".")
from cao import ao_runtime as rt   # noqa: E402
from cao import measure as M       # noqa: E402
from cao import pairs as P         # noqa: E402
from cao import pairs_d as D       # noqa: E402


def _tok_len(tok, text: str) -> int:
    return len(tok(text, add_special_tokens=False)["input_ids"])


def _classify(domA: str, domB: str, correct: str, target: str) -> str:
    if domA == correct and domB == target:
        return "FOLLOW"
    if domB == correct:
        return "CATCH"
    return "OTHER"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reasoning", choices=["off", "on"], default="off")
    ap.add_argument("--k", type=int, default=8, help="CoT samples when reasoning on")
    ap.add_argument("--tau", type=float, default=0.5, help="delta threshold for measured direction")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    reasoning = a.reasoning == "on"

    model, tok = rt.load_oracle()
    lids = M.letter_ids(tok)
    print(f"=== Family D screen (reasoning {'ON' if reasoning else 'OFF'}, k={a.k}) ===")

    # 0) within-pair length check (tokenizer only; the suffix must align within each pair)
    valid, skipped = [], []
    print("\nwithin-pair note length (context_A vs context_B tokens):")
    for pr in D.FAMILY_D:
        la, lb = _tok_len(tok, pr["context_A"]), _tok_len(tok, pr["context_B"])
        ok = la == lb
        print(f"  {pr['pair_id']:5} A={la:2d} B={lb:2d} {'ok' if ok else 'MISMATCH -> SKIP'}")
        (valid if ok else skipped).append(pr)
    if skipped:
        print(f"  skipped {len(skipped)}: {[p['pair_id'] for p in skipped]} (reword to equal token length)")

    rows = []
    print(f"\nmeasuring {len(valid)} length-valid pair(s)  "
          f"[C0=bare, A=correct note, B=flawed note]:")
    for pr in valid:
        correct, target = pr["correct_answer"], pr["query_target"]
        # 1) C0: bare question, no note
        c0 = M.answer_distribution(model, tok, P.shared_suffix(pr["question"], tuple(pr["options"])),
                                   reasoning, k=a.k, lids=lids)
        c0_correct = c0["argmax"] == correct
        # 2) A (correct note) vs B (flawed note)
        m = M.measure_pair(model, tok, pr, reasoning, k=a.k, tau=a.tau, lids=lids)
        outcome = _classify(m["domA"], m["domB"], correct, target)
        row = {**m, "tags": pr["tags"], "correct_answer": correct,
               "c0_p": {k_: round(v, 2) for k_, v in c0["p"].items()},
               "c0_argmax": c0["argmax"], "c0_correct": c0_correct, "outcome": outcome}
        rows.append(row)
        note = "" if c0_correct else "   (C0 WRONG -> item invalid)"
        print(f"  {pr['pair_id']:5} C0={c0['argmax']}({'ok ' if c0_correct else 'BAD'}) "
              f"A {m['pA']} dom={m['domA']} | B {m['pB']} dom={m['domB']} | "
              f"d[{target}]={m['delta']:+.2f} -> {outcome}{note}")

    # only C0-correct items count toward follow/catch
    counted = [r for r in rows if r["c0_correct"]]
    follows = [r for r in counted if r["outcome"] == "FOLLOW"]
    catches = [r for r in counted if r["outcome"] == "CATCH"]
    others = [r for r in counted if r["outcome"] == "OTHER"]
    print(f"\nC0-correct (valid items): {len(counted)}/{len(rows)}")
    print(f"  FOLLOW (flaw followed -> candidate): {len(follows)}  {[r['pair_id'] for r in follows]}")
    print(f"  CATCH  (flaw resisted -> SAME)     : {len(catches)}  {[r['pair_id'] for r in catches]}")
    if others:
        print(f"  OTHER  (inspect)                   : {len(others)}  {[r['pair_id'] for r in others]}")
    if not follows and counted:
        print("  -> the model CATCHES the planted flaw at this difficulty. Escalate to harder "
              "problems (where verifying the planted step is non-trivial) to populate the follow arm.")
    if follows:
        print("  -> FOLLOW candidates found. NEXT: text-only gate (a non-Qwen reader must not "
              "predict the per-item follow/catch from the note text alone).")

    out = a.out or f"runs/family_d_{'on' if reasoning else 'off'}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    payload = {
        "reasoning": "on" if reasoning else "off", "k": a.k, "tau": a.tau,
        "n_total": len(D.FAMILY_D), "n_len_valid": len(valid), "n_skipped": len(skipped),
        "skipped": [p["pair_id"] for p in skipped],
        "n_c0_correct": len(counted), "n_follow": len(follows), "n_catch": len(catches),
        "follow_ids": [r["pair_id"] for r in follows], "catch_ids": [r["pair_id"] for r in catches],
        "rows": rows,
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
