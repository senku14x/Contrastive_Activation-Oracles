#!/usr/bin/env python3
"""
subtype_yield.py — catch/miss/discard yield broken down by flaw_subtype and by authoring batch.

Ad hoc diagnostic (not part of the main pipeline). Reads data/candidates_labeled.jsonl (written by
scripts/label_candidates.py) and reports, per flaw_subtype, how many items landed clean_catch /
clean_miss / discard and the resulting catch rate. Motivation: the 137-item run showed a sharp catch-
yield drop in the newly-authored batches (L82-L137) relative to the original bank (L01-L81) -- this
tells us whether that drop concentrates in specific fallacy subtypes (actionable: bias future authoring
away from them) or is spread evenly (more likely just noise at this n).

    python3 scripts/subtype_yield.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

sys.path.insert(0, ".")


def main() -> int:
    recs = [json.loads(l) for l in open("data/candidates_labeled.jsonl")]
    L = [r for r in recs if r["family"] == "L"]

    by_sub = defaultdict(lambda: {"clean_catch": 0, "clean_miss": 0, "discard": 0})
    for r in L:
        sub = r.get("flaw_subtype") or "?"
        status = r.get("status", "discard")
        if status not in ("clean_catch", "clean_miss"):
            status = "discard"
        by_sub[sub][status] += 1

    print(f"{'subtype':24s} {'catch':>6} {'miss':>6} {'discard':>8} {'valid':>6} {'catch-rate':>11}")
    tot_catch = tot_miss = tot_discard = 0
    for sub, c in sorted(by_sub.items(), key=lambda kv: -sum(kv[1].values())):
        catch, miss, disc = c["clean_catch"], c["clean_miss"], c["discard"]
        valid = catch + miss
        rate = f"{catch/valid:.1%}" if valid else "n/a"
        print(f"{sub:24s} {catch:6d} {miss:6d} {disc:8d} {valid:6d} {rate:>11}")
        tot_catch += catch; tot_miss += miss; tot_discard += disc
    valid = tot_catch + tot_miss
    print(f"{'TOTAL':24s} {tot_catch:6d} {tot_miss:6d} {tot_discard:8d} {valid:6d} "
          f"{(tot_catch/valid if valid else 0):>10.1%}")

    # split by authoring batch (slug-number ranges from this session's commits)
    def batch_of(slug):
        try:
            n = int("".join(ch for ch in slug.split("_")[0] if ch.isdigit()))
        except ValueError:
            return "?"
        if n <= 32:
            return "L01-L32 (original bank)"
        if n <= 81:
            return "L33-L81 (workflow batch)"
        if n <= 105:
            return "L82-L105 (hand-authored, thin subtypes)"
        return "L106-L137 (hand-authored, thin subtypes)"

    by_batch = defaultdict(lambda: {"clean_catch": 0, "clean_miss": 0, "discard": 0})
    for r in L:
        b = batch_of(r.get("slug") or r.get("candidate_id", ""))
        status = r.get("status", "discard")
        if status not in ("clean_catch", "clean_miss"):
            status = "discard"
        by_batch[b][status] += 1

    print(f"\n{'batch':32s} {'catch':>6} {'miss':>6} {'discard':>8} {'valid':>6} {'catch-rate':>11}")
    for b in ("L01-L32 (original bank)", "L33-L81 (workflow batch)",
              "L82-L105 (hand-authored, thin subtypes)", "L106-L137 (hand-authored, thin subtypes)"):
        c = by_batch.get(b, {"clean_catch": 0, "clean_miss": 0, "discard": 0})
        catch, miss, disc = c["clean_catch"], c["clean_miss"], c["discard"]
        valid = catch + miss
        rate = f"{catch/valid:.1%}" if valid else "n/a"
        print(f"{b:32s} {catch:6d} {miss:6d} {disc:8d} {valid:6d} {rate:>11}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
