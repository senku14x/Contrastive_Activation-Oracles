#!/usr/bin/env python3
"""
match_catch_miss.py — match CATCH vs MISS and freeze the feasibility set (data_construction_v2 §13/§14).

Restricts to gate-passing clean items, then checks the groups are NOT separable by text-visible
confounds — the difficulty-leakage trap that sank the peer-hint set. Per spec §13: "If MISS items
remain systematically lower-margin than CATCH, the set is confounded — do not proceed until the
groups overlap." We report, per group, the neutral gold-margin and the prompt length (the two most
likely confounds), flag if they separate, and write data/feasibility_frozen.jsonl.

Tokenizer-free (uses char length as a length proxy; the exact token audit is verify_token_invariants).
"""
from __future__ import annotations

import argparse
import json
import sys
from statistics import mean, pstdev

sys.path.insert(0, ".")
from cao import dataset as D       # noqa: E402


def _len_proxy(r):
    return len(D.user_content(r["context_flawed"], r["shared_suffix"]))


def _summ(xs):
    return (mean(xs), pstdev(xs)) if xs else (float("nan"), float("nan"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gated", default="data/candidates_gated.jsonl",
                    help="for --require-gate false, point this at data/candidates_labeled.jsonl")
    ap.add_argument("--out", default="data/feasibility_frozen.jsonl")
    ap.add_argument("--family", default="L", choices=["L", "P", "all"])
    ap.add_argument("--subtype", default="all",
                    help="restrict to one flaw_subtype (e.g. affirming_consequent) -- use this when catch "
                         "rate is concentrated in a few subtypes: mixing subtypes with very different catch "
                         "rates lets a supervised text/activation probe fingerprint SUBTYPE as a proxy for "
                         "the label instead of reading disposition. Single-subtype makes it a constant, not "
                         "a variable. Default 'all' keeps the old behavior.")
    ap.add_argument("--require-gate", dest="require_gate", choices=["true", "false"], default="true",
                    help="false = use ALL clean items (gate NOT applied) for a pipeline dry-run / yield")
    a = ap.parse_args()
    require = a.require_gate == "true"

    recs = [json.loads(l) for l in open(a.gated)]
    sel = [r for r in recs if r["status"] in ("clean_miss", "clean_catch")
           and (a.family == "all" or r["family"] == a.family)
           and (a.subtype == "all" or r.get("flaw_subtype") == a.subtype)
           and (not require or r.get("text_only_gate", {}).get("passes_gate", False))]
    if not require:
        print("** --require-gate false: using ALL clean items, text gate NOT applied. The existence-gate "
              "result is NOT valid for the claim — pipeline dry-run / yield only.\n")
    miss = [r for r in sel if r["status"] == "clean_miss"]
    catch = [r for r in sel if r["status"] == "clean_catch"]
    print(f"gate-passing clean items (family={a.family}, subtype={a.subtype}): "
          f"MISS={len(miss)} CATCH={len(catch)}")

    flags = []
    for feat, fn in (("neutral_gold_margin", lambda r: r["derived"].get("neutral_gold_margin") or 0.0),
                     ("prompt_len_chars", _len_proxy)):
        mm, ms = _summ([fn(r) for r in miss])
        cm, cs = _summ([fn(r) for r in catch])
        # crude separation flag: group means differ by > 1 pooled SD
        pooled = (ms + cs) / 2 or 1e-9
        sep = abs(mm - cm) / pooled
        print(f"  {feat:20} MISS={mm:.3f}±{ms:.3f}  CATCH={cm:.3f}±{cs:.3f}  |Δ|/SD={sep:.2f}"
              f"{'  <- CONFOUND' if sep > 1.0 else ''}")
        if sep > 1.0:
            flags.append(feat)

    n_bal = 2 * min(len(miss), len(catch))
    print(f"\nbalanced usable = {n_bal} (go-bar: >= 30-40; spec §19).")
    if flags:
        print(f"** CONFOUND on {flags}: MISS/CATCH separate on text-visible features -> the existence gate "
              f"would be confounded. Rebalance/match (or report as a confounded supply result) before extraction.")
    if n_bal < 30:
        print("** UNDERPOWERED: < 30 balanced. A null here is 'inconclusive / couldn't build', NOT 'no signal' "
              "(spec §19). Report build-yield as a distinct outcome.")

    # freeze: keep all gate-passing clean items (label retained in 'status'); downstream balances as needed.
    with open(a.out, "w") as f:
        for r in sel:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {a.out} ({len(sel)} items) -> next: scripts/extract_activations.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
