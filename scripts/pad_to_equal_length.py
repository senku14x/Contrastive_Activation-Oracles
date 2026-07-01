#!/usr/bin/env python3
"""
pad_to_equal_length.py — equalize within-pair token length (data_construction_v2 §9; Colab/tokenizer).

verify_token_invariants flags c_A/c_B token-length mismatches (the correct-vs-flawed worked step, or the
weak-vs-strong P template, tokenize to different lengths). This pads the SHORTER of (context_correct,
context_flawed) with a PREDECLARED NEUTRAL prefix so both reach the pair's max full templated length →
the shared suffix lands at identical absolute positions with matched RoPE phase (invariant 1).

Design choices that keep this honest:
  - The filler is bland and answer-irrelevant, and sits at the very START of the context (far from the
    decision point at the suffix), so it does not steer the option choice.
  - It is a PREDECLARED bank, selected by exact token count — never tuned per item to force a label.
  - The ablated (bare-givens) condition is left UNPADDED: it is the competence/ablation probe, measured
    on its own, not part of the extracted contrastive pair.
  - UNIVERSAL_PREFIX is applied to BOTH sides of EVERY pair, unconditionally, before the old
    shorter-side-only logic runs. Root-cause fix for a text leak found via scripts/inspect_text_leak.py:
    the old behavior only padded the shorter side and skipped items whose two conditions already
    tokenized equal — so "does this item have a preface, and how long is it" was a direct function of
    how much context_correct/context_flawed happened to differ in length, which turned out to leak into
    a supervised text-feature probe even restricted to a single flaw_subtype. Making the preface
    identical and universal removes it as a variable entirely; only a small, generic residual top-up
    (the existing bank/coin logic) still varies per item, closing whatever gap remains after both sides
    already share the same long fixed opener.

Run AFTER build_candidates, BEFORE verify_token_invariants/measurement. Rewrites the file in place.
"""
from __future__ import annotations

import argparse
import json
import sys

sys.path.insert(0, ".")
from cao import dataset as D       # noqa: E402
from cao import ao_checks as ck    # noqa: E402

# Predeclared neutral prefix bank, increasing length (~+1 token each). Bland, answer-irrelevant.
BANK = [
    "Note.",
    "A note.",
    "Just a note.",
    "Here is a note.",
    "Here is a short note.",
    "Here is a short note first.",
    "Here is a short note for context.",
    "Here is a short note for some context.",
    "Here is a short note added for some context.",
    "Here is a short note added here for some context.",
    "Here is a short preliminary note added for some context.",
    "Here is a short preliminary note added below for some context.",
]
COIN = "the"  # 1-token top-up for any residual the bank can't hit exactly

# Applied identically to BOTH sides of EVERY pair before anything else -- see module docstring.
UNIVERSAL_PREFIX = "Here is a short preliminary note added below for some context."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="data/candidates_unfiltered.jsonl")
    a = ap.parse_args()
    tok = ck.load_tokenizer()

    def plen(context, suffix):
        s = tok.apply_chat_template([{"role": "user", "content": D.user_content(context, suffix)}],
                                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return len(tok(s, add_special_tokens=False)["input_ids"])

    def pad_to(context, suffix, target):
        """Prepend neutral filler until full templated length == target (exact)."""
        cur = plen(context, suffix)
        prefix = ""
        guard = 0
        while cur < target and guard < 4000:
            guard += 1
            need = target - cur
            # pick the largest bank phrase whose contribution does not overshoot; else a 1-token coin
            best = None
            for p in BANK:
                ctx = f"{prefix}{p} {context}"
                n = plen(ctx, suffix)
                contrib = n - cur
                if 0 < contrib <= need and (best is None or contrib > best[1]):
                    best = (p, contrib, n)
            if best is None:  # fall back to a single 1-token coin
                ctx = f"{prefix}{COIN} {context}"
                n = plen(ctx, suffix)
                if n - cur <= 0:
                    break  # cannot make progress
                prefix = f"{prefix}{COIN} "
                cur = n
            else:
                prefix = f"{prefix}{best[0]} "
                cur = best[2]
        return (prefix + context) if prefix else context, cur

    recs = [json.loads(l) for l in open(a.candidates)]
    fixed = stuck = already = 0
    for r in recs:
        sfx = r["shared_suffix"]
        r["context_correct"] = f"{UNIVERSAL_PREFIX} {r['context_correct']}"
        r["context_flawed"] = f"{UNIVERSAL_PREFIX} {r['context_flawed']}"
        lc = plen(r["context_correct"], sfx)
        lf = plen(r["context_flawed"], sfx)
        if lc == lf:
            already += 1
            continue
        target = max(lc, lf)
        if lc < target:
            r["context_correct"], lc = pad_to(r["context_correct"], sfx, target)
        if lf < target:
            r["context_flawed"], lf = pad_to(r["context_flawed"], sfx, target)
        if lc == lf:
            fixed += 1
        else:
            stuck += 1
            print(f"STUCK {r['candidate_id']}: {lc} vs {lf} (reword manually)")

    with open(a.candidates, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(f"\nalready equal: {already}  |  padded to equal: {fixed}  |  stuck: {stuck}  (of {len(recs)})")
    print("re-run scripts/verify_token_invariants.py — within-pair length/suffix should now pass.")
    return 1 if stuck else 0


if __name__ == "__main__":
    raise SystemExit(main())
