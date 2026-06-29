#!/usr/bin/env python3
"""
verify_token_invariants.py — invariant gate for the candidate set (tokenizer only; run on Colab).

data_construction_v2 §9 + spec_v4 §5. For each candidate, with the EXACT target chat template and
enable_thinking=False (matching the OFF extraction in extract_activations.py), assert per pair:
  - context_correct and context_flawed give equal FULL templated token length (invariant 1),
  - the shared suffix sits at the same absolute positions in both (invariant 2),
  - the suffix token-ids are identical across conditions,
  - the W8 (primary) and W20 (sensitivity) extraction windows lie inside the suffix (not the
    generation header) and decode to the intended trailing suffix text.

Mismatches are reported (not crashed) with the token gap, so they can be reworded / padded with the
predeclared neutral bank BEFORE behavioral screening (never tailor padding to make an item pass).
"""
from __future__ import annotations

import argparse
import json
import sys

sys.path.insert(0, ".")
from cao import dataset as D       # noqa: E402
from cao import ao_checks as ck    # noqa: E402

W8, W20 = 8, 20


def _span(tok, context: str, suffix: str):
    """(ids, (start,end)) for the suffix, char-anchored, with enable_thinking=False."""
    content = D.user_content(context, suffix)
    s = tok.apply_chat_template([{"role": "user", "content": content}],
                                tokenize=False, add_generation_prompt=True, enable_thinking=False)
    cs = s.rindex(suffix)
    ce = cs + len(suffix)
    enc = tok(s, return_offsets_mapping=True, add_special_tokens=False)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    idx = [i for i, (a, b) in enumerate(offs) if a is not None and a >= cs and b <= ce and b > a]
    if not idx:
        raise ValueError("suffix not found as a token span")
    return ids, (idx[0], idx[-1] + 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="data/candidates_unfiltered.jsonl")
    a = ap.parse_args()
    tok = ck.load_tokenizer()  # Qwen/Qwen3-8B

    recs = [json.loads(l) for l in open(a.candidates)]
    len_fail, suf_fail, win_fail, ok = [], [], [], []
    print(f"{'id':16}{'fam':4}{'lenA':>5}{'lenB':>5}{'sufA':>6}{'sufB':>6}  {'lenEq':6}{'sufId':6}  W8_decode")
    for r in recs:
        cid, fam, suffix = r["candidate_id"], r["family"], r["shared_suffix"]
        try:
            idsA, (a0, a1) = _span(tok, r["context_correct"], suffix)
            idsB, (b0, b1) = _span(tok, r["context_flawed"], suffix)
            len_eq = len(idsA) == len(idsB)
            suf_eq = (a0, a1) == (b0, b1) and idsA[a0:a1] == idsB[b0:b1]
            span = list(range(a0, a1))
            w8 = span[-W8:]
            w20 = span[-W20:]
            win_in_suffix = (len(span) >= 1) and (min(w8) >= a0)
            dec = tok.decode([idsA[i] for i in w8])
            if not len_eq:
                len_fail.append((cid, len(idsA) - len(idsB)))
            if not suf_eq:
                suf_fail.append(cid)
            if not win_in_suffix or len(span) < W8:
                win_fail.append(cid)
            if len_eq and suf_eq and win_in_suffix and len(span) >= W8:
                ok.append(cid)
            print(f"{cid:16}{fam:4}{len(idsA):>5}{len(idsB):>5}{a0:>6}{b0:>6}  "
                  f"{str(len_eq):6}{str(suf_eq):6}  {dec!r}")
        except Exception as e:  # noqa: BLE001
            len_fail.append((cid, "ERR"))
            print(f"{cid:16}{fam:4}  ERROR: {e}")

    print(f"\n{len(ok)}/{len(recs)} pass all invariants.")
    if len_fail:
        print(f"LENGTH mismatch ({len(len_fail)}): {len_fail}")
        print("  -> reword the shorter context or pad with the predeclared neutral bank to equal token length.")
    if suf_fail:
        print(f"SUFFIX position/id mismatch ({len(suf_fail)}): {suf_fail}  (fix length first)")
    if win_fail:
        print(f"WINDOW too short / outside suffix ({len(win_fail)}): {win_fail}")
    blocking = bool(len_fail or suf_fail or win_fail)
    print("\n** Blocking — fix before extraction. **" if blocking else "\nAll invariants satisfied. Ready to measure + extract.")
    return 1 if blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
