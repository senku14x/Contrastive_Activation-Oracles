#!/usr/bin/env python3
"""
verify_pairs.py — tokenizer-only invariant diagnostic for the Family-C pairs.

Spec gate (Stage 1): "Stop if suffix tokenization isn't identical." No GPU needed; run on
Colab (needs the Qwen3-8B tokenizer). It reports, per pair:
  - whether "(A)".."(D)" tokenize to equal length (the hint-constant-length premise),
  - the templated token lengths of conditions A vs B and the suffix start offsets,
  - whether the FINAL-N suffix tokens are identical across A and B (seam-robust; this is the
    invariant that must hold), and what those tokens decode to,
  - the required common context length L for padding (invariant 1).

Pre-padding, the suffix START offsets will differ within follow/resist pairs (filler vs hint
have different lengths) — that's expected and tells us the padding budget. The FINAL-N token
identity is the load-bearing check.
"""
import sys

sys.path.insert(0, ".")
from cao import ao_checks as ck   # noqa: E402  (reuses the char-offset suffix locator)
from cao import pairs as P        # noqa: E402

N_FINAL = 16  # end-weighted extraction window (spec note used 8; widened per STAGE0.md §4)


def main() -> int:
    tok = ck.load_tokenizer()  # Qwen/Qwen3-8B

    ol = {L: tok.encode(f"({L})", add_special_tokens=False) for L in "ABCD"}
    eq = len({len(v) for v in ol.values()}) == 1
    print("(A)-(D) token ids:", ol)
    print(f"  -> equal length: {eq}" + ("" if eq else "   ** FIX: hint length varies with target **"))

    starts, fails, within_fail = [], [], []
    print(f"\n{'id':4}{'type':7}{'lenA':>5}{'lenB':>5}{'sufA':>6}{'sufB':>6}  "
          f"f{N_FINAL}_idMatch  final{N_FINAL}_decode")
    for p in P.PAIRS:
        s = P.suffix_for(p)
        try:
            idsA, (a0, a1) = ck.suffix_token_span(tok, p["context_A"], s)
            idsB, (b0, b1) = ck.suffix_token_span(tok, p["context_B"], s)
            fa = list(range(a0, a1))[-N_FINAL:]
            fb = list(range(b0, b1))[-N_FINAL:]
            idmatch = [idsA[i] for i in fa] == [idsB[i] for i in fb]
            dec = tok.decode([idsA[i] for i in fa])
            starts += [a0, b0]
            if a0 != b0:
                within_fail.append(p["pair_id"])
            if not idmatch:
                fails.append(p["pair_id"])
            print(f"{p['pair_id']:4}{p['predicted_type']:7}{len(idsA):>5}{len(idsB):>5}"
                  f"{a0:>6}{b0:>6}  {str(idmatch):>11}  {dec!r}")
        except Exception as e:  # noqa: BLE001
            fails.append(p["pair_id"])
            print(f"{p['pair_id']:4}{p['predicted_type']:7}  ERROR: {e}")

    uniq = sorted(set(starts))
    uniform = (len(uniq) == 1)
    print(f"\nsuffix-start values across all pairs/conditions: {uniq}")
    print(f"(A)-(D) equal length:                 {eq}")
    print(f"within-pair sufA==sufB:               {'ALL OK' if not within_fail else 'FAIL ' + str(within_fail)}  (MANDATORY)")
    print(f"cross-pair common L (all equal):      {uniform}  (target of Option 1)")
    print(f"final-{N_FINAL} suffix-token identity:        {'ALL OK' if not fails else 'FAIL ' + str(fails)}  (MANDATORY)")
    if not (eq and not within_fail and not fails):
        print("\n** Blocking issue — fix before GPU work. **")
    elif not uniform:
        print(f"\nWithin-pair OK, but contexts not all the same length: {uniq}. Tune the wording of the "
              "off-length contexts to the common target so every sufStart matches (cleaner cross-pair controls).")
    else:
        print("\nAll invariants satisfied (within-pair aligned, common L, seam-robust). Ready for behavior labelling.")
    return 0 if (eq and not within_fail and not fails and uniform) else 1


if __name__ == "__main__":
    raise SystemExit(main())
