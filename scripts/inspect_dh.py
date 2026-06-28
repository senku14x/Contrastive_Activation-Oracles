#!/usr/bin/env python3
"""
inspect_dh.py — look at the contrastive object ΔH = H_B - H_A before trusting any readout.

Answers, empirically:
  - Are the extracted activations non-null?          (|H_A|, |H_B| > 0)
  - Does the hint change the activations even when    (|ΔH| > 0 and cos(H_A,H_B) < 1
    the BEHAVIOR does not? presence != disposition)    on resist/follow pairs)
  - Do a few sink/outlier dims dominate ΔH?           (top-k dims' share of ΔH energy)
  - Do min-diff nulls (N1,N2) have smaller ΔH than    (by-type means)
    max-diff nulls (N3,N4)?

This is the spec-required raw-activation / sink inspection (STAGE0 §5). GPU; TARGET
extraction only (adapter disabled) — the oracle is not involved. ΔH is reasoning-mode
independent: suffix activations attend only to context+suffix, not the later CoT.
"""
from __future__ import annotations

import statistics as st
import sys

sys.path.insert(0, ".")
import torch  # noqa: E402

from cao import ao_checks as ck   # noqa: E402  (char-offset suffix locator + templating)
from cao import ao_runtime as rt  # noqa: E402
from cao import pairs as P        # noqa: E402

LAYERS = [9, 18, 27]
N_FINAL = 16


def acts_for(model, tok, pair, cond):
    ctx = pair["context_A"] if cond == "A" else pair["context_B"]
    suf = P.suffix_for(pair)
    text = ck._templated_string(tok, ctx, suf)
    ids, (s0, s1) = ck.suffix_token_span(tok, ctx, suf)
    pos = list(range(s0, s1))[-N_FINAL:]
    return rt.extract_layer_major(model, tok, text, pos, LAYERS).float()  # [len(LAYERS)*N_FINAL, D]


def main() -> int:
    model, tok = rt.load_oracle()
    rows, energy = [], None
    print(f"{'id':4}{'type':7}{'|H_A|':>8}{'|H_B|':>8}{'|dH|':>8}{'dH/|H|':>8}{'cos(A,B)':>9}")
    for p in P.PAIRS:
        HA, HB = acts_for(model, tok, p, "A"), acts_for(model, tok, p, "B")
        dH = HB - HA
        nA, nB = HA.norm(dim=-1).mean().item(), HB.norm(dim=-1).mean().item()
        ndh = dH.norm(dim=-1).mean().item()
        cos = torch.nn.functional.cosine_similarity(HA, HB, dim=-1).mean().item()
        rel = ndh / max((nA + nB) / 2, 1e-6)
        rows.append((p["pair_id"], p["predicted_type"], ndh, rel, cos))
        e = (dH ** 2).mean(0)
        energy = e if energy is None else energy + e
        print(f"{p['pair_id']:4}{p['predicted_type']:7}{nA:8.1f}{nB:8.1f}{ndh:8.1f}{rel:8.2f}{cos:9.3f}")

    top = torch.topk(energy, 8)
    frac = top.values.sum().item() / energy.sum().item()
    print(f"\nTop-8 dims hold {frac * 100:.1f}% of total ΔH energy; dim indices={top.indices.tolist()}")
    for t in ("follow", "resist", "null"):
        vals = [r[2] for r in rows if r[1] == t]
        if vals:
            print(f"  mean |dH| {t:7}: {st.mean(vals):7.1f}")
    n12 = [r[2] for r in rows if r[0] in ("N1", "N2")]
    n34 = [r[2] for r in rows if r[0] in ("N3", "N4")]
    if n12 and n34:
        print(f"  min-diff nulls (N1,N2) mean |dH|={st.mean(n12):.1f}  vs  max-diff (N3,N4)={st.mean(n34):.1f}")
    print("\nReading:")
    print("  |H_A|,|H_B|>0 and |dH|>0  => activations are non-null AND the hint changes the state")
    print("                              even when the ANSWER doesn't (presence != disposition).")
    print("  cos(A,B) well below 1     => the contexts genuinely move the suffix representation.")
    print("  if top-8 dims hold most ΔH energy => sink dims dominate; handle before ΔH geometry")
    print("     (sensitivity analysis: zero those dims and recheck — do NOT pre-mask the AO input).")
    print("  N3,N4 |dH| > N1,N2        => max-difference nulls behave as designed (big ΔH, no behavior).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
