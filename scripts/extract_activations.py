#!/usr/bin/env python3
"""
extract_activations.py — extract pre-output activations for the frozen set (GPU; adapter DISABLED).

spec_v4 §5 / data_construction_v2 §15. For each frozen item, for the correct (H_A) and flawed (H_B)
conditions, template with enable_thinking=False (matching the OFF label), locate the shared suffix by
char offset, take the final W positions (W8 primary, W20 sensitivity) over layers {9,18,27}, and
extract layer-major activations with the adapter disabled. The window is INSIDE the suffix and
excludes the generation header (the position assertion mirrors verify_token_invariants).

Saves data/activations_w{W}.npz with H_A, H_B (float32 [n, W*nlayers, D]), labels (1=MISS,0=CATCH),
and ids; plus a sidecar data/activations_meta.json with the texts (for the text-feature probe in
stage1_5_probe.py). Diagnostics (per-position norms, cos(H_A,H_B), ||ΔH||) are logged — descriptive
only (PCA separation is not proof; spec §0).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from cao import ao_runtime as rt   # noqa: E402
from cao import dataset as D       # noqa: E402

LAYERS = (9, 18, 27)


def _final_positions(tok, context, suffix, w):
    """(templated_text, final-w suffix token positions) with enable_thinking=False, char-anchored.

    Positions are indices into tok(templated, add_special_tokens=False) — exactly what
    extract_layer_major re-tokenizes — so they line up."""
    content = D.user_content(context, suffix)
    templ = tok.apply_chat_template([{"role": "user", "content": content}],
                                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
    cs = templ.rindex(suffix)
    ce = cs + len(suffix)
    enc = tok(templ, return_offsets_mapping=True, add_special_tokens=False)
    offs = enc["offset_mapping"]
    span = [i for i, (s, e) in enumerate(offs) if s is not None and s >= cs and e <= ce and e > s]
    if len(span) < w:
        raise ValueError(f"suffix span {len(span)} < window {w}")
    return templ, span[-w:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", default="data/feasibility_frozen.jsonl")
    ap.add_argument("--window", type=int, default=8, help="W8 primary; pass 20 for the sensitivity run")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"data/activations_w{a.window}.npz"

    recs = [json.loads(l) for l in open(a.frozen)]
    model, tok = rt.load_oracle()
    print(f"extracting {len(recs)} items, window W{a.window}, layers {LAYERS} (adapter disabled)")

    HA, HB, labels, ids, meta = [], [], [], [], []
    for i, r in enumerate(recs):
        sfx = r["shared_suffix"]
        tA, pA = _final_positions(tok, r["context_correct"], sfx, a.window)
        tB, pB = _final_positions(tok, r["context_flawed"], sfx, a.window)
        hA = rt.extract_layer_major(model, tok, tA, pA, LAYERS).float().cpu().numpy()
        hB = rt.extract_layer_major(model, tok, tB, pB, LAYERS).float().cpu().numpy()
        HA.append(hA)
        HB.append(hB)
        labels.append(1 if r["status"] == "clean_miss" else 0)
        ids.append(r["candidate_id"])
        meta.append({"candidate_id": r["candidate_id"], "family": r["family"],
                     "flaw_subtype": r.get("flaw_subtype"), "status": r["status"],
                     "context_correct": r["context_correct"], "context_flawed": r["context_flawed"],
                     "shared_suffix": sfx})
        dh = hB - hA
        cos = float((hA * hB).sum() / (np.linalg.norm(hA) * np.linalg.norm(hB) + 1e-9))
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  {i+1}/{len(recs)} {r['candidate_id']:16} |ΔH|={np.linalg.norm(dh):8.1f} cos={cos:.3f}")

    HA, HB, labels = np.stack(HA), np.stack(HB), np.array(labels)
    os.makedirs("data", exist_ok=True)
    np.savez_compressed(out, H_A=HA, H_B=HB, labels=labels, ids=np.array(ids))
    json.dump({"window": a.window, "layers": list(LAYERS), "n": len(ids),
               "n_miss": int(labels.sum()), "n_catch": int((1 - labels).sum()), "items": meta},
              open("data/activations_meta.json", "w"), indent=2)
    print(f"\nwrote {out}  shape H_A={HA.shape} (n, W*nlayers, D); labels MISS={int(labels.sum())} "
          f"CATCH={int((1-labels).sum())}")
    print("next: scripts/stage1_5_probe.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
