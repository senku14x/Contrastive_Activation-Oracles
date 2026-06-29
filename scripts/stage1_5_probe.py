#!/usr/bin/env python3
"""
stage1_5_probe.py — THE EXISTENCE GATE (spec_v4 §11/§12; decides everything, cheap, no AO).

Empirically adjudicates read-time vs reason-time: can a linear probe on the PRE-OUTPUT state separate
CATCH from MISS above text — at the flawed-step readout (H_B / ΔH)?

  - probe(ΔH) and probe(H_B):  PCA-reduced, honest LOO (PCA fit on train only), AUC.
  - text-feature probe:        a MATCHED SUPERVISED baseline (char n-grams of the flawed context,
                               same PCA+LOO). This is the right control (my review fix): a supervised
                               activation probe must beat a supervised TEXT probe, not merely a
                               zero-shot LLM reader — else it may just be learning flaw-content.
  - shuffle control:           permute labels -> AUC must collapse to ~0.5 (else the CV leaks).

Decision (spec §12): activation probe >> text-feature AND >> 0.5 AND shuffle ~0.5 -> signal exists,
proceed to the AO. Activation ≈ text-feature or ≈ 0.5 -> clean negative (read-time trace absent /
text-legible). n < 30 -> underpowered, "inconclusive" not "absent".
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

sys.path.insert(0, ".")
from cao import probe as PB        # noqa: E402


def _text_features(meta_items, ids):
    by = {m["candidate_id"]: m for m in meta_items}
    return PB.text_features([by[i]["context_flawed"] for i in ids])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", default="data/activations_w8.npz")
    ap.add_argument("--meta", default="data/activations_meta.json")
    ap.add_argument("--k", type=int, default=10, help="PCA components")
    ap.add_argument("--perm", type=int, default=20, help="label-shuffle repeats")
    a = ap.parse_args()

    z = np.load(a.acts, allow_pickle=True)
    HA, HB, y, ids = z["H_A"], z["H_B"], z["labels"], [str(x) for x in z["ids"]]
    meta = json.load(open(a.meta))["items"]
    n = len(y)
    nmiss, ncatch = int(y.sum()), int((1 - y).sum())
    print(f"n={n}  MISS={nmiss} CATCH={ncatch}  (window from {a.acts})\n")
    if min(nmiss, ncatch) < 2:
        print("** too few in one class to probe.")
        return 1

    dH = PB.pool_layer_major(HB - HA)
    hB = PB.pool_layer_major(HB)
    auc_dh = PB.probe_auc(dH, y, a.k)
    auc_hb = PB.probe_auc(hB, y, a.k)
    auc_txt = PB.probe_auc(_text_features(meta, ids), y, a.k)
    rng = np.random.default_rng(0)
    shuf = [PB.probe_auc(dH, rng.permutation(y), a.k) for _ in range(a.perm)]
    shuf = float(np.nanmean(shuf))

    print(f"  probe(ΔH)            AUC = {auc_dh:.3f}   <- primary readout")
    print(f"  probe(H_B alone)     AUC = {auc_hb:.3f}")
    print(f"  text-feature probe   AUC = {auc_txt:.3f}   <- matched supervised baseline (must be beaten)")
    print(f"  shuffled-label ΔH    AUC = {shuf:.3f}   <- sanity (want ~0.50)")

    best_act = max(auc_dh, auc_hb)
    min_class = min(nmiss, ncatch)
    print("\nverdict:")
    # Power gate FIRST: at tiny n / tiny minority class, AUCs are noise (a few-point swing moves AUC
    # ~0.2). Neither 'signal' nor 'negative' is supportable -> do not evaluate either branch.
    if n < 30 or min_class < 8:
        print(f"  INCONCLUSIVE — UNDERPOWERED (n={n}, minority class={min_class}). At this size the AUCs "
              f"({auc_dh:.2f}/{auc_hb:.2f} vs text {auc_txt:.2f}, shuffle {shuf:.2f}) are noise: a single "
              f"misclassified item moves AUC ~0.2. This is NOT a negative — it cannot distinguish "
              f"'no signal' from 'not enough data'. Need >= ~30 total AND >= ~8-10 in the minority "
              f"(catch) class before reading the result.")
        return 0
    if best_act > 0.70 and best_act - auc_txt > 0.10 and shuf < 0.62:
        print("  SIGNAL likely present: activation probe beats the text-feature baseline and shuffle is flat.")
        print("  -> proceed to Stage 2 (AO zero-shot); detection AO <= probe is expected (spec §6/§12).")
    elif abs(best_act - auc_txt) <= 0.10 or best_act <= 0.60:
        print("  CLEAN NEGATIVE: activation probe ~ text-feature (or ~chance) -> the catch/miss trace is")
        print("  not in the pre-output state above text (read-time vs reason-time adjudicated negative).")
    else:
        print("  AMBIGUOUS: re-run W20, inspect raw ΔH, and confirm power before interpreting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
