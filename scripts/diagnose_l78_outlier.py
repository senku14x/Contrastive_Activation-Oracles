#!/usr/bin/env python3
"""
diagnose_l78_outlier.py — is the W8-vs-W20 Stage-1.5 verdict flip driven by one outlier item?

At n=61 (52 MISS / 9 CATCH), scripts/extract_activations.py logged L78_volane_chalk (one of only 9
CATCH items) as a 20-30x norm outlier at W20 (|ΔH|=9085.7, cos=0.226) while completely normal at W8
(|ΔH|=101.9, cos=0.994). With only 9 items in the minority class, a single extreme point can dominate
the LOO probe (cao/probe.py) and flip a verdict on its own. Two independent code-reading passes ruled
out a pipeline bug (window-span computation is independent of window size; no cross-call state leakage;
batch size 1 throughout, so no padding-position bug is possible) and ruled out the most obvious content
explanations (suffix length is not short — 271 chars vs a 256±23 bank mean; the window doesn't even
reach back far enough to touch the gold option's text for this item). Root cause is unresolved — this
script answers the question that actually matters for the decision instead: does the verdict on either
window hinge on this one item, in either direction.

Reads the ALREADY-EXTRACTED data/activations_w8.npz + data/activations_w20.npz — no GPU/model reload,
runs in seconds:

    python3 scripts/diagnose_l78_outlier.py

To reuse this on a different flagged item later, just change OUTLIER_ID below.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, ".")
from cao import probe as PB  # noqa: E402

OUTLIER_ID = "L78_volane_chalk"
WINDOWS = [("W8", "data/activations_w8.npz"), ("W20", "data/activations_w20.npz")]


def load(path):
    z = np.load(path, allow_pickle=True)
    return z["H_A"], z["H_B"], z["labels"], np.array([str(x) for x in z["ids"]])


def per_item_norms(HA, HB, ids):
    """|ΔH| and cos(H_A,H_B) per item -- the raw per-item extraction diagnostic, matching what
    scripts/extract_activations.py prints at capture time (un-pooled, flattened per item)."""
    n = len(ids)
    out = []
    for i in range(n):
        hA = HA[i].reshape(-1).astype(np.float64)
        hB = HB[i].reshape(-1).astype(np.float64)
        dh = hB - hA
        nrm = float(np.linalg.norm(dh))
        cos = float((hA @ hB) / (np.linalg.norm(hA) * np.linalg.norm(hB) + 1e-9))
        out.append((ids[i], nrm, cos))
    return out


def main() -> int:
    results = {}
    for label, path in WINDOWS:
        try:
            HA, HB, y, ids = load(path)
        except FileNotFoundError:
            print(f"** {path} not found -- run scripts/extract_activations.py first (skipping {label})")
            continue

        n = len(ids)
        if OUTLIER_ID not in ids:
            print(f"** {OUTLIER_ID} not found in {path} ids -- check the id spelling / bank version")
        outlier_mask = ids == OUTLIER_ID

        # ---- 1. before/after AUC with and without the outlier item ----
        dH_full = PB.pool_layer_major(HB - HA)
        auc_dh_full = PB.probe_auc(dH_full, y, k=10)

        keep = ~outlier_mask
        HA2, HB2, y2, ids2 = HA[keep], HB[keep], y[keep], ids[keep]
        dH_drop = PB.pool_layer_major(HB2 - HA2)
        auc_dh_drop = PB.probe_auc(dH_drop, y2, k=10)

        print(f"=== {label} ({path}) ===")
        print(f"  n={n}  MISS={int(y.sum())}  CATCH={int((1 - y).sum())}")
        print(f"  probe(ΔH) AUC   FULL (n={n})            = {auc_dh_full:.3f}")
        print(f"  probe(ΔH) AUC   DROP {OUTLIER_ID} (n={n - 1}) = {auc_dh_drop:.3f}")
        print(f"  delta (drop - full)                    = {auc_dh_drop - auc_dh_full:+.3f}")

        # ---- 2. norm / cosine outlier ranking for every item, both windows ----
        diag = per_item_norms(HA, HB, ids)
        diag_sorted = sorted(diag, key=lambda t: -t[1])
        norms = np.array([d[1] for d in diag])
        med = np.median(norms)
        mad = np.median(np.abs(norms - med)) + 1e-9
        print(f"  |ΔH| median={med:.1f}  MAD={mad:.1f}")
        print(f"  top 5 by |ΔH| (id, |ΔH|, cos, robust-z=(x-median)/MAD):")
        for iid, nrm, cos in diag_sorted[:5]:
            z = (nrm - med) / mad
            flag = f"  <== {OUTLIER_ID}" if iid == OUTLIER_ID else ""
            print(f"    {iid:24s} |ΔH|={nrm:10.1f}  cos={cos:6.3f}  z={z:7.1f}{flag}")
        rank = [i for i, (iid, _, _) in enumerate(diag_sorted, start=1) if iid == OUTLIER_ID]
        if rank and rank[0] > 5:
            iid, nrm, cos = diag_sorted[rank[0] - 1]
            z = (nrm - med) / mad
            print(f"    ... {OUTLIER_ID} rank {rank[0]}/{n}  |ΔH|={nrm:10.1f}  cos={cos:6.3f}  z={z:7.1f}")
        print()

        results[label] = dict(auc_full=auc_dh_full, auc_drop=auc_dh_drop, n=n)

    if "W8" in results and "W20" in results:
        print("=== summary ===")
        for label in ("W8", "W20"):
            r = results[label]
            print(f"  {label}: full={r['auc_full']:.3f}  drop-{OUTLIER_ID}={r['auc_drop']:.3f}  "
                  f"|delta|={abs(r['auc_drop'] - r['auc_full']):.3f}")
        w8_fragile = abs(results["W8"]["auc_drop"] - results["W8"]["auc_full"]) >= 0.15
        w20_fragile = abs(results["W20"]["auc_drop"] - results["W20"]["auc_full"]) >= 0.15
        print(f"\n  W8 result driven by a single item (|delta AUC| >= 0.15)?  {w8_fragile}")
        print(f"  W20 result driven by a single item (|delta AUC| >= 0.15)? {w20_fragile}")
        if w20_fragile and not w8_fragile:
            print("  -> supports: W20's CLEAN NEGATIVE verdict is fragile / an artifact of one item; "
                  "W8's SIGNAL verdict is comparatively robust to this one item.")
        elif w8_fragile and w20_fragile:
            print("  -> BOTH windows swing on this one item -- neither verdict (SIGNAL nor CLEAN "
                  "NEGATIVE) should be trusted yet; treat as underpowered regardless of window.")
        elif not w8_fragile and not w20_fragile:
            print("  -> neither verdict hinges on this single item; the W8/W20 divergence needs a "
                  "different explanation (e.g. real window-dependent signal, or noise at this n).")
        else:
            print("  -> W8 is the fragile one; re-examine the W8 SIGNAL verdict specifically.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
