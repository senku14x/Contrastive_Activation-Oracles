#!/usr/bin/env python3
"""Pure-logic tests that need NO model/tokenizer (runnable in any environment).

Covers the parts of the wiring that don't touch HuggingFace: the grouped oracle-
prefix builder + layer-major char spans, the parity/scoring math, and the dH
geometry presence-check. The model-dependent path is gated separately by
scripts/stage0_repro.py (GPU box only).
"""
import sys

import numpy as np

sys.path.insert(0, ".")
from cao import ao_checks as ck          # noqa: E402
from cao import ao_runtime as rt         # noqa: E402


def test_grouped_prefix():
    layers, k = [9, 18, 27], 2
    prefix, spans = rt.build_grouped_prefix(layers, k, placeholder=" ?")
    assert prefix == "L9: ? ? L18: ? ? L27: ? ?.\n", repr(prefix)
    assert len(spans) == len(layers) * k == 6
    for s, e in spans:
        assert prefix[s:e] == " ?", (s, e, prefix[s:e])
    # layer-major: the first k spans precede the "L18:" label, etc.
    assert spans[0][1] <= prefix.index("L18:")
    assert spans[2][0] >= prefix.index("L18:") and spans[2][1] <= prefix.index("L27:")
    print("  ok grouped_prefix")


def test_softmax_and_joint():
    p = ck._softmax({"A": 2.0, "B": 0.0})
    assert abs(sum(p.values()) - 1.0) < 1e-9 and p["A"] > p["B"]
    # joint directional logits over {A,B,SAME,UNCERTAIN}
    out = ck.joint_direction_logits({"A": 3.0, "B": 0.0, "SAME": 0.0, "UNCERTAIN": 0.0}, tau=0.15)
    assert out["direction"] == "A" and out["margin"] < 0, out
    out = ck.joint_direction_logits({"A": 0.0, "B": 3.0, "SAME": 0.0, "UNCERTAIN": 0.0}, tau=0.15)
    assert out["direction"] == "B" and out["margin"] > 0, out
    out = ck.joint_direction_logits({"A": 1.0, "B": 1.0, "SAME": 5.0, "UNCERTAIN": 0.0}, tau=0.15)
    assert out["direction"] == "SAME", out
    print("  ok softmax/joint")


def test_balanced_accuracy():
    yt = ["A", "B", "SAME", "SAME", "B"]
    yp = ["A", "B", "SAME", "B", "B"]           # A:1/1, B:2/2, SAME:1/2 -> (1+1+0.5)/3
    assert abs(ck.balanced_accuracy(yt, yp) - (1 + 1 + 0.5) / 3) < 1e-9
    print("  ok balanced_accuracy")


def test_parity_direction_with_fake_oracle():
    # monkeypatch the model call: map a per-trace "lean letter" to a score dict.
    leans = {}

    def fake(activations, query, letters=ck.LETTERS):
        lean = leans[id(activations)]
        return {L: (3.0 if L == lean else 0.0) for L in letters}

    ck.ao_letter_logits = fake
    aA, aB = object(), object()
    # A leans 'A', B leans 'B'; query target 'B' -> P_B(B) high, P_A(B) low -> direction 'B'
    leans[id(aA)], leans[id(aB)] = "A", "B"
    r = ck.parity_direction(aA, aB, target_option="B", tau=0.15)
    assert r["direction"] == "B" and r["margin"] > 0, r
    # both lean 'A' (hint ignored) -> no shift on 'B' -> SAME
    leans[id(aB)] = "A"
    r = ck.parity_direction(aA, aB, target_option="B", tau=0.15)
    assert r["direction"] == "SAME", r
    print("  ok parity_direction (fake oracle)")


def test_dh_geometry_asymmetric():
    rng = np.random.default_rng(0)
    n, nL, nP, d = 12, 1, 4, 16
    A = rng.normal(size=(n, nL, nP, d)).astype(np.float32)
    B = A.copy()
    labels = []
    sep = np.zeros(d, np.float32); sep[0] = 6.0     # separating direction for follow/resist
    for i in range(n):
        if i < 4:                                   # follow: +sep
            B[i] += sep; labels.append("follow")
        elif i < 8:                                 # resist: -sep
            B[i] -= sep; labels.append("resist")
        else:                                       # null: ~no change
            B[i] += rng.normal(scale=0.01, size=(nL, nP, d)); labels.append("null")
    geo = ck.dh_geometry(A, B, labels, pca_k=3, run_cv=True)
    rec = geo[0]
    nb = rec["mean_norm_by_label"]                              # per-label ||dH|| (no null=small assumption)
    assert nb["null"] < nb["follow"] and nb["null"] < nb["resist"]  # this synthetic null IS small by construction
    cv = rec["loo_cv_reduced"]
    assert cv is None or (0.0 <= cv <= 1.0)                    # runs & returns sane value
    print(f"  ok dh_geometry (by_label={ {k: round(v, 1) for k, v in nb.items()} }, cv={cv})")


if __name__ == "__main__":
    fns = [test_grouped_prefix, test_softmax_and_joint, test_balanced_accuracy,
           test_parity_direction_with_fake_oracle, test_dh_geometry_asymmetric]
    for f in fns:
        f()
    print(f"\nALL {len(fns)} PURE-LOGIC TESTS PASSED")
