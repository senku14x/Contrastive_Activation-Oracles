#!/usr/bin/env python3
"""Pure-logic tests for the v4 dataset scaffolding + probe math (no model/tokenizer needed)."""
import sys

import numpy as np

sys.path.insert(0, ".")
from cao import dataset as D       # noqa: E402
from cao import family_l as L      # noqa: E402
from cao import family_p as P      # noqa: E402
from cao import probe as PB        # noqa: E402


def test_rotate_preserves_correctness():
    opts = {"A": "wa", "B": "xb", "C": "yc", "D": "zd"}
    for shift in range(4):
        new, ng, nf = D.rotate_options(opts, "A", "C", shift)
        # the TEXT that was gold/flawed must still sit at the new gold/flawed letters
        assert new[ng] == opts["A"], (shift, ng, new)
        assert new[nf] == opts["C"], (shift, nf, new)
        assert set(new.values()) == set(opts.values())
    print("  ok rotate_options preserves correctness")


def test_balance_positions():
    raw = [{"options": {"A": str(i), "B": "b", "C": "c", "D": "d"},
            "gold_option": "A", "flawed_consistent_option": "C"} for i in range(12)]
    bal = D.balance_positions(raw)
    from collections import Counter
    gc = Counter(it["gold_option"] for it in bal)
    fc = Counter(it["flawed_consistent_option"] for it in bal)
    assert max(gc.values()) - min(gc.values()) <= 1, gc          # gold spread across all 4
    assert all(it["gold_option"] != it["flawed_consistent_option"] for it in bal)
    assert set(gc) == set(D.LETTERS) and set(fc) == set(D.LETTERS)
    print(f"  ok balance_positions (gold={dict(gc)}, flawed={dict(fc)})")


def test_suffix_and_content():
    s = D.make_suffix("Q?", {"A": "a", "B": "b", "C": "c", "D": "d"})
    assert s.endswith(D.ANSWER_INSTRUCTION) and "(A) a" in s and "(D) d" in s
    assert D.user_content("ctx", s) == "ctx" + D.SEP + s
    assert D.user_content("", s) == s                            # ablated/bare condition
    print("  ok make_suffix / user_content")


def test_candidate_record_schema():
    r = D.candidate_record("id1", "L", "Q?", {"A": "a", "B": "b", "C": "c", "D": "d"},
                           "C", "A", "ctxA", "ctxB", "suf")
    for key in ("candidate_id", "family", "gold_option", "flawed_consistent_option",
                "context_correct", "context_flawed", "context_ablated", "shared_suffix",
                "ablation_verified_load_bearing", "token_audit", "neutral_distribution",
                "flawed_distribution", "derived", "text_only_gate", "status"):
        assert key in r, key
    assert r["ablation_verified_load_bearing"] is None and r["status"] == "candidate"
    print("  ok candidate_record schema")


def test_family_l_invariants():
    assert len(L.FAMILY_L) >= 12
    for it in L.FAMILY_L:
        g, f = it["gold_option"], it["flawed_consistent_option"]
        assert g != f
        assert "cannot be determined" in it["options"][g].lower()   # gold IS the cannot-determine opt
        assert it["context_ablated"] in it["context_correct"]       # givens shared
        assert it["context_ablated"] in it["context_flawed"]
        assert it["context_correct"] != it["context_flawed"]
        c = L.conditions(it)
        assert set(c) == {"ablated", "correct", "flawed"}
    print(f"  ok family_l invariants ({len(L.FAMILY_L)} items)")


def test_family_p_pairs():
    pairs = list(P.enumerate_pairs())
    assert len(pairs) == len(P.FAMILY_P) * len(P.TEMPLATES)
    for it in P.FAMILY_P:
        assert it["gold_option"] != it["wrong_option"]
    rid, it, tpl = pairs[0]
    cond = P.conditions(it, tpl)
    assert it["wrong_option"] in cond["flawed"] or f"({it['wrong_option']})" in cond["flawed"]
    print(f"  ok family_p ({len(pairs)} item×template pairs)")


def test_probe_pool_and_auc():
    # pool: (n=2, R=6, D=4) layer-major, 3 layers -> (2, 12)
    H = np.arange(2 * 6 * 4, dtype=np.float32).reshape(2, 6, 4)
    pooled = PB.pool_layer_major(H, n_layers=3)
    assert pooled.shape == (2, 12)
    # AUC: separable blobs -> high; shuffled labels -> ~chance
    rng = np.random.default_rng(0)
    n, d = 40, 10
    y = np.array([0] * 20 + [1] * 20)
    X = rng.normal(size=(n, d)).astype(np.float32)
    X[y == 1, 0] += 4.0                                          # class signal in dim 0
    auc = PB.probe_auc(X, y, k=5)
    shuf = np.nanmean([PB.probe_auc(X, rng.permutation(y), k=5) for _ in range(8)])
    assert auc > 0.85, auc
    assert 0.3 < shuf < 0.7, shuf
    print(f"  ok probe (separable AUC={auc:.2f}, shuffled≈{shuf:.2f})")


if __name__ == "__main__":
    fns = [test_rotate_preserves_correctness, test_balance_positions, test_suffix_and_content,
           test_candidate_record_schema, test_family_l_invariants, test_family_p_pairs,
           test_probe_pool_and_auc]
    for f in fns:
        f()
    print(f"\nALL {len(fns)} DATASET-LOGIC TESTS PASSED")
