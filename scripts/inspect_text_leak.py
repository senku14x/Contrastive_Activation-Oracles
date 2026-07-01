#!/usr/bin/env python3
"""
inspect_text_leak.py — what is the text-feature probe actually keying on?

The Stage-1.5 text-feature baseline (cao/probe.py:text_features, char n-grams 2-4) has stayed
stubbornly high across every cut of the data (0.378 -> 0.623 -> 0.688 -> 0.702 as the subtype mix
narrowed), even restricted to a single flaw_subtype. That rules out "mixing fallacy shapes" as the
sole explanation. This fits the SAME char-ngram + logistic-regression pipeline on the FULL set (no
LOO -- this is a diagnostic, not another AUC estimate) and prints the top-weighted n-grams, so we can
see concretely whether the leak is a narrow, fixable artifact (e.g. one recurring phrase/proper-noun)
or a diffuse authorial-style signature (harder to construction-fix).

    python3 scripts/inspect_text_leak.py
    python3 scripts/inspect_text_leak.py --meta data/activations_meta.json
"""
from __future__ import annotations

import argparse
import json
import sys

sys.path.insert(0, ".")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="data/activations_meta.json")
    ap.add_argument("--top", type=int, default=20)
    a = ap.parse_args()

    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression

    items = json.load(open(a.meta))["items"]
    texts = [it["context_flawed"] for it in items]
    y = [1 if it["status"] == "clean_miss" else 0 for it in items]  # 1=MISS, 0=CATCH
    ids = [it["candidate_id"] for it in items]
    print(f"n={len(items)}  MISS={sum(y)}  CATCH={len(y) - sum(y)}\n")

    vec = CountVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
    X = vec.fit_transform(texts)
    clf = LogisticRegression(max_iter=2000).fit(X, y)

    vocab = vec.get_feature_names_out()
    coefs = clf.coef_[0]
    order = coefs.argsort()

    print(f"=== top {a.top} n-grams predicting MISS (positive weight) ===")
    for i in order[::-1][:a.top]:
        print(f"  {vocab[i]!r:20} weight={coefs[i]:+.3f}")

    print(f"\n=== top {a.top} n-grams predicting CATCH (negative weight) ===")
    for i in order[:a.top]:
        print(f"  {vocab[i]!r:20} weight={coefs[i]:+.3f}")

    # sanity: is this just re-detecting domain/proper-noun vocabulary, or something more like style
    # (short connector words, punctuation, sentence-length artifacts)?
    alpha_only = [(vocab[i], coefs[i]) for i in order[::-1][:a.top] if vocab[i].strip().isalpha()]
    print(f"\n({sum(1 for v, _ in alpha_only if len(v.strip()) <= 3)}/{len(alpha_only)} of the top "
          f"MISS n-grams above are short (<=3 char) fragments -- high count here points to a diffuse "
          f"stylistic/structural tell rather than a single fixable word or proper noun.)")

    print("\n=== example items, each class (first 2) ===")
    for label, name in ((1, "MISS"), (0, "CATCH")):
        shown = 0
        for it, yy in zip(items, y):
            if yy == label and shown < 2:
                print(f"  [{name}] {it['candidate_id']}: {it['context_flawed'][:160]}")
                shown += 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
