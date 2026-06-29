"""
probe.py — small, testable probe utilities for the Stage-1.5 existence gate.

Kept separate from the script so the math (honest LOO with PCA fit on the train fold only) is unit-
tested without a model. sklearn is imported lazily so the rest of the package imports without it.
"""
from __future__ import annotations

import numpy as np


def pool_layer_major(H, n_layers: int = 3):
    """(n, W*n_layers, D) in layer-major order -> per-layer mean-pool -> (n, n_layers*D).

    Falls back to a global mean-pool if the row count is not divisible by n_layers.
    """
    H = np.asarray(H, np.float32)
    n, R, Dm = H.shape
    if R % n_layers == 0:
        w = R // n_layers
        return H.reshape(n, n_layers, w, Dm).mean(2).reshape(n, n_layers * Dm)
    return H.mean(1)


def probe_auc(X, y, k: int = 10) -> float:
    """Leave-one-out AUC of logistic regression in PCA-reduced space.

    PCA is fit on the TRAIN fold only each iteration (no leakage). Returns NaN if a class is missing.
    Honest for small n; a raw d=4096 probe at n~30 would be trivially separable, hence the reduction.
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    X = np.asarray(X, np.float32)
    y = np.asarray(y)
    n = len(y)
    k = max(1, min(k, n - 2, X.shape[1]))
    probs = np.full(n, np.nan)
    for i in range(n):
        tr = np.arange(n) != i
        if len(set(y[tr].tolist())) < 2:
            continue
        pca = PCA(n_components=k).fit(X[tr])
        clf = LogisticRegression(max_iter=2000).fit(pca.transform(X[tr]), y[tr])
        probs[i] = clf.predict_proba(pca.transform(X[i:i + 1]))[0, list(clf.classes_).index(1)]
    m = ~np.isnan(probs)
    return float(roc_auc_score(y[m], probs[m])) if len(set(y[m].tolist())) == 2 else float("nan")


def text_features(texts, ngram=(2, 4)):
    """Char n-gram count features for the matched supervised text baseline."""
    from sklearn.feature_extraction.text import CountVectorizer
    return CountVectorizer(analyzer="char_wb", ngram_range=ngram, min_df=1).fit_transform(texts).toarray()
