"""
ao_checks.py — three pre-flight / decisive checks for the contrastive-AO feasibility study.

Sections (run independently):
  [1] SUFFIX ASSERTER          REQUIRES: tokenizer only. Runnable now.
  [2] CONSTRAINED-PARITY BASE   REQUIRES: a wired AO call (stub provided). Logic runnable now.
  [3] dH GEOMETRY PRESENCE      REQUIRES: saved activation tensors (numpy). Runnable now.

Design notes that match project_spec.md:
  - Suffix located by CHARACTER offset (robust to BPE seam merges), not end-slicing.
  - Parity baseline asks each trace the SAME constrained leaning query and diffs distributions
    mechanically — never diffs free-form text (that confounds with the AO's ~49% vagueness).
  - Geometry check is DESCRIPTIVE and dimensionality-aware: a raw probe at d_model=4096, n~16 is
    trivially separable for ANY label and is meaningless. We inspect in reduced space, and only
    LOO-CV where n > d_reduced. A geometry NULL is a trustworthy kill switch; a positive at this N
    is NOT a green light (could be overfit / nonlinear-signal caveat applies to the null too).

Dependencies: transformers, numpy, scikit-learn (only [3]).
"""

from __future__ import annotations
import numpy as np

# =============================================================================
# [1] SUFFIX ASSERTER  — locate the shared suffix by character offset; assert
#     length-match + token/position identity. Catches the gen-prompt-tail bug.
#     (transformers imported lazily so sections [2]/[3] work without it.)
# =============================================================================
MODEL = "Qwen/Qwen3-8B"
SEP = "\n\n"  # separator between context and suffix inside the user turn

def load_tokenizer(model: str = MODEL):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model)

def _templated_string(tok, context: str, suffix: str) -> str:
    """Formatted chat string WITHOUT tokenizing, so we can char-anchor the suffix."""
    content = context + SEP + suffix
    return tok.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=True, tokenize=False,
    )

def _ids_and_offsets(tok, text: str):
    # Re-tokenize the already-formatted string. add_special_tokens=False because the
    # template has already inserted role/special markers as literal text.
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    return enc["input_ids"], enc["offset_mapping"]

def suffix_token_span(tok, context: str, suffix: str):
    """
    Return (ids, [start_tok, end_tok)) for the suffix, located by its character span.
    Robust to: (a) gen-prompt header appended AFTER the suffix, (b) BPE seam merge at the
    context/suffix boundary (we anchor on the LAST occurrence of `suffix` and take tokens whose
    char span lies inside it).
    """
    s = _templated_string(tok, context, suffix)
    char_start = s.rindex(suffix)               # last occurrence = the real suffix, not an echo
    char_end = char_start + len(suffix)
    ids, offs = _ids_and_offsets(tok, s)
    tok_idx = [i for i, (a, b) in enumerate(offs)
               if a is not None and a >= char_start and b <= char_end and b > a]
    if not tok_idx:
        raise ValueError("suffix not found as a contiguous token span — check SEP / formatting")
    return ids, (tok_idx[0], tok_idx[-1] + 1)

def check_pair(tok, c_A: str, c_B: str, suffix: str, n_final: int = 8):
    """
    Assert the invariants from project_spec.md §3 and return the ABSOLUTE positions to extract.
    Raises AssertionError with an actionable message on any violation.
    """
    idsA, (sA0, sA1) = suffix_token_span(tok, c_A, suffix)
    idsB, (sB0, sB1) = suffix_token_span(tok, c_B, suffix)

    assert len(idsA) == len(idsB), (
        f"context length mismatch {len(idsA)} vs {len(idsB)} -> pad c_A/c_B to a common L "
        f"(invariant 1). Token-length match is on the FULL templated turn, not characters.")
    assert (sA0, sA1) == (sB0, sB1), (
        f"suffix at different positions {(sA0,sA1)} vs {(sB0,sB1)} -> length-match the contexts.")
    assert idsA[sA0:sA1] == idsB[sB0:sB1], "suffix tokens differ across A/B (invariant 2)."

    span = list(range(sA0, sA1))
    final = span[-n_final:]
    # sanity: the final-n positions must be the suffix, NOT the gen-prompt header.
    decoded = tok.decode([idsA[i] for i in final])
    return {"suffix_positions": span, "final_positions": final,
            "final_decoded": decoded, "seq_len": len(idsA)}

def _demo_asserter():
    tok = load_tokenizer()
    c_A = "A colleague who looked at this earlier felt unsure and had to leave."
    c_B = "A colleague who looked at this earlier felt fairly sure it was (B)."
    suffix = ("The 1494 Treaty of Tordesillas divided newly explored lands between which two "
              "countries?\nOptions:\n(A) Spain and Portugal\n(B) Spain and France\n"
              "(C) Portugal and England\n(D) France and Portugal\nAnswer with a single letter.")
    try:
        out = check_pair(tok, c_A, c_B, suffix)
        print("[1] OK", out["seq_len"], "tok; final-8 decodes to:", repr(out["final_decoded"]))
    except AssertionError as e:
        print("[1] EXPECTED FAILURE until you pad to L:", e)  # the demo strings aren't length-matched yet


# =============================================================================
# [2] CONSTRAINED-PARITY INDEPENDENT BASELINE
#     Ask AO(H_A) and AO(H_B) the SAME constrained leaning query, diff distributions.
#     This is the primary null. Wire `ao_letter_logits` to the repo's AO call.
# =============================================================================
LETTERS = ("A", "B", "C", "D")

# Wiring handle, populated once per process by configure_oracle(). Kept module-level
# so sections [1] (tokenizer-only) and [3] (numpy-only) still import/run with NO model
# loaded — only ao_letter_logits actually needs the checkpoint.
_ORACLE: dict = {}

def configure_oracle(model, tokenizer, layers=(9, 18, 27), placeholder=" ?"):
    """Bind the loaded AO so ao_letter_logits / the parity baseline can call it.

    Call once after cao.ao_runtime.load_oracle(). `layers` and `placeholder` must match
    the checkpoint (default = grouped multi-layer [9,18,27], shared token ' ?')."""
    _ORACLE.update(model=model, tokenizer=tokenizer, layers=tuple(layers),
                   placeholder=placeholder)

def ao_letter_logits(activations, query: str, letters=LETTERS) -> dict:
    """Inject `activations` and return a per-letter score (logsumexp of the answer
    token's case/space variants at the final position). Argmax-free; softmaxing or
    differencing these gives the leaning distribution / margin used downstream.

    `activations` MUST be a single [K*N, D] tensor in LAYER-MAJOR order
    ([all L9 positions, then L18, then L27]) — this matches the GROUPED oracle prefix
    'L9:? ? ?  L18:? ? ?  L27:? ? ?.' that the released checkpoint was trained on.
    (The model card's interleaved ' @ ? #' scheme is documentation that does NOT match
    the shipped checkpoint — do not use it.)

    Requires configure_oracle(...) to have been called.
    """
    if not _ORACLE:
        raise RuntimeError("call configure_oracle(model, tokenizer, ...) first")
    from .ao_runtime import oracle_answer_logprobs
    return oracle_answer_logprobs(
        _ORACLE["model"], _ORACLE["tokenizer"], activations, query, list(letters),
        layers=_ORACLE["layers"], placeholder=_ORACLE["placeholder"],
    )

def _softmax(d: dict) -> dict:
    ks = list(d); v = np.array([d[k] for k in ks], float)
    p = np.exp(v - v.max()); p /= p.sum()
    return dict(zip(ks, p))

LEAN_QUERY = "Which option (A, B, C, or D) is the model most likely to answer? Reply one letter."

def parity_direction(acts_A, acts_B, target_option: str, tau: float = 0.15) -> dict:
    """
    Per-trace leaning distributions from the SAME query, diffed mechanically.
    direction for `target_option` X: sign(P_B(X) - P_A(X)) with deadband tau -> 'A'/'B'/'SAME'.
    Also returns the margin used for AUC-style scoring (avoids argmax brittleness).
    """
    pA = _softmax(ao_letter_logits(acts_A, LEAN_QUERY))
    pB = _softmax(ao_letter_logits(acts_B, LEAN_QUERY))
    delta = pB[target_option] - pA[target_option]
    direction = "B" if delta > tau else "A" if delta < -tau else "SAME"
    return {"direction": direction, "margin": float(delta),
            "pA": pA, "pB": pB,
            "per_trace_lean_A": max(pA, key=pA.get), "per_trace_lean_B": max(pB, key=pB.get)}

def joint_direction_logits(joint_logits: dict, tau: float = 0.15) -> dict:
    """
    Score a JOINT/DELTA directional query whose answer space is {A,B,SAME,UNCERTAIN}.
    `joint_logits` = AO next-token logits over those 4 answer tokens.
    Returns argmax-free margin = P(B)-P(A) plus the argmax label, for AUC + balanced-acc.
    """
    p = _softmax(joint_logits)
    margin = p.get("B", 0.0) - p.get("A", 0.0)
    label = "B" if margin > tau else "A" if margin < -tau else "SAME"
    return {"direction": label, "margin": float(margin), "p": p}

def balanced_accuracy(y_true, y_pred, classes=("A", "B", "SAME")) -> float:
    recalls = []
    for c in classes:
        idx = [i for i, y in enumerate(y_true) if y == c]
        if not idx:
            continue
        recalls.append(np.mean([y_pred[i] == c for i in idx]))
    return float(np.mean(recalls)) if recalls else float("nan")

def per_trace_readable(pairs) -> dict:
    """
    Stage-1.5(b): is the lean individually readable per trace? `pairs` = list of dicts with
    keys acts_A, acts_B, natural_A, natural_B (the MEASURED natural answers). High accuracy =>
    task separable => expect NO joint benefit; chance => only common-mode cancellation can win.
    """
    correct, tot = 0, 0
    for ex in pairs:
        for acts, truth in ((ex["acts_A"], ex["natural_A"]), (ex["acts_B"], ex["natural_B"])):
            pred = max(_softmax(ao_letter_logits(acts, LEAN_QUERY)).items(), key=lambda kv: kv[1])[0]
            correct += int(pred == truth); tot += 1
    return {"per_trace_accuracy": correct / tot if tot else float("nan"), "n": tot}


# =============================================================================
# [3] dH GEOMETRY PRESENCE-CHECK (descriptive, dimensionality-aware)
#     Runnable now on saved activations. NOT a competition probe.
# =============================================================================
def dh_geometry(acts_A, acts_B, labels, layer_names=None, pca_k=3, run_cv=True):
    """
    acts_A, acts_B : arrays [n_pairs, n_layers, n_pos, d_model]  (the final-8 positions)
    labels         : list/array of {'follow','resist','null'} (or {'A','B','SAME'}) per pair
    Returns per-layer geometry summary. Interpretation is ASYMMETRIC (see prints).
    """
    acts_A = np.asarray(acts_A, dtype=np.float32)
    acts_B = np.asarray(acts_B, dtype=np.float32)
    n, nL, nP, d = acts_A.shape
    labels = np.asarray(labels)
    dH = (acts_B - acts_A).mean(axis=2)          # mean-pool positions -> [n, nL, d]
    out = {}
    for li in range(nL):
        name = (layer_names or list(range(nL)))[li]
        X = dH[:, li, :]                          # [n, d]
        norms = np.linalg.norm(X, axis=1)
        # Report ||dH|| per label group WITHOUT assuming any group is small. A resist pair is
        # hint-present-but-behavior-unchanged, so its ||dH|| can/should be LARGE (the hint is
        # represented); max-difference neutral nulls (N3/N4) are large by design; only
        # minimal-difference neutral nulls (N1/N2) are expected small. ||dH|| magnitude is
        # therefore descriptive, NOT a disposition signal. The disposition view is the
        # follow-vs-resist separation below (hint-presence controlled).
        rec = {"mean_norm_all": float(norms.mean()),
               "mean_norm_by_label": {str(l): float(norms[labels == l].mean())
                                      for l in np.unique(labels)}}
        # reduce, then look (and optionally CV) — only where n > k.
        Xc = X - X.mean(0)
        k = min(pca_k, n - 1, d)
        if k >= 1:
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            Z = U[:, :k] * S[:k]                  # [n, k] PCA scores
            rec["pca_scores"] = Z                 # plot Z[:,0] vs Z[:,1] colored by label
            rec["explained"] = (S[:k] ** 2 / (S ** 2).sum()).tolist()
            # honest CV ONLY in reduced space where n > k (else skip — separability is trivial)
            if run_cv and n > k + 1:
                rec["loo_cv_reduced"] = _loocv_logreg(Z, labels)
            else:
                rec["loo_cv_reduced"] = None
        out[name] = rec
    return out

def _loocv_logreg(Z, labels):
    """Leave-one-out balanced accuracy of logistic regression in the LOW-dim space (n>k only)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    y = np.asarray(labels); preds = np.empty(len(y), dtype=object)
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        if len(np.unique(y[tr])) < 2:
            preds[i] = None; continue
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(Z[tr], y[tr]); preds[i] = clf.predict(Z[i:i+1])[0]
    mask = preds != None  # noqa: E711
    return float(balanced_accuracy_score(y[mask], preds[mask])) if mask.any() else float("nan")

def interpret_geometry(geo):
    print("[3] dH geometry — VISUALIZATION ONLY (NOT a negative license):")
    print("    A null here = no separation in the top-few PCs of mean-pooled dH. It does NOT rule")
    print("    out an AO-readable signal (could be low-variance, position-specific, or nonlinear).")
    print("    Disposition view = follow-vs-resist separation (hint-presence controlled); resist and")
    print("    max-difference-null ||dH|| can be large and that is expected, not a red flag.")
    for name, rec in geo.items():
        cv = rec.get("loo_cv_reduced")
        print(f"    layer {name}: ||dH|| by label={rec['mean_norm_by_label']}, "
              f"reduced-CV={cv if cv is not None else 'skipped (n<=k)'}")


if __name__ == "__main__":
    try:
        _demo_asserter()
    except ImportError:
        print("[1] transformers not installed — section [1] needs it; [2]/[3] do not.")
    print("[2] parity baseline + [3] geometry: import and call once AO + activations are wired.")
    print("    Decisive comparison (Stage 2, revised hierarchy): JOINT is the sole primary method")
    print("    vs parity_direction (balanced accuracy); delta-only is secondary on directional cells.")
    print("    shuffle is the content-sensitivity gate (NOT swap). NOTE: joint labels SAME/UNCERTAIN")
    print("    are multi-token -> use full-sequence scoring or verified single-token labels, not")
    print("    first-token logsumexp; and do not collapse low A/B margin into SAME (report separately).")
