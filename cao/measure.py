"""
measure.py — behavioral labelling of the Family-C pairs (GPU-only).

Runs the TARGET model (Qwen3-8B, AO adapter DISABLED) on each pair's A/B contexts and
measures the answer distribution over {A,B,C,D} by FORCED-CHOICE LOGITS (not regex parsing
of free text — the pilot showed the model answers with words/values, breaking parsing).

Method:
- Prefill the assistant turn with "The answer is (" and read next-token logits over the
  four option letters -> answer distribution. Parse-free; mirrors how we score the oracle.
- Reasoning OFF: one forward per condition -> exact letter softmax (no sampling needed).
- Reasoning ON: sample K chains-of-thought, then force-read the post-CoT letter for each ->
  empirical distribution. (The follow/resist outcome forms during the CoT — the timing point.)

The measured label is ground truth: it re-sorts / discards predicted labels, never relabels.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import torch

from cao import pairs as P
from cao.ao_runtime import _model_device

LETTERS = ("A", "B", "C", "D")
OFF_CUE = "The answer is ("
ON_CUE = "\n\nThe answer is ("


def letter_ids(tok) -> dict[str, list[int]]:
    """First-token ids for each option letter across no-space/space variants."""
    out = {}
    for L in LETTERS:
        s = set()
        for v in (L, " " + L):
            e = tok.encode(v, add_special_tokens=False)
            if e:
                s.add(e[0])
        out[L] = sorted(s)
    return out


@torch.no_grad()
def _letter_scores(model, tok, prefill_text: str, lids: dict[str, list[int]]) -> dict[str, float]:
    """logsumexp of next-token logprobs over each letter's variants, at the prefill end."""
    dev = _model_device(model)
    enc = tok(prefill_text, return_tensors="pt", add_special_tokens=False).to(dev)
    out = model(**enc)
    lp = torch.log_softmax(out.logits[0, -1, :].float(), dim=-1)
    return {L: float(torch.logsumexp(lp[torch.tensor(lids[L], device=dev)], dim=0)) for L in LETTERS}


@torch.no_grad()
def _letter_scores_ids(model, ids_list: list[int], lids: dict[str, list[int]], dev) -> dict[str, float]:
    """Same as _letter_scores but on an exact token-ID sequence (no decode/re-tokenize).

    Used for reasoning ON: we keep the model's generated token IDs verbatim and append the
    cue's token IDs, so we score the answer from the exact trajectory the model produced.
    """
    enc = torch.tensor([ids_list], device=dev)
    out = model(input_ids=enc)
    lp = torch.log_softmax(out.logits[0, -1, :].float(), dim=-1)
    return {L: float(torch.logsumexp(lp[torch.tensor(lids[L], device=dev)], dim=0)) for L in LETTERS}


def _softmax(scores: dict[str, float]) -> dict[str, float]:
    m = max(scores.values())
    ex = {k: math.exp(v - m) for k, v in scores.items()}
    z = sum(ex.values())
    return {k: v / z for k, v in ex.items()}


@torch.no_grad()
def answer_distribution(model, tok, user_content: str, reasoning: bool, k: int = 8,
                        temperature: float = 0.7, top_p: float = 0.95, max_new: int = 640,
                        lids: dict | None = None, want_raw: bool = False) -> dict:
    """Distribution over {A,B,C,D} for the TARGET (adapter disabled), forced-choice."""
    lids = lids or letter_ids(tok)
    if not reasoning:
        fmt = tok.apply_chat_template([{"role": "user", "content": user_content}],
                                      tokenize=False, add_generation_prompt=True, enable_thinking=False)
        with model.disable_adapter():
            sc = _letter_scores(model, tok, fmt + OFF_CUE, lids)
        p = _softmax(sc)
        return {"p": p, "argmax": max(p, key=p.get), "method": "logit", "n": None}

    fmt = tok.apply_chat_template([{"role": "user", "content": user_content}],
                                  tokenize=False, add_generation_prompt=True, enable_thinking=True)
    dev = _model_device(model)
    ins = tok(fmt, return_tensors="pt").to(dev)
    plen = ins["input_ids"].shape[1]
    cue_ids = tok.encode(ON_CUE, add_special_tokens=False)  # appended at TOKEN level (faithful)
    with model.disable_adapter():
        gen = model.generate(**ins, max_new_tokens=max_new, do_sample=True,
                             temperature=temperature, top_p=top_p, num_return_sequences=k)
    counts, raw = Counter(), None
    for i in range(k):
        seq_ids = gen[i].tolist()  # exact prompt+generation token IDs (no decode/re-tokenize)
        if want_raw and raw is None:
            raw = tok.decode(gen[i][plen:], skip_special_tokens=False)  # for eyeballing only
        with model.disable_adapter():
            sc = _letter_scores_ids(model, seq_ids + cue_ids, lids, dev)
        counts[max(sc, key=sc.get)] += 1
    p = {L: counts.get(L, 0) / k for L in LETTERS}
    out = {"p": p, "argmax": max(p, key=p.get), "method": "sample+logit", "n": k, "counts": dict(counts)}
    if want_raw:
        out["raw"] = raw
    return out


@torch.no_grad()
def answer_distribution_nocue(model, tok, user_content: str, lids: dict | None = None) -> dict:
    """§10 OFF readout: letter distribution at the FIRST answer position, NO appended cue.

    The answer commitment lives in the suffix ("Answer with exactly one letter:"); we read the
    next-token distribution right after the chat template's generation prompt — i.e. the EXACT
    pre-output state activations are extracted from (scripts/extract_activations.py). This is the
    spec-v4 §10 readout: it labels the stored state, unlike answer_distribution(OFF) which appends
    "The answer is (" and thereby labels a different state. Adapter disabled (pure target behavior).
    """
    lids = lids or letter_ids(tok)
    fmt = tok.apply_chat_template([{"role": "user", "content": user_content}],
                                  tokenize=False, add_generation_prompt=True, enable_thinking=False)
    with model.disable_adapter():
        sc = _letter_scores(model, tok, fmt, lids)
    p = _softmax(sc)
    return {"p": p, "argmax": max(p, key=p.get), "method": "logit-nocue", "n": None}


def measure_pair(model, tok, pair: dict, reasoning: bool, k: int = 8, tau: float = 0.5,
                 temperature: float = 0.7, lids: dict | None = None, want_raw: bool = False) -> dict:
    """Measure one pair. direction (for query_target X): 'B' if cond B raises P(X) by > tau,
    'A' if cond A does, else 'SAME'. clean = near-deterministic flip / shared dominant option."""
    lids = lids or letter_ids(tok)
    dA = answer_distribution(model, tok, P.target_user_content(pair, "A"), reasoning,
                             k=k, temperature=temperature, lids=lids, want_raw=want_raw)
    dB = answer_distribution(model, tok, P.target_user_content(pair, "B"), reasoning,
                             k=k, temperature=temperature, lids=lids, want_raw=want_raw)
    qt = pair["query_target"]
    fA, fB = dA["p"][qt], dB["p"][qt]
    delta = fB - fA
    direction = "B" if delta > tau else "A" if delta < -tau else "SAME"
    domA, domB = max(dA["p"], key=dA["p"].get), max(dB["p"], key=dB["p"].get)
    if direction == "SAME":
        clean = (domA == domB) and dA["p"][domA] >= 0.75 and dB["p"][domB] >= 0.75
    else:
        clean = abs(delta) >= tau and max(dA["p"].values()) >= 0.75 and max(dB["p"].values()) >= 0.75
    out = {
        "pair_id": pair["pair_id"], "type": pair["predicted_type"], "reasoning": reasoning,
        "query_target": qt, "fA": fA, "fB": fB, "delta": delta,
        "pA": {k_: round(v, 2) for k_, v in dA["p"].items()},
        "pB": {k_: round(v, 2) for k_, v in dB["p"].items()},
        "domA": domA, "domB": domB, "measured_direction": direction,
        "predicted_direction": pair["predicted_direction"],
        "agree": direction == pair["predicted_direction"], "clean": clean, "method": dA["method"],
    }
    if want_raw:
        out["rawA"], out["rawB"] = dA.get("raw"), dB.get("raw")
    return out


# Kept only as an optional diagnostic for eyeballing free-text generations; NOT used for labels.
_ANS_PATTERNS = [
    re.compile(r"(?:answer|option|choose|select|correct)\b[^A-Da-d]{0,15}\(?([ABCD])\)?", re.I),
    re.compile(r"\(([ABCD])\)"),
    re.compile(r"\b([ABCD])\b"),
]


def parse_letter(text: str):
    seg = text.split("</think>")[-1] if "</think>" in text else text
    seg = seg.strip()
    if len(seg) == 1 and seg.upper() in "ABCD":
        return seg.upper()
    for rx in _ANS_PATTERNS:
        m = rx.findall(seg)
        if m:
            return m[-1].upper()
    return None
