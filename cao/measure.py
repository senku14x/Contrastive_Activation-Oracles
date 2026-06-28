"""
measure.py — behavioral labelling of the Family-C pairs (GPU-only).

Runs the TARGET model (Qwen3-8B, AO adapter DISABLED) on each pair's A/B contexts,
K samples/condition, parses the emitted MCQ letter, and computes the MEASURED behavioral
direction. The measured label is ground truth: it re-sorts / discards the predicted labels
(never relabels them). Used by the pilot and the full Stage-1 measurement.

Notes:
- Sampling (temperature>0) is required to estimate the answer distribution; the clean-flip
  filter then keeps only pairs whose flip is near-deterministic.
- Reasoning ON: enable_thinking=True; the follow/resist outcome is produced during the CoT
  (the timing concern). Reasoning OFF: enable_thinking=False; answer commits right after the
  suffix (the cleaner pre-output read).
"""

from __future__ import annotations

import re
from collections import Counter

import torch

from cao import pairs as P
from cao.ao_runtime import _model_device

# Answer-letter parsers, tried in order of reliability; we take the LAST match (final answer).
_ANS_PATTERNS = [
    re.compile(r"(?:answer|option|choose|select|correct)\b[^A-Da-d]{0,15}\(?([ABCD])\)?", re.I),
    re.compile(r"\*\*\(?([ABCD])\)?\*\*"),      # **A** / **(A)**
    re.compile(r"\(([ABCD])\)"),                 # (A)
    re.compile(r"\b([ABCD])\b"),                 # standalone A
]


def parse_letter(text: str):
    """Best-effort extraction of the committed answer letter (A-D), else None.

    Looks only after </think> when present (so option letters inside the CoT don't count),
    and prefers explicit 'answer/option' phrasing over a bare letter; takes the last match.
    """
    seg = text.split("</think>")[-1] if "</think>" in text else text
    seg = seg.strip()
    if len(seg) == 1 and seg.upper() in "ABCD":
        return seg.upper()
    for rx in _ANS_PATTERNS:
        m = rx.findall(seg)
        if m:
            return m[-1].upper()
    m = re.findall(r"\b([ABCD])\b", text)  # last-resort: scan everything
    return m[-1].upper() if m else None


@torch.no_grad()
def target_samples(model, tok, user_content: str, reasoning: bool, k: int = 8,
                   temperature: float = 0.7, top_p: float = 0.95, max_new: int | None = None):
    """K sampled completions from the TARGET (adapter disabled)."""
    if max_new is None:
        max_new = 640 if reasoning else 16
    formatted = tok.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False, add_generation_prompt=True, enable_thinking=reasoning,
    )
    dev = _model_device(model)
    inputs = tok(formatted, return_tensors="pt").to(dev)
    plen = inputs["input_ids"].shape[1]
    with model.disable_adapter():
        gen = model.generate(
            **inputs, max_new_tokens=max_new, do_sample=True,
            temperature=temperature, top_p=top_p, num_return_sequences=k,
        )
    return [tok.decode(gen[i][plen:], skip_special_tokens=True) for i in range(k)]


def measure_pair(model, tok, pair: dict, reasoning: bool, k: int = 8, tau: float = 0.5,
                 temperature: float = 0.7, keep_texts: bool = True) -> dict:
    """Measure one pair: letter distributions for A and B, signed delta, measured direction.

    direction (for the pair's query_target X): 'B' if condition B raises P(X) by > tau,
    'A' if condition A does, else 'SAME'. clean = near-deterministic flip (directional) or
    same dominant option both sides (null/resist).
    """
    letters, texts = {}, {}
    for cond in ("A", "B"):
        txts = target_samples(model, tok, P.target_user_content(pair, cond), reasoning,
                              k=k, temperature=temperature)
        texts[cond] = txts
        letters[cond] = [parse_letter(t) for t in txts]

    qt = pair["query_target"]

    def frac(ls):
        n = sum(1 for x in ls if x is not None)
        return (sum(1 for x in ls if x == qt) / n) if n else float("nan")

    fA, fB = frac(letters["A"]), frac(letters["B"])
    domA = Counter(x for x in letters["A"] if x).most_common(1)
    domB = Counter(x for x in letters["B"] if x).most_common(1)
    delta = (fB - fA) if (fA == fA and fB == fB) else float("nan")
    direction = "B" if (delta == delta and delta > tau) else "A" if (delta == delta and delta < -tau) else "SAME"

    # clean-flip: directional => near-deterministic opposite ends; null/resist => same dom both sides
    def dom_frac(c):
        cnt = Counter(x for x in letters[c] if x)
        n = sum(cnt.values())
        return (cnt.most_common(1)[0][1] / n) if n else 0.0
    if direction == "SAME":
        clean = bool(domA and domB and domA[0][0] == domB[0][0] and dom_frac("A") >= 0.75 and dom_frac("B") >= 0.75)
    else:
        clean = (delta == delta) and abs(delta) >= tau and dom_frac("A") >= 0.75 and dom_frac("B") >= 0.75

    parse_fail = sum(1 for c in ("A", "B") for x in letters[c] if x is None)
    out = {
        "pair_id": pair["pair_id"], "type": pair["predicted_type"], "reasoning": reasoning,
        "query_target": qt, "fA": fA, "fB": fB, "delta": delta,
        "domA": domA, "domB": domB, "measured_direction": direction,
        "predicted_direction": pair["predicted_direction"],
        "agree": direction == pair["predicted_direction"], "clean": clean,
        "parse_fail": parse_fail, "letters": letters,
    }
    if keep_texts:
        out["texts"] = texts
    return out
