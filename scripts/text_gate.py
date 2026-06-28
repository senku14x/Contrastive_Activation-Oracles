#!/usr/bin/env python3
"""
text_gate.py — the text-only leakiness gate for candidate FOLLOW pairs.

A candidate follow is only USABLE if the behavioral direction is NOT predictable from the
prompt text alone (otherwise the AO would just be doing text inversion, not reading state).
We give a STRONG, NON-Qwen reader the two full prompts (filler condition vs hint condition)
and ask it to predict the target's letter in each — neutrally, without flagging the hint.
If the reader predicts the measured shift well above chance, the pair LEAKS and is cut.

Why non-Qwen: the target is Qwen3-8B, so a Qwen reader would just reproduce the target's
own behavior (degenerate). We need a different model to test whether the direction is a
generic text-inferable property vs Qwen3-8B's idiosyncratic uncertainty.

Reader = any OpenAI-compatible chat endpoint. Set:
  GATE_API_KEY   (required)
  GATE_MODEL     (required; a strong non-Qwen reasoner, e.g. a frontier model)
  GATE_BASE_URL  (default https://openrouter.ai/api/v1)

Input: runs/uncertainty_screen_off.json (from uncertainty_screen.py). Gates the items whose
tip.flip is true. Writes runs/text_gate.json.
"""
from __future__ import annotations

import json
import os
import re
import sys

import requests

sys.path.insert(0, ".")
from cao import pairs as P  # noqa: E402

BASE_URL = os.environ.get("GATE_BASE_URL", "https://openrouter.ai/api/v1")
API_KEY = os.environ.get("GATE_API_KEY")
MODEL = os.environ.get("GATE_MODEL")

GATE_TMPL = """You are predicting how one specific AI assistant will answer a multiple-choice \
question. Below are two prompts that assistant received (they differ only in their opening \
context). For EACH prompt, predict the single option letter (A, B, C, or D) the assistant will \
choose. Do not explain. Respond exactly as: 1: <letter>, 2: <letter>

=== PROMPT 1 ===
{a}

=== PROMPT 2 ===
{b}
"""


def reader_predict(prompt_a: str, prompt_b: str) -> tuple[str | None, str | None]:
    r = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={"model": MODEL, "temperature": 0,
              "messages": [{"role": "user", "content": GATE_TMPL.format(a=prompt_a, b=prompt_b)}]},
        timeout=120,
    )
    r.raise_for_status()
    txt = r.json()["choices"][0]["message"]["content"]
    m = re.findall(r"[12]\s*[:.\-]\s*\(?([ABCD])\)?", txt)
    p1 = m[0].upper() if len(m) >= 1 else None
    p2 = m[1].upper() if len(m) >= 2 else None
    return p1, p2


def main() -> int:
    if not (API_KEY and MODEL):
        raise SystemExit("Set GATE_API_KEY and GATE_MODEL (strong non-Qwen reader). "
                         "Optionally GATE_BASE_URL (default openrouter).")
    src = sys.argv[1] if len(sys.argv) > 1 else "runs/uncertainty_screen_off.json"
    data = json.load(open(src))
    by_id = {it["id"]: it for it in data["items"]}
    cands = [it for it in data["items"] if it.get("tip", {}).get("flip")]
    print(f"Gating {len(cands)} candidate(s) with reader={MODEL}\n")

    results, leak = [], 0
    for it in cands:
        runner = it["tip"]["runner"]                         # the hinted option (measured shift target)
        suf = P.shared_suffix(it["question"], tuple(it["options"]))
        pa = P.FILLER + P.SEP + suf
        pb = P.HINT_TMPL.format(X=runner) + P.SEP + suf
        r1, r2 = reader_predict(pa, pb)
        # measured: filler argmax (a_star) vs hint -> runner. Reader "predicts the shift" if it
        # predicts the answer moves TO runner in prompt 2 but not in prompt 1.
        a_star = max(it["p_nohint"], key=it["p_nohint"].get)
        predicts_shift = (r2 == runner) and (r1 != runner)
        leak += int(predicts_shift)
        results.append({"id": it["id"], "cat": it["cat"], "runner": runner,
                        "measured_nohint_argmax": a_star, "reader_p1": r1, "reader_p2": r2,
                        "reader_predicts_shift": predicts_shift})
        print(f"{it['id']:4} {it['cat']:5} hint->{runner}: reader predicts P1={r1} P2={r2}  "
              f"predicts_shift={predicts_shift}  {'LEAKS' if predicts_shift else 'ok'}")

    n = len(cands)
    rate = leak / n if n else float("nan")
    print(f"\nreader predicted the shift on {leak}/{n} (rate={rate:.2f}). "
          f"Chance for a 4-way per-condition guess is well below this.")
    print("PASS (non-leaky) = reader at/near chance. LEAKS = reader predicts the shift -> cut that pair.")
    print("Note: this is one reader; confirm survivors with a second strong non-Qwen reader before locking.")
    os.makedirs("runs", exist_ok=True)
    json.dump({"reader": MODEL, "n": n, "n_leak": leak, "results": results},
              open("runs/text_gate.json", "w"), indent=2)
    print("wrote runs/text_gate.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
