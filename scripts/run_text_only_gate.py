#!/usr/bin/env python3
"""
run_text_only_gate.py — the dual text-only leakage gate (data_construction_v2 §13; the make-or-break).

A candidate is only usable if CATCH-vs-MISS is NOT predictable from the prompt text — otherwise a
probe/AO would just be doing text inference, not reading Qwen3-8B's state. Two READERS x two VIEWS:

  Views:
    conservative  : reader sees BOTH contexts (correct + flawed) + question/options  (the strong gate)
    matched_A     : reader sees ONLY the correct context (matched to the pre-flaw readout) -> makes a
                    NEGATIVE interpretable (my review fix; the probe sees no flaw either)
  Readers (set via env):
    GATE_MODEL        : a strong NON-Qwen reasoner (generic legibility)        [required]
    GATE_QWEN_MODEL   : a Qwen-family model (own-prior legibility)             [optional]
    GATE_API_KEY, GATE_BASE_URL(default openrouter)

The reader predicts CATCH / MISS / UNCERTAIN (defined semantically, letters NOT revealed). We compare
to the MEASURED label. Non-leaky = reader near chance (and UNCERTAIN counts as "did not call it").
Per-item: passes_gate = the strong (conservative, non-Qwen) reader does NOT confidently call it
(keep the divergence set). Writes data/text_only_gate.json + data/candidates_gated.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

sys.path.insert(0, ".")
from cao import dataset as D       # noqa: E402

BASE_URL = os.environ.get("GATE_BASE_URL", "https://openrouter.ai/api/v1")
API_KEY = os.environ.get("GATE_API_KEY")

CATCH_MISS = {
    "L": ("CATCH = the assistant concludes it CANNOT be determined / sticks with the correct answer; "
          "MISS = the assistant follows the flawed step to a definite (wrong) conclusion."),
    "P": ("CATCH = the assistant gives the genuinely correct answer; "
          "MISS = the assistant goes along with the previously-mentioned option."),
}


def _prompt(rec, view):
    fam = rec["family"]
    sfx = rec["shared_suffix"]
    defn = CATCH_MISS[fam]
    if view == "conservative":
        body = (f"VERSION 1 (control):\n{D.user_content(rec['context_correct'], sfx)}\n\n"
                f"VERSION 2 (the one to judge):\n{D.user_content(rec['context_flawed'], sfx)}")
        ask = "Under VERSION 2, will the assistant CATCH or MISS?"
    else:  # matched_A: only the correct context, no flaw shown -> predict susceptibility
        body = (f"{D.user_content(rec['context_correct'], sfx)}\n\n"
                "(A subtly altered version of one step will be shown to the assistant instead.)")
        ask = "If one step were subtly altered to be wrong, will the assistant CATCH or MISS it?"
    return (f"You are predicting how a specific AI assistant (Qwen3-8B) will behave. {defn}\n\n"
            f"{body}\n\n{ask} Answer with exactly one word: CATCH, MISS, or UNCERTAIN.")


def reader_predict(model, text, tries: int = 3):
    """Tolerant single prediction: retries transient errors with backoff, never raises (errors ->
    'UNCERTAIN', which counts as 'did not call it' = non-leaky, the safe default)."""
    for i in range(tries):
        try:
            r = requests.post(f"{BASE_URL}/chat/completions",
                              headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                              json={"model": model, "temperature": 0,
                                    "messages": [{"role": "user", "content": text}]}, timeout=120)
            r.raise_for_status()
            txt = r.json()["choices"][0]["message"]["content"].upper()
            m = re.search(r"\b(CATCH|MISS|UNCERTAIN)\b", txt)
            return m.group(1) if m else "UNCERTAIN"
        except Exception as e:  # noqa: BLE001  (transient API/rate-limit/parse error)
            if i == tries - 1:
                print(f"  [warn] reader call failed after {tries} tries ({type(e).__name__}); -> UNCERTAIN")
                return "UNCERTAIN"
            time.sleep(2 ** i)  # 1s, 2s backoff (eases 429s)
    return "UNCERTAIN"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default="data/candidates_labeled.jsonl")
    ap.add_argument("--out", default="data/candidates_gated.jsonl")
    ap.add_argument("--workers", type=int, default=8, help="concurrent reader calls (lower if you hit 429s)")
    a = ap.parse_args()
    nonqwen = os.environ.get("GATE_MODEL")
    qwen = os.environ.get("GATE_QWEN_MODEL")
    if not (API_KEY and nonqwen):
        raise SystemExit("Set GATE_API_KEY and GATE_MODEL (strong non-Qwen reader); "
                         "optionally GATE_QWEN_MODEL and GATE_BASE_URL.")

    recs = [json.loads(l) for l in open(a.labeled)]
    clean = [r for r in recs if r["status"] in ("clean_miss", "clean_catch")]
    readers = [("nonqwen", nonqwen)] + ([("qwen", qwen)] if qwen else [])
    print(f"gating {len(clean)} clean items x {len(readers)} reader(s) x 2 views\n")

    views = ("conservative", "matched_A")
    stats = {f"{rn}/{v}": {"n": 0, "correct": 0} for rn, _ in readers for v in views}
    by_id = {}
    truth_of = {r["candidate_id"]: ("MISS" if r["status"] == "clean_miss" else "CATCH") for r in clean}

    # all (item, reader, view) calls are independent I/O -> run them concurrently
    jobs = [(r["candidate_id"], rn, v, model, _prompt(r, v))
            for r in clean for rn, model in readers for v in views]
    print(f"dispatching {len(jobs)} reader calls across {a.workers} workers...")
    preds = {}
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        fut2key = {ex.submit(reader_predict, model, prompt): (cid, rn, v)
                   for (cid, rn, v, model, prompt) in jobs}
        done = 0
        for fut in as_completed(fut2key):
            preds[fut2key[fut]] = fut.result()
            done += 1
            if done % 10 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} done")
    print()

    for r in clean:
        cid = r["candidate_id"]
        truth = truth_of[cid]
        rp = {f"{rn}/{v}": preds[(cid, rn, v)] for rn, _ in readers for v in views}
        for k, p in rp.items():
            stats[k]["n"] += 1
            stats[k]["correct"] += int(p == truth)
        strong = rp["nonqwen/conservative"]                 # binding per-item gate
        passes = strong != truth                            # reader did NOT confidently call it
        r["text_only_gate"] = {"truth": truth, "preds": rp, "passes_gate": passes,
                               "conservative_both": strong, "matched_A_only": rp["nonqwen/matched_A"]}
        by_id[cid] = passes
        print(f"{cid:16} truth={truth:5} {rp}  {'PASS(non-leaky)' if passes else 'LEAKS'}")

    print("\nreader accuracy vs measured label (chance ~0.50; near chance = non-leaky):")
    for k, s in stats.items():
        acc = s["correct"] / s["n"] if s["n"] else float("nan")
        print(f"  {k:24} {s['correct']}/{s['n']} = {acc:.2f}")
    n_pass = sum(by_id.values())
    print(f"\nper-item: {n_pass}/{len(clean)} pass the gate (strong non-Qwen reader can't call them).")
    print("Aggregate gate: the conservative non-Qwen reader should be NEAR CHANCE for the family to be usable.")

    # write through (clean items get the gate field; others pass through unchanged)
    with open(a.out, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    os.makedirs("data", exist_ok=True)
    json.dump({"stats": stats, "n_clean": len(clean), "n_pass": n_pass, "readers": [r[0] for r in readers]},
              open("data/text_only_gate.json", "w"), indent=2)
    print(f"wrote {a.out} and data/text_only_gate.json -> next: scripts/match_catch_miss.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
