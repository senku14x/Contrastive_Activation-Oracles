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


TIMEOUT = 45   # per-call seconds; set from --timeout in main(). Short so stragglers fail-fast.
MAXTOK = 24    # cap reader output; the answer is ONE word, so this is the big latency win.


def reader_predict(model, text, tries: int = 2):
    """Tolerant single prediction: short timeout + a couple retries, never raises (errors ->
    'UNCERTAIN', which counts as 'did not call it' = non-leaky, the safe default). The short timeout
    keeps the tail from crawling on 120s hangs — a slow/rate-limited call just resolves to UNCERTAIN."""
    for i in range(tries):
        try:
            r = requests.post(f"{BASE_URL}/chat/completions",
                              headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                              json={"model": model, "temperature": 0, "max_tokens": MAXTOK,
                                    "reasoning": {"enabled": False},  # hybrid readers (DeepSeek V3.1+,
                                    # some Gemini/GLM modes) default to thinking-on via OpenRouter unless
                                    # told otherwise -> would burn MAXTOK on CoT, never emit the answer word.
                                    "messages": [{"role": "user", "content": text}]}, timeout=TIMEOUT)
            r.raise_for_status()
            txt = r.json()["choices"][0]["message"]["content"].upper()
            m = re.search(r"\b(CATCH|MISS|UNCERTAIN)\b", txt)
            return m.group(1) if m else "UNCERTAIN"
        except Exception as e:  # noqa: BLE001  (transient API/rate-limit/parse error)
            if i == tries - 1:
                print(f"  [warn] reader call failed after {tries} tries ({type(e).__name__}); -> UNCERTAIN")
                return "UNCERTAIN"
            time.sleep(1.5)
    return "UNCERTAIN"


def balanced(pred_truth):
    """Balanced accuracy = mean per-truth-class recall. A constant/biased reader (e.g. always 'MISS')
    scores high RAW accuracy on the majority class but ~0.5 here — which is the point: leakage means the
    reader can DISCRIMINATE catch from miss, not just guess the base rate."""
    classes = sorted({t for _, t in pred_truth})
    recalls = {}
    for c in classes:
        items = [p for p, t in pred_truth if t == c]
        recalls[c] = (sum(p == c for p in items) / len(items)) if items else float("nan")
    vals = [v for v in recalls.values() if v == v]  # drop NaN
    return (sum(vals) / len(vals) if vals else float("nan")), recalls


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default="data/candidates_labeled.jsonl")
    ap.add_argument("--out", default="data/candidates_gated.jsonl")
    ap.add_argument("--workers", type=int, default=8, help="concurrent reader calls (lower if you hit 429s)")
    ap.add_argument("--disc-thr", dest="disc_thr", type=float, default=0.65,
                    help="balanced-acc above which the reader is judged to DISCRIMINATE (family leaks)")
    ap.add_argument("--timeout", type=int, default=45, help="per reader-call seconds (stragglers fail-fast)")
    ap.add_argument("--max-tokens", dest="max_tokens", type=int, default=24,
                    help="cap reader output tokens (answer is one word; small = much faster)")
    a = ap.parse_args()
    global TIMEOUT, MAXTOK
    TIMEOUT, MAXTOK = a.timeout, a.max_tokens
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

    # DISCRIMINATION per (family, reader, view): balanced accuracy, NOT raw match. The binding signal
    # is the non-Qwen conservative reader per family.
    fams = sorted({r["family"] for r in clean})
    summary, family_disc = {}, {}
    print("\n=== discrimination (balanced acc; ~0.5 = reader CANNOT tell catch from miss = NON-LEAKY) ===")
    for fam in fams:
        fitems = [r for r in clean if r["family"] == fam]
        n_miss = sum(truth_of[r["candidate_id"]] == "MISS" for r in fitems)
        print(f"  Family {fam}  (n={len(fitems)}: MISS={n_miss}, CATCH={len(fitems)-n_miss})")
        for rn, _ in readers:
            for v in views:
                pt = [(preds[(r["candidate_id"], rn, v)], truth_of[r["candidate_id"]]) for r in fitems]
                bal, rec = balanced(pt)
                summary[f"{fam}/{rn}/{v}"] = {"balanced_acc": round(bal, 3),
                                              "recalls": {k: round(x, 3) for k, x in rec.items()}}
                if rn == "nonqwen" and v == "conservative":
                    family_disc[fam] = bal
                print(f"     {rn}/{v:12} balanced_acc={bal:.2f}  recalls={ {k: round(x,2) for k,x in rec.items()} }")

    # per-item gate: an item leaks ONLY if the reader DISCRIMINATES on its family AND calls it correctly.
    # If the family reader is at chance (e.g. constant 'MISS'), nothing leaks -> keep all clean items.
    print("\n=== per-item gate ===")
    for r in clean:
        cid, fam, truth = r["candidate_id"], r["family"], truth_of[r["candidate_id"]]
        rp = {f"{rn}/{v}": preds[(cid, rn, v)] for rn, _ in readers for v in views}
        strong = rp["nonqwen/conservative"]
        discriminates = family_disc.get(fam, 0.0) > a.disc_thr
        passes = (not discriminates) or (strong != truth)
        r["text_only_gate"] = {"truth": truth, "preds": rp, "passes_gate": passes,
                               "family_balanced_acc": round(family_disc.get(fam, float("nan")), 3),
                               "conservative_both": strong, "matched_A_only": rp["nonqwen/matched_A"]}
        by_id[cid] = passes
        print(f"{cid:16} {fam} truth={truth:5} cons={strong:9} {'PASS' if passes else 'LEAKS'}")

    print("\n=== verdict (non-Qwen conservative reader) ===")
    for fam in fams:
        bal = family_disc.get(fam, float("nan"))
        nclass = len({truth_of[r["candidate_id"]] for r in clean if r["family"] == fam})
        if nclass < 2:
            verdict = "N/A (only one behavioral class -> discrimination undefined, not 'non-leaky')"
        else:
            verdict = "NON-LEAKY (reader at chance)" if bal <= a.disc_thr else "LEAKY (reader discriminates)"
        print(f"  Family {fam}: balanced_acc={bal:.2f} ({nclass} class) -> {verdict}  "
              f"({sum(1 for r in clean if r['family']==fam and by_id[r['candidate_id']])}/"
              f"{sum(1 for r in clean if r['family']==fam)} items kept)")
    print("NOTE: at this n the verdict is preliminary; confirm on the scaled set (and with a 2nd reader).")

    with open(a.out, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    os.makedirs("data", exist_ok=True)
    json.dump({"summary": summary, "family_balanced_acc": {k: round(v, 3) for k, v in family_disc.items()},
               "n_clean": len(clean), "n_pass": sum(by_id.values()), "disc_thr": a.disc_thr,
               "readers": [r[0] for r in readers]}, open("data/text_only_gate.json", "w"), indent=2)
    print(f"\nwrote {a.out} and data/text_only_gate.json -> next: scripts/match_catch_miss.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
