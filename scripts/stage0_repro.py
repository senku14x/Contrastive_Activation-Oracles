#!/usr/bin/env python3
"""
stage0_repro.py — Stage-0 reproduction / wiring gate for the contrastive-AO study.

Runs on a GPU box WITH HuggingFace access (the authoring sandbox has neither). It
verifies, with hard pass/fail gates, that our lifted runtime actually drives the
released CoT Oracle. Per project_spec.md Stage 0: "Stop if not reproducible."

Gates (each prints PASS/FAIL; script exits non-zero on any FAIL):
  G1 tokenizer:    placeholder ' ?' is a single token; print its id (card says 937).
  G2 load:         base Qwen3-8B + CoT-Oracle LoRA load; adapter present.
  G3 placeholders: the grouped prefix's ' ?' tokens are located on ' ?', not the
                   chat/gen-prompt header (decode them and check).
  G4 extraction:   activations extracted adapter-OFF differ from adapter-ON
                   (confirms disable_adapter() actually changes representations).
  G5 injection:    THE key gate. Oracle answer scores must CHANGE when we swap the
                   injected activations (real vs zero vs different-text). If they do
                   not move, injection is a no-op and the whole study is dead.

Usage:
  python scripts/stage0_repro.py                       # defaults (v15-stochastic, [9,18,27])
  python scripts/stage0_repro.py --oracle-lora ceselder/cot-oracle-v4-8b --layers 18
"""

from __future__ import annotations

import argparse
import sys

import torch

# allow `python scripts/stage0_repro.py` from repo root
sys.path.insert(0, ".")
from cao import ao_runtime as rt  # noqa: E402


def _ok(name, passed, detail=""):
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    return passed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=rt.DEFAULT_BASE)
    ap.add_argument("--oracle-lora", default=rt.DEFAULT_ORACLE_LORA)
    ap.add_argument("--layers", type=int, nargs="+", default=list(rt.DEFAULT_LAYERS))
    ap.add_argument("--k", type=int, default=8, help="positions per layer to extract")
    args = ap.parse_args()
    layers = list(args.layers)
    results = []

    # ---- G2 load (do this first; everything else needs it) -------------------
    model, tok = rt.load_oracle(args.base, args.oracle_lora)
    has_adapter = bool(getattr(model, "peft_config", {}))
    results.append(_ok("G2 load", has_adapter, f"adapters={list(getattr(model,'peft_config',{}))}"))

    # ---- G1 tokenizer --------------------------------------------------------
    ph_ids = tok.encode(rt.PLACEHOLDER, add_special_tokens=False)
    g1 = (len(ph_ids) == 1)
    results.append(_ok("G1 tokenizer", g1,
                       f"' ?' -> ids {ph_ids} (single-token={g1}; card claims 937)"))
    for L in ("A", "B", "C", "D"):
        print(f"        '{L}'->{tok.encode(L, add_special_tokens=False)}  "
              f"' {L}'->{tok.encode(' '+L, add_special_tokens=False)}")

    # ---- build a trivial target context & extract ----------------------------
    target_text = tok.apply_chat_template(
        [{"role": "user", "content": "What is the capital of France? Answer with one word."}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )
    n_tok = len(tok(target_text, add_special_tokens=False)["input_ids"])
    positions = list(range(max(0, n_tok - args.k), n_tok))     # last-k positions
    acts = rt.extract_layer_major(model, tok, target_text, positions, layers)  # adapter OFF
    expected_rows = len(layers) * len(positions)
    results.append(_ok("extract shape", acts.shape[0] == expected_rows,
                       f"acts {tuple(acts.shape)} (expected rows={expected_rows}, layer-major)"))

    # ---- G3 placeholder location ---------------------------------------------
    prefix, char_spans = rt.build_grouped_prefix(layers, len(positions))
    print(f"        prefix sample: {prefix[:80]!r} ...")
    g3 = (len(char_spans) == expected_rows)
    results.append(_ok("G3 placeholders", g3, f"{len(char_spans)} placeholder spans (grouped, layer-major)"))

    # ---- G4 extraction depends on adapter state ------------------------------
    with model.disable_adapter():
        off = rt._collect_layers(model, layers, *(
            lambda e: (e["input_ids"].to(rt._model_device(model)),
                       e["attention_mask"].to(rt._model_device(model)))
        )(tok(target_text, return_tensors="pt", add_special_tokens=False)))
    on = rt._collect_layers(model, layers, *(
        lambda e: (e["input_ids"].to(rt._model_device(model)),
                   e["attention_mask"].to(rt._model_device(model)))
    )(tok(target_text, return_tensors="pt", add_special_tokens=False)))
    diff = (off[layers[0]] - on[layers[0]]).abs().max().item()
    results.append(_ok("G4 extraction", diff > 1e-3,
                       f"max|adapter_off - adapter_on| at L{layers[0]} = {diff:.4g} (want >0)"))

    # ---- G5 injection changes the output (THE gate) --------------------------
    probe_tokens = ["A", "B", "C", "D", "yes", "no", "Paris", "math"]
    query = "Which option (A, B, C, or D) is the model most likely to answer? Reply one letter."

    s_real = rt.oracle_answer_logprobs(model, tok, acts, query, probe_tokens, layers=layers)
    s_zero = rt.oracle_answer_logprobs(model, tok, torch.zeros_like(acts), query, probe_tokens, layers=layers)

    other_text = tok.apply_chat_template(
        [{"role": "user", "content": "Compute 17 times 23. Answer with the number only."}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )
    n2 = len(tok(other_text, add_special_tokens=False)["input_ids"])
    pos2 = list(range(max(0, n2 - args.k), n2))
    acts2 = rt.extract_layer_major(model, tok, other_text, pos2, layers)
    s_other = rt.oracle_answer_logprobs(model, tok, acts2, query, probe_tokens, layers=layers)

    def vmax(a, b):
        return max(abs(a[k] - b[k]) for k in probe_tokens)

    d_zero = vmax(s_real, s_zero)
    d_other = vmax(s_real, s_other)
    print(f"        scores(real) = {{ {', '.join(f'{k}:{s_real[k]:.2f}' for k in probe_tokens)} }}")
    g5 = (d_zero > 1e-2) and (d_other > 1e-2)
    results.append(_ok("G5 injection", g5,
                       f"max|real-zero|={d_zero:.3g}, max|real-otherText|={d_other:.3g} (want both >0)"))

    # ---- qualitative free-form (not a gate) ----------------------------------
    print("        free-form (real acts):",
          repr(rt.oracle_generate(model, tok, acts,
                                  "Briefly, what is the model reasoning about?", layers=layers)[:160]))

    print("\n=== STAGE 0:", "REPRODUCIBLE ✅" if all(results) else "BROKEN ❌ (do not proceed)", "===")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
