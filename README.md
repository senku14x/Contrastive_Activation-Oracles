# Contrastive_Activation-Oracles

This project explores whether an Activation Oracle can compare two matched activation traces from the same language model under different prompt conditions and describe the resulting change in observable behavior. For each pair, the target model is run on two contexts with an identical shared suffix, producing aligned activations (H_A) and (H_B), along with their difference (H_B-H_A). The work begins by testing whether a released AO can use these paired traces zero-shot, then fine-tunes it on contrastive examples where the target is a measured directional behavior change, such as selecting a different answer, following a hint, asking for clarification, or showing no stable change. The resulting contrastive AO is evaluated with condition swaps, shuffled-trace controls, null pairs, text-only baselines, and separate single-trace AO baselines to test whether its answers depend on the internal contrast rather than merely reconstructing the prompt or predicting the final output.

## Status: Stage 0 (wiring foundation)

The released **CoT Activation Oracle** runtime has been reverse-engineered from source and
reimplemented minimally; the experiment itself is **not** built yet (we gate on Stage 0 first).
The full verified wiring facts, corrections to the project spec, and the reproduction gate are in
**[`docs/STAGE0.md`](docs/STAGE0.md)** — read that first.

Key facts the code encodes: load `Qwen/Qwen3-8B` + a CoT-Oracle LoRA (default
`ceselder/cot-oracle-v15-stochastic`) standalone; extract residual activations with the adapter
**disabled**; inject at layer 1 via additive norm-matching `h' = h + ‖h‖·(v/‖v‖)`; oracle prompt uses
a **grouped, single-`" ?"`-per-layer** format with **layer-major** activation ordering (the model
card's interleaved-token scheme is documentation that does not match the shipped checkpoint); read
answers as logsumexp-of-logits → margin → AUC.

## Layout
```
cao/ao_runtime.py        verified runtime: load / extract / inject / read-logprobs
cao/ao_checks.py         pre-flight + decisive checks; ao_letter_logits wired to the runtime
scripts/stage0_repro.py  Stage-0 reproduction gate (run on a GPU box)
tests/test_pure_logic.py no-model tests (prefix builder, parity/scoring math, dH geometry)
docs/                    project_spec.md, contrastive_C_pairs.md, AO_knowledge_base.md, STAGE0.md
```

## Running
```bash
# No-model logic (runs anywhere):
python tests/test_pure_logic.py

# Stage-0 wiring gate — requires a GPU box WITH HuggingFace access:
pip install -r requirements.txt        # + torch/flash-attn for your CUDA
python scripts/stage0_repro.py
```
Note: the authoring/CI sandbox has **no HuggingFace egress**, so anything that loads the
tokenizer or checkpoint must run on the GPU box. First execution there is the real test of the
runtime (assertions are deliberately load-bearing). Attribution for the lifted semantics: see
[`NOTICE`](NOTICE).

Work in Progress!

***This might fail before it takes off KEKW***
