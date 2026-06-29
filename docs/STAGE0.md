# Stage 0 — Verified wiring ground truth & reproduction gate

This file is the load-bearing reference for talking to the released Qwen3-8B **CoT
Activation Oracle**. Everything here was confirmed by reading the actual source repos
(not the papers, not the model card alone). Where the model card and the code disagree,
**the code wins** and it is flagged. Claim tags: `[VERIFIED]` (read in code),
`[FROM-CARD]` (documentation only, unverified or contradicted), `[OPEN]` (decision/unknown).

## 0. Environment constraint (why nothing model-touching runs in CI here)
The authoring sandbox can reach GitHub but **HuggingFace is egress-blocked (403)**. So the
tokenizer and checkpoints cannot be downloaded here; no model code has been executed in-
sandbox. `cao/ao_runtime.py` and `scripts/stage0_repro.py` are **statically authored** —
their **first run on a GPU box with HF access is the test**. Pure-logic (no model) is tested
by `tests/test_pure_logic.py` (passes locally).

## 1. The repos and our relationship to them
- `adamkarvonen/activation_oracles` — Paper 1 (Karvonen). Base AO infra (`nl_probes`), single-layer.
- `japhba/activation_oracles` — Paper 3 (Bauer) **fork** of the above; adds AObench, `text_sft`, etc.
- `ceselder/cot-oracle` — the **CoT Oracle** (Celeste De Schamphelaere). Produces the released
  checkpoints. Runtime wrapper = `src/core/ao.py`. Vendors the Bauer fork as a submodule
  (`ao_reference`); the Bauer fork vendors `ceselder/cot-oracle` back (`third_party/cot-oracle`) —
  a cross-submodule tangle; neither is checked out in release tarballs.

**We do NOT fork.** We develop in this repo and **lift the ~5 self-contained runtime functions**
into `cao/ao_runtime.py` (semantics verified against `ceselder/cot-oracle/src/core/ao.py`,
Apache-2.0 — see `NOTICE`). We reuse their **checkpoint** (from HF) and their baseline **patterns**
(`baselines/linear_probe.py`, `baselines/no_act_oracle.py`, AObench scoring) as references.

## 2. The released checkpoint
- Primary: **`ceselder/cot-oracle-v15-stochastic`** `[VERIFIED]` — the authors' eval default
  (`configs/eval.yaml: method_config.our_ao.checkpoint`); trained on **stochastic, end-weighted
  arbitrary CoT positions** (not sentence boundaries), `fresh_lora: true` (a fresh LoRA on base).
- Alternative: `ceselder/cot-oracle-v4-8b` `[FROM-CARD]` — documented in the model card; "continues
  from Adam's base AO"; sentence-boundary corpus framing. Many other ckpts exist (last-pos-only,
  1-layer, DPO/GRPO, final-sprint-no-DPO).
- Loading is **standalone**: `PeftModel.from_pretrained(base Qwen3-8B, <lora>)` — NOT stacked on
  Adam's AO adapter. `[VERIFIED]`
- **[OPEN]** pin the exact revision hash once chosen.

## 3. Verified wiring facts (the ground truth `cao/ao_runtime.py` encodes)
- **Base**: `Qwen/Qwen3-8B`, bf16. Tokenizer `padding_side="left"`. `[VERIFIED]`
- **Extraction**: collect residual-stream **output of block L** with the adapter **DISABLED**
  (`model.disable_adapter()`), single forward, early-stop after the deepest layer. `[VERIFIED]`
- **Injection**: forward hook on the **output of block 1**, adapter **ENABLED**, additive
  norm-matched, coefficient **1.0**, **no** forced target-norm rescale (natural ≈√2), **no**
  position-encoding, **no** pooling: `h' = h + ‖h‖·(v/‖v‖)`. `[VERIFIED]`
  (An `AO_FINAL_NORM_SCALE`/`target_norm_scale` and the 2× variant exist but are inference-time
  overrides, OFF for the released ckpt. PE/pooling exist only as default-OFF ablations.)
- **Oracle-prompt format** `[VERIFIED]` — **grouped, one shared `" ?"` placeholder per layer**:
  `"L9:? ? ?  L18:? ? ?  L27:? ? ?.\n" + <query>`, then `apply_chat_template(add_generation_prompt=True,
  enable_thinking=False)`. ⚠️ The model card's interleaved distinct tokens `" @"/" ?"/" #"` is
  **documentation that does NOT match the shipped checkpoint** — a default-OFF ablation. Do not use it.
- **Activation tensor ordering**: **LAYER-MAJOR** — `[all L9 positions, then all L18, then all L27]`,
  a single `[K·N, D]` tensor; the i-th `" ?"` (left-to-right) receives row i. `[VERIFIED]`
- **Placeholder location**: by **character offset** (robust to BPE seam merges), not token-id match.
  `[VERIFIED]` (matches `ao_checks.py §1`.)
- **Reading answers**: single forward, `logits[0,-1]`, `log_softmax`, then per answer string
  **logsumexp over its case/space token variants** → score; downstream use **margin → AUC**
  (Paper 3's "AUC not accuracy"). `[VERIFIED]` — this is `oracle_answer_logprobs`, the wiring
  target behind `ao_checks.ao_letter_logits`.
- **`" ?"` token id**: card says **937**, code asserts single-token. `[OPEN]` verify on GPU box
  (gate G1 in `stage0_repro.py`).

## 4. Corrections / updates to `project_spec.md` (from reading the code)
1. **Extraction layers `{9,18,27}` are in-distribution** — the oracle trained on the 3-layer
   interleave *and* L50-only. Drop the spec's "9/27 may be OOD vectors" worry. Single-layer `{18}`
   is also in-regime. `[VERIFIED]`
2. **Arbitrary positions are fine** — v15 trained on stochastic (end-weighted) CoT positions, not
   sentence boundaries. So the spec's "final 8 suffix positions" is not special; **widen to ~16–20,
   end-weighted** (Paper 3 saw big gains 1→50 tokens). The genuine OOD axis is **pre-generation**
   (training positions are *inside* the CoT; ours are *before* it) — this sharpens the timing concern
   into THE crux. `[VERIFIED]`
3. **"AOs match but never beat probes" is paper-text-only** — the repo's AO-vs-probe table is an
   unfilled TODO (`plot_baselines.py:68 oracle_scores = {}`; `data/baseline_comparison.txt` has `?`
   in every CoT-Oracle cell; no eval-cache db committed). The probe numbers that DO exist show hint
   tasks are **highly linearly decodable from CoT activations** (truthfulqa_hint 0.93, hint_admission
   0.82, sycophancy 0.82) — so our hint signal is likely linearly readable *mid-CoT*; the open
   question is whether it's there *pre-generation*. Reinforces deferring the probe + the timing crux. `[VERIFIED]`
4. **`activation_sensitivity` (Paper 3's flagship anti-text-inversion eval) scores answer CHANGE,
   not new-state correctness** — an LLM judge labels divergent_meaningful/divergent_noise/same;
   `ground_truth_missing_info` is plumbed in but never read by the scorer. Vindicates our
   correctness-not-sensitivity + shuffle-not-swap stance. `[VERIFIED]`
5. **No-activation baselines are inconsistent in-repo** — three constructions (eval-time *with*
   question, train-time *without* question, p3 full-transcript) and a train/eval question-inclusion
   mismatch. For our decisive control: define ONE format, keep train==eval, and run **both** our
   clean no-act (query only) AND a **text-context** baseline (feed the text the activations encode,
   per Paper 3 fn5; `baselines/no_act_oracle.py` shows the construction). `[VERIFIED]`

## 5. Locked evaluation-hierarchy changes (from review, to apply in Stage 2+)
- Primary disposition metric = balanced accuracy on the **12 hint-present pairs only**
  (follow-A / follow-B / resist-SAME); **resist-SAME recall** as co-primary diagnostic; neutral-null
  FPR reported **separately**; all-16 balanced accuracy descriptive only.
- Scale **resist** pairs, not follows, for any zero-shot positive power.
- **Reasoning-OFF is the clean feasibility read**; ON is the realistic condition layered on top.
- Geometry check (`ao_checks.py §3`) is **visualization, not a negative license**.
- Inspect sink/outlier dominance of `‖ΔH‖` **before** analysis; **do not pre-mask** the AO input
  (use masking only as a sensitivity analysis).
- **Delete** the natural-ΔH-norm-vs-steering-floor numeric inference (norm-matching makes it
  non-comparable); keep the steering sweep only as a plumbing/sensitivity check.
- **Joint** is the sole primary method; **delta-only** is directional-cells-only secondary.
- SFT (if reached) decisive control = paired-FT > {no-act-FT, text-context-FT} on held-out cue
  templates with shuffle collapse; enforce query-letter ⊥ label balance (trivial classifier at chance).

## 6. Reproduction gate — run on the GPU box
```bash
pip install -r requirements.txt   # + torch/flash-attn per your CUDA
python scripts/stage0_repro.py    # defaults: v15-stochastic, layers [9,18,27]
```
Gates (all must PASS, else **stop and fix wiring**): G1 `" ?"` single-token (print id);
G2 base+LoRA load; G3 grouped placeholders land on `" ?"`; G4 adapter-off ≠ adapter-on activations;
**G5 (key)** oracle scores move when activations are swapped (real vs zero vs different-text) — if
injection is a no-op, the study is dead.

## 7. File map
```
cao/ao_runtime.py   verified runtime (load/extract/inject/read); torch-guarded so pure helpers import anywhere
cao/ao_checks.py    user scaffold; ao_letter_logits wired to ao_runtime (grouped, layer-major)
scripts/stage0_repro.py  the Stage-0 wiring gate (GPU box)
tests/test_pure_logic.py no-model tests (prefix builder, parity/scoring math, dH geometry) — pass locally
docs/                project_spec.md, contrastive_C_pairs.md, AO_knowledge_base.md, STAGE0.md
```

## 8. Open decisions
- **[OPEN]** checkpoint: default `v15-stochastic` (recommended) vs `v4-8b`.
- **[OPEN]** reasoning ON vs OFF as the *primary* feasibility read (OFF recommended for clean
  label↔activation coupling; ON is less position-OOD for this CoT-trained ckpt — a real trade-off).
- **[OPEN]** verify `" ?"`→ token id 937 (single token) on the GPU box.
