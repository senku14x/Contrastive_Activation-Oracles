# Contrastive Activation Oracle — project memory & status

Orientation doc for this repo. Read this first. The full plan is in `docs/project_spec_v4.md`
(plan) and `docs/data_construction_v2.md` (dataset recipe); how to run everything is in
`docs/COLAB.md`. This file is the **living status** + a condensed map.

---

## TL;DR (where we are right now — 2026-06-29)

We built the dataset + the whole Stage-0→1.5 pipeline, ran it on an A100, and hit the **expected
binding constraint, not a bug**: the scientific class we need (Qwen *catching* a planted fallacy) is
**rare (~18%)**, so the existence probe is **underpowered (4 catches), hence inconclusive — NOT a
negative.** Everything that should be green is green (wiring, invariants, non-leakiness). The open
decision is purely **how to get more "catch" examples** (or whether to stop and write up).

**Nothing is broken. The pipeline is fully reproducible from committed code.**

---

## The question (registered, existence-first)

> On problems at Qwen3-8B's competence boundary, where a **load-bearing flawed step** changes the
> answer, is "did the model **catch or miss** the flaw" readable from the **pre-output activations** —
> above a text-only baseline, up to a linear probe — and can an AO **verbalize** it?

Split of labor (so it can't collapse into a known lost race):
- **probe carries detection** (is the signal there?), **AO carries verbalization** (the contribution),
  **SFT improves verbalization, not detection.**
- A **clean negative is a complete result** (and the spec's expected modal outcome).

The data structure that makes this well-posed (worked out empirically, see `cao/family_l.py`):
an **invalid inference on a genuinely under-determined fictional problem** — gold = "cannot be
determined", **MISS** = following the fallacy to a definite wrong answer, **CATCH** = holding the line.
This is the only shape that is *both* load-bearing (ablation → under-determined, can't be sidestepped)
*and* catchable (recognizing an unjustified leap, not recomputing).

---

## Status by stage

### Stage 0 — AO wiring — ✅ DONE (reproduced on A100)
Released oracle `ceselder/cot-oracle-v15-stochastic` reverse-engineered + reimplemented
(`cao/ao_runtime.py`). Verified: placeholder `" ?"` = token **937**; extraction layers **{9,18,27}**;
injection at layer 1, additive norm-matched, coeff **1.0**; grouped layer-major format; **G1–G5 pass**
(`scripts/stage0_repro.py`). Bauer's 2×/62% are a *different checkpoint* — out of scope.

### Stage 1 — build data + behavioral screen — ✅ DONE (first bank)
- **Family L** (primary, claim-carrier): **32 items**, 10 fallacy types, all ablation-verified,
  80/80 token-invariants pass. OFF no-cue screen result:
  - **4 clean CATCH, 18 clean MISS, 10 discard** (9 `competence_fail` + 1 `middle`). 22 valid.
  - ⇒ **Qwen catches the fallacy ~18% of valid items; follows it ~82%** (misses with `p≈1.0`).
    This matches the literature (models follow flawed "therefore" steps). **CATCH is the scarce,
    binding class.**
  - The OFF readout is **deterministic** (reproduces bit-for-bit across runs).
- **Family P** (validation shakedown, NOT a claim-carrier): 8 MCQs × 6 commitment templates = 48.
  Screen: **34 CATCH, 0 MISS, 14 discard** → Qwen resists commitment cues entirely (re-confirms why
  peer-hint was abandoned). Its job was to prove the pipeline runs end-to-end; it did.

### Stage 1.5 — existence probe — ⚠️ RAN, UNDERPOWERED → INCONCLUSIVE
On the 22 valid L items (4 catch / 18 miss):
| readout | W8 AUC | W20 AUC |
|---|---|---|
| probe(ΔH) | 0.43 | 0.32 |
| probe(H_B alone) | 0.17 | 0.23 |
| text-feature probe (baseline to beat) | 0.32 | 0.32 |
| shuffled-label ΔH (sanity, want ~0.5) | 0.55 | 0.48 |

**All within noise of chance.** With 4 catches the AUC CI is ≈ ±0.25 — a single misclassified item
moves AUC ~0.2. So this is **"not enough data," not "no signal."** The script now power-gates the
verdict (`n<30` or minority class `<8` → INCONCLUSIVE, never "negative").

Also flagged: the 4 catches are **systematically longer prompts** (431 vs 408 chars,
`|Δ|/SD=1.33`) → a confound to length-match away once we have more catches.

### Stage 2 (AO zero-shot), Stage 3 (SFT) — NOT STARTED (gated behind a positive Stage 1.5).

---

## THE OPEN DECISION (this is what we're on)

We need ~**15 catches** for a defensible probe (we have 4). At the ~20% catch rate that's ~**120 more
items**.

- **B — reasoning-ON shortcut — TESTED, RULED OUT (2026-06-30).** ON catch rate = **7/32 = 22%** ≈ the
  OFF 18%. Plain CoT does *not* balance the classes (consistent with PCBench: only an explicit
  "check the premises" instruction lifts critique, and we don't add one — it collapses the gap). So the
  ~20% catch rate is **robust across OFF/ON**; there is no cheap shortcut to power.

Remaining paths:
- **A — Scale (completes the registered experiment).** Bulk-author ~100–120 Family-L items (volume, not
  cleverness — catch is Qwen's disposition at ~20% regardless of author) → harvest ~15 catches → the
  real probe. Authoring is the **generate-and-verify workflow's** job (sandbox; workflow infra failed
  twice, would retry). A *powered* result — even a null — is the spec's intended deliverable.
- **C — Bank it.** Write up the real behavioral finding (*Qwen follows fallacious "therefore" bridges
  ~80% OFF and ON, and that catch/miss split is non-leaky*) + the underpowered probe + build-yield as a
  documented supply constraint. Partial (doesn't answer the existence question) but honest and complete.

**Now: A (complete the experiment; authoring cost is the workflow's, not the user's) vs C (stop here).**

---

## Key facts to not re-learn the hard way

- **"Good prompts" ≠ "balanced."** Three independent axes: construction validity (✅ 80/80 invariants),
  non-leakiness (✅ GLM can't predict catch/miss), behavioral yield (⚠️ 18% catch). Only the third is
  the problem, and it's a property of **Qwen**, not the prompts. No prompt edit raises the catch rate.
- **The gate metric is balanced-accuracy, not raw match.** A constant/biased reader (GLM defaulted to
  "MISS" on all L) scores high raw accuracy on the majority class but ~0.5 balanced → that IS the
  non-leaky result. Earlier "LEAKS" labels were a metric bug (fixed).
- **`competence_fail` ≠ bad construction in the structural sense.** It means Qwen commits to a definite
  answer on the bare givens (doesn't see the item as under-determined). Reworking the *flaw* doesn't
  fix it; the *givens* must read as open to Qwen. ~28% of items fail this.
- **The probe at tiny minority-class is uninterpretable.** Don't read AUC until ≥ ~8–10 catches.
- **OFF readout is deterministic**; ON needs sampling (K≥16). Don't finalize labels from K=8.
- **Generated data files are gitignored** (`data/candidates_*.jsonl`, `*.npz`, etc.). Source of truth =
  `cao/family_l.py` / `cao/family_p.py`. Re-run `build_candidates.py` to regenerate.

---

## Repo layout

```
cao/
  ao_runtime.py        load oracle; extract activations (adapter off); inject; oracle scoring (Stage 0)
  ao_checks.py         tokenizer suffix-asserter; parity/geometry helpers
  measure.py           target behavior: forced-choice logits; answer_distribution_nocue (§10 OFF readout)
  dataset.py           shared suffix/record schema; option-position counterbalancing
  family_l.py          PRIMARY bank: 32 invalid-inference items (gold = "cannot be determined")
  family_p.py          validation shakedown: 8 MCQ × 6 weak/strong commitment templates
  probe.py             Stage-1.5 probe math (PCA + honest LOO AUC; text-feature baseline)
scripts/  (pipeline order)
  build_candidates.py        -> data/candidates_unfiltered.jsonl (§20 schema)
  pad_to_equal_length.py     neutral-prefix pad so c_A/c_B token lengths match (tokenizer)
  ablation_verify.py         structural load-bearing flag (no model)
  verify_token_invariants.py within-pair length/suffix identity (tokenizer)
  measure_target_logits.py   OFF no-cue measurement (+ --on cross-check, --family filter)
  label_candidates.py        Stage A–D -> clean_catch / clean_miss / discard
  run_text_only_gate.py      dual text gate; balanced-acc discrimination; parallel + retry
  match_catch_miss.py        confound check + freeze feasibility set (--require-gate)
  extract_activations.py     H_A/H_B/ΔH over W8 + W20, layers {9,18,27}, adapter off
  stage1_5_probe.py          THE existence gate: probe(ΔH/H_B) vs text-feature + shuffle
  stage0_repro.py            Stage-0 wiring gate (G1–G5)
tests/   test_pure_logic.py, test_dataset_logic.py  (no model; run anywhere)
docs/    project_spec_v4.md, data_construction_v2.md, COLAB.md (runbook), STAGE0.md
```

Run order (from `docs/COLAB.md`): build → pad → ablation_verify → verify_token_invariants →
measure_target_logits → label_candidates → run_text_only_gate → match_catch_miss →
extract_activations (W8 + W20) → stage1_5_probe.

---

## Condensed spec (full versions in `docs/`)

**Primary family L** = correct-step vs subtly-flawed-step on a novel fictional under-determined problem;
gold = "cannot be determined"; label CATCH (flip 0) / MISS (flip 1), measured by OFF forced-choice
logits with the answer cue *inside* the suffix (no post-hoc cue). **Load-bearing + ablation-verified**
(delete the step → under-determined, else sidesteppable → discard). **The squeeze** is the registered
binding constraint: subtle enough that a text reader can't call catch/miss, salient enough to move the
forward pass. **Power bar: 30–40 balanced + a large probe-vs-text gap, shuffle-surviving.** Family P
(commitment) is validation-only (difficulty-leaky, not a claim).

**Decision rule:** probe(ΔH or H_B) beats text-only by a large margin on 30–40 balanced items, shuffle
kills it → signal exists. probe ≈ text → clean negative. < 30 / small gap → inconclusive (where we are).

---

## Conventions

- **Branch:** `claude/mechanistic-interpretability-qditqg`. Commit + push there; PR #1 is the draft.
- **Commit trailer:** `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` + the `Claude-Session:` line.
- Do **not** put the model identifier in committed artifacts.
- Push with `-u origin <branch>`, retry with backoff on network errors.
- All model runs happen on the user's Colab A100 (HuggingFace is egress-blocked in the dev sandbox).
- Gate reader: GLM-4.6 via OpenRouter (`GATE_API_KEY`, `GATE_MODEL`), a strong non-Qwen reader.
