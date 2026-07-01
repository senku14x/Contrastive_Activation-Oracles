# Contrastive Activation Oracle — project memory & status

Orientation doc for this repo. Read this first. The full plan is in `docs/project_spec_v4.md`
(plan) and `docs/data_construction_v2.md` (dataset recipe); how to run everything is in
`docs/COLAB.md`. This file is the **living status** + a condensed map.

---

## TL;DR (where we are right now — 2026-07-01)

**→ The full, nothing-hidden status is in `docs/STATUS.md`. Read that. This is the condensed version.**

We scaled Family L to **167 items**, cleared the power floor (**15 catches, 30 balanced**), and got an
**interpretable — and confounded — probe result**: at the primary window W8 `probe(ΔH)=0.804` beats the
text baseline `0.686`, but this **does not survive scrutiny**: (a) the pre-registered sensitivity window
**W20 reverses it** (probe 0.653 < text 0.684); (b) `probe(H_B alone)` is at **chance** in every powered
run; (c) within a single subtype, **text beats the probe**; (d) the text baseline **rose with the
"signal"** (0.32→0.69) as we scaled. Net: the probe is reading the **text/subtype contrast between the
two conditions, not Qwen's disposition.** The registered "readable above text, robustly" claim is
**not supported** on this design.

**This is NOT "we failed" and NOT "underpowered."** It is a *confounded/negative read on Family L
specifically*, and several load-bearing assumptions were **never tested** (ON-trajectory activations,
layer/position sweep, the steering detection-floor control, the AO itself). Those — not more data — are
where the answer, if any, still lives. See `docs/STATUS.md` §6–§7.

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

### Stage 1 — build data + behavioral screen — ✅ DONE (scaled to 167)
- **Family L** (primary, claim-carrier): **167 items**, 10 fallacy types, all ablation-verified, token
  invariants pass. Scaled 32 → 81 (generate-and-verify workflow) → 105 → 137 → 167 (hand-authored).
  OFF no-cue screen on the 167-item bank: **~128 clean (113 MISS / 15 CATCH), ~39 discard.**
  - ⇒ **Qwen catches ~15–18% of valid items; follows the fallacy ~82–85%** (misses with `p≈1.0`).
    Robust across OFF (18%) and ON (22%). **CATCH is the scarce class.**
  - **Catch is NOT uniform — it concentrates by subtype:** affirming_consequent ~43%, composition ~21%,
    division ~8%, the other 7 subtypes **~0%** (see `scripts/subtype_yield.py`). Subtype is a text
    property, so this is the root of the text leak (Stage 1.5).
  - The OFF readout is **deterministic** (reproduces bit-for-bit across runs).
- **Family P** (validation shakedown, NOT a claim-carrier): 8 MCQs × 6 templates = 48. Qwen resists
  commitment cues entirely (34 CATCH / 0 MISS) — re-confirms why peer-hint was abandoned. Pipeline
  shakedown only; it did its job.

### Stage 1.5 — existence probe — ⚠️ POWERED, CONFOUNDED → registered claim NOT supported
Final powered run (167 items → 128 valid, 15 catch / 113 miss, **30 balanced = power floor cleared**):
| readout | **W8 (primary)** | **W20 (pre-registered sensitivity)** |
|---|---|---|
| probe(ΔH) | **0.804** | 0.653 |
| probe(H_B alone) | 0.539 (chance) | 0.443 (chance) |
| text-feature probe (baseline to beat) | 0.686 | 0.684 |
| shuffled-label ΔH (want ~0.5) | 0.454 | 0.471 |

**The W8 "SIGNAL" does not survive scrutiny** (full detail + the run-by-run table in `docs/STATUS.md`
§3–§5): W20 reverses it (probe < text); `H_B alone` is at chance everywhere; the text baseline **rose
with the probe** as we scaled (0.32→0.38→0.62→0.69); and within a single subtype **text beats the
probe** (AC-only: probe 0.652 < text 0.702). Signature = the probe reads the **text/subtype contrast**
in `ΔH = H_flawed − H_correct`, **not** Qwen's disposition. (Because H_A/H_B differ in *wording*, ΔH
conflates content-difference with disposition — Paper 2's splice confound, in reverse.)

### Stage 2 (AO zero-shot), Stage 3 (SFT) — NOT STARTED. So is the steering detection-floor control,
the ON-trajectory (mid-CoT) probe, and the layer/position sweep — the assumptions that were never tested.

---

## THE OPEN DECISION (this is what we're on)

**We are no longer power-limited — we are confound-limited and design-limited.** More items will NOT
manufacture an above-text signal. Path "A — scale" is **off the table** as a way to flip the verdict.
Full menu in `docs/STATUS.md` §7; the live ones:

- **B (cheap, high-info) — test "signal forms during CoT."** Only pre-output/OFF activations were ever
  extracted. Extract ON-trajectory (mid/post-CoT) activations and probe those. `H_B`-at-chance is
  *consistent* with the disposition forming during reasoning, not before it. Either finds the signal
  (→ redirect the AO to read mid-CoT) or cleanly confirms it's absent pre-output.
- **D (cheap) — steering detection-floor control.** Never run. Tells us if natural ‖ΔH‖ is even above
  the interface's resolution floor — the assumption under everything.
- **C (cheap, exploratory) — layer/position sweep.** {9,18,27} are the AO's *injection* layers; a probe
  can read anywhere. Paper 3 says reading is best ~62% depth (layer 22). Re-extract all layers, probe.
- **A (real fix, most work) — redesign for text-orthogonality.** Same-surface-text, different-upstream-
  state contrast so ΔH isolates disposition. The only path that fixes the structural leak.
- **E — write up** the behavioral finding + non-leaky dataset + the honest negative. Complete, but the
  user's goal (ΔH signal → AO verbalizing shifts) is not yet achieved, and B/C/D are untested.

**User's stated goal is still to get the ΔH signal and then train an AO on the shifts — so the live
work is B/C/D (test the untested assumptions), not E.**

---

## Key facts to not re-learn the hard way

- **Read the probe columns, not the script's canned verdict.** `stage1_5_probe.py`'s "SIGNAL" gate
  (`ΔH>0.70 & ΔH−text>0.10 & shuffle flat`) fired at W8 on a **confounded positive**. It does NOT check
  `H_B`-alone, subtype structure, or W8/W20 agreement — all three of which say the signal is text-borne.
- **`probe(H_B alone)` at chance is the sharpest clue.** Catch/miss is decided in the flawed condition,
  so if it were in the pre-output state, `H_B` should carry it. It doesn't (0.44–0.54 everywhere). The
  only thing that scores is ΔH, which is dominated by the correct-vs-flawed **text difference**.
- **The text baseline rose WITH the "signal"** (0.32→0.38→0.62→0.69) as we scaled/concentrated. Catch
  concentrates in 2–3 subtypes; subtype is a text tell; a probe can fingerprint it. Within one subtype,
  **text beats the probe.** This is the confound, and it's **structural to Family L** (H_A/H_B differ in
  wording) — not fixable by more data.
- **W8/W20 disagreement is the pre-registration doing its job.** W20 is the *declared* sensitivity check
  precisely so W8 can't be cherry-picked. A signal that reverses across them is not robust.
- **Padding that changes what the model reads changes the behavior.** A universal neutral prefix
  (attempted leak-fix) collapsed catches 15→2 on the same bank; reverted (`d91b413`). Mock-tokenizer
  code-correctness ≠ behavioral safety — any change to model input needs a GPU behavioral re-check.
- **"Good prompts" ≠ "balanced" ≠ "non-confounded."** Four independent axes: construction validity
  (✅ invariants), non-leaky-to-a-zero-shot-reader (✅ GLM/DeepSeek balanced-acc ~0.40), behavioral yield
  (⚠️ ~15% catch, subtype-concentrated), and **non-leaky-to-a-supervised-text-probe (❌ — the new
  failure).** The gate reader test passing does NOT imply the supervised-text-probe baseline is beaten.
- **The gate metric is balanced-accuracy, not raw match** (a constant predictor → 0.5 → non-leaky).
- **`competence_fail`** = Qwen commits to a definite answer on the bare givens; ~25–30% of items. The
  *givens* must read as open, not the flaw.
- **We optimized the kill-switch, not the experiment.** The probe was only ever the existence gate; the
  AO comparison + steering control (the actual project) were skipped. Don't repeat that.
- **OFF readout is deterministic**; ON needs sampling (K≥16). **We only ever extracted OFF/pre-output
  activations** — the "forms during CoT" hypothesis is untested.
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
  match_catch_miss.py        confound check + freeze feasibility set (--require-gate, --subtype)
  extract_activations.py     H_A/H_B/ΔH over W8 + W20, layers {9,18,27}, adapter off
  stage1_5_probe.py          THE existence gate: probe(ΔH/H_B) vs text-feature + shuffle (progress-printed)
  stage0_repro.py            Stage-0 wiring gate (G1–G5)
  subtype_yield.py           DIAGNOSTIC: catch/miss/discard by flaw_subtype + authoring batch
  inspect_text_leak.py       DIAGNOSTIC: top n-grams the text-feature probe keys on
  diagnose_l78_outlier.py    DIAGNOSTIC: leave-one-out AUC sensitivity to the W20 outlier item
  run_all.sh                 one-shot clone-to-probe pipeline (HF login + parallel gate)
tests/   test_pure_logic.py, test_dataset_logic.py  (no model; run anywhere)
docs/    STATUS.md (← full honest status), project_spec_v4.md, data_construction_v2.md, COLAB.md, STAGE0.md
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
kills it → signal exists. probe ≈ text → clean negative. < 30 / small gap → inconclusive. **Where we
are:** 30 balanced (floor cleared), but the W8 gap is confound-driven (W20 reverses, H_B at chance,
within-subtype text wins) → registered claim NOT supported on Family L. Untested assumptions (ON-CoT,
layer sweep, steering floor, the AO itself) remain — see `docs/STATUS.md`.

---

## Conventions

- **Branch:** `claude/mechanistic-interpretability-qditqg`. Commit + push there. **PR #1 was MERGED;
  the live draft is PR #2** (rebased follow-up work onto main after the merge).
- **Commit trailer:** `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` + the `Claude-Session:` line.
- Do **not** put the model identifier in committed artifacts.
- Push with `-u origin <branch>`, retry with backoff on network errors.
- All model runs happen on the user's GPU box (Vast/Colab A100/H100; HuggingFace is egress-blocked in
  the dev sandbox). The Stage-1.5 probe stage is **CPU-only** (~2,900 sklearn fits; slow, not hung).
- Gate reader: **`deepseek/deepseek-chat-v3-0324`** via OpenRouter (`GATE_API_KEY`, `GATE_MODEL`,
  `GATE_WORKERS`), a strong non-Qwen non-reasoning reader. (Was GLM-4.6; switched for cost/speed.
  `run_text_only_gate.py` now sends `reasoning:{enabled:false}` so hybrid readers don't burn the
  24-token cap on hidden CoT. Avoid `deepseek-chat` — deprecating 2026-07-24 — and thinking-mode
  models, which the output cap truncates before the answer word.)
