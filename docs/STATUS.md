# Contrastive Activation Oracle — Full Status (nothing hidden)

**Date:** 2026-07-01. **Branch:** `claude/mechanistic-interpretability-qditqg`. **PR:** #2 (draft; #1 merged).

This document states the actual state of the project outright — the good, the bad, the confounded, and
the untested. It is written so that further research has a complete, honest map. If something is a
guess, it is labeled a guess. If something failed, it says so.

---

## 0. The goal (unchanged, stated plainly)

Get a **behavioral shift** (here: did Qwen *catch* or *miss* a planted load-bearing fallacy) to be
**readable from the pre-output activation contrast `ΔH = H_flawed − H_correct`, above a text-only
baseline** — and then **train an Activation Oracle (AO) to verbalize that shift**. The linear probe is
only the **existence gate**: if a full-access linear probe can't read it above text, the lossy AO
almost certainly can't either, so the probe gates the (expensive) AO/SFT work.

**This is the thing we have not yet achieved.** Read §3 and §5 for exactly why, and §6 for the
untested assumptions that might be why.

---

## 1. Bottom line up front (BLUF)

- On the current dataset (Family L, 167 items) the registered claim — *catch/miss readable from
  pre-output ΔH **above a text baseline, robustly*** — **is not supported.**
- **But this is a result about *this design + this probe config*, not a proof the AO idea is dead.**
  Several load-bearing assumptions were **never tested** (§6). We are **no longer power-limited**
  (we cleared the 30-balanced floor); we are **confound-limited and design-limited.**
- The single most important empirical fact: **`probe(H_B alone)` sits at chance (~0.44–0.54) in every
  powered run**, while `probe(ΔH)` looks high only at the W8 window and only on the full multi-subtype
  set. That pattern says the probe is reading the **text/subtype contrast between the two conditions**,
  not Qwen's internal disposition. See §5.
- Honest current options are in §7. They are **not** "scale more data" — more items will not
  manufacture an above-text signal that isn't there.

---

## 2. What is solid (does not evaporate, regardless of the probe verdict)

1. **Stage 0 — AO wiring reproduced on an A100.** Released oracle `ceselder/cot-oracle-v15-stochastic`
   reverse-engineered and reimplemented (`cao/ao_runtime.py`): placeholder `" ?"` = token 937,
   extraction layers {9,18,27}, injection at layer 1 additive norm-matched coeff 1.0, grouped
   layer-major format, G1–G5 pass (`scripts/stage0_repro.py`).
2. **A real behavioral finding.** Qwen3-8B **follows a planted fallacious "therefore" step ~80–85% of
   the time** on genuinely under-determined fictional problems; it *catches* the flaw (holds "cannot be
   determined") only ~15–18%. This is **robust across reasoning OFF (18%) and ON (22%)** — plain CoT
   does not rebalance it. Deterministic under the OFF readout (reproduces bit-for-bit).
3. **A non-leaky-to-a-zero-shot-reader dataset.** A strong non-Qwen reader (GLM-4.6, then DeepSeek-V3)
   shown *both* conditions cannot call catch-vs-miss above chance (balanced-acc ~0.40–0.44). So the
   label is not trivially guessable by an LLM reading the prompts.
4. **All construction invariants pass.** 167/167 Family-L items ablation-verified load-bearing
   (delete the step → under-determined); token/position within-pair invariants pass on the real
   tokenizer; gold/flawed option positions counterbalanced across A–D.
5. **A reproducible pipeline + a diagnostics suite**, all committed. Everything regenerates from
   `cao/family_l.py` via `scripts/run_all.sh`.

---

## 3. What we ran, and every number (the honest chronology)

Family L was scaled 32 → 81 → 105 → 137 → 167 items over this work. Each full pipeline run:

| bank | valid (catch/miss) | window | probe(ΔH) | probe(H_B) | **text baseline** | shuffle | script verdict |
|---|---|---|---|---|---|---|---|
| 32  | 22 (4/18)  | W8  | 0.43 | 0.17 | 0.32 | 0.55 | UNDERPOWERED |
| 32  |            | W20 | 0.32 | 0.23 | 0.32 | 0.48 | UNDERPOWERED |
| 81  | 61 (9/52)  | W8  | 0.733 | 0.152 | 0.378 | 0.464 | "SIGNAL" |
| 81  |            | W20 | 0.415 | 0.177 | 0.235 | 0.423 | "CLEAN NEGATIVE" |
| 137 | 100 (10/90)| W8  | 0.717 | 0.294 | **0.623** | 0.468 | ~negative (gap 0.094) |
| 167 | 128 (15/113)| W8 | **0.804** | 0.539 | **0.686** | 0.454 | "SIGNAL" |
| 167 |            | W20 | 0.653 | 0.443 | **0.684** | 0.471 | "CLEAN NEGATIVE" |
| 167 | AC-only 29 (11/18) | W8 | 0.652 | 0.237 | **0.702** | 0.396 | (text > probe) |

Read the columns, not the last cell:

- **The text baseline climbed with the "signal":** 0.32 → 0.38 → 0.62 → 0.69. As we scaled and
  concentrated on productive subtypes, the label became *more* text-legible, not less.
- **`probe(H_B alone)` never left chance** in any powered run (0.15–0.54). The disposition is
  **not** linearly in the flawed-condition pre-output state at these layers.
- **W8 vs W20 disagree on the same items.** W8 (primary) shows probe > text; W20 (the *pre-registered*
  sensitivity window) shows probe **below** text. A signal that reverses across the two declared
  windows is not robust — and the pre-registration is exactly what stops us from cherry-picking W8.
- **Within a single subtype (affirming_consequent), text (0.702) beats the probe (0.652).** The
  probe's apparent edge exists only *across* subtypes.

**The final powered run (167 items, 15 catches, 30 balanced) cleared the power floor**, so the
verdict is now interpretable and it is: **the activation probe does not robustly beat text.**

---

## 4. The catch-yield structure (why the confound exists)

`scripts/subtype_yield.py` on the 137-item run:

| subtype | catch-rate |
|---|---|
| affirming_consequent | ~43% |
| composition | ~21% |
| division | ~8% |
| unsupported_assumption, illicit_conversion, denying_antecedent, equivocation, false_cause, overgeneralization | **~0%** (0 catches across ~60 valid items) |

Catch is **not** a uniform disposition — it is concentrated in 2–3 fallacy types. Since fallacy type is
a **text property**, any probe (activation or text) can predict catch/miss by fingerprinting subtype.
That is the mechanism behind the text leak, and it is why the probe's edge vanishes when subtype is
held constant.

---

## 5. Honest diagnosis: it is (almost certainly) a text/subtype confound, not disposition-reading

The signature is unambiguous when all four facts are read together:

1. `probe(ΔH)` high **only** at W8, **only** on the full multi-subtype set.
2. `probe(H_B alone)` at chance — the flawed-condition state itself does not carry the label. (The spec
   names H_B the *primary* claim-carrier; it is at chance.)
3. `text-feature probe` ~0.685 — catch/miss is substantially readable from the flawed context's raw
   text n-grams alone.
4. Within one subtype, text ≥ probe; across the two pre-registered windows, the probe's edge reverses.

`ΔH = H_flawed − H_correct` is dominated by the **wording difference** between the correct and flawed
steps. `ΔH` carrying the label while `H_B` does not = the probe is reading **how the two texts differ**
(which correlates with subtype, which correlates with catch-rate), **not** Qwen's internal state. This
is Paper 2's splice confound ("same local text, different upstream state" is the hard case; we did the
*opposite* — different text) reappearing because Family L's two conditions differ in wording.

`inspect_text_leak.py` confirmed the text-feature probe keys on short structural/stylistic n-grams and
per-batch authoring artifacts (an earlier padding-preface artifact was found and fixed; see §8).

---

## 6. What we might be missing (the live threads — this is the part for further research)

The user's instinct ("we are missing something") is **partly right**: the W8 0.804 is real, and the
following load-bearing assumptions were **never tested**. Ranked by (plausibility it's the issue ×
cheapness to test):

1. **[DESIGN — biggest] The family violates "same text, different upstream state."** Because H_A/H_B
   differ in wording, ΔH conflates *content difference* with *disposition*. The original spec was built
   around Paper 2's splice — hold the surface text (near-)identical, vary only upstream state. Family C
   did this (identical hint syntax, only the letter varies) but collapsed because Qwen ignores peer
   hints. **A design where near-identical text yields catch vs miss is the highest-value missing
   piece.** Until then the text leak is structural, not fixable by more data.

2. **[TIMING — the spec's central open risk, untested] The disposition may form *during* the CoT, not
   pre-output.** We only ever extracted **reasoning-OFF, pre-output** activations. We tested ON
   *behaviorally* (22% catch) but **never extracted ON-trajectory activations and probed them.** If
   catch/miss crystallizes mid-CoT, the pre-output state genuinely won't contain it — `H_B` at chance
   is *consistent* with this. This is directly testable: extract at a mid/post-CoT position and probe.
   If it reads there and not pre-output, that is a real, publishable finding — and it redirects the AO
   to read mid-CoT instead of pre-output.

3. **[MEASUREMENT — cheap-ish] Layer & position were never swept for the probe.** We used {9,18,27}
   (the AO's *injection*-training layers) and pooled the final 8/20 suffix positions. But a **probe**
   needs no AO-compatibility — it can read any layer/position. Paper 3 finds reading is best at ~62%
   depth (layer 22). The catch/miss signal could live at a layer/position we never extracted. Requires
   re-extraction with all layers; clearly label as exploratory (don't p-hack the window).

4. **[INTERFACE — cheap, never run] The steering-vector detection-floor control (spec Stage 1 / §5).**
   Add a known concept vector to condition B's saved activations at coefficient c·‖h‖, sweep c down,
   find the AO's detection floor, then compare to the measured norm of natural ΔH. If natural ΔH sits
   **below** the floor, the honest conclusion is "**signal too small for this interface to resolve**" —
   a clean, quantified explanation instead of a guess. We never did this.

5. **[THE ACTUAL OBJECT — never run] We have never run the AO itself.** The entire project is about the
   Activation Oracle; every number above is a *probe*. Given the literature regularity (AOs match but
   never beat probes), a probe negative *predicts* an AO negative — but the AO's **verbalization** (the
   user's actual end goal) has not been tested at all, and the plumbing on real contrastive pairs is
   unconfirmed beyond Stage 0's single-task repro.

6. **[LABEL] Catch is rare (~15%) and subtype-correlated.** Even setting the leak aside, the minority
   class is small and structurally tied to fallacy type. A behavioral lever that is (a) more balanced
   and (b) orthogonal to surface form would sidestep both the power and the confound problems. Ties to
   thread #1.

---

## 7. Options from here (honest menu — pick with eyes open)

- **A. Redesign for text-orthogonality (highest scientific value, most work).** Build a contrast where
  near-identical surface text yields catch vs miss, so ΔH isolates upstream state. This is the only path
  that directly fixes the structural leak and gives the registered claim a fair test.
- **B. Test the "forms during CoT" hypothesis (cheap-ish, high information).** Extract ON-trajectory
  activations at a mid/post-CoT position; probe those. Either finds the signal (→ redirect the AO to
  read there) or cleanly confirms it's genuinely absent from the pre-output state.
- **C. Sweep layers/positions on the existing data (cheap, exploratory).** Re-extract all layers; probe
  per-layer/per-position. Rules in/out "we looked in the wrong place."
- **D. Run the steering detection-floor control (cheap).** Answers whether *any* activation reader could
  resolve natural ΔH at this interface — the assumption sitting under everything.
- **E. Write up what we have (no more GPU).** Behavioral finding + non-leaky dataset + the honest
  negative (catch/miss is text-legible but not robustly readable from pre-output ΔH above text;
  window-fragile; absent in H_B-alone). A complete, registered feasibility result — the modal outcome
  both specs predicted.

**Recommendation:** B and D are the cheap, high-information moves that could still change the story
(they test the two biggest untested assumptions). A is the real fix if the goal is to make the
registered claim work. E is legitimate and complete but is *not* required yet — the user's instinct to
dig is scientifically justified specifically because B, C, and D were never run.

---

## 8. Mistakes and lessons from this work (stated outright)

- **Padding disaster (found + reverted).** An attempt to kill the text leak by prepending a *universal*
  neutral prefix to every item (`pad_to_equal_length.py`) **changed Qwen's behavior** — catches
  collapsed 15 → 2 on the same bank. Lesson: any change to *what the model reads* must be
  behaviorally re-checked, not just verified for code-correctness. Reverted (`d91b413`); a mock
  tokenizer can verify padding *logic* but cannot catch a behavioral shift — that needs the GPU.
- **We optimized the kill-switch, not the experiment.** Days went into scaling the dataset to power the
  *probe existence check* — which was only ever a gate, never the deliverable. The AO comparison
  (the actual project) and the steering control (Stage 1) were skipped. Documented so it isn't repeated.
- **"SIGNAL" from the script is not a verdict.** `stage1_5_probe.py`'s gate
  (`ΔH>0.70 & ΔH−text>0.10 & shuffle flat`) fired at W8 on a confounded positive. It does not check
  H_B-alone, subtype structure, or W8/W20 agreement. Read the columns, not the canned verdict.
- **A joint-mode AO null would be confounded with format-OOD** (the released AO never saw two tagged
  blocks) — so even when we do run the AO, a null there cannot alone close the project; the probe +
  steering control are what license a real negative. (Carried from the spec; still true.)

---

## 9. What is pending / not started

- Stage 2 (AO zero-shot: joint / delta / constrained-parity independent) — **not started.**
- Stage 3 (contrastive SFT + the no-activation ablation) — **not started** (correctly gated).
- Steering-vector detection-floor control (Stage 1 leftover) — **not started.**
- ON-trajectory (mid-CoT) activation extraction + probe — **not started.**
- Layer/position sweep for the probe — **not started.**
- Length/margin confound matching in `match_catch_miss.py` — currently **flagged, not corrected**
  (the `neutral_gold_margin` confound: catch items cluster at ~0.999 vs miss ~0.95, |Δ|/SD ≈ 0.9–1.1,
  hovering at the threshold). `--subtype` filter added; margin-matching itself not implemented.

---

## 10. How to reproduce / run

```bash
git checkout claude/mechanistic-interpretability-qditqg
export HF_TOKEN=... GATE_API_KEY=... GATE_MODEL=deepseek/deepseek-chat-v3-0324 GATE_WORKERS=16
bash scripts/run_all.sh 2>&1 | tee run.log
```

Diagnostics (all read cached data, no GPU reload):
- `scripts/subtype_yield.py` — catch/miss/discard by subtype and authoring batch.
- `scripts/inspect_text_leak.py` — what the text-feature probe keys on (top n-grams).
- `scripts/diagnose_l78_outlier.py` — leave-one-out sensitivity to the W20 outlier item.
- `scripts/match_catch_miss.py --subtype affirming_consequent` — single-subtype frozen set (removes
  subtype as a text tell before the probe).

Run order: build → pad → ablation_verify → verify_token_invariants → measure_target_logits →
label_candidates → run_text_only_gate → match_catch_miss → extract_activations (W8+W20) →
stage1_5_probe.
