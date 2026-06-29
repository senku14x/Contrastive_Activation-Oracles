# Project Spec v4 — Contrastive Activation Oracle Feasibility

**Supersedes v1–v3.** Stage 0 is complete (§3). The registered question, the primary item family, and
the binding constraint are now settled by the pilot, the mentor review, and the literature dig. v1's
common-mode-cancellation hypothesis stays retired. The AO's three-claim role and the
existence-gate-first ordering carry from v2/v3. **Companion:** `data_construction_v2.md` — the full
dataset protocol; this spec is the plan, that is the recipe.

---

## 0. What's settled (the anchors)

- **Pilot:** Qwen3-8B forced-choice is bimodal (confident-or-saturated); a committed cue overshoots
  uncertain items to ~1.0 (copy/leakage). Hunting for *naturally* ~50/50 items fails — they're scarce
  and uncertain-because-obscure (text-legible). Retired.
- **Literature:** manufacture the boundary with **difficulty** (GSM-Symbolic: competence is a dial);
  the manipulation that lands there is PCBench **Flawed Solution Completion** (hardest to resist,
  text-illegible). Difficulty makes diffuse-AND-text-illegible simultaneously, which natural ambiguity
  cannot.
- **Mentor + review:** the manipulation must be **load-bearing** (the conclusion depends on it, so the
  model can't sidestep by independent computation), **ablation-verified**, and the binding constraint is
  **the squeeze** (subtle enough to fool a text reader, salient enough to move the forward pass) — not a
  favorable mechanism to lean on. Stage 0 is done; the OFF label is deterministic; the power bar is
  30–40 balanced, not 8.

---

## 1. Registered question (existence-first; two levels)

> On problems at Qwen3-8B's competence boundary, where a **load-bearing** flawed step changes the
> answer, is "did the model catch or miss the flaw" readable from the **pre-output** activations —
> above a text-only baseline, up to a linear probe — and can an AO **verbalize** it?

- **The probe carries detection; the AO carries verbalization.** A probe on the pre-output state
  beating text-only establishes the signal exists; the AO's non-redundant job is language.
- **Binding constraint (registered, not assumed away):** the **squeeze**. Text-illegibility wants the
  flaw subtle; pre-output readability wants it to perturb the read-time state; subtler → less
  perturbation. The usable band is narrow and it *is* the bet — do not lean on §5's "catch leaves a
  read-time trace" mechanism to escape it.
- **A clean negative is a complete result** and the likely modal outcome.

---

## 2. Fixed decisions

| Decision | v4 value | Why |
|---|---|---|
| Primary family | **Family L — load-bearing content flaw** (correct step vs subtly flawed step) | only load-bearing, ablation-verified flaws can't be sidestepped; reads catch-vs-miss, which is text-illegible |
| Validation family | **Family P — preemptive commitment** (wrong-option-in-both, vary commitment), MCQ | buildable now; clean machinery shakedown; **not** a claim-carrier (difficulty leakage unsolved) |
| Manipulation axis | **vary content** (correct vs flawed), not commitment | commitment-only is sidesteppable and reads social pressure, not catch/miss |
| Behavioral label | **catch (flip=0) vs miss (flip=1)**, deterministic under reasoning **OFF** | OFF per-item flip is deterministic; balance = ~half the *items* flip across the pool; K reserved for ON cross-check |
| Existence instrument | **linear probe on flawed-step ΔH (and H_B alone) vs text-only** | the cheap detector; gate is injection-free so checkpoint-config-independent |
| Text-only gate | **dual:** both-prompts (conservative) **+** prompt-A-only (matched to the pre-hint probe) | makes a negative interpretable; two baselines (non-Qwen + Qwen-family) |
| Readout | **flawed-step ΔH = primary claim-carrier**; H_B-alone tested alongside | for Family L the flawed-step state is the proximate cause of the label, not a leakage floor (§5) |
| Power bar | **30–40 balanced items AND a large probe-vs-text gap** | at n≈15 the AUC SE is too large; "couldn't build 35" ≠ "no signal in 35" |
| AO role | **classifier-floor + verbalizer + SFT-better** (§6) | AO-as-classifier ≈/≤ probe (known); its lane is verbalization |
| SFT | gated behind a positive probe; three pass-conditions (§13) | SFT can't put signal into activations that don't carry it |
| Stage 0 | **DONE** (§3) | v15-stochastic config verified, G1–G5 pass |

---

## 3. Setup — Stage 0 COMPLETE

- **Target:** `Qwen/Qwen3-8B`, AO LoRA **disabled**, for target activations.
- **Oracle:** `ceselder/cot-oracle-v15-stochastic`, adapter **enabled**. **Verified config:** placeholder
  `" ?"` = token 937; extraction layers `{9, 18, 27}`; injection coefficient **1.0**; grouped
  (layer-major) placeholder format; additive norm-matched; **plumbing checks G1–G5 PASS.**
- The v3 "Stage-0 unknowns" are **stale** — do not reopen. Bauer's 2×/62% are a *different checkpoint*;
  switch only if we change checkpoints.
- **Two scope notes (my review #3):** (a) the existence gate (Stage 1.5) reads target activations
  **directly, no injection** → coefficient is irrelevant there; the decisive test is config-independent.
  (b) "G1–G5 pass" means injection *works*, not that 1.0 is *optimal*; so any weak **Stage-2 AO** number
  must rule out **under-injection** before it's called a true negative. Keep coeff-1.0 on a
  "revisit-if-AO-underperforms" list, not "closed forever."
- **Hardware:** one A100 80GB; sequential extraction/oracle on the shared base; BF16; cache to disk.

---

## 4. The two families

### Family L — load-bearing content flaw (PRIMARY, claim-carrier)
`c_A` = a **correct** load-bearing step/premise; `c_B` = a **subtly flawed** one, length-matched; shared
suffix = "continue; final answer?". Label = **catch (flip=0)** vs **miss (flip=1)**.

- **Load-bearing + ablation-verified (the binding rule):** delete the manipulated element entirely; if
  the question is still answerable to the same conclusion, there is an independent path → it's
  sidesteppable (arithmetic always is — recompute) → **discard**. Keep only items that go
  **under-determined** without the manipulated premise. The verifier is the *ablation*, not "is it
  semantic."
- **Not "semantic" as a category.** Validity/syllogism items pass through a memorized **form-prior**
  even when fully semantic. Require **novel-scenario** premise/scope/definitional flaws that ablate to
  ambiguity.
- Subsumes the mentor's B2-shape (scope: "every eligible resident" vs "every registered member") and
  PCBench Flawed Solution Completion. Excludes B1-shape arithmetic ("36+8=44") — sidesteppable.

### Family P — preemptive commitment (VALIDATION PASS, not a claim-carrier)
Wrong option present in **both** conditions; vary only commitment ("considered (w)" vs "concluded (w)
correct"). Shared MCQ suffix. The wrong-option-in-both trick cleanly kills **presence** leakage — a real
improvement and a clean **pipeline shakedown** (build it from MMLU-Redux now). **But it does not solve
*difficulty* leakage** (on trivia, follow-vs-resist tracks Qwen's certainty ≈ question obscurity ≈
text-legible), so it is **not** the claim-carrier. Use it to validate the full pipeline end-to-end
before authoring Family L; cap commitment so a "follow" isn't driven to ~1.0 (copy/leakage).

---

## 5. Contrastive object, extraction, and the Family-L readout map

For each pair: `x_A = c_A ⊕ s`, `x_B = c_B ⊕ s`, identical tokenized suffix. Extract over `{9,18,27}`,
final-**8** positions (W8 primary) and final-**20** (W20 pre-registered sensitivity), **pre-answer**
(after the final suffix token, before the first answer token — do not append a new answer cue after
extraction). Store `H_A` (correct), `H_B` (flawed), `ΔH = H_B − H_A`. Invariants 1–4 (one fixed L;
suffix token+position identity; manipulation in `c`; AO sees activations + query only) asserted per pair.

**Readout map for Family L (the mentor's fix — this differs from Family-A):**
- The **flawed-step state (`H_B` / `ΔH`) is the PRIMARY claim-carrier**, *not* a leakage floor. Unlike
  Family-A post-hint, no option is named and the answer isn't computed yet — `H_B` is the *proximate
  cause* of the label (catch vs miss). Its leak check is therefore different (next bullet), not
  "post-manipulation = discard."
- **Leak check (this is the experiment, not hygiene):** `ΔH` is dominated by the surface difference of
  the step text itself ("+8"/"−8", "eligible"/"registered"); the catch/miss signal is a *small
  modulation* on a large, text-legible `ΔH`. So: **(i) shuffle must kill any positive** (pair *i*'s query
  with pair *j*'s traces → toward base rate; a positive that survives shuffle is surface-fingerprinting);
  **(ii) construct flip and no-flip items with matched flaw structure** so the probe can't cheat on the
  flaw text; **(iii) test `H_B` alone** as a readout alongside `ΔH` — it drops the explicit
  "the-texts-differ" component and may isolate the catch-state.
- **The "catch → larger ‖ΔH‖" direction is a guess — measure the sign, don't assume.** A catcher may
  *converge* flawed→correct late, inverting it. The geometry presence-check reports the sign; the spec
  does not assume it.

---

## 6. The AO's three jobs (carried from v2/v3)

| Claim | Instrument | Baseline | Honest ceiling | If it fails |
|---|---|---|---|---|
| **(a) Detection (classifier)** | AO emits catch/miss, AUC | **text-only**, then **probe** | matches probe; loses on this task | report AO ≤ probe; it's the floor, not the headline |
| **(b) Verbalization (the point)** | AO free-form description of the catch/miss | **no-activation twin** + resist/null + edit test | none — a probe can't make a sentence | description is fingerprint/story |
| **(c) SFT (make it better)** | SFT'd AO vs zero-shot AO, held-out | the same three | improves *verbalization*, not detection | "better" = more fluent narration |

The **probe carries detection, the AO carries explanation, SFT improves explanation not detection.**
**probe→template** is the verbalization competitor — promote it from deferred at Stage 2 (scoped: it's
the competitor for *known-phenomenon* verbalization, not for the discovery / hypothesis-generator
framing where no template exists).

---

## 7. Input modes and the contrastive mechanism

Run for detection AUC: `H_A`, `H_B`, `ΔH`, `[H_A;H_B]`. For Family L the contrast has a **motivated
mechanism** (replacing cancellation): a model that **catches** the flaw should process the flawed step
differently from the correct step (flags the error → larger ‖ΔH‖); a model that **misses** it processes
them similarly (→ smaller ‖ΔH‖). So `ΔH` is a candidate catch/miss predictor — but **test `H_B` alone
alongside it** (§5), and **measure the sign** (§5). Do not commit to `ΔH` as the sole object before
comparing to `H_B`-alone.

---

## 8. Controls (inference-only; all required)

- **Dual text-only gate.** (a) *Conservative gate:* a strong non-Qwen reasoner gets **both** prompts +
  the query → must be ≈chance. (b) *Matched reader:* a **prompt-A-only** reader matched to the pre-hint
  probe → makes a negative interpretable. Two baselines: non-Qwen (generic legibility) + Qwen-family
  (own-prior legibility). Dev model filters; a **reserved** model reports on the frozen set.
- **Shuffle = the experiment** (§5). **Matched flaw structure** across flip/no-flip (§5). **Ablation-
  verify** at construction (§4). **`H_B`-alone vs `ΔH`** comparison (§7).
- **no-activation twin** (decisive for verbalization/SFT). **resist/null discriminator** (resist =
  flaw present, caught; neutral = no flaw, report separately). **edit test** (change the flaw's load so
  the catch/miss target flips, surface held fixed; the readout/description must track it).
- **Steering positive control + sweep** (one-sided floor: below → likely unreadable; above → unknown).

---

## 9. Metrics

- **Detection:** catch/miss **AUC** for the probe, both text-only baselines, and each AO input mode, on
  W8 (primary) and W20 (sensitivity). Report all side by side.
- **Per-cell, never one aggregate:** separate **FOLLOW(miss) / RESIST(catch) / NEUTRAL-min / NEUTRAL-max**.
  *A method that identifies the flaw but fails RESIST is a manipulation-presence detector, not a
  disposition reader.*
- **Verbalization:** scored against measured catch/miss (parser or ≥40-example-calibrated judge,
  agreement reported); pass = beats no-activation twin **and** gets resist/null right **and** tracks the
  edit test. Specificity and correctness reported separately.
- **The squeeze accounting:** report build-yield (how many load-bearing, gate-passing items existed) as
  a **distinct** outcome from detector performance. "Couldn't build 35" ≠ "no signal in 35."

---

## 10. Dataset (summary → `data_construction_v2.md`)

Primary = **Family L** via difficulty-manufactured boundary (GSM8K → OlympiadBench → Omni-MATH>6; tune
to catch-rate band) + Flawed Solution Completion + **ablation-verify** + **matched flaw structure**.
Validation pass = **Family P** from MMLU-Redux (wrong-option-in-both, capped commitment). **OFF
deterministic label** (balance = ~half items flip). Power: **30–40 balanced + large effect**. Staged
screening with **frozen thresholds**, **dual text-only gate**, **three SFT ablations** (paired /
no-activation / text-context), **label-shortcut tests** (P(y|template), P(y|domain), classifiers on
template/domain/text), **orientation coherence**, **per-cell reporting** — all in the companion doc.

---

## 11. Staged plan

- **Stage 0 — DONE** (§3).
- **Stage 1 — Build the sets.** First **Family P** as an end-to-end **pipeline shakedown** (it's
  buildable now and validates token-invariants, screening, gates, extraction). Then author **Family L**:
  difficulty screen to the catch-rate band, **ablation-verify** every item, matched flaw structure, dual
  gate. Run the steering positive control + sweep. Record build-yield. Stop if Family L can't reach
  ~30–40 load-bearing gate-passing items — that's a **supply** result (report it as such).
- **Stage 1.5 — THE EXISTENCE GATE (decides everything; we measure, not argue).** Probe on **flawed-step
  ΔH** (and `H_B` alone) → catch/miss, vs **text-only**. *This is the empirical adjudication of
  read-time vs reason-time* — if the pre-output state separates catch/miss above text-only, the trace is
  read-time; if not (and behavior is clean), the decision is reason-time. Shuffle must kill any positive.
- **Stage 2 — AO zero-shot.** Classifier floor (vs probe + text-only) + verbalization (three gates).
- **Stage 3 — SFT for verbalization (gated; §13).**
- **Stage 4 — Decision / writeup (§12).**

---

## 12. Decision rule (pre-registered)

> **Existence (primary, Stage 1.5):** the probe on **flawed-step ΔH (or `H_B`)** beats **text-only**
> (both gate models ≈chance) by a **large** margin on **30–40 balanced** items, **and the positive
> survives shuffle**.

- **Signal exists** → run §6(a/b) zero-shot, then consider §13 SFT. Detection: AO ≤ probe is expected
  and fine; the contribution is (b)/(c).
- **Clean negative (a complete result):** probe ≈ text-only, or text-only solves it. Adjudicates
  read-time vs reason-time empirically; extends Jakkli's null to the catch/miss setting.
- **Underpowered ≠ absent:** < 30–40 items, or a small gap → "inconclusive," not "no signal."
- **NOT a success:** survives only **without** shuffle (fingerprinting); or an AO mode beats text-only
  only because the probe does (report AO ≤ probe).

---

## 13. SFT arm (Stage 3) — verbalization target, gated, three pass-conditions

**Precondition (hard):** run only if Stage 1.5 found probe > text-only (shuffle-surviving). SFT cannot
put signal into activations that don't carry it; before a positive probe, "describe the catch/miss" is
fluent narration, not reading. **Target:** catch/miss *description*. **Claim scope:** contrastive SFT
improves the AO's verbalization of the catch/miss, and the improvement **depends on the injected
traces**. **Three pass-conditions (all required):** no-activation twin **fails**; resist/null **right**
(train + held-out); edit test **tracks**. **Three ablations** (from the dataset doc): paired-activation
vs no-activation vs text-context — paired must beat **both** on held-out templates *and* held-out
questions. Detection AUC reported as a constant you're **not** trying to move.

---

## 14. Compute budget (one A100 80GB; order-of-magnitude)

| Stage | Scale | Cost |
|---|---|---|
| 1 build (Family P shakedown + Family L) | source pulls + screen + gates | < 1 GPU-hr compute (authoring is the cost) |
| 1.5 existence gate (probe vs text-only) | 30–40 items | < 1 GPU-hr |
| 2 AO zero-shot (modes + verbalization) | set × modes | a few GPU-hrs |
| 3 SFT (gated) | grouped splits + ablations | hours |

---

## 15. Risk register

- **Sidestep confound** (the pilot's real lesson): catch and sidestep both give pA≈pB and small ΔH →
  **ablation-verify** every item; keep only ablate-to-ambiguity flaws.
- **The squeeze** (subtle vs salient): registered as the binding constraint; build-yield reported
  separately from detector performance.
- **Surface fingerprinting** (ΔH dominated by flaw text): **shuffle is the experiment**; matched flaw
  structure; test `H_B` alone.
- **Sign may invert** (catcher converges late): **measure** the catch/miss sign, don't assume.
- **Read-time vs reason-time:** not argued — **measured** at Stage 1.5 (OFF-first; the geometry of the
  pre-output state is the adjudicator).
- **Difficulty leakage on trivia:** why Family P is validation-only, not the claim.
- **Form-prior on logic items:** require novel-scenario flaws, not memorizable forms.
- **Commitment overshoot to ~1.0:** cap commitment (Family P); prefer content-flaws (Family L).
- **Power + effect size:** 30–40 balanced + large gap; "couldn't build" ≠ "no signal."
- **Under-injection masquerading as AO-null** (Stage 2 only): rule out before calling an AO number a
  true negative.
- **Verbalization = narration:** no-activation twin + resist/null + edit test; SFT gated.
- **Judge as untrusted instrument:** ≥40-example calibration, agreement reported, parser primary.

---

## 16. Non-claims

Does not establish: latent-goal reading; mechanism; faithful explanation; that ΔH *causes* the output;
unique-layer localization; that the AO beats a probe on detection (not the claim); generalization beyond
this family. **A positive establishes only:** the catch-vs-miss of a load-bearing flaw is statistically
predictable from the pre-output activations above a text-only baseline (probe-carried, shuffle-surviving)
— and, if §6(b/c) pass, that an AO can *verbalize* it in a trace-dependent way. Nothing past that sentence.
