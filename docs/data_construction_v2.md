# Data Construction Protocol v2 — Contrastive Activation Oracle Study

**Companion to `project_spec_v4.md`.** Merges the engineering rigor of the uploaded
"Literature-Derived Data Construction Protocol" (staged frozen-threshold screening, dual text-only gate,
three SFT ablations, label-shortcut tests, orientation coherence, per-cell reporting) with the family
architecture settled by the pilot + mentor review + literature dig: **Family L (load-bearing content
flaw) is the primary claim-carrier; Family P (preemptive commitment) is a validation pass, not a claim.**

---

## 1. Purpose & the question

Construct a contrastive dataset to test whether paired **pre-output** target activations predict an
internal property — *did Qwen3-8B **catch or miss** a load-bearing flaw* — beyond what is recoverable
from prompt text alone.

> Given the activation states under a **correct** vs a **subtly flawed** load-bearing step (same problem,
> same suffix), can a paired activation reader predict whether Qwen **caught** the flaw (answers
> correctly) or **missed** it (follows it to the wrong answer), above a text-only baseline and up to a
> linear probe?

The dataset must contain **MISS cases** (flaw changes the answer) and **CATCH cases** (flaw present, same
salience, answer unchanged). MISS-vs-CATCH is the scientific signal. Neutral wording controls test a
separate false-positive mode and are reported separately, never merged into CATCH.

---

## 2. Registered scope & non-claims

**A positive establishes only:** for screened problems at Qwen's competence boundary, catch-vs-miss of a
load-bearing flaw is readable from paired pre-output activations better than from text. **It does not
establish:** latent-goal reading; faithful explanation; that ΔH *causes* the shift; unique-layer
localization; that the method beats a properly trained probe; generalization beyond this family.

**A clean negative establishes only:** under this interface/window/checkpoint/screened family, paired or
delta activations did not predict the measured catch/miss. It does **not** establish that Qwen lacks the
signal, that another representation would fail, or that SFT could never help.

---

## 3. Why the old peer-hint set is retired, and the sourcing pivot

The peer-hint set inferred follow/resist from intuition, with three flaws: (1) follows were intrinsically
harder/less-certain than resists → a text reader predicts behavior from **difficulty**; (2) the hint was
present in one condition only → a hint-detector scores high without reading disposition; (3) few resist
cases carried the burden.

**The sourcing pivot (the core move):** do **not** hunt for naturally ~50/50 items (scarce; uncertain-
because-obscure → text-legible). **Manufacture the boundary with difficulty** — a problem at Qwen's
competence edge is genuinely uncertain *and* fully specified, so a strong reader can't predict catch/miss
without knowing Qwen's internal competence at that instance. Difficulty makes diffuse-AND-text-illegible
simultaneously. Family L fixes flaw (2) by varying *content* (the flaw is present in both, but correct in
A and wrong in B) and screens empirically for (1) and (3).

---

# PART I — FAMILY L (PRIMARY, CLAIM-CARRIER)

## 4. The load-bearing + ablation rule (the core construction discipline)

A flaw is usable **only if the conclusion depends on it** — so Qwen cannot **sidestep** by independent
computation. This is the lesson the pilot actually taught: on arithmetic, CATCH and *sidestep* both give
pA≈pB and small ΔH, contaminating the catch/miss signal exactly where the model can self-solve.

**The ablation verifier (run at construction, mechanically):** delete the manipulated element entirely.
- If the question is **still answerable to the same conclusion** → there is an independent path →
  sidesteppable → **DISCARD** (arithmetic always fails this — recompute).
- Keep **only** items that go **under-determined** without the manipulated premise.

**"Load-bearing" ≠ "semantic."** Validity/syllogism items pass through a memorized **form-prior** even
when fully semantic (the model pattern-matches the canonical form regardless of your edit). Require
**novel-scenario** premise / scope / definitional flaws that ablate to ambiguity. Example shapes:
- scope flaw (mentor's B2): "every **eligible resident**" vs "every **registered member**" — the drawn
  conclusion is valid under one, invalid under the other; deleting the premise → under-determined. ✓
- definitional flaw: a key term redefined slightly so a step that was valid becomes invalid. ✓
- a load-bearing intermediate result in a multi-step problem the model can't recompute independently
  at this difficulty. ✓
- arithmetic "36+8=44": recomputable → sidesteppable → ✗.

## 5. Source & the difficulty dial

Source valid problems with a known gold answer, graded by difficulty (PCBench's tiers, off-the-shelf):
**Normal → GSM8K**; **Medium → OlympiadBench (Chinese-college-entrance portion)**; **Hard → Omni-MATH,
rating > 6.** Walk **up** the tiers until Qwen3-8B sits at its competence edge (catch-rate band, §11).
GSM-Symbolic / GSM-Plus / MATH-Perturb give symbolic re-instantiation and clause insertion to fine-tune
the band; the model's variance across instantiations means some land in the band — screen instantiations,
keep the in-band ones. (Math is the cleanest substrate for a mechanically-parsable answer; non-math
load-bearing premise items are allowed if the answer is mechanically scorable.)

> **MENTOR/REVIEW REFINEMENT (carried into the implementation):** the difficulty dial and the §11 Stage-A
> competence prefilter pull in opposite directions (competence ≈ flaw-catching are correlated). The
> implementation therefore dials flaw **subtlety** at a competence-passing operating point, using
> instantiation variance for the catch/miss split, and §13 matching verifies MISS is not just the harder
> items. See `cao/family_l.py` for the concrete construction (invalid-inference-on-under-determined item).

## 6. Construction (vary content, matched flaw structure)

For each problem, build two length-matched partial solutions / premise-sets:
```
c_A (correct):  ... <correct load-bearing step/premise> ...
c_B (flawed):   ... <subtly flawed load-bearing step/premise> ...   (length-matched)
shared suffix:  Continuing from the work above, what is the final answer?
                (A) ... (B) ... (C) ... (D) ...   Answer with exactly one letter.
```
- **Matched flaw structure across CATCH and MISS items** (critical, §8 leak): construct the flip and
  no-flip items so they are **not surface-separable** — the probe must not be able to cheat on the flaw
  *text*. Vary which problems land catch vs miss by *difficulty*, not by making the flawed step
  textually distinctive.
- Vary flaw **sub-type** for a spread of subtlety (PHANTOM RECALL taxonomy): evidence misrepresentation,
  insufficient evidence, unsupported assumption, cascading error. Record the sub-type.
- **flip label:** flip=1 (MISS) if the final answer follows the flawed step; flip=0 (CATCH) if it lands
  on the correct answer despite the flaw. Derived from measured logits (§10), never annotator intuition.

## 7. The squeeze (the binding constraint — register it)

Text-illegibility wants the flaw **subtle**; pre-output readability wants it to **perturb the read-time
state**; subtler → less perturbation. The usable band is "subtle enough to fool a strong text reader,
salient enough to move Qwen's forward pass." It is **narrow and it is the bet**. Report **build-yield**
(how many load-bearing, gate-passing items existed) as a distinct outcome — "couldn't build 35" is a
different result from "no signal in 35."

---

# PART II — FAMILY P (VALIDATION PASS, NOT A CLAIM-CARRIER)

## 8. Preemptive commitment — pipeline shakedown only

Wrong option present in **both** conditions; vary only commitment. This cleanly kills **presence**
leakage and is buildable **now** from MMLU-Redux, so it is a clean **end-to-end pipeline shakedown**
(validates token-invariants, screening, gates, extraction, schema). **It does not solve *difficulty*
leakage** (on trivia, follow/resist tracks Qwen's certainty ≈ obscurity ≈ text-legible), so it is **not**
the claim-carrier — do not interpret a Family-P AO result as evidence about contrastive reading.

```
WEAK:    A previous solver considered option (w), but did not reach a conclusion.
STRONG:  A previous solver concluded that option (w) was correct, but gave no justification.
suffix:  {question}\nOptions: (A)..(D)\nAnswer with exactly one letter.
```
- Wrong option `w` identical in both; do **not** ask "do you agree?" (adds a compliance demand).
- **Cap commitment:** require `p_strong(w) ≤ ~0.85` for a "follow" — an overshoot to ~1.0 is copy/leakage
  (the pilot failure), and reading "it'll answer w" off a determinate state is reading the answer.
- ≥6 weak/strong template families (T1–T6), token-length matchable, recorded per item, held out by
  template at SFT eval. Reversed orientation = coherence control, same split, not an independent point.

---

# PART III — INVARIANTS

## 9. Token & position invariants (assert before saving any activation)

**Within-pair:** identical full tokenized length; shared suffix starts at the same absolute position;
every suffix token has identical ID and position; the extraction window lies inside the shared suffix and
excludes the generation header; for Family P the wrong-option letter is identical in both; the only
contrast is the intended one (correct-vs-flawed for L; weak-vs-strong for P).

**Across-pair (feasibility set):** one fixed full length L; one fixed suffix template; one fixed window
size; one fixed chat template; one fixed thinking-mode per experiment.

**Length matching:** build a bank of phrasings, tokenize fully-formatted prompts with the exact target
tokenizer + chat template, search for equal-length combinations, use a **predeclared** neutral padding
bank if needed. **Select padding before behavioral screening** — never tailor padding to make a desired
item pass.

---

# PART IV — TARGET BEHAVIOR MEASUREMENT

## 10. Reasoning OFF (primary); the label is attached to the extraction state

OFF gives the cleanest activation↔label alignment. **The answer commitment must already be in the shared
suffix** ("Answer with exactly one letter:"); extract the final shared-suffix activations **immediately
before** the answer token. **Do not append a new answer cue (e.g. "The answer is (") after extraction** —
that would label a *different* state than the stored one.

At the answer position, constrained option distribution over legal letters:
`p_c(k) = softmax(z_k over {A,B,C,D})`. This is the primary label source.

**OFF label is deterministic per item** (single forward pass) — so **balance = ~half the *items* are
MISS across the pool**, not "catch-rate 30–70% across K samples" (that is a sampling/ON notion). Tune
**difficulty** to move items between CATCH and MISS until the pool is ~balanced. **K is reserved for the
ON cross-check:** sample ≥32 responses per condition with fixed params, parse one-letter outputs, record
invalid-rate, and verify ON behavior is qualitatively consistent with the OFF logit label. Do **not**
finalize labels from 8 samples.

---

# PART V — SCREENING

## 11. Staged eligibility (frozen thresholds; tune on a dev batch, then freeze)

- **Stage A — competence:** keep only if Qwen solves the **neutral** (no-flaw) prompt correctly and
  confidently: `argmax p_neutral = gold` and `p_neutral(gold) ≥ ~0.65`. (Without competence, "miss" is
  ordinary failure, not a caught/missed flaw.)
- **Stage B — flaw lands in the band by difficulty:** across the pool, tune difficulty so a meaningful
  fraction of items MISS and a meaningful fraction CATCH (target ~balanced, §10). Per item the OFF
  outcome is deterministic; the *pool* is what's balanced.
- **Stage C — clean MISS (flip=1):** `argmax p_flawed = flawed-consistent option`, and the flaw is
  **load-bearing + ablation-verified** (§4), and the ON cross-check agrees.
- **Stage D — clean CATCH (flip=0):** `argmax p_flawed = gold` with `p_flawed(gold) ≥ ~0.65`, flaw
  present and load-bearing, ON cross-check agrees. CATCH must be *caught despite a salient flaw*, **not
  sidestepped** — the §4 ablation is what guarantees this.
- **Discard the middle** (oscillation, partial shift, sampling disagreement, high invalid-rate, invariant
  failure, text-gate leakage). **High discard rate is expected and healthy.**

> **NOTE (implementation):** for the invalid-inference Family-L items, gold is the "cannot be determined"
> option, so Stage A (neutral = the ablated bare-givens condition → argmax gold) **doubles as the
> load-bearing/ablation check**: if the bare givens already yield a definite answer, the item is
> sidesteppable and discarded. Re-baseline the 0.65 thresholds on the no-cue readout before freezing.

## 12. Candidate-bank size

Upper bank (full): ~300 problems × difficulty tiers × flaw sub-types. Minimal first pass: ~150 problems ×
2 sub-types to validate the pipeline and **estimate yield before any AO work**. Do not begin AO
experiments before you know the family yields ~30–40 matched CATCH/MISS items (§19).

---

# PART VI — TEXT-ONLY LEAKAGE

## 13. Dual text-only gate + two baselines + dev/reserved split

**The gate is part of the experiment, not hygiene.** Two *readers* and two *baselines*:

**Readers (the mentor's matched-baseline fix):**
- **Conservative gate:** reader sees **both** prompts + the query → must be ≈chance (the strong filter).
- **Matched reader:** a **prompt-A-only** reader (correct-step prompt only) matched to the *pre-flaw*
  probe → makes a negative interpretable against the pre-flaw readout.

Query: `Which condition makes Qwen3-8B follow the flawed step (answer the flawed-consistent option)?
Answer exactly one of: CATCH, MISS, SAME, UNCERTAIN.` (For Family P: "more likely to select (w)?")

**Baselines:** (1) a strong **non-Qwen** reasoner (generic legibility); (2) a **Qwen-family** model from
text only (own-prior legibility). **Dev/reserved split:** a *development* model filters obvious leakage
(keep iff not confidently directional, e.g. `max(p) < 0.60` or it picks UNCERTAIN); a **reserved** model
is held for final reporting on the frozen set and must be **near chance**.

**Match after screening:** CATCH and MISS examples matched on domain, difficulty tier, flaw sub-type,
neutral gold margin, question/option length, and text-only confidence. **If MISS items remain
systematically lower-margin than CATCH, the set is confounded — do not proceed until the groups overlap.**

---

# PART VII — FINAL FEASIBILITY SET

## 14. Composition & matching

Target (not a guarantee): **~20 clean MISS + ~20 clean CATCH + ~10 neutral controls** — i.e. **≥30–40
balanced** (the go-bar, §19). Neutral controls = same problem/suffix, two neutral paraphrases (no flaw,
no option, no commitment, matched length); test spurious shift from arbitrary wording; **reported
separately from CATCH**. Reversed orientation = coherence check, grouped with its candidate, not an
independent point. **The set is a *screened feasibility set*, not a held-out test** — say so explicitly.

---

# PART VIII — ACTIVATION EXTRACTION & THE FAMILY-L READOUT MAP

## 15. Extraction, storage, and what carries the claim

Extract at the OFF state of §10 (after the final suffix token, before the answer token; no appended cue),
adapter **disabled**. Windows: **W8 (final 8, primary)** and **W20 (final 20, pre-registered
sensitivity)** — do not search windows post hoc. Layers `{9,18,27}` (verified in-distribution, §spec 3).

Store `H_A` (correct-step), `H_B` (flawed-step), `ΔH = H_B − H_A`, plus diagnostics: per-position L2
norms, `cos(H_A,H_B)`, `‖ΔH‖` by position, top-dimension energy. **Diagnostics are descriptive — PCA
separation (or not) is not proof of readable (or unreadable) signal** (spec §0).

**The Family-L readout map (differs from Family-A — the mentor's fix):**
- **`H_B` / `ΔH` (the flawed-step state) is the PRIMARY claim-carrier, not a leakage floor.** No option
  is named and the answer isn't computed yet; `H_B` is the *proximate cause* of catch/miss.
- **Its leak check (the experiment):** (i) **shuffle must kill any positive** (pair-*i* query with
  pair-*j* traces → base rate; surviving shuffle = surface-fingerprinting); (ii) **matched flaw
  structure** across CATCH/MISS (§6) so the probe can't cheat on the flaw text; (iii) **test `H_B` alone**
  alongside `ΔH` (drops the explicit "texts differ" component, may isolate the catch-state); **(iv) a
  matched supervised text-feature probe** (the activation probe must beat a supervised text probe, not
  only a zero-shot reader — review fix).
- **Measure the sign, don't assume it.** "Catch → larger ‖ΔH‖" is a guess (a catcher may converge
  flawed→correct late and invert it). The geometry presence-check reports the sign.

---

# PART IX — SPLITS

## 16. Feasibility vs SFT splits

Feasibility set = behaviorally screened, **not** a generalization test. For **SFT**, split **by source
problem** (never split wrong-option variants / orientations / template instantiations of the same problem
across train/test): ~70% train / 10% val / 20% held-out test, **and hold out entire flaw sub-types /
templates** (e.g. train sub-types 1–3, test sub-type 4). This creates two generalization axes: unseen
problems under known flaw-types, and unseen flaw-types. Do not claim broad generalization from held-out
problems under the same flaw-type only.

---

# PART X — SFT DATASET

## 17. Structured labels, shortcut tests, three ablations

**Labels** (structured, not free-form): `CATCH / MISS / SAME / UNCERTAIN`, derived from measured
distributions. Fixed query (above).

**Label-shortcut tests (before training):** check `P(y | flaw sub-type)`, `P(y | domain)`,
`P(y | difficulty tier)`, and train trivial classifiers on **sub-type-only / domain-only / text-only** —
none may be meaningfully above chance, else rebalance/regenerate.

**Three required ablations:** (1) **paired-activation SFT** (paired traces + query); (2) **no-activation
SFT** (same placeholders/tags/query/labels/schedule, activations omitted or fixed-noninformative); (3)
**text-context SFT** (both prompts + query, no activations). The claim requires **paired > no-activation
AND paired > text-context** on **held-out templates AND held-out problems**. Otherwise the system has not
demonstrated activation-dependent contrast reading.

---

# PART XI — DATA QUALITY GATES (all must hold before AO evaluation)

1. **Formatting:** suffix-position assertions pass; windows decode to intended suffix tokens; no
   generation header in the window; equal tokenized length.
2. **Behavior:** enough clean MISS and CATCH; groups overlap on neutral margin/difficulty; ON cross-check
   agrees with OFF labels; low invalid-rate; **every MISS/CATCH item ablation-verified load-bearing**.
3. **Text leakage:** dev gate not confidently directional; **reserved** baselines near chance on the
   frozen set; catch/miss not predictable from difficulty/domain alone; groups matched.
4. **Activation sanity:** adapter disabled for extraction, enabled only for AO readout; injected rows
   align to AO placeholder positions; raw ΔH diagnostics logged; any sink-dimension dominance reported
   before interpreting delta-only results.
5. **Orientation coherence (Family P):** behavior reverses appropriately under ordering; order not
   spuriously predictive; reversed orientations grouped with their candidate.

---

# PART XII — DECISION RULES

## 18. The existence gate is the adjudicator (we measure read-time vs reason-time)

Stage 1.5 = **probe on flawed-step `ΔH` (and `H_B` alone) → catch/miss, vs the dual text-only gate.** This
*measures* whether the catch/miss trace is read-time (separable in the pre-output state above text) or
reason-time (not). We do not argue it.

## 19. Continue / stop

**Continue to zero-shot AO iff:** ≥ **30–40 balanced** CATCH/MISS items survive **AND** a **large**
probe-vs-text gap **AND** the positive **survives shuffle**; reserved text baselines near chance; OFF
labels stable, ON-consistent; invariants pass. (n≈15 is too few — the AUC SE is too large.)

**Do not continue / report the negative if:** MISS only on systematically low-confidence problems; CATCH
only on obviously easy ones; text models reliably predict catch/miss; weak/flawed condition already
adopts the wrong answer for most; labels unstable; **or behavior changes only after long reasoning while
the study is restricted to pre-output states.** **Underpowered (< 30–40, or small gap) ≠ absent.**

**Continue to SFT only if** zero-shot shows paired evidence that **survives shuffle and the no-activation
check**, OR clear activation-sensitivity on the screened set, OR a documented reason the zero-shot null is
format-OOD (stable behavior + nonleaky data). **A zero-shot null alone does not justify SFT.**

---

# PART XIII — SCHEMA & FILES

## 20. Record schema (one per underlying candidate, pre-orientation)

```json
{
  "candidate_id": "gsm_hard_017_flawScope_subUnsupported",
  "family": "L",                                 // "L" (primary) or "P" (validation)
  "source_dataset": "omni_math",
  "source_item_id": "...", "domain": "...", "difficulty_tier": "hard",
  "flaw_subtype": "unsupported_assumption",      // L only
  "question": "...", "options": {"A":"...","B":"...","C":"...","D":"..."},
  "gold_option": "B", "flawed_consistent_option": "A",
  "context_correct": "...", "context_flawed": "...", "shared_suffix": "...",
  "ablation_verified_load_bearing": null,        // delete element -> under-determined?  REQUIRED true
  "token_audit": {"full_len_A": null, "full_len_B": null, "suffix_start_A": null,
                  "suffix_start_B": null, "suffix_positions": [], "window_positions": [],
                  "suffix_ids_match": null, "window_decoded": null},
  "target_config": {"model":"Qwen/Qwen3-8B","thinking_mode":"off","chat_template_version":null,
                    "dtype":"bfloat16"},
  "neutral_distribution": {}, "correct_distribution": {}, "flawed_distribution": {},
  "derived": {"delta_flawed_consistent": null, "neutral_gold_margin": null,
              "flawed_argmax": null, "label": null, "eligibility_reason": null},
  "on_cross_check": {"n_correct":32,"n_flawed":32,"correct_letter_counts":{},
                     "flawed_letter_counts":{},"invalid":0,"on_agrees_with_off":null},
  "text_only_gate": {"conservative_both": null, "matched_A_only": null,
                     "reader_nonqwen": null, "reader_qwen": null, "passes_gate": null},
  "status": "candidate"
}
```
Store activation tensors separately (path/content-hash). **Store discarded candidates too** — the discard
distribution tells you whether the family yields a continuum or a few brittle artifacts.

## 21. Files & execution order

```
data/  raw_source_items.jsonl  template_bank.json  candidates_unfiltered.jsonl
       candidates_measured.jsonl  feasibility_frozen.jsonl  sft_{train,valid,test}.jsonl
artifacts/  target_logits/  target_samples/  token_audits/  activations/  text_only_gates/  diagnostics/
scripts/  build_candidates.py  verify_token_invariants.py  ablation_verify.py  measure_target_logits.py
          sample_target_outputs.py  label_candidates.py  run_text_only_gate.py  match_catch_miss.py
          freeze_feasibility_set.py  extract_activations.py  inspect_raw_dh.py  build_sft_splits.py
```
Order: **(0) Family-P shakedown** end-to-end first → (1) load source bank → (2) build correct/flawed
candidates → (3) token + **ablation** verify → (4) measure neutral/correct/flawed logits →
(5) label catch/miss/discard → (6) ON cross-check on retained → (7) dual text-only gate (dev) →
(8) match CATCH/MISS → (9) freeze → (10) extract activations → (11) AO plumbing recheck → (12) **Stage 1.5
existence gate (probe vs text-only, shuffle)** → (13) decide AO → (14) grouped SFT splits → (15) three
ablations.

---

# PART XIV — REPORTING

## 22. Report every stage separately; never one aggregate

**Candidate-bank:** source count, raw count, invariant pass-rate, competence pass-rate, **ablation
pass-rate**, clean-MISS / clean-CATCH yield, discard rate, ON-disagreement rate. **Feasibility set:**
MISS/CATCH/neutral counts, domains, difficulty tiers, flaw sub-types, neutral-margin distributions,
**reserved** text-baseline performance, **build-yield**. **AO:** independent baseline; `H_B`-alone; `ΔH`;
`[H_A;H_B]`; no-activation; **shuffle**; orientation coherence; per-cell **MISS / CATCH / NEUTRAL-min /
NEUTRAL-max**; confusion matrices; raw log-prob margins. **A method that identifies the flaw but fails
CATCH is a manipulation-presence detector, not a disposition reader.**

---

## 23. Final operational rule

> The data is ready only when: the flaw is present in both paired prompts (correct in A, wrong in B);
> every item is **ablation-verified load-bearing** (deleting the element → under-determined, no sidestep);
> CATCH and MISS are **measured** (not assumed) and **matched** on text-visible difficulty and baseline
> margin, with **matched flaw structure**; the OFF labels are attached to the **exact pre-output state**
> from which activations are extracted; and **both** text-only readers (conservative + matched-A-only)
> cannot reliably infer catch/miss. If any clause is false, do not interpret AO performance as evidence
> about contrastive activation reading.

---

## Implementation map (this repo)

| Protocol step | File |
|---|---|
| Family-L bank (invalid-inference, ablation-safe) | `cao/family_l.py` |
| Family-P shakedown (commitment) | `cao/family_p.py` |
| Shared suffix / record schema / counterbalancing | `cao/dataset.py` |
| §10 OFF no-cue readout | `cao/measure.py:answer_distribution_nocue` |
| (2) build candidates → JSONL | `scripts/build_candidates.py` |
| (3) token invariants | `scripts/verify_token_invariants.py` |
| (4) measure neutral/correct/flawed | `scripts/measure_target_logits.py` |
| (3/5) ablation + Stage A–D label | `scripts/label_candidates.py` |
| (7) dual text-only gate | `scripts/run_text_only_gate.py` |
| (8/9) match + freeze | `scripts/match_catch_miss.py` |
| (10) extract activations (W8/W20) | `scripts/extract_activations.py` |
| (12) Stage 1.5 existence gate | `scripts/stage1_5_probe.py` + `cao/probe.py` |
