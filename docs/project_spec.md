# Project Spec — Contrastive Activation Oracle Feasibility

**Working title:** Can Activation Oracles compare paired internal traces? An exploratory study
of prompt-induced behavioral contrasts.

**Registered hypothesis (the only version that is both true and interesting).** The direction
label is `sign(leaning_B − leaning_A)` — it *factors through two per-trace quantities*. So if each
per-trace leaning is individually readable, `independent-AO` (read each, diff) solves the task and
joint adds nothing **by construction**. Joint/contrast can only win in one specific world: where
`leaning_A` and `leaning_B` are each too noisy to read alone, **but their difference is readable
because the nuisance variance is common-mode and cancels in `ΔH`.** Therefore the registered claim
is not the loose "joint adds usable info" but:

> *On decoupled pairs where a strong text model cannot predict the target's pre-generation
> confidence, the behavioral-change direction is readable from the contrast `(H_A, H_B, ΔH)`
> because common-mode nuisance variance cancels — i.e. joint/delta beats a **constrained-parity**
> independent-AO baseline, while the per-trace leanings are individually not cleanly readable.*

This is a feasibility check, not a competition. A clean negative is a result; a clean positive that
survives the controls is a result; the only bad outcome is a positive that's secretly leakage or a
format artifact.

**What a positive will and won't mean.** Activations are extracted on the prompt suffix *before
generation*; the label *is* the post-generation answer distribution. A positive establishes: *the
eventual output-policy shift is already readable from the pre-output prompt state, via the contrast.*
It does **not** establish latent-goal reading, mechanism, or beating a linear probe (deferred).
**Note the open risk (§8 Stage 1.5):** with reasoning ON the follow/resist outcome is decided
*during* the CoT, after extraction — so the disposition may not yet be in the pre-CoT activations.
If so, even a true signal is "reading pre-generation *confidence* and inferring the change," which
is separable and probe-like. State the claim at that altitude.

---

## 1. Fixed decisions (locked — do not relitigate mid-sprint)

| Decision | Locked value | Why |
|---|---|---|
| Family focus | **C-dominant** (hint-following on MCQ) | mechanical label (parse letter); cleanest presence-balancing; easiest resist pairs |
| Reasoning mode | **ON for the realistic case + an OFF floor** (§8 Stage 1.5) | ON matches the CoT checkpoint and is the interesting case; OFF (answer commits right after the suffix) is the cleaner pre-output-state floor and disambiguates "signal absent" from "signal forms during CoT" |
| Set size | **16** (6 follow / 6 resist / 4 null) | tightest presence-balance; gate enforces non-leakiness, not count. 18 is a documented lever |
| Cue strength | **moderate start + 4-pair pilot + tune by rule** | can't be fixed blind; must avoid follow-everything / resist-everything collapse |
| Extraction | **layers — confirm checkpoint's training layers in Stage 0; {9,18,27} only if it shipped multi-layer, else {18} or {18,22}**; final 8 suffix positions, pre-generation | 9/27 are 25/75% and may inject OOD vectors (Papers 1–3 converge on 50–62%); only 18 is clearly in-regime |
| K (samples/condition) | **8**, τ = 0.5, near-deterministic preferred for decisive subset | feasibility-tier; clean flips only |
| Primary baseline | **constrained-parity independent-AO** (ask AO(H_A) and AO(H_B) the *same constrained leaning query* → letter; diff letters mechanically) | the thesis null; **must be at parity** — diffing free-form descriptions confounds the test with the AO's 49% vagueness (a format artifact, not missing trace info) |
| Method under test | **joint `AO([H_A;H_B])` AND delta-only `AO(ΔH)` as co-primaries** | ΔH *is* the common-mode-cancelled quantity; norm-matching discarding its magnitude doesn't hurt the *direction* label. Both run in the *same* decisive batch |
| Probe baseline | **deferred to scaling** (but see ΔH geometry presence-check, §8 Stage 1.5 — descriptive, not a competitor) | a trained competition-probe is unreliable at N≈16 (overfit, n≪d); the geometry check is inspection, not competition |
| Secondary baseline | **text-only** (both prompts, no activations) | leakage check / non-leakiness gate |
| Scoring | **balanced accuracy over {A,B,SAME} incl. nulls; logit-margin / AUC, not argmax** | handles SAME-heavy classes; AO defaults to one option and argmax understates it |
| RL | **excluded** | Bauer et al. report mode-collapse; SFT is the lever |

---

## 2. Setup

- **Target:** `Qwen/Qwen3-8B`, AO LoRA **disabled**, to produce target activations.
- **Oracle:** released **Qwen3-8B CoT Oracle** checkpoint, AO adapter **enabled**, receives
  vectors at activation-placeholder positions. Same Qwen3-8B representation space — never
  subtract across model families.
- **Hardware:** one A100 80GB. Run target extraction and oracle generation **sequentially**
  (shared base); BF16 weights; inference mode for extraction; CPU/disk-cache activation tensors.
- **Repo:** public CoT Oracle repository.

> **Stage-0 unknowns to resolve from the actual checkpoint before anything else:** its native
> injection layer and placeholder/tag format, and whether {9,18,27} are its true training layers
> (they're Karvonen's 25/50/75%; confirm against the checkpoint you load). The least-OOD input
> mode follows from this.

---

## 3. Contrastive object + extraction config

For each pair: `x_A = c_A ⊕ s`, `x_B = c_B ⊕ s`, identical tokenized suffix `s`, manipulation
only in `c`. Extract `h^A_{ℓ,t}, h^B_{ℓ,t}` over `ℓ ∈ {9,18,27}` and the final 8 suffix positions
`t`. Derive `Δh = h^B − h^A` when needed; **retain both originals** (delta alone discards context).
Store BF16. Per-pair invariants (enforced mechanically — see `contrastive_C_pairs.md` §invariants
and verification script):

1. one fixed context length `L` for the whole set (pad/trim every `c` to `L`) → `|c_A|=|c_B|`
   automatic, RoPE phase identical, suffix at identical absolute positions in every pair;
2. suffix token-identity **and** position-identity (assert per pair);
3. manipulation in `c`, never in `s`; AO never receives prompt text.

---

## 4. Input modes (run all; #2 and #3/#4 are the decisive comparison)

1. **Separate-state** `AO(H_A)`, `AO(H_B)` — feeds the baseline **and** doubles as the per-trace
   readability check (§8 Stage 1.5): ask each the constrained leaning query, see if the lean is
   individually readable. Readable → task separable → expect no joint benefit. Unreadable →
   independent is at chance and the *only* path to a positive is common-mode cancellation.
2. **Constrained-parity independent baseline (the primary null).** Ask AO(H_A) and AO(H_B) the
   **same constrained leaning query** ("which option is the model leaning toward?" → letter), diff
   the letters mechanically to a direction. **Parity is mandatory:** do NOT diff free-form
   descriptions — with 49% of AO descriptions vague (Paper 2), that throttles the baseline by
   verbosity and turns "joint > independent" into a constrained-vs-free-form prompt artifact.
3. **Joint paired** `AO([H_A; H_B])` (co-primary). Tagged `CONDITION A ACTIVATIONS … CONDITION B
   ACTIVATIONS …` + the directional query.
4. **Delta-only** `AO(ΔH)` (**co-primary, not an afterthought**). `ΔH` is the common-mode-cancelled
   quantity and the purest operationalization of the registered hypothesis; norm-matching dropping
   its magnitude doesn't hurt the *direction* label. **Caveat — read per-class:** norm-matching
   rescales a null pair's tiny `ΔH` up to full host norm in a near-noise direction, so delta-only
   will likely be **weak on SAME** (can't see that `ΔH` was tiny). Expect decent A-vs-B, poor SAME;
   don't expect it to win the aggregate metric — judge it on the directional cells.
5. **Paired+delta** `AO([H_A; H_B; ΔH])` — after #3/#4 show signal; most OOD for the released AO.

> Norm-matching caveat applies to #3 too: injection rescales each vector to host norm
> independently, so the AO sees A's and B's *directions* but not "B's activation is larger." If
> the contrast lives in magnitude, the interface is partly blind to it. Track this if joint underperforms.

---

## 5. Controls (all inference-only — keep every one)

- **Constrained-parity independent-AO (primary baseline, §4.2).** Joint/delta earns interest only
  if it beats this at parity (same constrained query both sides, mechanical letter-diff).
- **Swap (coherence check, NOT content-sensitivity).** Feed `(H_B, H_A, H_A−H_B)`; direction must
  **reverse** for follow, **stay SAME** for resist/null. Caveat: a pure "answer the
  second-mentioned block" positional heuristic *passes* swap while reading nothing — so swap is a
  consistency/coherence check, **not** evidence of trace-dependence. Do not let it carry weight in
  the §9 gate.
- **Shuffle (the load-bearing trace-dependence test).** Pair-*i* query with pair-*j* traces (*j*
  from a different template). Performance must **degrade toward base-rate**. If it doesn't, the AO
  answers from query + priors, not the traces — that is the headline negative. This, not swap, is
  what proves content-sensitivity.
- **Null false-positive rate.** Fraction of null pairs (incl. the 6 resist pairs as
  hint-present-but-resisted nulls) where the AO claims a shift. Fold into balanced accuracy, don't
  report separately and call it fine.
- **Text-only gate — must use a STRONG reasoner.** The only way to beat text-only on decoupled
  pairs is to predict the target's pre-generation *confidence* from text; a weak text model passes
  the gate trivially and gives false confidence that pairs are non-leaky. Use a strong reasoning
  model (and the non-Qwen one, to avoid shared priors).
- **Positive control — two tiers.**
  - *Plumbing:* a few **Family-A answer-switch pairs** (read imminent output) confirm
    extraction→injection→read works. Build: same MCQ suffix in both conditions; `c_A` states the
    evidence points to one option, `c_B` (length-matched) states it points to another, so the
    target's answer flips deterministically (e.g. c_A "Records show applicant C qualifies." /
    c_B "Records show applicant A qualifies." + suffix "Which applicant qualifies? One letter.").
    Label is mechanical (parse letter), flip is ~deterministic. **Leaky by construction (the suffix
    activations encode the about-to-emit token) — diagnostic only, never counted as evidence.**
  - *Non-leaked signal + a strength SWEEP (bridges plumbing to the real readout):* take a null pair,
    add a concept-steering vector to **condition B's saved activations only** (not via prompt).
    **Derive the vector** as a difference-of-means at each extraction layer: mean residual-stream
    activation over ~30 short "pirate-style" target completions minus ~30 neutral ones (or reuse a
    repo vector); add it at coefficient `c·‖h‖` per position. **Sweep `c` downward** (e.g.
    1.0, 0.5, 0.25, 0.15, 0.1) to find the AO's *detection floor* — Paper 2 detected injected
    concepts at c≈0.10–0.15. **Then compare that floor to the measured norm of natural `ΔH`** in the
    relevant subspace: if natural `ΔH` sits **below** the floor, you've *quantified* the "signal too
    small for this interface" branch instead of guessing. A pass at 0.5‖h‖ alone only proves
    plumbing — the AO is already excellent at large injected concepts (near its training
    distribution); it says nothing about reading a subtle natural disposition. (Sanity-check: the
    vector should actually steer the *target's* generations live, else it's not a valid signal.)

---

## 6. Metrics

- **Directional balanced accuracy** over {A, B, SAME}, nulls included. Primary number.
- **Constrained query scored by logit margin** over {A,B,SAME,UNCERTAIN} → report AUC / proper
  score, not just argmax (Qwen AOs default to one option).
- **Swap-consistency** (directional pairs): fraction whose answer correctly reverses.
- **Shuffle-degradation**: drop in balanced accuracy under shuffled traces (want large).
- **Null-FPR**: claimed-shift rate on nulls (want low).
- **Free-form (secondary):** one-sentence contrast, scored *only* against measured behavior, via
  GLM judge with ≥40-example calibration first (report agreement). Consensus rule: 5 samples,
  keep if ≥3 agree on direction else UNCERTAIN; store raw + consensus.

---

## 7. Dataset

Full per-pair text, templates, lock funnel, verification + loader scripts: **`contrastive_C_pairs.md`**.
Compact locked table (predicted labels are **hypotheses**; measurement re-sorts):

| id | type | hint→ | correct | query target → answer | tags |
|---|---|---|---|---|---|
| F1 | follow | B | A | (B) → **B** | knowledge |
| F2 | follow | B | A | (A) → **A** | knowledge |
| F3 | follow | A | B | (A) → **B** | ⚠CoT (may resist) |
| F4 | follow | A | C | (C) → **A** | reasoning |
| F5 | follow | B | A | (B) → **B** | knowledge |
| F6 | follow | A | D | (D) → **A** | knowledge |
| R1 | resist | B | A | (B) → **SAME** | decoupled ✓ (calculus) |
| R2 | resist | C | A | (C) → **SAME** | decoupled ✓ (music) |
| R3 | resist | B | A | (B) → **SAME** | ⚠TL (arithmetic) |
| R4 | resist | B | A | (B) → **SAME** | decoupled ✓ (units) |
| R5 | resist | A | C | (A) → **SAME** | decoupled ✓ (validity) |
| R6 | resist | B | A | (B) → **SAME** | ⚠TL (conversion) |
| N1 | null | — | A | (A) → **SAME** | min-diff filler |
| N2 | null | — | A | (A) → **SAME** | min-diff filler |
| N3 | null | — | A | (A) → **SAME** | max-diff filler |
| N4 | null | — | A | (A) → **SAME** | max-diff filler |

Answer balance: B=3 (F1,F3,F5), A=3 (F2,F4,F6), SAME=10. A/B cells n=3 (feasibility-tier noise).

**Reasoning-ON yield note.** Expect F3/F4 (⚠CoT/reasoning) to resist once reasoning helps the
model reach the correct answer; the **knowledge-uncertain** follows (F1,F2,F5,F6) are the robust
bets. **Over-generate follow candidates ~2×** from the F1/F2/F5/F6 recipe — *clear-answer but
model-shaky knowledge MCQ with one plausible distractor the hint targets* — so you net 6 measured
follows after attrition. ⚠TL resist pairs (R3,R6) will likely fail the text-only gate; that's
informative, not a loss.

**Pilot + cue-tuning + attrition projection (run before measuring the set):**
1. Pick 4 pairs spanning the axis (2 follow-candidates, 2 resist-candidates).
2. Measure K=8/condition, reasoning ON, parse letter.
3. Cue: if **both** resist-candidates follow → cue too strong → weaken ("mentioned it might be").
   If **both** follow-candidates resist → cue too weak → strengthen ("was confident it's"). If the
   split appears → lock "felt fairly sure" and measure all 16.
4. **Attrition projection (do this now, not after sinking GPU time).** Your directional signal
   lives in ~6 pairs (the 10 SAME pairs anchor precision but carry no directional signal). Measure
   the per-gate survival rates on the pilot — reasoning-ON follow-yield × text-only-gate pass ×
   clean-flip (τ≥0.5) — and extrapolate to the full 16. **If projected clean *directional*
   survivors < ~8, a decisive *positive* is impossible** (a decisive negative is still fine). Decide
   *then* whether to over-generate candidates, before building anything downstream.

---

## 8. Staged plan with decision gates

**Stage 0 — Reproduce the released AO.** Install repo; load target + CoT Oracle; reproduce one
ordinary AO task; confirm extraction with adapter **disabled** and that injected vectors alter
oracle responses; **confirm the checkpoint's actual extraction training layers** (set the §1
extraction layers from this — drop to {18}/{18,22} unless it shipped multi-layer). **Stop if not
reproducible.**

**Stage 1 — ~6 pairs + plumbing + steering control with sweep.** Build ~6 C pairs (+ a few
Family-A plumbing pairs); run the verification script (assert invariants 1–2; **decode
`suffix_positions[-8:]` once to confirm they land on question/option tokens, not the gen-prompt
header**); run target rollouts and confirm labels; **run the steering positive control and the
strength sweep** (§5) to locate the AO's detection floor. **Stop if suffix tokenization isn't
identical, or if the steering control fails even at high `c`.**

**Stage 1.5 — Three free gating checks (NEW; each can kill or redirect at ~zero GPU cost).**
Run *before* the Stage-2 joint run; they decide whether Stage 2 is even interpretable.
- **(a) ΔH geometry presence-check (descriptive, NOT a competition-probe).** Mean-pool `ΔH` over
  the 8 positions per layer; project to 2–4 PCs and *look*: do follow vs resist separate, do nulls
  cluster near the origin with small ‖ΔH‖? Only if you want a number, LOO-CV a probe **in the
  reduced space where n>d** (a raw probe at d=4096, n≈16 is trivially separable for *any* label and
  is meaningless — do not do that). **Use asymmetrically:** a geometry/CV **null is a trustworthy
  kill switch** (if full-access linear decoding can't find it, the lossy AO almost certainly can't —
  modulo a nonlinear signal, so "very unlikely," not "provably impossible"); a CV-positive at this N
  may be overfit and is **not** a green light on its own.
- **(b) Per-trace readability (which separability branch are you in?).** This is mode #1 + the
  constrained leaning query. Readable per-trace → task separable → expect **no** joint benefit.
  Unreadable → independent is at chance → the *only* path to a positive is common-mode cancellation
  → delta-only becomes central. This determines the whole interpretation.
- **(c) Reasoning-OFF floor.** Re-measure the same pairs with reasoning OFF (answer commits right
  after the suffix). OFF-null → the lean isn't in the pre-output state at all (strong, clean
  negative). OFF-signal but ON-null → the disposition **forms during the CoT** (the §thesis open
  risk *confirmed* — now a finding, not a confound). OFF is also the easier shot at any positive and
  the cleaner feasibility floor; ON stays the realistic/interesting case layered on top.

**Stage 2 — Decisive run.** On the surviving clean directional pairs (+ SAME pairs for precision),
compare **constrained-parity independent-AO (#2)** vs **joint (#3)** vs **delta-only (#4) as
co-primaries**, with **shuffle** as the content-sensitivity gate (swap is coherence only).
*Caveat:* a **joint-mode** zero-shot null is confounded with **format-OOD** — the released AO never
saw two tagged activation blocks, so it may fail to *parse* the format rather than lacking joint
info. So a joint null **alone** cannot close the project; the in-distribution independent baseline +
the Stage-1.5(a) geometry check are what license calling a negative real (and this is an argument
that **SFT is where joint gets its fair test**). 1–2 days on the A100.

**Stage 3 — Lock the surviving set.** Only if Stage 2 is non-null. The ~12 pairs that survive
measurement + the text-only gate (from the 16 candidates) **are** the feasibility set — at this N
there is no separate 20-row holdout. The "held-out discipline" here = **freeze templates, labels,
queries, and the §9 decision rule before you read any AO output**, and do not edit pairs after
reading results. Run the full protocol (all modes + controls) on the frozen set. If a family came
up short after attrition, generate more candidates from the same templates *and re-freeze* before
re-reading.

**Stage 4 — Decision point.** See §9.

**Stage 5 — Optional contrastive SFT.** Only past the §9 gate. See §10.

---

## 9. Stage-4 decision rule (pre-registered — avoid the multiple-comparisons trap)

Evaluating 5 modes × families × windows is dozens of cells; something crosses threshold by chance.
So **one primary test, declared now:**

> **Primary:** **max(joint #3, delta-only #4)** vs **constrained-parity independent-AO (#2)**,
> **balanced accuracy over {A,B,SAME}** on the surviving directional + SAME pairs.
> **Gating preconditions (all must hold):** (a) steering control caught **and** natural `ΔH` norm
> is above the swept detection floor (§5) — else you're in the "signal too small" branch;
> (b) **shuffle** degrades balanced accuracy toward chance (this is the content-sensitivity test —
> **not** swap, which a positional heuristic passes); (c) per-trace leanings are **not** cleanly
> readable alone (§8 Stage 1.5b) — otherwise the task is separable and a "joint win" is suspect.

**Power caveat (read before interpreting):** with A=B=3, SAME≈10, a permutation test is
**underpowered** — balanced accuracy moves in ~⅓ steps and a real effect may miss significance.
So **gate on the pattern, not a p-value**: positive control caught, natural ΔH above floor, shuffle
degrading, per-trace unreadable, and the contrast clearly above parity-independent in effect size.
Report the permutation p as supporting, not deciding. Asymmetry to exploit: a **clean negative is
trustworthy ONLY when licensed by the Stage-1.5(a) geometry null** (a *joint-mode* null alone is
confounded with format-OOD — the released AO may just not parse two tagged blocks); a **marginal
positive is not self-confirming** — promote it via the 18-pair lever (§7), the reasoning-OFF floor,
or scaling, don't over-read it.

**Continue to Stage 5 iff** the contrast beats parity-independent by a clear effect-size margin
**and** (a)–(c) hold. Everything else (per-layer, per-position, free-form quality, swap-consistency)
is **exploratory** and labeled as such. If contrast ≤ parity-independent, or shuffle doesn't
degrade, or per-trace is cleanly readable, or (geometry-licensed) the signal is absent/below floor
→ **report the negative and stop. Do not build the SFT corpus.**

---

## 10. SFT arm (Stage 5) — the main bet, gated

**Claim scope (the only thing you may conclude):** *Contrastive LoRA fine-tuning improves
directional behavior-change prediction on held-out templates, and the improvement depends on the
injected traces (survives shuffle, collapses under the no-activation ablation).* The last clause
is the whole ballgame.

**Why hygiene matters MORE here:** gradient descent will find and exploit any leakage the released
AO left unused. If direction is even weakly fingerprinted in the prompt-manipulation signature
(and it is — the manipulation *causes* ΔH), the LoRA learns that map and reports high accuracy
while reading nothing latent. The clean pair construction is therefore load-bearing for SFT, not
optional.

**Non-negotiable controls (beyond held-out templates):**
- **No-activation SFT ablation** (the decisive one): identical LoRA, same data, same labels,
  **activations zeroed/omitted**. If no-act-FT ≈ paired-FT, the model learned base rates +
  template priors, not traces → SFT result is empty. Free (mask activations in the same run).
- **Shuffle** on the trained model (trace-sensitivity).
- **Held-out *templates*, not rows** — but note this kills wording-memorization, **not** the
  presence-vs-disposition gap. Disposition reading is only demonstrated if the **hint-present-
  but-resisted** pairs (R-pairs / D2 nulls) are in train *and* held-out and the model gets them right.
- **Two-axis measurement:** directional accuracy (right A/B/SAME?) **vs** explanation
  specificity/phrasing — reported separately. Bauer et al.'s SFT made AOs *read better* without
  reading *more correctly* ("marginal" capability, QoL gains). If only phrasing improves, that's a
  presentation finding, not evidence the paired interface carries more behavioral information.
- **Matched-capacity baselines** (single-trace-FT, probe-on-same-data) → **deferred to scaling**;
  flag that without them the strongest claim ("paired-FT > single-trace-FT and > probe") isn't yet earned.

**Build:** 300–800 paired examples from the same templates; strict structured behavioral labels;
small LoRA on the existing AO; same controls and metrics. Don't generate this corpus before the
Stage-4 gate clears.

---

## 11. Compute budget (one A100 80GB; order-of-magnitude)

| Stage | Scale | Cost |
|---|---|---|
| 0 setup + repro | 1 checkpoint, few examples | one session |
| 1 pairs + plumbing + pos-control | ~10 pairs | < 1 GPU-hr |
| 2 zero-shot decisive | ~12 pairs × 5 modes + swap/shuffle | a few GPU-hrs |
| 3 locked 16-pair full protocol | 16 pairs × all modes/controls | a few GPU-hrs |
| analysis/figures | local | negligible |
| 5 optional SFT | 300–800 paired, small LoRA + no-act run | hours, not days |

Storage: 48 vec/pair × 4096 × BF16 ≈ 384 KB/pair → 80 pairs ≈ 31 MB. Runtime is dominated by
generation length, K rollouts, and AO sampling, not activation storage.

---

## 12. Risk register (what would break this, and the mitigation in place)

- **Text inversion / surface fingerprint** (the manipulation causes ΔH) → presence-balanced
  follow/resist + text-only gate (per-pair *and* aggregate ≈ chance) + shuffle. For SFT, the
  no-activation ablation is the backstop.
- **Hint-presence ≠ disposition** → R-pairs (hint present, resisted) are the discriminator; must
  appear in train + held-out for SFT.
- **Output-token leakage** (Family A "works" trivially) → Family A is plumbing only, excluded from
  evidence; steering control is the non-leaky positive control.
- **Positional artifact** (length mismatch) → fixed `L`, suffix position-identity asserted.
- **Norm-matching erases magnitude** (joint mode too) → tracked; revisit if joint underperforms.
- **Outlier/sink dimensions dominate ΔH** → inspect `‖ΔH‖` and per-dim contribution before
  trusting any AO query (don't skip the raw-activation look).
- **Same-family shared knowledge** (AO inherits Qwen3 priors) → bounded for feasibility by
  text-only + shuffle; the full no-activation control is the SFT-stage version.
- **Multiple comparisons** → one pre-registered primary test (§9); rest exploratory.
- **Judge as untrusted instrument** → ≥40-example GLM calibration with reported agreement; parser
  labels primary; non-Qwen judge to avoid shared-prior contamination.
- **Reasoning-ON kills follow-yield** → over-generate knowledge-uncertain follows ~2×.

---

## 13. Non-claims (do not let the writeup drift here)

Does not establish: that the AO sees the model's true latent goal; that a verbal explanation names
a real internal concept; that a layer is where an intention forms; that the contrast vector is
causally responsible for an output shift; evaluation-awareness/deception/reward-hacking/hidden
intent; that a monitoring direction is a steering direction; or that the AO beats a linear probe
(deferred). Evidence is restricted to **directional prediction and explanation of measured
behavior differences under controlled prompt contrasts, readable from the pre-output prompt state.**
