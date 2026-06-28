# Activation Oracles — Knowledge Base

**Purpose.** Standing literature context for the contrastive-AO project. Load this so an
assistant (or a future you) understands the methods, math, datasets, and results of the three
foundational papers without re-deriving them. The *project plan* lives separately in
`project_spec.md`; the *dataset* in `contrastive_C_pairs.md`. This file is the *background*.

**Provenance.** Distilled and paraphrased from the three papers' text. All numeric results below
were grep-verified against the extracted text. Figures/plots were **not** re-rendered, so treat
any single number as "verify against the figure if it becomes load-bearing for a decision."

**How to read claim strength.** Throughout, results are tagged:
`[OBS]` narrow observation · `[PATTERN]` recurring across settings · `[SUPPORTED]` survives a
threatening baseline · `[INTERP]` proposed interpretation · `[SPEC]` untested speculation.

---

## 0. Glossary

- **Activation Oracle (AO):** an LLM (here LoRA-finetuned from an instruct model) trained to take
  another model's residual-stream activations as an *input modality* and answer natural-language
  questions about them. A learned verbalizer of internal state.
- **Target model:** the model whose activations are being read (here Qwen3-8B, adapter off).
- **Oracle / AO:** the reader model (here the Qwen3-8B CoT Oracle checkpoint, adapter on).
- **Injection:** writing target activations into the AO's residual stream at placeholder ("?")
  token positions.
- **Extraction layer:** layer in the *target* the activations are read from (≈50–62% depth).
- **Injection layer:** layer in the *AO* the vectors are written to (fixed, ≈layer 1–2).
- **Lens / context-prediction task:** self-supervised training where the AO predicts tokens
  before/after an activation span. Mechanically this is **trained text inversion**.
- **Splice / A-vs-C test:** hold the surface tokens at the probe fixed, vary the upstream state,
  test whether the AO's answer changes. The cleanest text-inversion control (from Paper 2).
- **Text inversion:** recovering the input tokens (or answer) from activations — the dominant
  confound, since most "reading" can be explained as paraphrasing decodable text.

---

## 1. Shared mechanism + math (common to all three papers)

An AO reads activations by **injection at placeholder positions**. For a placeholder with host
activation `h_i` (the AO's own residual stream at the injection layer) and injected target vector
`v_i`, the injection is **additive and norm-matched**:

```
h'_i = h_i + ||h_i|| · (v_i / ||v_i||)
```

- The injected term is unit-normalized then rescaled to the host norm `||h_i||`, so its magnitude
  is independent of `v_i`'s provenance (raw residual, activation *difference*, or SAE feature).
- Post-addition norm is bounded by `2||h_i||` (triangle inequality) → numerically stable.

**Why additive, not overwrite.** The original LatentQA recipe *replaced* `h_i` with `v_i`. Paper 1
reports this caused placeholder-position **norm growth of ~20× (inject at AO layer 0) to ~100,000×
(layer 1)** by the time activations reached the injection point → instability. Additive
norm-matching fixes it. `[SUPPORTED]` Paper 3 confirms overwrite/NLA-style injection is
consistently worse than additive.

**Two distinct layers — do not conflate:**
- *Injection layer (into AO):* fixed at ≈layer 1 (0-indexed), i.e. after the AO's 2nd block.
  Rationale: the 1st residual layer has anomalously low cosine similarity to neighbors; injecting
  after it lands the vector in a "stable basis." `[INTERP]` (not tested causally).
- *Extraction layer (from target):* 50% depth in Paper 1; Paper 3 finds ≈62% and multi-layer better.

**Injection strength (Paper 3 tweak):** scale the injected term by a coefficient `c ≈ 2`:
`h'_i = h_i + c·||h_i||·(v_i/||v_i||)`. Rationale: LMs weight tokens ≈equally by default, so a
single activation is rarely load-bearing; a larger norm overcomes that prior. `[INTERP]`

**Norm-matching has a cost (important for the project):** because each vector is rescaled to host
norm independently, the AO sees the *direction* of injected vectors but **not their relative
magnitude**. So "B's activation is larger than A's" is invisible. This bites delta-only inputs and
also paired `[H_A; H_B]` inputs if the behavioral signal lives in magnitude.

---

## 2. Paper 1 — Karvonen et al., "Activation Oracles" (arXiv 2512.15674)

**The method/foundational paper.** Trains AOs as general-purpose activation explainers and
evaluates on four auditing tasks.

### 2.1 Training data (~1M examples, ~65M tokens)
| Family | Count | What it is |
|---|---|---|
| Context-prediction ("lens") | **600K (~60%)** | predict J∈[1,50] tokens before/after a span of K∈[1,50] activations; input/target non-overlapping; half FineWeb / half LMSYS. **= trained text inversion** |
| Classification | **336K** | 7 binary datasets × 48K (Geometry-of-Truth, SST-2, MD-Gender, SNLI, NER, Tense, Relations), recast yes/no |
| System-Prompt QA (SPQA) | **64K** | from Pan et al. |

**Tell that matters:** the authors **upweight single-token classification inputs** "because many
classification tasks become solvable via text inversion when provided full-sequence context," and
add positional jitter to avoid brittle generalization. → They *know* text inversion is the
dominant channel and engineered training to suppress it. Yet the **secret-keeping evals feed the
full sequence** (more invertible). `[OBS]`

### 2.2 The four evaluations + baselines
1. **Secret-keeping** (Cywinski et al. organisms): Taboo (Gemma models hiding a word), User Gender,
   Secret Side Constraint (SSC). Baselines = Cywinski's manually-tuned agent pipelines (5 white-box
   + 5 black-box tools).
2. **Emergent-misalignment model-diffing** (Minder et al. scaffold): AO reads base−finetuned
   activation *difference* on first ~10 tokens; graded 1–5 by gpt-5-mini.
3. **PersonaQA** (Li et al.): models finetuned on 100 shuffled personas; prompt has only the name;
   AO reads attributes.
4. **OOD classification:** 7 held-out binary datasets, single-token inputs.

### 2.3 Results (verified numbers)
| Task | AO | Best baseline | Note |
|---|---|---|---|
| Taboo | **21%** | 4% (best white-box) | impressive ratio, low absolute |
| Secret Side Constraint | **59%** | 52% | answers post-processed by **GPT-4.1** (extra LLM in loop) |
| User Gender | **0.97** | — | uses last ~10 tokens (0.970→0.995) |
| Misalignment (1–5) | **2/5** | ties ADL | **ceiling ≈2/5**: organisms rarely misbehave (1/6 shows it) |
| PersonaQA open-ended | 5→**14%** (open), **35%** (Claude) | ~25% (related probe) | knowledge brittle |
| PersonaQA binary | **0.63** (Llama), **0.69–0.70** (Qwen) | source model ≈0.55 | **AO > the model it reads** |

### 2.4 Confounds / what it does NOT establish
- **Per-task token tuning** contradicts the "out-of-the-box, no scaffolding" framing (Taboo = single
  token; Gender = last 10; SSC = full seq, selected on validation). `[OBS]`
- **No linear-probe baseline anywhere** on the headline tasks.
- **Organisms are narrow finetunes**; Minder et al. (cited) showed narrow finetuning leaves
  **readable activation traces** → these tasks may be unusually easy for *any* method.
- **PersonaQA binary anomaly** `[OBS]→[INTERP]`: AO scoring higher than the source model on the
  source's own knowledge most parsimoniously means the constrained yes/no format lets the AO exploit
  its **own priors / label structure**, not that it "reads info the source can't use" (the paper's
  charitable reading). Binary PersonaQA cannot discriminate reading from guessing.
- **Best result uses a different method:** the Claude Haiku AO (35% PersonaQA) uses *overwrite*
  injection + *full* finetuning + *single* layer — not the paper's formalized recipe.

**Net:** `[SUPPORTED]` a diversely-trained verbalizer generalizes across readable-trace organism
tasks and beats specific prior pipelines. `[NOT ESTABLISHED]` activation-specific reading beyond
surface text or beyond a linear probe.

---

## 3. Paper 2 — Jakkli, Rajamanoharan, Nanda, "Current AOs Are Hard to Use on Safety-Relevant Tasks"

**The critique.** Five safety tasks on *unmodified* Qwen3 using the *released* AO, built so text
inversion can't produce the answer, and adding the probe + no-activation baselines Paper 1 omitted.
LessWrong: `lesswrong.com/posts/LXQBcztrWKhtcgQfJ/current-activation-oracles-are-hard-to-use`

### 3.1 The three text-inversion-controlled experiments (judge-free, most robust)
- **Number prediction:** ask what number Qwen3 will emit, from activations *before any answer
  token*. AO emits the same handful ("the number 10", "value of 12") across true answers spanning
  **−8369 to 909**; fails 1+1. Clean null. `[SUPPORTED]`
- **Missing-information splice (the key result):** condition A = complete prompt; C = incomplete
  prompt with A's tokens **spliced in**, so **identical surface tokens at the probe, different
  upstream state**. An activation-reader must separate them; the AO gives the **same A-vs-C
  classification in 53/54** cases and identifies the missing info in **0/54**. B-vs-C gap = 21 pts
  (what reading would close); observed A-vs-C gap = 2 pts. `[SUPPORTED]` **This is the benchmark the
  contrastive project is built around** ("same local text, different upstream state").
- **Sycophancy (Scruples flip-pairs):** AO best **AUC 0.60** vs a **dense linear probe 0.833** on
  the same activations (N=20, leave-one-out). The probe is the correct threatening baseline; **the
  AO loses to it.** `[SUPPORTED]`

### 3.2 Backtracking edit experiment + no-activation ablation
- **Backtracking funnel:** ~2,300 turns → ~10% judged correct-and-specific → ~5% after removing
  text-recoverable cases. On the 15-case slice where the AO had succeeded, edits to the *upstream*
  CoT (probe tokens held fixed) → AO updates correctly in 8, **all 8 restatements of a noun phrase
  from immediately upstream** ("last digit" appears two sentences before → AO says "uncertain about
  the result beyond the last digit"). Survivors indistinguishable from text inversion. `[SUPPORTED]`
- **No-activation ablation (CCP-censored topics):** run the AO with activations **omitted**. The
  no-activation AO **matches or beats** the with-activation AO (**7/9 vs 5/9**); text-only Qwen3 also
  ties at 5/9 with more specific descriptions. Since AO and target share the base, "read it from
  activations" is unidentifiable from "already knew it." → **Recommends no-activation AO as a
  mandatory baseline whenever AO and target are same-family.** `[SUPPORTED]` (the single most useful
  methodological fix across the three papers; patches Paper 1's biggest gap.)

### 3.3 Positive signals (their own)
- **Steering detection:** AO flags an *injected* concept-steering ("pirate") vector before it
  surfaces in text. Sweep: detect at c≈0.10–0.15, overt text at c≈0.5, collapse at c=1.0. Real but
  narrow, large-signal regime. `[SUPPORTED, narrow]`
- **Constrained 4-way MCQ over reasons:** 66% vs 47% (**N=15**). `[OBS]`
- **Previous-token prediction:** 0.45 vs 0.20 — but that's the *training objective*. Next-token
  smaller (0.38 vs 0.30). `[OBS]`

### 3.4 Its own weaknesses (hold symmetrically)
- **Vagueness 49%**, the backtracking funnel, and sycophancy scoring route through an **LLM judge
  (Claude Sonnet 4.5) with no reported inter-annotator agreement** — an unvalidated instrument by
  its own standard (acknowledged). Robust results are the judge-free ones.
- Several positive-direction results are N≈9–15 → directional only.
- Single recipe, single target family, no retraining → says nothing about *better* AOs.

---

## 4. Paper 3 — Bauer et al., "Building Better Activation Oracles" (arXiv 2606.02609)

**The follow-up.** Four ablated improvements + a new benchmark (AObench) + measurement-methodology
fixes that partly rebut Paper 2.

### 4.1 The four changes (each ablated, 50M-token budget)
| Change | Effect (chance-adjusted AObench) | Rationale |
|---|---|---|
| **Conversational dataset** (Sonnet splits a CoT; writes a question about the suffix hard to answer from prefix *text* but plausible from prefix *activations*) | +0.244 → **+0.310** (largest single step) | trains *out* text inversion at the data level |
| **Extraction layer** ≈62% (layer 22/Qwen3-8B), or 5 layers (21–25) | best on model-diffing | — |
| **On-policy lens** (CoT rollouts replace FineWeb) | +0.244 → +0.274 | predicting on-policy generation needs model state, not just prior text |
| **2× injection strength** | hallucination 79%→85% | single activation rarely load-bearing |
| **Full recipe** | **+0.244 → +0.435** | — |

Hallucination (separately reported): **68.8% → 84.6%**. Vagueness (chance-adj): 0.076 → 0.205.

### 4.2 AObench (11 tasks)
Activation Sensitivity / "Not Just Reading Tokens" (**= the critique's A-vs-C splice generalized**:
same tokens, different upstream context, does the answer change), Hallucination, Missing
Information, Sycophancy, Taboo, PersonaQA, MMLU-prediction, Number prediction, Vagueness, Domain
Confusion, Backtracking. **Building the splice test into a standard benchmark is the correct
institutional response to Paper 2.**

### 4.3 Measurement methodology (the most reusable part)
- **Use AUC not accuracy:** Qwen AOs default to "No"/one option, so fixed-threshold accuracy looks
  like chance while the logit margin carries signal. On a sycophancy-from-CoT task: **0.50 accuracy
  but 0.83 AUC.** `[SUPPORTED]` → **adopt logit-margin/AUC scoring in the project.**
- **Sweep the AO's context window:** backtracking 1.26/5 (final activation) → **2.10/5 (last 50
  tokens)**, above black-box. Default ≥20-token windows.
- **Consensus sampling:** Taboo, k=10, consensus≥0.8 → keep 19.4% at 94.3% precision.

### 4.4 Two tensions to carry (important)
- **The sycophancy "rebuttal" is not clean.** Bauer says Jakkli's near-chance sycophancy is "a
  calibration artifact" (0.50 acc / 0.83 AUC). But **Jakkli already reported AUC (0.60), not
  accuracy**, on its *flip-pair* construction; Bauer's 0.83 is on a *different* "sycophancy-from-CoT"
  task. Plausibly apples-to-oranges. And even at 0.83 it merely **equals** the 0.833 dense probe
  Jakkli used — i.e. the null Jakkli was worried about, not a refutation. `[OBS]`
- **The through-line, stated by Bauer in their own appendix:** when narrowly training on hard
  CoT-interp tasks, they could **match linear-probe performance but never significantly exceed it**,
  and recommend "aim to match probes." Capability gains "marginal"; real wins are **quality-of-life**
  (less vague, less hallucinatory). `[PATTERN]`

---

## 5. Cross-paper synthesis (the load-bearing summary)

**Graded state of the evidence:**
- `[SUPPORTED]` AOs detect a *deliberately injected* steering vector before it surfaces (large
  signal, model-organism regime). AOs do previous-token prediction (their objective). Paper 3's
  conversational swap + multi-layer extraction reduce vagueness/hallucination (judge-measured → soft).
- `[PATTERN]` **AOs match dense linear probes on activation-readable properties and never exceed
  them.** Three independent convergences: Jakkli 0.60 vs 0.833 (doesn't even match), Bauer "match,
  never exceed," Paper 1's missing probe baseline. **A verbalizer that never beats a linear probe on
  the same activations is an interface, not new information.** This is the single most important
  regularity, and it's unflattering.
- `[NOT ESTABLISHED / CONFOUNDED]` that AOs read activation-specific content beyond surface text on
  OOD safety tasks. The splice (53/54), no-activation (7/9≥5/9), constant numbers, and 8/8
  keyword-restatements all point the other way.

**Two structural confounds contaminate nearly every positive result:**
1. **Text inversion** — guaranteed by ~60% lens training; most "reading" = paraphrasing decodable text.
2. **Shared-family knowledge** — AO finetuned from the same base as the target, so the AO's priors
   are an uncontrolled channel ("read it" vs "already knew it" is unidentifiable).

**Discriminative tests none of the three actually run (where new work can bite):**
- (a) **Probe-vs-AO head-to-head on every AObench task**, same activations, same split. Decision: if
  the AO never clears a dense probe by more than seed noise on a text-inversion-controlled task, the
  "more useful than probes" claim is dead.
- (b) **Cross-family AO** (AO base ≠ target base) + the no-activation floor, to break the
  shared-knowledge channel.
- (c) **Generalize the A-vs-C splice onto Paper 1's own secret-keeping organisms** — do 21%/59%/97%
  survive holding surface tokens fixed, or are they readable-trace + text inversion?

---

## 6. Failure-mode catalogue (apply these to any AO experiment)

| Failure mode | What happens | Control |
|---|---|---|
| **Text inversion** | AO paraphrases decodable tokens, not state | splice/A-vs-C (same tokens, diff state); text-only baseline; balance surface ⊥ label |
| **Shared-family knowledge** | AO answers from inherited priors | **no-activation ablation** (omit activations, same everything); cross-family AO |
| **Output-token leakage** | "works" because the answer token is about to be emitted | extract *before* generation; exclude answer-switch tasks from evidence |
| **Presence-vs-disposition** | AO detects *that* a manipulation is present, not the model's response to it | include manipulation-present-but-ignored pairs (label = no change) |
| **Norm-matching magnitude loss** | injection rescales each vector → relative magnitude invisible | track when magnitude-carried signal underperforms; don't rely on delta-only |
| **Positional artifact** | length-mismatched contexts → RoPE phase differs → ΔH conflates content + position | fix one context length; assert suffix at identical absolute positions |
| **Outlier/sink dimensions** | a few massive-activation dims dominate ΔH | inspect ‖ΔH‖ + per-dim contribution before trusting any readout |
| **LLM judge as instrument** | unvalidated grader inflates/deflates rates | ≥40-example calibration w/ reported agreement; parser labels primary; non-same-family judge |
| **Multiple comparisons** | many cells (modes×tasks×layers) → chance crossings | one pre-registered primary test; rest exploratory |

---

## 7. Connection to the contrastive-AO project

**The bet:** give the AO *paired* traces `(H_A, H_B, H_B−H_A)` from one target under two controlled
conditions sharing an identical suffix, and ask for the *direction* of a measured behavioral change
— then SFT it to do so. Directly motivated by the **"same local text, different upstream state"**
failure (Paper 2's splice, §3.1): the contrastive framing pairs activations instead of labeling
them in isolation, precisely the regime where single-trace AOs collapse (53/54).

**Sharpened registered hypothesis (the only true+interesting version).** The direction label is
`sign(leaning_B − leaning_A)`, which factors through two per-trace quantities — so if each
per-trace leaning is readable, independent-AO solves the task and joint adds nothing *by
construction*. Joint/contrast can only win where each leaning is individually too noisy to read but
**the difference is readable because nuisance variance is common-mode and cancels in `ΔH`**. Two
consequences: (i) the independent baseline must be at **constrained parity** (same letter query
both sides, mechanical diff) — diffing free-form descriptions confounds the test with the AO's 49%
vagueness; (ii) **delta-only is a co-primary, not an afterthought** — `ΔH` *is* the cancelled
quantity, and norm-matching dropping its magnitude doesn't hurt the *direction* label (though it
cripples SAME detection — read delta-only per-class).

**Open risk that caps the claim:** with reasoning ON, the follow/resist outcome is produced *during*
the CoT, after pre-generation extraction — so the disposition may not be in the activations yet
(Paper 2's number-prediction null in a new costume). A reasoning-OFF floor disambiguates "signal
absent" from "signal forms during CoT." If it's the latter, the honest claim is "reading
pre-generation *confidence* and inferring the change" — separable and probe-like.

**Literature → design decisions (all in `project_spec.md`):**
- AOs never beat probes → trained competition-**probe deferred** at feasibility N; primary baseline
  is **constrained-parity independent-AO**. (A *descriptive*, dimensionality-aware `ΔH` geometry
  check — PCA + look, or LOO-CV only in reduced space where n>d — is used as a kill switch, since a
  raw probe at d=4096/n≈16 is trivially separable and meaningless. A geometry **null** is
  trustworthy; a positive at this N is not.)
- Text inversion + presence-vs-disposition → **presence-balanced follow/resist C pairs**; the
  hint-present-but-resisted pairs are the disposition discriminator. Text-only gate must use a
  **strong reasoner** (the only way to beat it is to predict confidence from text).
- Shared-family knowledge → **no-activation SFT ablation is the decisive control**.
- Output-token leakage → **answer-switch (Family A) is plumbing only, never evidence**; the
  **steering-injection control + a strength sweep** locate the AO's detection floor, which is then
  compared to natural `ΔH` norm to test the "signal too small for this interface" branch.
- AUC > accuracy (Paper 3) → **logit-margin/balanced-accuracy scoring**, not argmax.
- Content-sensitivity is carried by **shuffle, not swap** (a positional "answer the second block"
  heuristic passes swap while reading nothing).
- A **joint-mode zero-shot null is confounded with format-OOD** (the released AO never saw two
  tagged blocks) → can't alone close the project; this is an argument that **SFT is where joint gets
  its fair test**.
- Multiple comparisons → **one pre-registered primary test**: max(joint, delta) vs parity-independent.

**Scope discipline (the non-claims):** a positive shows *the eventual output-policy shift is
readable from the pre-output prompt state, via the contrast* — **not** latent-goal reading, not
mechanism, not beating a probe. Don't let the writeup drift past that sentence.

---

## 8. One-line citations
- **Paper 1** — Karvonen et al., *Activation Oracles*, arXiv 2512.15674. Method/foundational.
- **Paper 2** — Jakkli, Rajamanoharan, Nanda, *Current AOs Are Hard to Use on Safety-Relevant Tasks*. Critique. (LessWrong link above.)
- **Paper 3** — Bauer, De Schamphelaere, Karvonen, Luick, Nanda, *Building Better Activation Oracles*, arXiv 2606.02609. Follow-up.
