"""
dataset.py — shared scaffolding for the contrastive-AO study (spec v4 / data_construction v2).

Holds what both item families and every build/screen script share:
  - the user-turn assembly: context + SEP + shared suffix (invariant 3: manipulation in context only),
  - the §10 answer commitment that lives INSIDE the suffix ("Answer with exactly one letter:") so the
    extracted pre-output state is the state we label — we never append "The answer is (" after
    extraction,
  - option-block formatting and deterministic option-position counterbalancing,
  - the §20 candidate record schema (measured fields default to null; filled on the GPU box).

The suffix (question + options + cue) is IDENTICAL across a pair's conditions; only the context
differs. Token-length match of the contexts is the invariant verified on the REAL tokenizer by
scripts/verify_token_invariants.py (invariant 1); this module is tokenizer-free so it imports anywhere.
"""

from __future__ import annotations

SEP = "\n\n"
LETTERS = ("A", "B", "C", "D")
ANSWER_INSTRUCTION = "Answer with exactly one letter:"


def options_block(options: dict) -> str:
    return "\n".join(f"({L}) {options[L]}" for L in LETTERS)


def make_suffix(question: str, options: dict) -> str:
    """Shared suffix = question + options + the in-suffix answer commitment (§10, no post-hoc cue)."""
    return f"{question}\nOptions:\n{options_block(options)}\n{ANSWER_INSTRUCTION}"


def user_content(context: str, suffix: str) -> str:
    """User-turn text. context='' (the ablated/bare condition) yields the suffix alone."""
    return f"{context}{SEP}{suffix}" if context else suffix


def rotate_options(options: dict, gold: str, flawed: str, shift: int):
    """Cyclically shift option *positions* by `shift`; return (new_options, new_gold, new_flawed).

    Permuting the letter->text mapping preserves correctness (gold/flawed are relabelled to follow
    their text), so this is a safe, deterministic way to counterbalance which letter is gold /
    flawed-consistent across the bank without changing any item's meaning. Text at position p moves
    to position p+shift.
    """
    new = {L: options[LETTERS[(LETTERS.index(L) - shift) % 4]] for L in LETTERS}
    ng = LETTERS[(LETTERS.index(gold) + shift) % 4]
    nf = LETTERS[(LETTERS.index(flawed) + shift) % 4]
    return new, ng, nf


def balance_positions(items: list[dict], gold_key="gold_option", flawed_key="flawed_consistent_option",
                      opts_key="options") -> list[dict]:
    """Return copies of `items` with options rotated to JOINTLY balance gold AND flawed positions.

    A single rotation moves gold and flawed together (their offset is fixed per item), so the two
    marginals can't be set independently. We greedily pick, per item, the shift (of 4) that minimises
    the running sum-of-squares of both position histograms — spreading gold and flawed across all four
    letters (so no letter, e.g. D, is a permanent distractor, and 'miss' can't be read off a fixed
    option position). Deterministic; rotation is correctness-preserving so semantics are untouched.
    """
    gold_c = {L: 0 for L in LETTERS}
    flaw_c = {L: 0 for L in LETTERS}
    out = []
    for it in items:
        best = None
        for shift in range(4):
            new_opts, ng, nf = rotate_options(it[opts_key], it[gold_key], it[flawed_key], shift)
            cost = (sum((gold_c[L] + (L == ng)) ** 2 for L in LETTERS)
                    + sum((flaw_c[L] + (L == nf)) ** 2 for L in LETTERS))
            if best is None or cost < best[0]:
                best = (cost, new_opts, ng, nf)
        _, new_opts, ng, nf = best
        gold_c[ng] += 1
        flaw_c[nf] += 1
        out.append({**it, opts_key: new_opts, gold_key: ng, flawed_key: nf})
    return out


# --------------------------------------------------------------------------- #
# §20 candidate record schema (one per underlying candidate, pre-orientation)
# --------------------------------------------------------------------------- #
def candidate_record(candidate_id: str, family: str, question: str, options: dict,
                     gold_option: str, flawed_consistent_option: str | None,
                     context_correct: str, context_flawed: str, shared_suffix: str,
                     *, source_dataset="hand_authored", source_item_id=None, domain=None,
                     difficulty_tier="authored", flaw_subtype=None, context_ablated=None,
                     thinking_mode="off", extra=None) -> dict:
    """Build a §20-schema record with measured fields defaulted to null (filled on the GPU box)."""
    rec = {
        "candidate_id": candidate_id,
        "family": family,                                  # "L" (primary) or "P" (validation)
        "source_dataset": source_dataset,
        "source_item_id": source_item_id,
        "domain": domain,
        "difficulty_tier": difficulty_tier,
        "flaw_subtype": flaw_subtype,                      # L only
        "question": question,
        "options": dict(options),
        "gold_option": gold_option,
        "flawed_consistent_option": flawed_consistent_option,
        "context_correct": context_correct,
        "context_flawed": context_flawed,
        "context_ablated": context_ablated,
        "shared_suffix": shared_suffix,
        "ablation_verified_load_bearing": None,            # REQUIRED true after ablation_verify
        "token_audit": {"full_len_A": None, "full_len_B": None, "suffix_start_A": None,
                        "suffix_start_B": None, "suffix_positions": [], "window_positions": [],
                        "suffix_ids_match": None, "window_decoded": None},
        "target_config": {"model": "Qwen/Qwen3-8B", "thinking_mode": thinking_mode,
                          "chat_template_version": None, "dtype": "bfloat16"},
        "neutral_distribution": {},                        # the ablated/bare-givens condition
        "correct_distribution": {},                        # context_correct condition
        "flawed_distribution": {},                         # context_flawed condition
        "derived": {"delta_flawed_consistent": None, "neutral_gold_margin": None,
                    "flawed_argmax": None, "label": None, "eligibility_reason": None},
        "on_cross_check": {"n_correct": 32, "n_flawed": 32, "correct_letter_counts": {},
                           "flawed_letter_counts": {}, "invalid": 0, "on_agrees_with_off": None},
        "text_only_gate": {"conservative_both": None, "matched_A_only": None,
                           "reader_nonqwen": None, "reader_qwen": None, "passes_gate": None},
        "status": "candidate",
    }
    if extra:
        rec.update(extra)
    return rec
