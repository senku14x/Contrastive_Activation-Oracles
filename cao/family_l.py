"""
family_l.py — Family L (load-bearing content flaw): the PRIMARY claim-carrier bank.

THE STRUCTURE (worked out from the ablation rule, spec_v4 §4 / data_construction_v2 §4):
A flaw must be BOTH load-bearing (delete it -> the problem is under-determined) AND catchable
(the model can tell it's wrong). A bare premise swap is neither catch nor miss — the model just
follows whichever premise is given. The ONLY structure that satisfies both at once is an
INVALID INFERENCE on a genuinely under-determined problem:

  givens (fictional, shared)         : insufficient to fix a definite answer on their own
  context_correct (c_A)              : a worked step that correctly notes the info is insufficient
  context_flawed  (c_B)              : a worked step that commits a subtle FALLACY (affirming the
                                       consequent, illicit conversion, unsupported bridge, ...) to
                                       reach a DEFINITE (wrong) conclusion
  context_ablated (c_0)              : the givens alone, no worked step

  gold_option              = "cannot be determined" (the honest answer; what a competent reader gives
                              on c_0 and c_A)
  flawed_consistent_option = the fallacy's definite conclusion
  label (measured on c_B)  = CATCH (answers gold, i.e. holds the line at "cannot be determined")
                           vs MISS  (follows the fallacy to flawed_consistent_option)

Why this is the right structure:
  - LOAD-BEARING: removing the worked step -> givens only -> under-determined (gold). The definite
    wrong answer is NOT reachable without the fallacious step, so the model cannot sidestep it by
    independent computation (the lesson the arithmetic pilot taught).
  - CATCHABLE BY RECOGNITION (not reconstruction): catching = recognising the step is an unjustified
    leap; it needs no recomputation. Recognition is plausibly a read-time process -> the best case
    for the pre-output readout (spec §1 "the squeeze").
  - NOVEL FICTIONAL scenarios -> no real-world recall shortcut, no canonical-syllogism form-prior.

KNOWN LIMITATION (documented, not hidden): gold is structurally the "cannot be determined" option, so
gold is text-identifiable. That is fine for THIS study: the probe/AO predict CATCH vs MISS (did the
model follow the fallacy), which is orthogonal to which option is gold. The binding requirement is
that CATCH/MISS not be text-legible — enforced by the dual text-only gate (run_text_only_gate.py),
not by hiding gold. Gold-identifiability also cancels in ΔH (the options sit in the shared suffix).

predicted CATCH/MISS is NOT pre-annotated — it is MEASURED on the target (label_candidates.py).
gold/flawed positions are counterbalanced across A–D at load (dataset.balance_gold_positions).
"""

from __future__ import annotations

from cao import dataset as D


def _L(slug, subtype, domain, givens, step_correct, step_flawed, question, options, gold, flawed):
    """c_A/c_B share `givens`; they differ only in the worked step (correct vs fallacious).
    c_0 (ablated) is the givens alone. Keep the two steps close in length (token match on Colab)."""
    return {
        "slug": slug, "flaw_subtype": subtype, "domain": domain,
        "context_correct": f"{givens} {step_correct}",
        "context_flawed": f"{givens} {step_flawed}",
        "context_ablated": givens,
        "question": question, "options": dict(options),
        "gold_option": gold, "flawed_consistent_option": flawed,
    }


_RAW = [
    _L("L01_ballot", "unsupported_assumption", "civic eligibility",
       "In Brevin, every registered guild member was sent a ballot. Sara is an eligible resident of Brevin.",
       "Eligibility does not by itself imply guild registration, so her ballot status is unsettled.",
       "Eligible residents are registered guild members, so she was sent a ballot.",
       "Based on the information above, was Sara sent a ballot?",
       {"A": "Yes, she was sent a ballot", "B": "No, she was not sent a ballot",
        "C": "It cannot be determined from the information given", "D": "Only if she requested one"},
       "C", "A"),

    _L("L02_locker", "affirming_consequent", "club membership",
       "At Club Aral, all premier members receive a reserved locker. Devi has a reserved locker.",
       "Non-premier members can also be assigned lockers, so her tier cannot be inferred.",
       "Since she holds a reserved locker, Devi must therefore be a premier member.",
       "Based on the information above, is Devi a premier member?",
       {"A": "Cannot be determined from the information given", "B": "Yes, she is a premier member",
        "C": "No, she is not a premier member", "D": "Only during peak season"},
       "A", "B"),

    _L("L03_token", "unsupported_assumption", "courier guild",
       "Every Tarn courier carries a blue token. Olen is not a Tarn courier.",
       "Non-couriers may still happen to carry blue tokens, so Olen's case is unsettled.",
       "Only couriers are ever issued blue tokens, so Olen does not carry one.",
       "Based on the information above, does Olen carry a blue token?",
       {"A": "Yes, he carries one", "B": "No, he does not carry one",
        "C": "It cannot be determined from the information given", "D": "Only on delivery days"},
       "C", "B"),

    _L("L04_vendor", "illicit_conversion", "supplier taxonomy",
       "In the Hale system, every verified vendor is also a listed supplier. Maro is a listed supplier.",
       "Listed suppliers are not necessarily verified vendors, so Maro's status is open.",
       "Listed suppliers are exactly the verified vendors, so Maro is a verified vendor.",
       "Based on the information above, is Maro a verified vendor?",
       {"A": "No, he is not a verified vendor", "B": "It cannot be determined from the information given",
        "C": "Yes, he is a verified vendor", "D": "Only if recently audited"},
       "B", "C"),

    _L("L05_batch", "affirming_consequent", "factory process",
       "At the Renn plant, batches that fail inspection are always rerun. Batch 7 was rerun.",
       "Batches are also rerun for scheduling reasons, so we cannot say batch 7 failed.",
       "Since batch 7 was rerun, it must therefore have failed inspection.",
       "Based on the information above, did batch 7 fail inspection?",
       {"A": "Yes, it failed inspection", "B": "No, it passed inspection",
        "C": "It cannot be determined from the information given", "D": "Only its first run failed"},
       "C", "A"),

    _L("L06_combo", "composition", "invented card game",
       "In Skein, a 'combo' requires that the hand contain a matched pair. Each of Pim's cards is individually legal.",
       "Each card being legal does not guarantee a matched pair, so a combo is not established.",
       "Each card is legal, so the whole hand counts as a valid combo.",
       "Based on the information above, does Pim's hand form a combo?",
       {"A": "Cannot be determined from the information given", "B": "Yes, it forms a combo",
        "C": "No, it does not form a combo", "D": "Only with a wild card"},
       "A", "B"),

    _L("L07_archived", "equivocation", "library classification",
       "In Orla's library, a book is 'archived' only if it has been unborrowed for two full years. The ledger marks 'The Tide' as inactive.",
       "'Inactive' in the ledger merely means off-display, which is not the same as archived.",
       "'The Tide' is inactive, and inactive books are archived, so it is archived.",
       "Based on the information above, is 'The Tide' archived?",
       {"A": "Yes, it is archived", "B": "It cannot be determined from the information given",
        "C": "No, it is not archived", "D": "Only after a review"},
       "B", "A"),

    _L("L08_advance", "affirming_consequent", "contract clause",
       "Under the Velin contract, vendors paid in advance must deliver within ten days. Toft Co. delivered within ten days.",
       "Timely delivery does not establish advance payment, so its terms are unclear.",
       "Toft delivered within ten days, so the company must have been paid in advance.",
       "Based on the information above, was Toft Co. paid in advance?",
       {"A": "No, it was not paid in advance", "B": "Yes, it was paid in advance",
        "C": "It cannot be determined from the information given", "D": "Only a partial advance"},
       "C", "B"),

    _L("L09_harbor", "illicit_conversion", "town bylaw",
       "Marlow's charter grants harbor access to every chartered fisher. Bex has harbor access.",
       "Access is also granted to ferry operators, so Bex need not be a chartered fisher.",
       "Only chartered fishers receive harbor access, so Bex is a chartered fisher.",
       "Based on the information above, is Bex a chartered fisher?",
       {"A": "Yes, Bex is a chartered fisher", "B": "No, Bex is not a chartered fisher",
        "C": "It cannot be determined from the information given", "D": "Only during the season"},
       "C", "A"),

    # L10: the flaw is an UNSUPPORTED assumption (asserts a second-stage pass never given), NOT an
    # explicit contradiction of the rule — the explicit version was surface-catchable (review fix).
    _L("L10_assay", "unsupported_assumption", "lab process",
       "In the Dovre assay, a sample is 'cleared' only after both stages pass. Sample K passed stage one.",
       "Passing stage one alone does not establish stage two, so clearance is undetermined.",
       "Sample K passed stage one and its second-stage check, so the sample is cleared.",
       "Based on the information above, is sample K cleared?",
       {"A": "It cannot be determined from the information given", "B": "Yes, it is cleared",
        "C": "No, it is not cleared", "D": "Only provisionally cleared"},
       "A", "B"),

    _L("L11_gala", "affirming_consequent", "college eligibility",
       "At Auster College, all scholarship holders attend the spring gala. Nat attends the spring gala.",
       "Many non-scholarship students attend too, so Nat's status stays open.",
       "Gala attendees are the scholarship holders, so Nat holds a scholarship.",
       "Based on the information above, does Nat hold a scholarship?",
       {"A": "Yes, Nat holds a scholarship", "B": "It cannot be determined from the information given",
        "C": "No, Nat does not hold a scholarship", "D": "Only a partial scholarship"},
       "B", "A"),

    _L("L12_highland", "equivocation", "geographic index",
       "In the Pell index, a region is 'highland' only above 800 metres. Region Q is described as 'elevated'.",
       "'Elevated' means above the local mean, which can be under 800 metres, so it is unsettled.",
       "Region Q is elevated, and elevated regions are highland, so Q is highland.",
       "Based on the information above, is region Q highland?",
       {"A": "No, region Q is not highland", "B": "Yes, region Q is highland",
        "C": "It cannot be determined from the information given", "D": "Only its northern part"},
       "C", "B"),

    _L("L13_vault", "affirming_consequent", "access policy",
       "At Hollis, staff with level-three clearance may enter the vault. Priya entered the vault.",
       "Entry is permitted on other grounds too, so Priya's clearance is not established.",
       "Priya entered the vault, so she therefore holds level-three clearance.",
       "Based on the information above, does Priya hold level-three clearance?",
       {"A": "It cannot be determined from the information given", "B": "No, she does not hold it",
        "C": "Yes, she holds level-three clearance", "D": "Only temporary clearance"},
       "A", "C"),

    _L("L14_banner", "false_cause", "town economics",
       "In Calden, shops that join the festival receive a banner. The Pike Bakery has a banner.",
       "Banners also go to long-standing shops, so we cannot conclude Pike joined.",
       "Pike Bakery has a banner, so the bakery must have joined the festival.",
       "Based on the information above, did Pike Bakery join the festival?",
       {"A": "Yes, it joined the festival", "B": "It cannot be determined from the information given",
        "C": "No, it did not join the festival", "D": "Only as a sponsor"},
       "B", "A"),

    # ---- scale batch 2 (varied fallacy types x fresh fictional domains) ----
    _L("L15_parking", "denying_antecedent", "institute policy",
       "At the Selby Institute, all fellows receive a parking pass. Rao is not a fellow.",
       "Non-fellows can also receive passes, so Rao's pass status stays open.",
       "Only fellows are given passes, so Rao has no parking pass.",
       "Based on the information above, does Rao have a parking pass?",
       {"A": "It cannot be determined from the information given", "B": "No, Rao has no parking pass",
        "C": "Yes, Rao has a parking pass", "D": "Only a visitor pass"}, "A", "B"),

    _L("L16_badge", "illicit_conversion", "workplace credentials",
       "Every certified analyst at Voss holds a green badge. Lena holds a green badge.",
       "Others may hold green badges too, so Lena's certification is unconfirmed.",
       "Green-badge holders are the certified analysts, so Lena is certified.",
       "Based on the information above, is Lena a certified analyst?",
       {"A": "Yes, Lena is certified", "B": "It cannot be determined from the information given",
        "C": "No, Lena is not certified", "D": "Only provisionally certified"}, "B", "A"),

    _L("L17_motor", "false_cause", "factory process",
       "At the Korr mill, overheated motors are shut down. Motor 4 was shut down.",
       "Motors are shut down for maintenance too, so we cannot say motor 4 overheated.",
       "Motor 4 was shut down, so it must have overheated.",
       "Based on the information above, did motor 4 overheat?",
       {"A": "It cannot be determined from the information given", "B": "Yes, motor 4 overheated",
        "C": "No, it did not overheat", "D": "Only briefly"}, "A", "B"),

    _L("L18_set", "composition", "invented card game",
       "In Vance, a 'set' needs three cards of one suit. Tovo's three cards are each high-ranked.",
       "Being high-ranked says nothing about suit, so a set is not established.",
       "All three cards are high-ranked, so together they make a set.",
       "Based on the information above, do Tovo's cards make a set?",
       {"A": "No, they do not make a set", "B": "Yes, the cards make a set",
        "C": "It cannot be determined from the information given", "D": "Only with a joker"}, "C", "B"),

    _L("L19_vote", "division", "committee procedure",
       "The Aldous committee as a whole approved the budget. Pim is on the committee.",
       "A body approving as a whole need not mean each member voted yes, so Pim's vote is unknown.",
       "The committee approved it, so Pim voted to approve.",
       "Based on the information above, did Pim vote to approve?",
       {"A": "Yes, Pim voted to approve", "B": "No, Pim voted against",
        "C": "It cannot be determined from the information given", "D": "Pim abstained"}, "C", "A"),

    _L("L20_confirm", "equivocation", "lab procedure",
       "At Reed Labs, a result is 'confirmed' only after two independent trials. Result R is marked 'verified by the lead'.",
       "A lead's sign-off is one check, not two trials, so R is not confirmed.",
       "R is verified, and verified results are confirmed, so R is confirmed.",
       "Based on the information above, is result R confirmed?",
       {"A": "It cannot be determined from the information given", "B": "Yes, result R is confirmed",
        "C": "No, R is not confirmed", "D": "R is still pending"}, "A", "B"),

    _L("L21_volunteer", "affirming_consequent", "college eligibility",
       "Every scholarship student at Bram volunteers weekly. Hugo volunteers weekly.",
       "Many non-scholarship students volunteer too, so Hugo's status stays open.",
       "Hugo volunteers weekly, so he is a scholarship student.",
       "Based on the information above, is Hugo a scholarship student?",
       {"A": "Yes, Hugo is a scholarship student", "B": "It cannot be determined from the information given",
        "C": "No, Hugo is not", "D": "Only a partial award"}, "B", "A"),

    _L("L22_lane", "scope_quantifier", "town bylaw",
       "In Dell, every licensed cart may use the north lane. A blue cart is using the north lane.",
       "Other carts may use the lane too, so the blue cart need not be licensed.",
       "Only licensed carts use the north lane, so the blue cart is licensed.",
       "Based on the information above, is the blue cart licensed?",
       {"A": "It cannot be determined from the information given", "B": "Yes, the blue cart is licensed",
        "C": "No, it is not licensed", "D": "Licensed only temporarily"}, "A", "B"),

    _L("L23_discount", "unsupported_assumption", "lease terms",
       "Under the Mire lease, tenants who prepay get a discount. The Olsen account shows a discount applied.",
       "Discounts apply on other grounds too, so prepayment is not established.",
       "A discount was applied, so the Olsen account prepaid.",
       "Based on the information above, did the Olsen account prepay?",
       {"A": "No, the account did not prepay", "B": "It cannot be determined from the information given",
        "C": "Yes, the account prepaid", "D": "It prepaid in part"}, "B", "C"),

    _L("L24_form9", "denying_antecedent", "audit policy",
       "Every audited branch files a Form 9. The Tace branch was not audited.",
       "Unaudited branches may still file a Form 9, so Tace's filing is unknown.",
       "Since it was not audited, the Tace branch filed no Form 9.",
       "Based on the information above, did the Tace branch file a Form 9?",
       {"A": "Yes, the Tace branch filed one", "B": "No, the Tace branch filed none",
        "C": "It cannot be determined from the information given", "D": "It filed one late"}, "C", "B"),

    _L("L25_air", "illicit_conversion", "shipping classification",
       "All priority shipments at Wexel travel by air. Crate 12 travelled by air.",
       "Non-priority items also travel by air, so crate 12's priority is unconfirmed.",
       "Air-travelled crates are the priority shipments, so crate 12 is priority.",
       "Based on the information above, is crate 12 a priority shipment?",
       {"A": "It cannot be determined from the information given", "B": "Yes, crate 12 is priority",
        "C": "No, it is not priority", "D": "It is possibly priority"}, "A", "B"),

    _L("L26_responder", "false_cause", "fictional trial",
       "In the Honce trial, responders all showed a temperature rise. Subject 7 showed a temperature rise.",
       "Temperature can rise for unrelated reasons, so subject 7's response is unclear.",
       "Subject 7's temperature rose, so subject 7 is a responder.",
       "Based on the information above, is subject 7 a responder?",
       {"A": "Yes, subject 7 is a responder", "B": "It cannot be determined from the information given",
        "C": "No, subject 7 is not", "D": "A mild responder"}, "B", "A"),

    _L("L27_panel", "composition", "panel rules",
       "A 'full panel' at Strake needs a member from each of three units. Two named members are both from unit A.",
       "Two members from one unit do not cover three units, so a full panel is not shown.",
       "Both are valid members, so together they form a full panel.",
       "Based on the information above, do they form a full panel?",
       {"A": "It cannot be determined from the information given", "B": "Yes, they form a full panel",
        "C": "No, they do not", "D": "Only an acting panel"}, "A", "B"),

    _L("L28_vessel", "equivocation", "port clearance",
       "At Pell Port, a vessel is 'cleared' only with both customs and health sign-off. Vessel M has customs sign-off.",
       "Customs alone is not both sign-offs, so M's clearance is undetermined.",
       "M has a sign-off, and signed-off vessels are cleared, so M is cleared.",
       "Based on the information above, is vessel M cleared?",
       {"A": "Yes, vessel M is cleared", "B": "No, vessel M is not cleared",
        "C": "It cannot be determined from the information given", "D": "Cleared provisionally"}, "C", "A"),

    _L("L29_finalist", "overgeneralization", "competition history",
       "Last season every Lark finalist had trained at the academy. Quss is this season's finalist.",
       "A past season's pattern does not fix this one, so Quss's training is unknown.",
       "Finalists train at the academy, so Quss trained there.",
       "Based on the information above, did Quss train at the academy?",
       {"A": "It cannot be determined from the information given", "B": "Yes, Quss trained there",
        "C": "No, Quss did not", "D": "Trained only briefly"}, "A", "B"),

    _L("L30_keyholder", "affirming_consequent", "access policy",
       "Every keyholder at Tarn can open the annex. Sable opened the annex.",
       "The annex can be opened by other means, so Sable need not be a keyholder.",
       "Sable opened the annex, so Sable is a keyholder.",
       "Based on the information above, is Sable a keyholder?",
       {"A": "Yes, Sable is a keyholder", "B": "It cannot be determined from the information given",
        "C": "No, Sable is not", "D": "Only with help"}, "B", "A"),

    _L("L31_bonus", "unsupported_assumption", "staff policy",
       "At Cove, staff who pass the audit get a bonus. Devra received a bonus this cycle.",
       "Bonuses are given for other reasons too, so passing the audit is not established.",
       "Devra got a bonus, so Devra passed the audit.",
       "Based on the information above, did Devra pass the audit?",
       {"A": "No, Devra did not pass", "B": "It cannot be determined from the information given",
        "C": "Yes, Devra passed the audit", "D": "Passed in part"}, "B", "C"),

    _L("L32_stall", "scope_quantifier", "market rule",
       "In Ardel, every metered stall must display a tag. Stall 9 displays a tag.",
       "Unmetered stalls can display tags too, so stall 9 need not be metered.",
       "Only metered stalls display tags, so stall 9 is metered.",
       "Based on the information above, is stall 9 metered?",
       {"A": "It cannot be determined from the information given", "B": "Yes, stall 9 is metered",
        "C": "No, it is not metered", "D": "Metered seasonally"}, "A", "B"),
]

# Counterbalance gold AND flawed across A–D (kills position bias; rotation preserves correctness).
FAMILY_L = D.balance_positions(_RAW)


def suffix_for(item: dict) -> str:
    return D.make_suffix(item["question"], item["options"])


def conditions(item: dict) -> dict:
    """The three user-turn strings per item (shared suffix; only the context differs).

    - 'ablated'  : givens only (no worked step) -> the load-bearing + competence check (expect gold)
    - 'correct'  : givens + correct worked step  -> the H_A reference (expect gold)
    - 'flawed'   : givens + fallacious step      -> the H_B / label condition (CATCH gold | MISS flawed)
    """
    s = suffix_for(item)
    return {
        "ablated": D.user_content(item["context_ablated"], s),
        "correct": D.user_content(item["context_correct"], s),
        "flawed": D.user_content(item["context_flawed"], s),
    }


if __name__ == "__main__":
    from collections import Counter

    errs = []
    for it in FAMILY_L:
        g, f = it["gold_option"], it["flawed_consistent_option"]
        if set(it["options"]) != set(D.LETTERS):
            errs.append(f"{it['slug']}: options != A-D")
        if g == f:
            errs.append(f"{it['slug']}: gold == flawed")
        if len(set(it["options"].values())) != 4:
            errs.append(f"{it['slug']}: duplicate option text")
        if it["context_correct"] == it["context_flawed"]:
            errs.append(f"{it['slug']}: correct == flawed context")
        if it["context_ablated"] not in it["context_correct"] or it["context_ablated"] not in it["context_flawed"]:
            errs.append(f"{it['slug']}: ablated givens not shared by both conditions")
        # gold must be the 'cannot be determined' option (the structural invariant of this family)
        if "cannot be determined" not in it["options"][g].lower():
            errs.append(f"{it['slug']}: gold is not the 'cannot be determined' option")

    print(f"{len(FAMILY_L)} Family-L items")
    print("value/structure errors:", errs or "none")
    print("gold positions:   ", dict(Counter(it["gold_option"] for it in FAMILY_L)))
    print("flawed positions: ", dict(Counter(it["flawed_consistent_option"] for it in FAMILY_L)))
    print("flaw subtypes:    ", dict(Counter(it["flaw_subtype"] for it in FAMILY_L)))
    print("NOTE: within-pair token-length equality (c_A vs c_B) is checked on the REAL tokenizer by "
          "scripts/verify_token_invariants.py; reword/pad any mismatches there.")
