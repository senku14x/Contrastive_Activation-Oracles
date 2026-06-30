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

    # ---- scale batch 3 (generate-and-verify workflow; 49 adversarially-verified items) ----
    _L("L33_door_red", "affirming_consequent", "town bylaw",
       "Under Merrow's heritage bylaw, every house on the register may paint its door red. The Crane house has a red door.",
       "Other houses may paint a door red too, so the Crane house need not be on the register.",
       "The Crane house has a red door, so the Crane house is on the register.",
       "Based on the information above, is the Crane house on the register?",
       {"A": "Yes, the Crane house is on the register", "B": "No, the Crane house is not on the register", "C": "It cannot be determined from the information given", "D": "Only its front door is registered"}, "C", "A"),
    _L("L34_wristband", "affirming_consequent", "town bylaw",
       "By Ostry's market rule, every trader licensed before noon wears a green wristband. Hesper wears a green wristband.",
       "Wristbands also go to traders licensed later, so Hesper's licence time is unsettled.",
       "Hesper wears a green wristband, so Hesper was licensed before noon.",
       "Based on the information above, was Hesper licensed before noon?",
       {"A": "It cannot be determined from the information given", "B": "Yes, Hesper was licensed before noon", "C": "No, Hesper was licensed after noon", "D": "Only a provisional licence was issued"}, "A", "B"),
    _L("L35_hedge", "affirming_consequent", "council rule",
       "Falden's ordinance states that every council-approved hedge exceeds four feet. The Dray hedge exceeds four feet.",
       "Hedges can exceed four feet without approval, so the Dray hedge's status is open.",
       "The Dray hedge exceeds four feet, so the Dray hedge is council-approved.",
       "Based on the information above, is the Dray hedge council-approved?",
       {"A": "No, the Dray hedge is not approved", "B": "Yes, the Dray hedge is council-approved", "C": "It cannot be determined from the information given", "D": "Only its street-side section is approved"}, "C", "B"),
    _L("L36_placard", "affirming_consequent", "council rule",
       "Under Quill's parade rules, every entrant registered at the hall receives a numbered placard. The Voss float carries a numbered placard.",
       "Placards are also given out at the gate, so the Voss float need not have registered.",
       "The Voss float carries a numbered placard, so it registered at the hall.",
       "Based on the information above, did the Voss float register at the hall?",
       {"A": "It cannot be determined from the information given", "B": "Yes, it registered at the hall", "C": "No, it did not register at the hall", "D": "It registered only on the day"}, "A", "B"),
    _L("L37_slate_roof", "affirming_consequent", "town bylaw",
       "Brae's statute requires every property in the conservation zone to have a slate roof. The Lund cottage has a slate roof.",
       "Slate roofs are common outside the zone too, so the Lund cottage's zone status is unsettled.",
       "The Lund cottage has a slate roof, so the Lund cottage lies in the conservation zone.",
       "Based on the information above, does the Lund cottage lie in the conservation zone?",
       {"A": "Yes, the Lund cottage lies in the zone", "B": "It cannot be determined from the information given", "C": "No, the Lund cottage lies outside the zone", "D": "Only its rear extension is in the zone"}, "B", "A"),
    _L("L38_seed_packet", "affirming_consequent", "council rule",
       "Cresk's council sends a spring seed packet to every allotment holder in good standing. Wynn received a spring seed packet.",
       "Seed packets also go to new applicants, so Wynn's standing is not established.",
       "Wynn received a seed packet, so Wynn is an allotment holder in good standing.",
       "Based on the information above, is Wynn an allotment holder in good standing?",
       {"A": "No, Wynn is not in good standing", "B": "It cannot be determined from the information given", "C": "Yes, Wynn is an allotment holder in good standing", "D": "Wynn is only on the waiting list"}, "B", "C"),
    _L("L39_loomwright", "denying_antecedent", "club membership",
       "At the Loomwright Society, every full member receives a brass pin. Odra is not a full member.",
       "Brass pins also go to honorary guests, so Odra's pin status is unsettled.",
       "Brass pins go only to full members, so Odra has no brass pin.",
       "Based on the information above, does Odra have a brass pin?",
       {"A": "Yes, Odra has a brass pin", "B": "No, Odra has no brass pin", "C": "It cannot be determined from the information given", "D": "Only as a guest"}, "C", "B"),
    _L("L40_meridian", "denying_antecedent", "program eligibility",
       "Every Meridian Scholar is granted lab access. Pell was not named a Meridian Scholar.",
       "Lab access is granted on other grounds too, so Pell's access is undetermined.",
       "Lab access goes only to Meridian Scholars, so Pell has no lab access.",
       "Based on the information above, does Pell have lab access?",
       {"A": "It cannot be determined from the information given", "B": "No, Pell has no lab access", "C": "Yes, Pell has lab access", "D": "Only on weekdays"}, "A", "B"),
    _L("L41_verdance", "denying_antecedent", "garden club policy",
       "At the Verdance Circle, all tier-one growers get a plot in the walled garden. Sennet is not a tier-one grower.",
       "Lower tiers can be allotted walled plots as well, so Sennet's plot is unsettled.",
       "Walled plots are reserved for tier-one growers, so Sennet gets no walled plot.",
       "Based on the information above, does Sennet get a walled plot?",
       {"A": "Yes, Sennet gets a walled plot", "B": "It cannot be determined from the information given", "C": "No, Sennet gets no walled plot", "D": "Only a shared plot"}, "B", "C"),
    _L("L42_halcyon", "denying_antecedent", "athletic program",
       "In the Halcyon program, every certified swimmer is issued a lane card. Tibb is not a certified swimmer.",
       "Lane cards are issued to trainees too, so Tibb's lane-card status stays open.",
       "Lane cards are given out only to certified swimmers, so Tibb holds no lane card.",
       "Based on the information above, does Tibb hold a lane card?",
       {"A": "No, Tibb holds no lane card", "B": "Yes, Tibb holds a lane card", "C": "It cannot be determined from the information given", "D": "Only a guest card"}, "C", "A"),
    _L("L43_orrery", "denying_antecedent", "society membership",
       "Every patron of the Orrery Guild may borrow a star chart. Cael is not a patron of the Orrery Guild.",
       "Non-patrons are sometimes allowed to borrow charts, so Cael's case is unsettled.",
       "Star charts may be borrowed only by patrons, so Cael may not borrow one.",
       "Based on the information above, may Cael borrow a star chart?",
       {"A": "It cannot be determined from the information given", "B": "Yes, Cael may borrow one", "C": "No, Cael may not borrow one", "D": "Only with a deposit"}, "A", "C"),
    _L("L44_kindling", "denying_antecedent", "youth program eligibility",
       "At the Kindling Initiative, every enrolled mentee gets a travel stipend. Roon is not an enrolled mentee.",
       "Stipends are also awarded to volunteers, so Roon's stipend status is undetermined.",
       "Travel stipends are reserved for enrolled mentees, so Roon gets no stipend.",
       "Based on the information above, does Roon get a travel stipend?",
       {"A": "Yes, Roon gets a travel stipend", "B": "No, Roon gets no travel stipend", "C": "Only a partial stipend", "D": "It cannot be determined from the information given"}, "D", "B"),
    _L("L45_glint", "illicit_conversion", "invented card game",
       "In the card game Glint, every card in the Ember suit is a trump. Yara has just played a trump.",
       "Frost-suit cards can also be trumps, so Yara's card need not be an Ember card.",
       "Trumps are exactly the Ember cards, so Yara's card must be an Ember card.",
       "Based on the information above, is Yara's card an Ember card?",
       {"A": "Yes, it is an Ember card", "B": "No, it is not an Ember card", "C": "It cannot be determined from the information given", "D": "Only if it is face-up"}, "C", "A"),
    _L("L46_tile", "illicit_conversion", "invented board game",
       "In the board game Sarn, every Keep tile counts as a stronghold. The blue piece is standing on a stronghold.",
       "Some Tower tiles count as strongholds too, so the tile need not be a Keep tile.",
       "Strongholds are just the Keep tiles, so the blue piece sits on a Keep tile.",
       "Based on the information above, is the blue piece on a Keep tile?",
       {"A": "It cannot be determined from the information given", "B": "Yes, it is on a Keep tile", "C": "No, it is not on a Keep tile", "D": "Only at the start of play"}, "A", "B"),
    _L("L47_volt", "illicit_conversion", "invented card game",
       "In the deck game Volt, every Surge card is worth double points. Renn is holding a card worth double points.",
       "Other cards can be worth double points too, so Renn's card need not be a Surge card.",
       "The double-point cards are exactly the Surge cards, so Renn's card is a Surge card.",
       "Based on the information above, is Renn holding a Surge card?",
       {"A": "No, it is not a Surge card", "B": "It cannot be determined from the information given", "C": "Yes, it is a Surge card", "D": "Only in the final round"}, "B", "C"),
    _L("L48_moot", "illicit_conversion", "invented board game",
       "In the strategy game Moot, every Vanguard piece may cross the river. A piece of Tovi's has just crossed the river.",
       "Scout pieces may cross the river too, so Tovi's piece need not be a Vanguard.",
       "Only Vanguard pieces may cross the river, so Tovi's piece is a Vanguard.",
       "Based on the information above, is Tovi's piece a Vanguard?",
       {"A": "It cannot be determined from the information given", "B": "Yes, it is a Vanguard", "C": "No, it is not a Vanguard", "D": "Only while in check"}, "A", "B"),
    _L("L49_zelt", "illicit_conversion", "invented card game",
       "In the casino game Zelt, every gold chip is a wager chip. Bel has set down a wager chip.",
       "Silver chips can serve as wager chips too, so Bel's chip need not be gold.",
       "The wager chips are exactly the gold chips, so Bel's chip must be gold.",
       "Based on the information above, is Bel's chip a gold chip?",
       {"A": "Yes, it is a gold chip", "B": "It cannot be determined from the information given", "C": "No, it is not a gold chip", "D": "Only above the table limit"}, "B", "A"),
    _L("L50_brack", "illicit_conversion", "invented board game",
       "In the tile game Brack, every Oath rune scores a bonus. Pell has placed a rune that scores a bonus.",
       "Ward runes score a bonus as well, so Pell's rune need not be an Oath rune.",
       "The bonus-scoring runes are just the Oath runes, so Pell's rune is an Oath rune.",
       "Based on the information above, is Pell's rune an Oath rune?",
       {"A": "No, it is not an Oath rune", "B": "Yes, it is an Oath rune", "C": "It cannot be determined from the information given", "D": "Only on a corner space"}, "C", "B"),
    _L("L51_fringe", "unsupported_assumption", "herbarium taxonomy",
       "In the Vesh herbarium, every plant of the coronate class bears a fringed leaf. Specimen 12 bears a fringed leaf.",
       "Plants outside the coronate class can bear fringed leaves too, so its class is unsettled.",
       "Only coronate plants bear fringed leaves, so specimen 12 is of the coronate class.",
       "Based on the information above, is specimen 12 of the coronate class?",
       {"A": "Yes, specimen 12 is of the coronate class", "B": "It cannot be determined from the information given", "C": "No, specimen 12 is not of the coronate class", "D": "Only if it also flowers"}, "B", "A"),
    _L("L52_nocturnal", "unsupported_assumption", "fauna catalogue",
       "In the Welk catalogue, every tier-two beast is nocturnal. The keeper records that the grendel is nocturnal.",
       "Beasts in other tiers may be nocturnal as well, so the grendel's tier is open.",
       "Only tier-two beasts are nocturnal, so the grendel is a tier-two beast.",
       "Based on the information above, is the grendel a tier-two beast?",
       {"A": "It cannot be determined from the information given", "B": "Yes, the grendel is a tier-two beast", "C": "No, the grendel is not a tier-two beast", "D": "Only in winter"}, "A", "B"),
    _L("L53_marshtype", "unsupported_assumption", "fern taxonomy",
       "In the Pell taxonomy, every marsh-type fern is also classed low-light. Frond R is classed low-light.",
       "A fern can be low-light without being marsh-type, so frond R's type stays open.",
       "Only marsh-type ferns are classed low-light, so frond R is a marsh-type fern.",
       "Based on the information above, is frond R a marsh-type fern?",
       {"A": "Yes, frond R is marsh-type", "B": "It cannot be determined from the information given", "C": "No, frond R is not marsh-type", "D": "Only its outer fronds"}, "B", "A"),
    _L("L54_charter", "unsupported_assumption", "guild rank classification",
       "The Anwen guild classes a member as 'chartered' only if the member holds a seal and has served three terms. Roan holds a seal.",
       "Holding a seal does not establish three terms served, so Roan's class is unsettled.",
       "Roan holds a seal and has served three terms, so Roan is classed chartered.",
       "Based on the information above, is Roan classed as chartered?",
       {"A": "It cannot be determined from the information given", "B": "Yes, Roan is chartered", "C": "No, Roan is not chartered", "D": "Only as an associate"}, "A", "B"),
    _L("L55_anneal", "equivocation", "metal-forging lab process",
       "At the Vornek forge, an ingot is 'tempered' only after it has been quenched and aged for ten days. The shift log notes that ingot 12 was 'hardened' on Tuesday.",
       "'Hardened' on the log refers to a single quench, not the ten-day tempering, so its status is open.",
       "Ingot 12 was hardened, and any hardened ingot is tempered, so ingot 12 is tempered.",
       "Based on the information above, is ingot 12 tempered?",
       {"A": "Yes, ingot 12 is tempered", "B": "It cannot be determined from the information given", "C": "No, ingot 12 is not tempered", "D": "Only its core is tempered"}, "B", "A"),
    _L("L56_polymer", "equivocation", "polymer-plant factory process",
       "At the Kessil plant, a resin is 'cured' only after a four-hour bake at full heat. Drum 3 is marked 'set' on the line sheet.",
       "'Set' on the sheet means firm to the touch, not the four-hour bake, so curing is open.",
       "Drum 3 is set, and any set resin is cured, so the resin in drum 3 is cured.",
       "Based on the information above, is the resin in drum 3 cured?",
       {"A": "It cannot be determined from the information given", "B": "No, the resin is not cured", "C": "Yes, the resin is cured", "D": "Cured only near the surface"}, "A", "C"),
    _L("L57_lease_load", "composition", "fictional lease clause",
       "Under the Vellmoor lease, a building is 'load-compliant' only if its total stored weight stays within the structural cap. Each unit in the building is within its own permitted weight limit.",
       "Units within their own limits can still sum past the cap, so compliance is unsettled.",
       "Each unit is within its weight limit, so the whole building is load-compliant.",
       "Based on the information above, is the building load-compliant?",
       {"A": "Yes, the building is load-compliant", "B": "No, the building is not load-compliant", "C": "It cannot be determined from the information given", "D": "Only its lower floors are compliant"}, "C", "A"),
    _L("L58_contract_tier", "composition", "fictional contract clause",
       "The Harrow supply contract deems a consignment 'complete' only if it includes at least one item from every catalogue tier. Each crate in the consignment holds items drawn from the catalogue.",
       "Catalogued items need not span every tier, so completeness is unsettled.",
       "Every crate holds catalogued items, so the consignment is complete.",
       "Based on the information above, is the consignment complete?",
       {"A": "It cannot be determined from the information given", "B": "Yes, the consignment is complete", "C": "No, the consignment is not complete", "D": "Only the first crate is complete"}, "A", "B"),
    _L("L59_lease_noise", "composition", "fictional lease clause",
       "The Dunmere lease grants a block 'quiet status' only if the tenants' combined noise stays under a fixed ceiling. Each tenant's individual noise is under the per-tenant limit.",
       "Per-tenant compliance can still combine over the ceiling, so quiet status is unsettled.",
       "Each tenant is under the per-tenant limit, so the whole block has quiet status.",
       "Based on the information above, does the block hold quiet status?",
       {"A": "Yes, the block holds quiet status", "B": "It cannot be determined from the information given", "C": "No, the block does not hold quiet status", "D": "Only at night"}, "B", "A"),
    _L("L60_contract_ledger", "composition", "fictional partnership contract",
       "The Castle partnership deed calls a ledger 'balanced' only if total credits equal total debits across all accounts. Each partner's own account has been internally reconciled.",
       "Per-account reconciliation need not balance the combined totals, so balance is unsettled.",
       "Each partner's account is reconciled, so the whole ledger is balanced.",
       "Based on the information above, is the ledger balanced?",
       {"A": "No, the ledger is not balanced", "B": "Yes, the ledger is balanced", "C": "It cannot be determined from the information given", "D": "Only the senior account is balanced"}, "C", "B"),
    _L("L61_lease_occupancy", "composition", "fictional estate lease",
       "Under the Pelham estate lease, an estate is 'renewal-eligible' only if its aggregate yearly occupancy exceeds a set threshold. Each cottage on the estate was occupied at some point during the year.",
       "Scattered occupancy need not clear the aggregate threshold, so eligibility is unsettled.",
       "Each cottage was occupied this year, so the whole estate is renewal-eligible.",
       "Based on the information above, is the estate renewal-eligible?",
       {"A": "It cannot be determined from the information given", "B": "Yes, the estate is renewal-eligible", "C": "No, the estate is not renewal-eligible", "D": "Only the north cottages qualify"}, "A", "B"),
    _L("L62_contract_insured", "composition", "fictional freight contract",
       "The Orvell freight contract treats a shipment as 'warranted' only if its total declared value falls within the insured band. Each parcel's declared value is within the insurer's per-parcel cap.",
       "Per-parcel caps can still total past the insured band, so warranty is unsettled.",
       "Each parcel is under the per-parcel cap, so the whole shipment is warranted.",
       "Based on the information above, is the shipment warranted?",
       {"A": "Yes, the shipment is warranted", "B": "No, the shipment is not warranted", "C": "It cannot be determined from the information given", "D": "Only the lighter parcels are warranted"}, "C", "A"),
    _L("L63_route", "division", "transit scheduling",
       "At Vernal Transit, the Coral route as a whole is classed as express. Stop Dunmore lies on the Coral route.",
       "An overall express class need not hold at each stop, so Dunmore's class stays unsettled.",
       "The route is classed as express, so the Dunmore stop is itself classed express.",
       "Based on the information above, is the Dunmore stop express?",
       {"A": "Yes, the Dunmore stop is express", "B": "No, the Dunmore stop is not express", "C": "It cannot be determined from the information given", "D": "Only at peak hours"}, "C", "A"),
    _L("L64_shift", "division", "staff rostering",
       "At the Orrin depot, the night shift overall is rated fully staffed. The 3 a.m. block falls within the night shift.",
       "A shift rated fully staffed overall need not make each block so, leaving 3 a.m. unsettled.",
       "The night shift is rated fully staffed, so the 3 a.m. block is itself fully staffed.",
       "Based on the information above, is the 3 a.m. block fully staffed?",
       {"A": "It cannot be determined from the information given", "B": "Yes, the 3 a.m. block is fully staffed", "C": "No, the 3 a.m. block is short-staffed", "D": "Only the first hour is staffed"}, "A", "B"),
    _L("L65_quota", "division", "performance targets",
       "At Soraine Labs, the analytics department met its quarterly quota overall. The mapping team sits inside that department.",
       "A department meeting its quota overall need not mean each team did, leaving the mapping team open.",
       "The department met its quota, so the mapping team inside it met its quota too.",
       "Based on the information above, did the mapping team meet its quota?",
       {"A": "Yes, the mapping team met its quota", "B": "No, the mapping team missed its quota", "C": "It cannot be determined from the information given", "D": "It met half its quota"}, "C", "A"),
    _L("L66_tariff", "false_cause", "fictional country economic rule",
       "In the republic of Vantil, any ledger flagged for irregularity is frozen for review. The Merrow ledger is currently frozen.",
       "Ledgers are also frozen during routine season-end closing, so we cannot conclude the Merrow ledger was flagged.",
       "The Merrow ledger is frozen for review, so it must have been flagged for irregularity.",
       "Based on the information above, was the Merrow ledger flagged for irregularity?",
       {"A": "Yes, it was flagged for irregularity", "B": "No, it was not flagged for irregularity", "C": "It cannot be determined from the information given", "D": "Only its prior entries were flagged"}, "C", "A"),
    _L("L67_subsidy", "false_cause", "fictional country economic rule",
       "In Doleun, a farm that fails the yield quota loses its subsidy for the year. The Ashfen farm received no subsidy this year.",
       "Subsidies are also withheld from farms that filed late, so the missing subsidy does not show Ashfen failed the quota.",
       "The Ashfen farm received no subsidy, so it must have failed the yield quota this year.",
       "Based on the information above, did the Ashfen farm fail the yield quota?",
       {"A": "It cannot be determined from the information given", "B": "Yes, the farm failed the yield quota", "C": "No, the farm met the yield quota", "D": "It failed only the spring quota"}, "A", "B"),
    _L("L68_levy", "false_cause", "fictional company economic rule",
       "At Corwyth Trading, an account that overdraws its credit line is charged a penalty levy. The Brandt account was charged a penalty levy this month.",
       "The levy is also charged for late settlement, so the charge does not establish that Brandt overdrew its credit line.",
       "The Brandt account was charged a penalty levy, so it must have overdrawn its credit line this month.",
       "Based on the information above, did the Brandt account overdraw its credit line?",
       {"A": "No, the account did not overdraw its credit line", "B": "It cannot be determined from the information given", "C": "Yes, the account overdrew its credit line", "D": "It overdrew only briefly"}, "B", "C"),
    _L("L69_audit", "false_cause", "fictional country economic rule",
       "In Selvory, a firm that underreports revenue is placed on the watch list. Halren Mills appears on the watch list this quarter.",
       "Firms are also watch-listed after a change of ownership, so the listing does not show Halren underreported revenue.",
       "Halren Mills appears on the watch list, so the firm must have underreported its revenue this quarter.",
       "Based on the information above, did Halren Mills underreport its revenue?",
       {"A": "It cannot be determined from the information given", "B": "Yes, the firm underreported its revenue", "C": "No, the firm reported its revenue accurately", "D": "It underreported only export revenue"}, "A", "B"),
    _L("L70_rebate", "false_cause", "fictional company economic rule",
       "At Pelham Foods, a supplier that misses a delivery window forfeits its quarterly rebate. The Oster supplier forfeited its quarterly rebate.",
       "The rebate is also forfeited for failing a quality check, so the forfeit does not show Oster missed a delivery window.",
       "The Oster supplier forfeited its quarterly rebate, so it must have missed a delivery window.",
       "Based on the information above, did the Oster supplier miss a delivery window?",
       {"A": "Yes, the supplier missed a delivery window", "B": "It cannot be determined from the information given", "C": "No, the supplier met every delivery window", "D": "It missed only one delivery window"}, "B", "A"),
    _L("L71_ceiling", "false_cause", "fictional country economic rule",
       "In Cathmar, goods that breach the price ceiling are moved to the restricted tariff band. The Vell crates were moved to the restricted tariff band.",
       "Goods are also moved to that band when imported off-season, so the move does not show the Vell crates breached the ceiling.",
       "The Vell crates were moved to the restricted tariff band, so they must have breached the price ceiling.",
       "Based on the information above, did the Vell crates breach the price ceiling?",
       {"A": "No, the crates did not breach the price ceiling", "B": "Yes, the crates breached the price ceiling", "C": "It cannot be determined from the information given", "D": "Only some crates breached it"}, "C", "B"),
    _L("L72_carrel", "overgeneralization", "fictional library policy",
       "At the Veld Athenaeum, last term every reserved carrel was assigned to a graduate fellow. Toller holds a reserved carrel this term.",
       "Last term's assignment rule need not hold this term, so Toller's standing is unsettled.",
       "Reserved carrels go to graduate fellows, so this term Toller is a graduate fellow.",
       "Based on the information above, is Toller a graduate fellow?",
       {"A": "Yes, Toller is a graduate fellow", "B": "It cannot be determined from the information given", "C": "No, Toller is not a graduate fellow", "D": "Only as a visiting fellow"}, "B", "A"),
    _L("L73_eastwing", "overgeneralization", "fictional archive policy",
       "In the Marrow Archive's east wing, every sealed folio is a wartime record. A sealed folio has just been catalogued in the west wing.",
       "What holds of east-wing folios need not hold in the west wing, so its content is open.",
       "Sealed folios are wartime records, so the west-wing folio is a wartime record too.",
       "Based on the information above, is the west-wing folio a wartime record?",
       {"A": "It cannot be determined from the information given", "B": "Yes, it is a wartime record", "C": "No, it is not a wartime record", "D": "Only in part a wartime record"}, "A", "B"),
    _L("L74_quill", "overgeneralization", "fictional library policy",
       "At every other branch of the Quill Lending Library, members who reserve a study room pay a deposit. Esma has reserved a study room at the new Harbour branch.",
       "A practice at the other branches need not bind Harbour, so Esma's deposit is unsettled.",
       "Reserving a study room means paying a deposit, so at Harbour Esma paid a deposit.",
       "Based on the information above, did Esma pay a deposit?",
       {"A": "No, Esma did not pay a deposit", "B": "It cannot be determined from the information given", "C": "Yes, Esma paid a deposit", "D": "Only a partial deposit"}, "B", "C"),
    _L("L75_overdue", "overgeneralization", "fictional library policy",
       "Last season at the Pellow Reading Rooms, every overdue title was fined. The Pellow ledger lists 'The Salt Gate' as overdue this season.",
       "Last season's fining practice need not carry into this season, so the fine is unsettled.",
       "Overdue titles are fined, so this season 'The Salt Gate' has been fined as well.",
       "Based on the information above, has 'The Salt Gate' been fined?",
       {"A": "It cannot be determined from the information given", "B": "Yes, it has been fined", "C": "No, it has not been fined", "D": "Only a reduced fine"}, "A", "B"),
    _L("L76_calmuth_manifest", "affirming_consequent", "fictional port customs",
       "At Calmuth Port, every bonded vessel is issued a green manifest stamp. The Wren carries a green manifest stamp.",
       "Manifest stamps are also issued on other grounds, so the Wren's bonded status is not established.",
       "The Wren carries a green manifest stamp, so the Wren is therefore a bonded vessel.",
       "Based on the information above, is the Wren a bonded vessel?",
       {"A": "Yes, the Wren is a bonded vessel", "B": "It cannot be determined from the information given", "C": "No, the Wren is not a bonded vessel", "D": "Only while in harbor"}, "B", "A"),
    _L("L77_drev_lanyard", "affirming_consequent", "fictional port customs",
       "Every customs officer at Drevport wears a copper lanyard. Hollin wears a copper lanyard.",
       "Copper lanyards are worn by others at the port too, so Hollin's role cannot be inferred.",
       "Hollin wears a copper lanyard, so Hollin must therefore be a customs officer.",
       "Based on the information above, is Hollin a customs officer?",
       {"A": "It cannot be determined from the information given", "B": "Yes, Hollin is a customs officer", "C": "No, Hollin is not a customs officer", "D": "Only on inspection days"}, "A", "B"),
    _L("L78_volane_chalk", "affirming_consequent", "fictional port customs",
       "At Volane wharf, every duty-paid crate is marked with a chalk cross. Crate 31 bears a chalk cross.",
       "Chalk crosses are also used to flag crates for other reasons, so crate 31's duty status is open.",
       "Crate 31 bears a chalk cross, so the duty on crate 31 must therefore have been paid.",
       "Based on the information above, has the duty on crate 31 been paid?",
       {"A": "Yes, the duty on crate 31 has been paid", "B": "No, the duty on crate 31 is unpaid", "C": "Only the import duty", "D": "It cannot be determined from the information given"}, "D", "A"),
    _L("L79_moss_ledger", "affirming_consequent", "fictional port customs",
       "Every registered trawler at Mossport appears in the harbor dues ledger. The Pike appears in the harbor dues ledger.",
       "The ledger also lists vessels that are not registered trawlers, so the Pike's status is unclear.",
       "The Pike appears in the dues ledger, so the Pike must therefore be a registered trawler.",
       "Based on the information above, is the Pike a registered trawler?",
       {"A": "Yes, the Pike is a registered trawler", "B": "No, the Pike is not a registered trawler", "C": "It cannot be determined from the information given", "D": "Only for the current season"}, "C", "A"),
    _L("L80_harne_seal", "affirming_consequent", "fictional port customs",
       "At Harne Quay, every inspected container leaves with a wax seal. Container 14 left with a wax seal.",
       "Wax seals are applied for other reasons too, so whether container 14 was inspected is unsettled.",
       "Container 14 left with a wax seal, so container 14 must therefore have been inspected.",
       "Based on the information above, was container 14 inspected?",
       {"A": "It cannot be determined from the information given", "B": "Yes, container 14 was inspected", "C": "No, container 14 was not inspected", "D": "Only its exterior"}, "A", "B"),
    _L("L81_studio", "denying_antecedent", "college facilities policy",
       "At Merrowmoor College, every enrolled architecture student is assigned a studio desk. Pell is not an enrolled architecture student.",
       "Students outside the architecture programme are sometimes assigned studio desks too, so Pell's desk status stays open.",
       "Studio desks are handed out only to enrolled architecture students, so Pell has not been assigned a studio desk.",
       "Based on the information above, has Pell been assigned a studio desk?",
       {"A": "Yes, Pell has been assigned a studio desk", "B": "No, Pell has not been assigned a studio desk", "C": "It cannot be determined from the information given", "D": "Only for the first term"}, "C", "B"),
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
