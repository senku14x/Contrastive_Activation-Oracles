"""Contrastive Activation Oracle (CAO) — feasibility study package.

Modules:
  ao_runtime : minimal, self-contained runtime for the released Qwen3-8B CoT
               Activation Oracle (load / extract / inject / read-logprobs).
  ao_checks  : pre-flight + decisive checks (suffix asserter, constrained-parity
               baseline, dH geometry presence-check). `ao_letter_logits` is wired
               to ao_runtime.

See docs/STAGE0.md for the verified wiring facts and the reproduction gate.
"""
