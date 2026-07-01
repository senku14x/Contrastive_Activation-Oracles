#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Full Stage-1 + Stage-1.5 pipeline on a fresh GPU box (Vast / RunPod / Colab terminal).
#
# Run from the repo root AFTER clone+checkout. Set these env vars first:
#   export HF_TOKEN=hf_...            # required (gated model download / login)
#   export GATE_API_KEY=sk-or-...     # OpenRouter key for the text gate (optional but recommended)
#   export GATE_MODEL=z-ai/glm-4.6    # a strong NON-Qwen reader
#   export GATE_WORKERS=8             # optional; parallel gate calls (raise to 12-16 if no 429s)
#
# Everything is regenerated from committed code — nothing needs to be preserved across runs.
# ---------------------------------------------------------------------------
set -euo pipefail

echo "==== [0/6] deps ===="
pip install -q -U "transformers>=4.55,<5" "peft>=0.17,<0.19" accelerate "scikit-learn>=1.3" \
    requests huggingface_hub

echo "==== [1/6] HuggingFace login ===="
: "${HF_TOKEN:?set HF_TOKEN first  (export HF_TOKEN=hf_...)}"
huggingface-cli login --token "$HF_TOKEN" >/dev/null 2>&1 \
  || python -c "import os; from huggingface_hub import login; login(os.environ['HF_TOKEN'])"
echo "  ok"

echo "==== [2/6] build candidates + invariants (no model) ===="
python scripts/build_candidates.py
python scripts/pad_to_equal_length.py        # neutral-prefix pad to equal within-pair token length
python scripts/ablation_verify.py            # structural load-bearing flag
python scripts/verify_token_invariants.py    # expect 80/80 pass

echo "==== [3/6] measure target behavior (OFF no-cue) + label ===="
python scripts/measure_target_logits.py
python scripts/label_candidates.py           # <- catch / miss / discard yield

echo "==== [4/6] text-only gate (parallel) ===="
if [[ -n "${GATE_API_KEY:-}" && -n "${GATE_MODEL:-}" ]]; then
  python scripts/run_text_only_gate.py --workers "${GATE_WORKERS:-8}"
  python scripts/match_catch_miss.py
else
  echo "  GATE_API_KEY/GATE_MODEL not set -> skipping gate; freezing all clean items (dry-run only)."
  python scripts/match_catch_miss.py --gated data/candidates_labeled.jsonl --require-gate false
fi

echo "==== [5/6] extract activations (W8 + W20, adapter off) ===="
python scripts/extract_activations.py
python scripts/extract_activations.py --window 20

echo "==== [6/6] Stage-1.5 existence probe ===="
python scripts/stage1_5_probe.py
python scripts/stage1_5_probe.py --acts data/activations_w20.npz

echo
echo "==== DONE. Read: the label yield ([3/6]) and the probe verdict ([6/6]). ===="
echo "Next decision (see CLAUDE.md): catch is the scarce class (~18%); to power the probe either"
echo "  B) test if reasoning-ON balances it (cheap):"
echo "     python scripts/measure_target_logits.py --on --family L --k 16 --out data/measured_on_L.jsonl"
echo "  A) scale to ~120 items, or  C) write up the behavioral finding + underpowered probe."
