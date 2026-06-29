# Colab runbook — Stage 1 (yield probe) + Stage 1.5 (existence gate)

Copy each cell into a Colab notebook on an **A100 (High-RAM)** runtime, in order. Cells 0–9 need only
the model (your HF login); the text gate (cell 10) needs an external reader API; cells 11–13 finish the
existence gate. All `python` scripts must run from the repo root (the `%cd` in cell 2 handles that).

> **Scope note:** with 14 Family-L items this is a **yield probe**, not a powered test. `stage1_5_probe`
> will say `UNDERPOWERED (n<30)` — that's expected. The decisive outputs of this first run are
> `label_candidates` (how many clean MISS/CATCH survive) and `verify_token_invariants` (which pairs need
> rewording). Those numbers tell us how many more items to author.

---

```python
# Cell 0 — GPU check
!nvidia-smi
```

```python
# Cell 1 — HuggingFace login (set HF_TOKEN in Colab > Secrets first)
from google.colab import userdata
from huggingface_hub import login
login(userdata.get('HF_TOKEN'))
```

```python
# Cell 2 — clone + checkout the working branch
# (public repo; if private, use: https://<GITHUB_PAT>@github.com/senku14x/Contrastive_Activation-Oracles.git)
!git clone https://github.com/senku14x/Contrastive_Activation-Oracles.git
%cd Contrastive_Activation-Oracles
!git checkout claude/mechanistic-interpretability-qditqg
!git pull origin claude/mechanistic-interpretability-qditqg
```

```python
# Cell 3 — deps (do NOT reinstall torch; Colab's CUDA torch is fine and >=2.6)
!pip install -q -U "transformers>=4.55,<5" "peft>=0.17,<0.19" accelerate "scikit-learn>=1.3" requests
```

```python
# Cell 4 — no-model sanity (runs in seconds)
!python tests/test_pure_logic.py
!python tests/test_dataset_logic.py
```

```python
# Cell 5 — (optional) Stage 0 re-repro: confirms the AO wiring still reproduces (G1-G5)
!python scripts/stage0_repro.py
```

```python
# Cell 6 — build the candidate set + structural ablation flag (no model)
!python scripts/build_candidates.py
!python scripts/ablation_verify.py
```

```python
# Cell 7 — token invariants (needs the tokenizer). Read the output:
#   any LENGTH/SUFFIX mismatch must be reworded in cao/family_l.py before extraction (cells 11-13).
#   Measurement (cells 8-9) is fine to run regardless.
!python scripts/verify_token_invariants.py
```

```python
# Cell 8 — measure target behavior, OFF no-cue readout (loads Qwen3-8B + the oracle LoRA; ~minutes)
#   add --on for the reasoning-ON cross-check (K=32, slower)
!python scripts/measure_target_logits.py
```

```python
# Cell 9 — THE YIELD RESULT: catch / miss / discard per family
#   thresholds default to 0.65; re-baseline on this no-cue readout if the pool looks mis-cut
#   (e.g. --comp 0.6 --catch 0.6). Paste this output back.
!python scripts/label_candidates.py
```

```python
# Cell 10 — (optional, needs an external reader API) the dual text-only gate
import os
os.environ['GATE_API_KEY'] = ''          # e.g. an OpenRouter key
os.environ['GATE_MODEL']   = ''          # a strong NON-Qwen reasoner
# os.environ['GATE_QWEN_MODEL'] = ''     # optional Qwen-family baseline
# os.environ['GATE_BASE_URL'] = 'https://openrouter.ai/api/v1'
!python scripts/run_text_only_gate.py
```

```python
# Cell 11 — match CATCH/MISS + freeze the feasibility set
# WITH the gate (cell 10 ran):
!python scripts/match_catch_miss.py
# WITHOUT a gate API (pipeline dry-run / yield only; existence-gate result NOT valid for the claim):
# !python scripts/match_catch_miss.py --gated data/candidates_labeled.jsonl --require-gate false
```

```python
# Cell 12 — extract pre-output activations (adapter disabled); W8 primary, W20 sensitivity
!python scripts/extract_activations.py
!python scripts/extract_activations.py --window 20
```

```python
# Cell 13 — Stage 1.5 existence gate: probe(ΔH / H_B) vs matched text-feature probe + shuffle
!python scripts/stage1_5_probe.py
!python scripts/stage1_5_probe.py --acts data/activations_w20.npz
```

---

### Family-P shakedown (validates the full machinery end-to-end)
The measurement/label cells already cover Family P. To freeze + probe it on its own:
```python
!python scripts/match_catch_miss.py --gated data/candidates_labeled.jsonl --require-gate false --family P --out data/feasibility_P.jsonl
!python scripts/extract_activations.py --frozen data/feasibility_P.jsonl
!python scripts/stage1_5_probe.py
```

### What to send back
`verify_token_invariants` (cell 7) + `label_candidates` (cell 9) output. Those give the attrition rate
and the clean MISS/CATCH yield — from which we scale Family-L toward the 30–40 balanced go-bar.
