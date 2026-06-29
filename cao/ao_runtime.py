"""
ao_runtime.py — minimal, self-contained runtime for the released Qwen3-8B CoT
Activation Oracle.

PROVENANCE. The injection/extraction semantics here are a clean reimplementation of
the verified behaviour in two Apache-2.0 repositories:
  - ceselder/cot-oracle  (src/core/ao.py)         — the released CoT Oracle runtime
  - adamkarvonen/activation_oracles (nl_probes/*) — the base AO infrastructure
We lift the *semantics*, not the files, to avoid their submodule tangle and heavy
training deps (Unsloth/vLLM/Ray). Attribution: see NOTICE.

GROUND TRUTH this encodes (confirmed against the repos, see docs/STAGE0.md):
  - Load: PeftModel.from_pretrained(base Qwen3-8B, <cot-oracle LoRA>) — standalone.
  - Extract activations with the adapter DISABLED (pure base representations).
  - Inject at layer 1 (output of block 1), ADDITIVE norm-matched, coefficient 1.0:
        h' = h + ||h|| * (v / ||v||)
    No forced target-norm rescale (natural ~sqrt(2)), no position-encoding, no pooling.
  - Oracle prompt placeholder format is GROUPED, ONE shared token " ?" per layer
    (the model card's interleaved " @ ? #" scheme is NOT what shipped):
        "L9:? ? ?  L18:? ? ?  L27:? ? ?.\n" + <query>
  - Activation tensor ordering is LAYER-MAJOR: [all L9 positions, then L18, then L27].
  - Generate/score with the adapter ENABLED; oracle answers with enable_thinking=False.
  - Read answers from a single forward pass: logits[0, -1], logsumexp over the
    case/space token variants of each answer string -> per-answer score; AUC over margins.

!!! UNTESTED IN THE AUTHORING SANDBOX: HuggingFace is egress-blocked there, so this
    module has never been executed against a real checkpoint. The FIRST run on a GPU
    box with HF access is the test. Assertions are deliberately load-bearing so a
    wiring mistake fails loudly rather than silently degrading. Run scripts/stage0_repro.py.
"""

from __future__ import annotations

import contextlib
from typing import Iterable

try:
    import torch
except ModuleNotFoundError:  # keep pure-Python helpers (build_grouped_prefix) importable
    torch = None             # torch-dependent functions fail only when actually called

DEFAULT_BASE = "Qwen/Qwen3-8B"
# The authors' eval default (configs/eval.yaml: method_config.our_ao.checkpoint).
# The model card documents the alternative `ceselder/cot-oracle-v4-8b`.
DEFAULT_ORACLE_LORA = "ceselder/cot-oracle-v15-stochastic"
PLACEHOLDER = " ?"          # single-token placeholder (verify id on the GPU box)
INJECTION_LAYER = 1         # inject at the OUTPUT of block 1 ("after the 2nd layer")
DEFAULT_LAYERS = (9, 18, 27)  # 25/50/75% of Qwen3-8B's 36 layers; in-distribution


# ----------------------------------------------------------------------------- #
# Model loading
# ----------------------------------------------------------------------------- #
def load_oracle(
    base: str = DEFAULT_BASE,
    oracle_lora: str = DEFAULT_ORACLE_LORA,
    dtype=None,                 # defaults to torch.bfloat16 at call time
    device_map: str = "auto",
):
    """Load base Qwen3-8B + the CoT-Oracle LoRA as a standalone adapter.

    Returns (model, tokenizer). The adapter is active by default; extraction code
    disables it via `model.disable_adapter()`.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    if dtype is None:
        dtype = torch.bfloat16
    tok = AutoTokenizer.from_pretrained(base)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    attn = "eager" if "gemma" in base.lower() else "sdpa"
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=dtype, device_map=device_map, attn_implementation=attn
    )
    model = PeftModel.from_pretrained(model, oracle_lora, is_trainable=False)
    model.eval()
    return model, tok


def get_block(model, layer: int):
    """Return transformer block `layer` across common HF/PEFT wrapper variants.

    The same module object is used whether or not the adapter is enabled, so it is
    valid both for extraction (adapter off) and injection (adapter on).
    """
    candidates = (
        lambda: model.base_model.model.model.layers[layer],
        lambda: model.base_model.model.layers[layer],
        lambda: model.model.model.layers[layer],
        lambda: model.model.layers[layer],
        lambda: model.base_model.language_model.layers[layer],
        lambda: model.language_model.layers[layer],
    )
    for fn in candidates:
        try:
            return fn()
        except (AttributeError, IndexError):
            continue
    raise ValueError(f"Could not locate transformer block {layer} on {type(model).__name__}")


def _model_device(model) -> torch.device:
    return next(model.parameters()).device


# ----------------------------------------------------------------------------- #
# Extraction (adapter DISABLED)
# ----------------------------------------------------------------------------- #
class _EarlyStop(Exception):
    pass


def _collect_layers(model, layers: Iterable[int], input_ids, attention_mask):
    """One forward pass; capture residual-stream output of each requested block.

    Returns {layer: tensor[1, seq_len, d_model]} (the block OUTPUT, i.e. the
    residual stream after that block). Stops early after the deepest layer.
    """
    layers = list(layers)
    grabbed: dict[int, torch.Tensor] = {}
    max_layer = max(layers)
    handles = []

    def make_hook(layer_idx):
        def hook(_module, _inp, out):
            grabbed[layer_idx] = (out[0] if isinstance(out, tuple) else out)
            if layer_idx == max_layer:
                raise _EarlyStop
        return hook

    for l in layers:
        handles.append(get_block(model, l).register_forward_hook(make_hook(l)))
    try:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
    except _EarlyStop:
        pass
    finally:
        for h in handles:
            h.remove()
    return grabbed


def extract_layer_major(
    model,
    tokenizer,
    templated_text: str,
    positions: list[int],
    layers: Iterable[int] = DEFAULT_LAYERS,
):
    """Extract activations at `positions` for each layer, adapter DISABLED.

    `templated_text` MUST already be the fully chat-templated TARGET string (the
    caller controls the target's chat template / enable_thinking). `positions` are
    absolute token indices into that string's tokenization.

    Returns a single tensor [len(layers) * len(positions), d_model] in LAYER-MAJOR
    order: [L0_pos0..L0_posK-1, L1_pos0.., ...] matching the grouped oracle prefix.
    """
    layers = list(layers)
    enc = tokenizer(templated_text, return_tensors="pt", add_special_tokens=False)
    dev = _model_device(model)
    input_ids = enc["input_ids"].to(dev)
    attn = enc["attention_mask"].to(dev)
    seq_len = input_ids.shape[1]
    for p in positions:
        assert 0 <= p < seq_len, f"position {p} out of range [0,{seq_len})"

    with model.disable_adapter():
        per_layer = _collect_layers(model, layers, input_ids, attn)

    pos = torch.tensor(positions, device=dev, dtype=torch.long)
    parts = [per_layer[l][0, pos, :] for l in layers]  # each [K, D]
    return torch.cat(parts, dim=0).detach()             # [K*N, D], layer-major


# ----------------------------------------------------------------------------- #
# Injection (adapter ENABLED) — additive, norm-matched, at layer 1
# ----------------------------------------------------------------------------- #
@contextlib.contextmanager
def _hook(module, fn):
    handle = module.register_forward_hook(fn)
    try:
        yield
    finally:
        handle.remove()


def _additive_steering_hook(vectors: torch.Tensor, positions: list[int], coefficient: float = 1.0):
    """h' = h + ||h|| * normalize(v) * coefficient, applied at `positions`.

    Only fires on the prompt forward pass (seq_len > 1), not autoregressive steps.
    `vectors[i]` is injected at the i-th entry of `positions` (same order).
    """
    normed = torch.nn.functional.normalize(vectors, dim=-1).detach()

    def fn(_module, _inp, out):
        is_tuple = isinstance(out, tuple)
        resid = out[0] if is_tuple else out
        rest = out[1:] if is_tuple else ()
        _b, L, _d = resid.shape
        if L <= 1:                              # decode step: do nothing
            return out
        pos = torch.tensor(positions, dtype=torch.long, device=resid.device)
        orig = resid[0, pos, :]
        norms = orig.norm(dim=-1, keepdim=True)
        steered = (normed.to(resid.device, resid.dtype) * norms * coefficient).to(resid.dtype)
        resid[0, pos, :] = (steered.detach() + orig)
        return (resid, *rest) if is_tuple else resid

    return fn


# ----------------------------------------------------------------------------- #
# Oracle prompt assembly (GROUPED, same-token; char-offset placeholder location)
# ----------------------------------------------------------------------------- #
def build_grouped_prefix(layers: list[int], k_positions: int, placeholder: str = PLACEHOLDER):
    """Return (prefix_str, char_spans).

    prefix_str: "L{l0}:{ph*k} L{l1}:{ph*k} ... .\n"  (grouped; one shared token).
    char_spans: list of (start,end) char offsets of each placeholder within prefix_str,
                in LAYER-MAJOR order (so char_spans[i] <-> activation row i).
    """
    prefix = ""
    char_spans: list[tuple[int, int]] = []
    for i, l in enumerate(layers):
        if i > 0:
            prefix += " "
        prefix += f"L{l}:"
        for _ in range(k_positions):
            start = len(prefix)
            prefix += placeholder
            char_spans.append((start, len(prefix)))
    prefix += ".\n"
    return prefix, char_spans


def _answer_first_token_ids(tokenizer, answer: str) -> list[int]:
    """Unique first-token ids over case/space variants of `answer`."""
    ids = set()
    for v in (answer, " " + answer, answer.upper(), " " + answer.upper(),
              answer.lower(), " " + answer.lower()):
        enc = tokenizer.encode(v, add_special_tokens=False)
        if enc:
            ids.add(enc[0])
    return sorted(ids)


def oracle_answer_logprobs(
    model,
    tokenizer,
    activations: torch.Tensor,          # [K*N, D], LAYER-MAJOR
    query: str,
    answer_tokens: list[str],
    layers: Iterable[int] = DEFAULT_LAYERS,
    injection_layer: int = INJECTION_LAYER,
    placeholder: str = PLACEHOLDER,
) -> dict[str, float]:
    """Inject `activations` and return a logsumexp score per answer string.

    Single forward pass with the adapter ENABLED; reads logits at the last position,
    log_softmax, then for each answer string logsumexp over its case/space token
    variants. (Matches the AObench scoring: margin = score[a] - score[b] -> AUC.)
    """
    layers = list(layers)
    P = activations.shape[0]
    N = len(layers)
    assert P % N == 0, f"{P} activation rows not divisible by {N} layers"
    K = P // N

    prefix, char_spans = build_grouped_prefix(layers, K, placeholder)
    full_prompt = prefix + query
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": full_prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    content_start = formatted.index(full_prompt)
    enc = tokenizer(formatted, add_special_tokens=False, return_offsets_mapping=True)
    input_ids, offsets = enc["input_ids"], enc["offset_mapping"]

    # Locate placeholder tokens by CHARACTER overlap (robust to BPE seam merges).
    positions: list[int] = []
    for rel_s, rel_e in char_spans:
        a, b = content_start + rel_s, content_start + rel_e
        hit = [i for i, (ts, te) in enumerate(offsets) if ts < b and te > a]
        if not hit:
            raise ValueError(f"no token for placeholder span {(rel_s, rel_e)!r}")
        positions.append(hit[0])
    assert len(positions) == P, f"located {len(positions)} placeholders, expected {P}"

    dev = _model_device(model)
    input_tensor = torch.tensor([input_ids], device=dev)
    attn = torch.ones_like(input_tensor)

    hook = _additive_steering_hook(activations.to(dev), positions)
    with torch.no_grad(), _hook(get_block(model, injection_layer), hook):
        out = model(input_ids=input_tensor, attention_mask=attn)

    logprobs = torch.log_softmax(out.logits[0, -1, :].float(), dim=-1)
    scores: dict[str, float] = {}
    for ans in answer_tokens:
        ids = _answer_first_token_ids(tokenizer, ans)
        if not ids:
            scores[ans] = float("-inf")
            continue
        scores[ans] = float(torch.logsumexp(logprobs[torch.tensor(ids, device=dev)], dim=0))
    return scores


def oracle_generate(
    model,
    tokenizer,
    activations: torch.Tensor,
    query: str,
    answer_tokens: list[str] | None = None,  # unused; kept for symmetry
    layers: Iterable[int] = DEFAULT_LAYERS,
    injection_layer: int = INJECTION_LAYER,
    placeholder: str = PLACEHOLDER,
    max_new_tokens: int = 64,
) -> str:
    """Free-form oracle answer (greedy). Useful for Stage-0 qualitative checks."""
    layers = list(layers)
    P = activations.shape[0]
    N = len(layers)
    assert P % N == 0
    K = P // N
    prefix, char_spans = build_grouped_prefix(layers, K, placeholder)
    full_prompt = prefix + query
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": full_prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    content_start = formatted.index(full_prompt)
    enc = tokenizer(formatted, add_special_tokens=False, return_offsets_mapping=True)
    input_ids, offsets = enc["input_ids"], enc["offset_mapping"]
    positions = []
    for rel_s, rel_e in char_spans:
        a, b = content_start + rel_s, content_start + rel_e
        hit = [i for i, (ts, te) in enumerate(offsets) if ts < b and te > a]
        positions.append(hit[0])
    dev = _model_device(model)
    input_tensor = torch.tensor([input_ids], device=dev)
    attn = torch.ones_like(input_tensor)
    hook = _additive_steering_hook(activations.to(dev), positions)
    with torch.no_grad(), _hook(get_block(model, injection_layer), hook):
        gen = model.generate(input_ids=input_tensor, attention_mask=attn,
                             max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(gen[0][len(input_ids):], skip_special_tokens=True)
