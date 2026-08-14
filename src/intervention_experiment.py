import os
import re
import json
import time
import torch
import torch.nn.functional as F
import hashlib
from functools import partial
from heapq import nlargest
from accelerate import Accelerator
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    set_seed,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from src.NIF import (
    build_train_dataset,
    build_single_sample_dataset,
    NewInferenceFunction,
    DatasetWrapper,
    CustomCollator,
    round_floats,
    _find_subseq_start,
)
from src.process_data import process_func_chatml
from src.export_real_ttav_bundle import token_surfaces_for_display
from src.loss import (
    compute_alti_correlation_gradient,
    compute_alti_match_and_probe_gradients,
    compute_alti_saliency_vector,
    compute_last_layer_correlation_gradient,
    compute_last_layer_match_and_probe_gradients,
    compute_last_layer_saliency_vector,
    compute_lm_head_ce_gradient_no_backward,
    compute_lm_head_ce_gradient_scores_no_backward,
    compute_lm_head_ce_gradient_sketches_no_backward,
)
from src.bank_loss import load_bank_loss_config, compute_bank_loss

# ====== DATA LOADING (auto-detects format) ======

def load_train_samples(jsonl_path: str) -> list[dict]:
    """Load **compact** train rows only (no ChatML re-encode later).

    Required per line:
      - ``input_ids``: list[int]  (ChatML already tokenized)
      - ``label`` or ``labels``: list[int], same length (prompt positions = -100)

    Optional:
      - ``attention_edges`` / ``edges``
      - ``uid`` / ``task_id`` / ``raw_id``
    """
    samples: list[dict] = []
    skipped = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            ids = obj.get("input_ids")
            labs = obj.get("label", obj.get("labels"))
            # Reject string gold "label" — that is eval text, not compact masks.
            if not isinstance(ids, list) or not ids:
                skipped += 1
                continue
            if not isinstance(labs, list) or len(labs) != len(ids):
                skipped += 1
                continue
            sample: dict = {
                "input_ids": [int(x) for x in ids],
                "labels": [int(x) for x in labs],
                "sample_index": len(samples),
                "task_id": str(obj.get("task_id", "") or obj.get("uid", "") or f"row_{line_no}"),
                "uid": str(obj.get("uid") or ""),
                "raw_id": str(obj.get("raw_id") or ""),
            }
            edges = obj.get("attention_edges")
            if edges is None:
                edges = obj.get("edges")
            if edges is not None:
                sample["attention_edges"] = edges
            samples.append(sample)
    if not samples:
        raise ValueError(
            f"No compact train samples in {jsonl_path!r}. "
            "Need JSONL rows with input_ids + label/labels (int lists). "
            "ChatML messages / prompt+response are not accepted for train data "
            "(re-encoding would misalign attention_edges)."
        )
    if skipped:
        print(
            f"[DEBUG] load_train_samples: kept {len(samples)}, skipped {skipped} "
            f"non-compact rows in {jsonl_path}",
            flush=True,
        )
    return samples


def load_samples(jsonl_path: str, *, compact_only: bool = False) -> list[dict]:
    """Load samples from a JSONL file.

    **Train data:** use ``load_train_samples`` or ``compact_only=True`` —
    compact ``input_ids`` + ``label``/``labels`` lists only.

    **Test / eval data** (``compact_only=False``) may also be text formats for
    generation:

    **Format A – messages array:**
    ``{"messages": [{"role": "system"|"user"|"assistant", ...}, ...]}``

    **Format B/C – flat fields:**
    ``{"prompt": "...", "response"|"label"|"predict": "..."}``

    **Format D – compact** (preferred when present):
    ``{"input_ids": [...], "label"|"labels": [...], "attention_edges": [...]}``
    """
    if compact_only:
        return load_train_samples(jsonl_path)

    samples: list[dict] = []
    seen_inputs: set[str] = set()

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)

            # ── Format D: compact (prefer; no re-encode) ──────────────────
            ids = obj.get("input_ids")
            labs = obj.get("label", obj.get("labels"))
            if isinstance(ids, list) and ids and isinstance(labs, list) and len(labs) == len(ids):
                sample = {
                    "input_ids": [int(x) for x in ids],
                    "labels": [int(x) for x in labs],
                    "sample_index": len(samples),
                    "task_id": str(obj.get("task_id", "") or obj.get("uid", "") or f"row_{line_no}"),
                    "uid": str(obj.get("uid") or ""),
                    "system": obj.get("system") or "",
                    "input": obj.get("input") or obj.get("prompt") or "",
                    "output": (
                        obj.get("output")
                        if isinstance(obj.get("output"), str)
                        else (obj.get("response") if isinstance(obj.get("response"), str) else "")
                    ),
                }
                edges = obj.get("attention_edges")
                if edges is None:
                    edges = obj.get("edges")
                if edges is not None:
                    sample["attention_edges"] = edges
                if isinstance(obj.get("predict"), str) and obj["predict"]:
                    sample["predict"] = obj["predict"]
                samples.append(sample)
                continue

            # ── Format A: messages array ──────────────────────────────────
            if "messages" in obj:
                msgs = obj["messages"]
                if len(msgs) < 3 or not msgs[2]["content"]:
                    continue
                system = msgs[0]["content"]
                inp = msgs[1]["content"]
                output = msgs[2]["content"]
                task_id = obj.get("task_id", "") or obj.get("uid", "")
                predict = obj.get("predict")

            # ── Format B/C: flat prompt + gold (+ optional predict) ───────
            elif "prompt" in obj and (
                "response" in obj or "label" in obj or "predict" in obj
            ):
                system = obj.get("system", "")
                inp = obj["prompt"]
                output = obj.get("response")
                if output is None or output == "":
                    # String label only (not list — list already handled as compact).
                    lab = obj.get("label", "")
                    output = lab if isinstance(lab, str) else ""
                task_id = str(obj.get("task_id", "") or obj.get("uid", ""))
                predict = obj.get("predict")
                if not output and not predict:
                    continue

            else:
                continue  # unrecognised format, skip

            if inp in seen_inputs:
                continue
            seen_inputs.add(inp)

            sample = {
                "system": system,
                "input": inp,
                "output": output or "",
                "task_id": task_id,
            }
            if isinstance(predict, str) and predict:
                sample["predict"] = predict
            edges = obj.get("attention_edges")
            if edges is None:
                edges = obj.get("edges")
            if edges is not None:
                sample["attention_edges"] = edges
            samples.append(sample)

    return samples


def _render_qwen_eval_prompt(
    tokenizer,
    user_text: str,
    system: str = "",
    *,
    enable_thinking: bool = False,
) -> str:
    """Match AI4Go ``generate_local.render_model_prompt`` (thinking off by default)."""
    sys_msg = system or "You are a helpful assistant."
    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_text},
    ]
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_template):
        try:
            return apply_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except (TypeError, ValueError):
            if not enable_thinking:
                messages[-1] = {
                    "role": "user",
                    "content": f"{user_text}\n/no_think",
                }
            try:
                return apply_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except (TypeError, ValueError):
                pass
    suffix = "" if enable_thinking else "\n/no_think"
    return (
        f"<|im_start|>system\n{sys_msg}<|im_end|>\n"
        f"<|im_start|>user\n{user_text}{suffix}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def _greedy_generate_on_prompt(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    *,
    max_new_tokens: int,
) -> tuple[list[int], str]:
    """AI4Go-style greedy decode (``run_sample44_local.generate``).

    Temporarily switches to ``eval()``: last_layer attribution keeps the model in
    ``train()`` for checkpointing, which + ``use_cache=True`` often yields garbage
    when used for ``generate``.
    """
    if prompt_ids.dim() == 1:
        prompt_batch = prompt_ids.unsqueeze(0)
    else:
        prompt_batch = prompt_ids
    was_training = bool(model.training)
    model.eval()
    try:
        im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        eos_ids = sorted({
            i for i in [tokenizer.eos_token_id, im_end] if i is not None
        })
        with torch.inference_mode():
            gen_out = model.generate(
                input_ids=prompt_batch,
                attention_mask=torch.ones_like(prompt_batch),
                max_new_tokens=max(1, int(max_new_tokens)),
                do_sample=False,
                num_beams=1,
                use_cache=True,
                eos_token_id=eos_ids if eos_ids else tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        plen = int(prompt_batch.size(1))
        new_ids = gen_out[0, plen:].tolist()
        text = tokenizer.decode(new_ids, skip_special_tokens=False)
        del gen_out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return new_ids, text
    finally:
        if was_training:
            model.train()
            for m in model.modules():
                if isinstance(m, torch.nn.Dropout):
                    m.eval()


def _left_truncate_prompt_for_attr(
    prompt_ids_list: list[int],
    pred_ids_list: list[int],
    attr_cap: int | None,
) -> list[int]:
    """Keep prompt+predict under ``attr_cap`` by left-truncating the prompt only."""
    if attr_cap is None or int(attr_cap) <= 0:
        return prompt_ids_list
    n_pred = len(pred_ids_list)
    if n_pred >= int(attr_cap):
        raise ValueError(
            f"completion tokens ({n_pred}) >= --attr-max-seq-len ({attr_cap}); "
            "raise the cap or shorten the completion."
        )
    max_prompt = int(attr_cap) - n_pred
    if len(prompt_ids_list) > max_prompt:
        dropped = len(prompt_ids_list) - max_prompt
        print(
            f"[DEBUG] Left-truncated prompt by {dropped} tokens so "
            f"prompt+completion <= {attr_cap} (train-aligned; "
            f"attn memory scales with T^2).",
            flush=True,
        )
        return prompt_ids_list[-max_prompt:]
    return prompt_ids_list


# ====== MODEL LOADING (generic — supports Qwen2, Qwen3, Qwen3-MoE, etc.) ======

def _patch_model_with_attn_hook(model: torch.nn.Module) -> torch.nn.Module:
    """Replace the Qwen2-specific subclass approach with a generic forward hook.

    After patching, any call to ``model(..., save_last_attention=True)`` will:
      1. Register a hook on the *last* self-attention module before the forward pass.
      2. Capture the attention weight tensor returned by that module.
      3. Remove the hook and return a :class:`CausalLMOutputWithPast` whose
         ``attentions`` field holds the captured weight for legacy callers that
         still use ``save_last_attention=True``.

    Works for dense models (Qwen2, Qwen3) **and** MoE models (Qwen3.5 35B A3B)
    because it only touches the attention module output, not the MoE routing.

    Requires ``attn_implementation="eager"`` so that the attention module actually
    computes and returns weight tensors (flash-attention variants return ``None``).
    """
    model.last_attention = None
    _orig_forward = model.forward

    def _patched_forward(*args, save_last_attention: bool = False, **kwargs):
        model.last_attention = None
        handle = None

        if save_last_attention:
            # Support both bare CausalLM and PeftModel wrappers.
            m = model
            if hasattr(m, "get_base_model"):
                try:
                    m = m.get_base_model()
                except Exception:
                    pass
            if hasattr(m, "model") and hasattr(m.model, "layers"):
                last_attn = m.model.layers[-1].self_attn
            elif hasattr(m, "layers"):
                last_attn = m.layers[-1].self_attn
            else:
                raise ValueError(f"Cannot find last self_attn on {type(model).__name__}")

            def _hook(_module, _inputs, output):
                # Attention modules return (attn_output, attn_weights[, past_kv, ...]).
                # attn_weights is a Tensor when attn_implementation="eager",
                # and None for flash/sdpa variants.
                if isinstance(output, tuple) and len(output) >= 2:
                    model.last_attention = output[1]

            handle = last_attn.register_forward_hook(_hook)

        try:
            outputs = _orig_forward(*args, **kwargs)
        finally:
            if handle is not None:
                handle.remove()

        if save_last_attention:
            return CausalLMOutputWithPast(
                loss=outputs.loss,
                logits=outputs.logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=(model.last_attention,) if model.last_attention is not None else None,
            )
        return outputs

    model.forward = _patched_forward
    return model


def load_model_and_tokenizer(
    model_path: str | None = None,
    attn_implementation: str = "eager",
    max_gpu_memory: str | None = None,
    base_model_path: str | None = None,
):
    """Load a full CausalLM checkpoint OR a PEFT LoRA adapter (+ base).

    If ``model_path`` contains ``adapter_config.json``, loads base then
    ``PeftModel.from_pretrained`` (viz-aligned). Tokenizer always comes from the
    base / full checkpoint, not the adapter folder alone.
    """
    if model_path is None:
        model_path = os.path.join(
            os.path.dirname(__file__), "sft", "scripts", "nif-checkpoints", "checkpoint-full"
        )

    model_path = os.path.abspath(os.path.expanduser(model_path))
    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"model_path is not a local directory: {model_path!r}. "
            "Gold/Unlearn only load local checkpoints (not HuggingFace hub repo ids)."
        )

    adapter_dir = _is_peft_adapter_dir(model_path)
    max_memory = (
        {i: max_gpu_memory for i in range(torch.cuda.device_count())}
        if max_gpu_memory and torch.cuda.is_available()
        else None
    )

    if adapter_dir:
        from peft import PeftModel

        base_path = _resolve_base_model_path(model_path, base_model_path)
        print(f"Loading PEFT adapter from {model_path}", flush=True)
        print(f"  base model: {base_path}", flush=True)
        print(f"Using attention implementation: {attn_implementation}", flush=True)

        tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
        config = AutoConfig.from_pretrained(
            base_path,
            attn_implementation=attn_implementation,
            output_attentions=False,
            use_cache=False,
        )
        base = AutoModelForCausalLM.from_pretrained(
            base_path,
            config=config,
            device_map="auto",
            max_memory=max_memory,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        model = PeftModel.from_pretrained(
            base,
            os.path.abspath(model_path),
            local_files_only=True,
        )
        # Only LoRA params participate in attribution grads (matches viz).
        for n, p in model.named_parameters():
            p.requires_grad = ("lora_" in n)
        n_lora = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
        print(f"  LoRA trainable params: {n_lora / 1e6:.2f}M", flush=True)
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.eval()
        model = _patch_model_with_attn_hook(model)
        model._eif_grad_space = "lora"
        model._eif_base_model_path = base_path
        return model, tokenizer

    print(f"Loading model from {model_path}...")
    print(f"Using attention implementation: {attn_implementation}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    config = AutoConfig.from_pretrained(
        model_path,
        attn_implementation=attn_implementation,
        output_attentions=False,
        use_cache=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        device_map="auto",
        max_memory=max_memory,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.eval()
    model = _patch_model_with_attn_hook(model)
    model._eif_grad_space = "fine_attn"
    model._eif_base_model_path = model_path
    return model, tokenizer


def _is_peft_adapter_dir(path: str | None) -> bool:
    if not path:
        return False
    return os.path.isfile(os.path.join(path, "adapter_config.json"))


def _resolve_base_model_path(adapter_path: str, base_model_path: str | None) -> str:
    """Resolve base LM for a LoRA adapter (handles relative base_model_name_or_path)."""
    if base_model_path:
        if not os.path.isdir(base_model_path):
            raise FileNotFoundError(f"--base-model-path not found: {base_model_path}")
        return os.path.abspath(base_model_path)

    from peft import PeftConfig

    try:
        cfg = PeftConfig.from_pretrained(adapter_path, local_files_only=True)
    except TypeError:
        cfg = PeftConfig.from_pretrained(adapter_path)
    recorded = getattr(cfg, "base_model_name_or_path", None) or ""
    candidates: list[str] = []
    if recorded and os.path.isdir(recorded):
        candidates.append(recorded)

    adapter_abs = os.path.abspath(adapter_path)
    # Walk up looking for relative recorded path, e.g. models/Qwen2.5-Coder-7B-Instruct
    if recorded and not os.path.isabs(recorded):
        cur = adapter_abs
        for _ in range(8):
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cand = os.path.join(parent, recorded)
            if os.path.isdir(cand) and os.path.isfile(os.path.join(cand, "config.json")):
                candidates.append(cand)
            cur = parent

    # Common layout: <repo>/models/Qwen2.5-Coder-7B-Instruct next to outputs/
    for root_name in ("code-corr-annotation", "Empirical-Influence-Function"):
        pass
    # Sibling / nested defaults
    for rel in (
        "models/Qwen2.5-Coder-7B-Instruct",
        "code-corr-annotation/models/Qwen2.5-Coder-7B-Instruct",
    ):
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cand = os.path.join(here, rel)
        if os.path.isdir(cand) and os.path.isfile(os.path.join(cand, "config.json")):
            candidates.append(cand)
        parent = os.path.dirname(here)
        cand2 = os.path.join(parent, "code-corr-annotation", "models", "Qwen2.5-Coder-7B-Instruct")
        if os.path.isdir(cand2) and os.path.isfile(os.path.join(cand2, "config.json")):
            candidates.append(cand2)

    for cand in candidates:
        if os.path.isdir(cand) and os.path.isfile(os.path.join(cand, "config.json")):
            return os.path.abspath(cand)

    raise FileNotFoundError(
        f"Could not resolve base model for adapter {adapter_path!r} "
        f"(adapter_config base_model_name_or_path={recorded!r}). "
        f"Pass --base-model-path explicitly."
    )


def make_lora_param_filter(model=None, last_n_layers: int | None = None):
    """Select PEFT LoRA parameters (optionally restricted to the last N layers).

    When ``last_n_layers`` is set, bank / probe / ALTI-∇C share the same last-N
    LoRA subspace (needed on 24GB GPUs; avoids full-stack LoRA ALTI graphs).
    """

    allowed_layers = None
    if model is not None and last_n_layers is not None and int(last_n_layers) > 0:
        decoder = _get_decoder_layers_module(model)
        num_layers = len(decoder.layers)
        start_layer = max(0, num_layers - int(last_n_layers))
        allowed_layers = set(range(start_layer, num_layers))

    def _filter(name, param):
        if "lora_" not in name:
            return False
        if allowed_layers is None:
            return True
        match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
        if match is None:
            return False
        return int(match.group(1)) in allowed_layers

    return _filter


def _get_decoder_layers_module(model):
    """Return the module that owns ``.layers`` for Qwen (+ optional Peft unwrap)."""
    m = model
    if hasattr(m, "get_base_model"):
        try:
            m = m.get_base_model()
        except Exception:
            pass
    # CausalLM: .model.layers ; already-decoder: .layers
    if hasattr(m, "model") and hasattr(m.model, "layers"):
        return m.model
    if hasattr(m, "layers"):
        return m
    raise ValueError(
        f"Cannot locate decoder .layers on {type(model).__name__} "
        f"(unwrapped={type(m).__name__})."
    )


# ====== CONFIGURATION ======
SEED = 42
SELECTED_TEST_SAMPLE_INDEX = 58

TOP_K_PROMPT_TOKENS = 4        # How many test correlation features to extract
TOP_K_PROMPT_OFFSET = 0        # Skip this many higher-ranked sources (0→ranks 1..K; 4→ranks 5..8 if K=4)
TOP_K_TRAIN_SAMPLES = 10       # How many top train samples from coarse screening
TOP_TARGETS = None             # None = all valid train answer tokens; use --train-scan-3x3 for 3×3
TOP_K_SOURCE_PER_TARGET = 3    # Top source tokens per train target (saliency top-3)
# last_layer: viz/precompute-style single-layer ALTI (cheaper). full_alti: multi-layer rollout.
SALIENCY_MODE = "last_layer"
CONTEXT_WINDOW_SIZE = 3        # Tokens shown on each side of source/target for annotation
FINE_MATCH_LAST_N_LAYERS = 1   # LoRA / fine-attn grads: last N layers (bank + probe + ∇C)
FINE_MATCH_PROJ = "qk"         # Attention projections used for fine matching: qk, qkvo, vo, q/k/v/o, all
ALTI_CHUNK_SIZE = 8            # Query chunk size for ALTI contribution computation
ALTI_GRAD_CHUNK_SIZE = 32      # Pair-gradient starts fast and falls back on OOM
ALTI_GRAD_MAX_SEQ_LEN = None   # Skip ALTI-gradient pairs beyond this prefix length; <=0 disables

# All-tokens mode parameters
MAX_OUTPUT_TOKENS = 40         # Max response tokens to analyze in all-tokens mode
SEQUENCE_LENGTH_LIMIT = 3000   # Default guard for gradient-heavy train sample scans
# Cap prompt+completion for attribution to match training max_len (attn ~ T^2).
ATTR_MAX_SEQ_LEN = 3000
# Global pre-screen pool size. The full training set is scanned ONCE with the full-response
# CE gradient to obtain this pool, then each per-token re-ranking only scans the pool
# (COARSE_POOL_SIZE samples) instead of the full training set.
# Cost: 1 × N_train (global) + N_tokens × COARSE_POOL_SIZE (per-token re-rank)
COARSE_POOL_SIZE = 100
PRESCREEN_BATCH_SIZE = 1       # Increase via --prescreen-batch-size when GPU memory allows
PRESCREEN_SAMPLE_LIMIT = None  # Limit coarse prescreen scan for quick/debug runs
PRESCREEN_MAX_SEQ_LEN = 3000   # Skip longer train samples during prescreen/rerank; <=0 disables
PRESCREEN_LENGTH_SWEEP = (3000, 2500, 2000)  # Report skipped counts before prescreen starts
PRESCREEN_SKETCH_DIM = 8192    # <=0 disables cached TensorSketch coarse retrieval
PRESCREEN_SKETCH_SEED = 42
PRESCREEN_SKETCH_CACHE_DIR = ".cache/prescreen_sketch"  # legacy CE lm_head cache (unused by new retrieval)
# Train-sample retrieval bank: CE grads on the same fine-attn params as ALTI-gradient features.
# Queried by sketched f_test(s→t) (viz-style influence). First run builds; later runs reuse.
SALIENCY_TRAIN_BANK_CACHE_DIR = ".cache/saliency_train_bank"
TRAIN_RETRIEVAL_METHOD = "saliency_probe_bank"  # replaces CE Stage-0/2 for Top-K trains

# Token strings (after strip) that carry no semantic content and should be skipped
# in all-tokens mode. Single non-alphanumeric characters are also skipped.
_TRIVIAL_STRIPPED = {"{", "}", "(", ")", "[", "]", ",", ";"}
_CHAT_TEMPLATE_STRIPPED = {
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "system",
    "user",
    "assistant",
}


def lm_head_filter(name, param):
    """Only select the LM head weight — a single dense matrix that
    projects the last hidden state to vocabulary logits.
    This gives a compact, task-agnostic feature for every token prediction.
    """
    return name == "lm_head.weight"


def _fine_match_projection_names(proj_mode: str) -> tuple[str, ...]:
    """Map a compact projection mode such as 'qk' to module name fragments."""
    normalized = (proj_mode or "").lower().replace("_", "").replace("-", "").replace(",", "")
    if normalized == "all":
        normalized = "qkvo"
    if not normalized or any(ch not in "qkvo" for ch in normalized):
        raise ValueError(
            f"Unsupported fine-match projection mode: {proj_mode!r}. "
            "Use a combination of q/k/v/o, e.g. 'qk', 'vo', 'qkvo', or 'all'."
        )

    selected = set(normalized)
    return tuple(f"{ch}_proj" for ch in "qkvo" if ch in selected)


def make_attention_projection_filter(model, last_n_layers: int, proj_mode: str):
    """Select requested attention projection parameters in the final N decoder layers."""
    decoder = _get_decoder_layers_module(model)
    num_layers = len(decoder.layers)
    start_layer = max(0, num_layers - last_n_layers)
    projection_names = _fine_match_projection_names(proj_mode)

    def _filter(name, param):
        for layer_idx in range(start_layer, num_layers):
            prefix = f"model.layers.{layer_idx}.self_attn."
            # Peft names often look like ...base_model.model.model.layers.N...
            if f"layers.{layer_idx}.self_attn." in name and any(
                proj in name for proj in projection_names
            ):
                if "lora_" in name:
                    return False  # fine-attn mode ignores LoRA tensors if both exist
                return True
            if name.startswith(prefix) and any(proj in name for proj in projection_names):
                return True
        return False

    return _filter


def is_trivial_token(tokenizer, token_id: int) -> bool:
    """Return True for tokens that carry no semantic meaning.

    Skips chat-template tokens, role labels, pure whitespace, lone punctuation
    ({, }, (, ), [, ], comma, semicolon), and any single non-alphanumeric /
    non-underscore character.
    """
    all_special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    if token_id in all_special_ids:
        return True

    tok_str = tokenizer.decode([token_id])
    stripped = tok_str.strip()
    if not stripped:
        return True
    if stripped in _CHAT_TEMPLATE_STRIPPED:
        return True
    raw_tok = tokenizer.convert_ids_to_tokens([token_id])[0]
    if raw_tok.strip() in _CHAT_TEMPLATE_STRIPPED:
        return True
    if stripped in _TRIVIAL_STRIPPED:
        return True
    if len(stripped) == 1 and not (stripped.isalnum() or stripped == "_"):
        return True
    return False


def top_nontrivial_saliency_sources(
    tokenizer,
    input_ids_1d,
    sal_vec,
    k: int,
    *,
    offset: int = 0,
):
    """Return saliency sources ranked ``offset+1 .. offset+k`` (1-based ranks).

    Trivial / chat-template tokens are excluded before ranking.
    ``offset=0, k=4`` → ranks 1–4; ``offset=4, k=4`` → ranks 5–8.
    """
    k = max(1, int(k))
    offset = max(0, int(offset))
    candidates = (
        (idx, score)
        for idx, score in enumerate(sal_vec)
        if not is_trivial_token(tokenizer, int(input_ids_1d[idx].item()))
    )
    ranked = nlargest(offset + k, candidates, key=lambda x: x[1])
    return ranked[offset : offset + k]


def saliency_rank_filename_tag(
    k: int | None = None,
    offset: int | None = None,
) -> str:
    """Suffix for report files when not using the default ranks 1–4."""
    k = max(1, int(TOP_K_PROMPT_TOKENS if k is None else k))
    offset = max(0, int(TOP_K_PROMPT_OFFSET if offset is None else offset))
    if offset == 0 and k == 4:
        return ""
    start = offset + 1
    end = offset + k
    return f"_salr{start}-{end}"


def find_first_valid_token_index(tokenizer, input_ids_tensor, start_idx):
    """
    Skip formatting characters like \\n, \\t, spaces, {, } to find the
    first token that carries actual semantic meaning.
    """
    valid_token_index = start_idx
    input_len = input_ids_tensor.size(1)
    while valid_token_index < input_len:
        tok_id = input_ids_tensor[0, valid_token_index].item()
        tok_str = tokenizer.decode([tok_id])
        if tok_str.strip() not in ["", "{", "}"]:
            break
        valid_token_index += 1
    return valid_token_index


def get_context_window(tokenizer, input_ids_1d, idx, window=CONTEXT_WINDOW_SIZE):
    """
    Return a list of token strings centered on `idx`.
    The focal token is wrapped in →[...]← for easy visual identification during annotation.
    """
    seq_len = input_ids_1d.size(0)
    tokens = []
    for i in range(max(0, idx - window), min(seq_len, idx + window + 1)):
        tok_str = tokenizer.decode([input_ids_1d[i].item()])
        tokens.append(f"→[{tok_str}]←" if i == idx else tok_str)
    return tokens


def model_tag_from_path(model_path: str | None) -> str:
    """Derive a short model tag from a checkpoint path (e.g. .../merged/ce_saliency → ce_saliency)."""
    if not model_path:
        return "model"
    tag = os.path.basename(os.path.normpath(str(model_path))).strip()
    tag = re.sub(r"[^\w.\-]+", "_", tag).strip("._")
    return tag or "model"


def _safe_filename_token(value: str | None, fallback: str = "x") -> str:
    """Filesystem-safe token for report filenames."""
    if value is None:
        return fallback
    tag = re.sub(r"[^\w.\-]+", "_", str(value)).strip("._")
    return tag or fallback


def _report_base_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _write_json_report(report_json: dict, report_filename: str, accelerator) -> str | None:
    """Atomically write a report JSON from the main process and return its path.

    Uses a unique ``.tmp`` name then ``os.replace``. On NFS (e.g. /mnt/md*),
    replace can race / lose the temp file; fall back to a direct rewrite so a
    checkpoint flush never aborts a long all-tokens run.
    """
    if not accelerator.is_main_process:
        return None
    report_json = round_floats(report_json, 5)
    report_path = os.path.join(_report_base_dir(), report_filename)
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)

    tmp_path = (
        f"{report_path}.{os.getpid()}.{int(time.time() * 1000)}.tmp"
    )
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(report_json, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.replace(tmp_path, report_path)
        except OSError as exc:
            # NFS / concurrent cleaner: temp vanished or replace refused.
            print(
                f"[checkpoint][WARN] atomic replace failed ({exc}); "
                f"writing directly → {report_path}",
                flush=True,
            )
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report_json, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            if os.path.isfile(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
    except Exception:
        if os.path.isfile(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise
    return report_path


def _public_train_sample_details(train_sample_cache: dict) -> dict:
    """Strip in-memory Stage3 keys (tensors / candidate lists) before JSON dump."""
    return {
        k: {fk: fv for fk, fv in v.items() if not fk.startswith("_")}
        for k, v in train_sample_cache.items()
    }


def _resume_fingerprint(
    *,
    task_id: str,
    model_path: str | None,
    model_tag: str,
    test_index: int,
    prompt_len: int,
    full_token_ids: list,
    max_output_tokens: int,
    saliency_mode: str,
    bank_loss_mode: str,
    top_k_prompt: int,
    top_k_prompt_offset: int,
    top_k_train: int,
) -> dict:
    """Stable fields used to decide whether an on-disk report can be resumed."""
    ids = list(full_token_ids or [])
    return {
        "task_id": task_id,
        "model_path": str(model_path or ""),
        "model_name": model_tag,
        "test_sample_index": int(test_index),
        "prompt_len": int(prompt_len),
        "attr_token_count": max(0, len(ids) - int(prompt_len)),
        "attr_sha1_12": hashlib.sha1(
            ",".join(str(i) for i in ids[int(prompt_len):]).encode("utf-8")
        ).hexdigest()[:12],
        "max_output_tokens": int(max_output_tokens),
        "saliency_mode": str(saliency_mode),
        "bank_loss_mode": str(bank_loss_mode),
        "top_k_prompt_tokens": int(top_k_prompt),
        "top_k_prompt_offset": int(top_k_prompt_offset),
        "top_k_train_samples": int(top_k_train),
    }


def _try_load_all_tokens_progress(
    report_filename: str,
    expected_fp: dict,
) -> tuple[list, dict, set[int]] | None:
    """Load partial/final report for resume if fingerprint matches.

    Returns ``(per_token_results, train_sample_details, done_target_indices)``
    or ``None`` when missing / incompatible / complete with nothing left.
    """
    report_path = os.path.join(_report_base_dir(), report_filename)
    if not os.path.isfile(report_path):
        return None
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[resume] ignore unreadable report {report_path}: {exc}", flush=True)
        return None

    meta = data.get("experiment_meta") or {}
    cfg = meta.get("config") or {}
    baseline = data.get("test_sample_baseline") or {}
    ids = baseline.get("full_token_ids") or []
    prompt_len = int(baseline.get("prompt_len") or 0)
    got_fp = {
        "task_id": meta.get("task_id"),
        "model_path": str(meta.get("model_path") or ""),
        "model_name": meta.get("model_name"),
        "test_sample_index": int(meta.get("test_sample_index") or -1),
        "prompt_len": prompt_len,
        "attr_token_count": max(0, len(ids) - prompt_len),
        "attr_sha1_12": hashlib.sha1(
            ",".join(str(i) for i in ids[prompt_len:]).encode("utf-8")
        ).hexdigest()[:12],
        "max_output_tokens": int(meta.get("max_output_tokens") or 0),
        "saliency_mode": str(cfg.get("SALIENCY_MODE") or ""),
        "bank_loss_mode": str(cfg.get("BANK_LOSS_MODE") or ""),
        "top_k_prompt_tokens": int(cfg.get("TOP_K_PROMPT_TOKENS") or 0),
        "top_k_prompt_offset": int(cfg.get("TOP_K_PROMPT_OFFSET") or 0),
        "top_k_train_samples": int(cfg.get("TOP_K_TRAIN_SAMPLES") or 0),
    }
    mismatches = [k for k in expected_fp if expected_fp[k] != got_fp.get(k)]
    if mismatches:
        print(
            f"[resume] existing report fingerprint mismatch "
            f"({', '.join(mismatches)}); starting fresh → {report_path}",
            flush=True,
        )
        return None

    per_token = list(data.get("per_token_results") or [])
    details = dict(data.get("train_sample_details") or {})
    done = {
        int(r["target_token_index"])
        for r in per_token
        if isinstance(r, dict) and "target_token_index" in r
    }
    if not done:
        return None
    is_partial = bool(meta.get("is_partial") or meta.get("is_checkpoint"))
    if not is_partial and meta.get("stage") not in (None, "all_tokens_partial"):
        # Finished report with same fingerprint: still allow resume if caller
        # expects more tokens (e.g. raised --max-output-tokens).
        pass
    print(
        f"[resume] loaded {len(done)} finished target token(s) from {report_path}",
        flush=True,
    )
    return per_token, details, done


def _flat_grad_on_device(
    grads: list,
    filtered_params: list,
    target_device: torch.device,
) -> torch.Tensor:
    """Flatten and concatenate gradients onto `target_device` without moving to CPU."""
    return torch.cat([
        g.reshape(-1).to(target_device) if g is not None
        else torch.zeros(p.numel(), dtype=p.dtype, device=target_device)
        for g, p in zip(grads, filtered_params)
    ])


def _gather_scores(
    accelerator,
    local_scores: list[tuple[int, float]],
    device: torch.device,
) -> list[tuple[int, float]]:
    """Gather (train_idx, score) pairs from all processes into the main process.

    Handles variable-length lists per process by padding with sentinel -1 indices.
    In single-process mode this is a no-op.
    """
    if accelerator.num_processes == 1:
        return local_scores

    n = len(local_scores)
    all_lens = accelerator.gather(torch.tensor(n, device=device, dtype=torch.long))
    max_len = int(all_lens.max().item())

    padded_scores = torch.full((max_len,), float("nan"), device=device, dtype=torch.float32)
    padded_indices = torch.full((max_len,), -1, device=device, dtype=torch.long)
    if n > 0:
        padded_scores[:n] = torch.tensor([s for _, s in local_scores], device=device, dtype=torch.float32)
        padded_indices[:n] = torch.tensor([i for i, _ in local_scores], device=device, dtype=torch.long)

    all_scores = accelerator.gather(padded_scores)
    all_indices = accelerator.gather(padded_indices)

    valid = all_indices != -1
    return list(zip(
        all_indices[valid].cpu().tolist(),
        all_scores[valid].cpu().tolist(),
    ))


def _is_cuda_alloc_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return (
        "out of memory" in msg
        or "cublas_status_alloc_failed" in msg
        or "cublascreate" in msg
        or "cuda error: cublas" in msg
    )


def _clear_cuda_after_oom():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _format_prescreen_skip(train_indices: list[int], seq_lens: list[int]) -> str:
    pairs = ", ".join(
        f"{int(idx)}(len={int(seq_len)})"
        for idx, seq_len in zip(train_indices, seq_lens)
    )
    return pairs or "<empty>"


def _dataset_item_seq_len(item) -> int:
    attention_mask = item.get("attention_mask")
    if attention_mask is not None:
        if isinstance(attention_mask, torch.Tensor):
            return int(attention_mask.sum().item())
        return int(sum(attention_mask))

    input_ids = item["input_ids"]
    if isinstance(input_ids, torch.Tensor):
        return int(input_ids.numel())
    return len(input_ids)


def _prescreen_length_sweep_report(
    train_ds,
    thresholds: tuple[int, ...] | list[int] | None,
    active_max_seq_len: int | None,
    accelerator,
) -> list[dict]:
    """Print and return how many tokenized train samples each length limit excludes."""
    if not thresholds:
        return []
    if not accelerator.is_main_process:
        return []

    normalized = []
    for limit in thresholds:
        limit = int(limit)
        if limit > 0 and limit not in normalized:
            normalized.append(limit)
    if active_max_seq_len is not None and active_max_seq_len > 0 and active_max_seq_len not in normalized:
        normalized.append(int(active_max_seq_len))
    normalized.sort(reverse=True)
    if not normalized:
        return []

    lengths = [_dataset_item_seq_len(train_ds[i]) for i in range(len(train_ds))]
    total = len(lengths)
    if total == 0:
        print("[DEBUG] Prescreen length sweep: train set is empty.", flush=True)
        return []

    stats = []
    previous_skipped = None
    print("\n=== Prescreen length-limit sweep ===", flush=True)
    print(f"  Tokenized train samples: {total}", flush=True)
    for limit in normalized:
        skipped = sum(1 for length in lengths if length > limit)
        kept = total - skipped
        skipped_pct = 100.0 * skipped / total
        kept_pct = 100.0 * kept / total
        extra = 0 if previous_skipped is None else skipped - previous_skipped
        current = "  <-- active" if active_max_seq_len == limit else ""
        print(
            f"  max_seq_len={limit:>5}: keep {kept:>6}/{total} ({kept_pct:5.1f}%), "
            f"skip {skipped:>6} ({skipped_pct:5.1f}%), "
            f"extra_vs_prev {extra:>6}{current}",
            flush=True,
        )
        stats.append({
            "max_seq_len": int(limit),
            "kept": int(kept),
            "skipped": int(skipped),
            "kept_pct": kept_pct,
            "skipped_pct": skipped_pct,
            "extra_skipped_vs_previous": int(extra),
            "active": active_max_seq_len == limit,
        })
        previous_skipped = skipped
    print("=====================================\n", flush=True)
    return stats


def _screen_training_set(
    model,
    test_ce_grad: torch.Tensor,   # flat tensor already on lm_head_device
    train_loader,
    accel_device: torch.device,   # accelerator.device for batch loading
    lm_head_device: torch.device, # device where lm_head lives (grad computed here)
    desc: str = "Scanning Train Samples",
    allowed_indices: set | None = None,
    max_seq_len: int | None = None,
) -> list[tuple[int, float]]:
    """Compute cosine similarity on GPU (lm_head_device). Returns LOCAL scores for this process."""
    sample_scores: list[tuple[int, float]] = []
    empty_ignored = torch.tensor([], device=accel_device)
    skipped_long = 0
    skipped_oom = 0

    for batch in tqdm(train_loader, desc=desc, leave=False):
        train_indices = batch["sample_index"].view(-1).tolist()
        keep_rows = []
        for row, train_idx in enumerate(train_indices):
            seq_len = int(batch["attention_mask"][row].sum().item())
            if allowed_indices is not None and train_idx not in allowed_indices:
                continue
            if max_seq_len is not None:
                if seq_len > max_seq_len:
                    skipped_long += 1
                    continue
            keep_rows.append(row)

        if not keep_rows:
            continue

        pending_groups = [keep_rows]
        while pending_groups:
            group_rows = pending_groups.pop()
            rows = torch.tensor(group_rows, dtype=torch.long, device=batch["input_ids"].device)
            batch_kept = {
                k: v.index_select(0, rows).to(accel_device)
                for k, v in batch.items()
                if isinstance(v, torch.Tensor) and k != "sample_index"
            }
            try:
                scores = compute_lm_head_ce_gradient_scores_no_backward(
                    model=model,
                    batch=batch_kept,
                    device=accel_device,
                    ignored_token_ids=empty_ignored,
                    test_ce_grad=test_ce_grad,
                    score_device=lm_head_device,
                )
            except (torch.OutOfMemoryError, RuntimeError) as exc:
                if not _is_cuda_alloc_error(exc):
                    raise
                del batch_kept, rows
                _clear_cuda_after_oom()
                if len(group_rows) > 1:
                    pending_groups.extend([[row] for row in reversed(group_rows)])
                    print(
                        f"[WARN] {desc}: OOM on batch of {len(group_rows)} samples; "
                        "retrying one sample at a time.",
                        flush=True,
                    )
                    continue

                skipped_oom += 1
                row = group_rows[0]
                print(
                    f"[WARN] {desc}: skipping train sample after OOM: "
                    f"{_format_prescreen_skip([train_indices[row]], [int(batch['attention_mask'][row].sum().item())])}",
                    flush=True,
                )
                continue

            for row, score in zip(group_rows, scores):
                sample_scores.append((int(train_indices[row]), float(score)))

            del batch_kept, scores, rows

    if skipped_long:
        print(f"[DEBUG] {desc}: skipped {skipped_long} samples longer than {max_seq_len} tokens.", flush=True)
    if skipped_oom:
        print(f"[WARN] {desc}: skipped {skipped_oom} samples after CUDA OOM.", flush=True)

    return sample_scores


def _dataset_fingerprint(train_ds, *, max_seq_len: int | None, sketch_dim: int, sketch_seed: int) -> str:
    """Stable-enough fingerprint for the tokenized train set used by the sketch cache."""
    h = hashlib.sha1()
    h.update(f"n={len(train_ds)}|max_seq_len={max_seq_len}|dim={sketch_dim}|seed={sketch_seed}".encode())
    for i in range(len(train_ds)):
        item = train_ds[i]
        ids = item["input_ids"]
        labels = item.get("labels")
        sample_index = int(item.get("sample_index", i))
        if not isinstance(ids, torch.Tensor):
            ids = torch.tensor(ids)
        h.update(str(sample_index).encode())
        h.update(str(int(ids.numel())).encode())
        h.update(str(int(ids[: min(32, ids.numel())].long().sum().item())).encode())
        h.update(str(int(ids[-min(32, ids.numel()):].long().sum().item())).encode())
        if isinstance(labels, torch.Tensor):
            valid = int(labels.ne(-100).sum().item())
        else:
            valid = 0
        h.update(str(valid).encode())
    return h.hexdigest()[:16]


def _prescreen_sketch_cache_path(
    model,
    train_ds,
    *,
    max_seq_len: int | None,
    sketch_dim: int,
    sketch_seed: int,
    cache_dir: str,
) -> str:
    model_name = str(getattr(getattr(model, "config", None), "_name_or_path", "model"))
    model_hash = hashlib.sha1(model_name.encode()).hexdigest()[:8]
    data_hash = _dataset_fingerprint(
        train_ds,
        max_seq_len=max_seq_len,
        sketch_dim=sketch_dim,
        sketch_seed=sketch_seed,
    )
    return os.path.join(
        cache_dir,
        f"lmhead_sketch_{model_hash}_{data_hash}_d{sketch_dim}_s{sketch_seed}.pt",
    )


def _load_or_build_prescreen_sketch_cache(
    model,
    train_ds,
    train_loader,
    accelerator,
    *,
    max_seq_len: int | None,
    sketch_dim: int,
    sketch_seed: int,
    cache_dir: str,
):
    """Build/load cached low-dimensional train LM-head gradient sketches."""
    if sketch_dim <= 0:
        return None
    if accelerator.num_processes != 1:
        print("Prescreen sketch cache disabled for multi-process Accelerator runs.")
        return None

    cache_path = _prescreen_sketch_cache_path(
        model,
        train_ds,
        max_seq_len=max_seq_len,
        sketch_dim=sketch_dim,
        sketch_seed=sketch_seed,
        cache_dir=cache_dir,
    )
    if os.path.exists(cache_path):
        print(f"Loading prescreen sketch cache: {cache_path}")
        return torch.load(cache_path, map_location="cpu")

    os.makedirs(cache_dir, exist_ok=True)
    print(f"Building prescreen sketch cache: {cache_path}")
    print("  First run is expected to be slow; later test samples reuse this file.")
    sample_ids = []
    sketch_chunks = []
    empty_ignored = torch.tensor([], device=accelerator.device)
    skipped_long = 0
    skipped_oom = 0

    for batch in tqdm(train_loader, desc="Build Prescreen Sketch Cache", leave=False):
        train_indices = batch["sample_index"].view(-1).tolist()
        keep_rows = []
        for row, train_idx in enumerate(train_indices):
            seq_len = int(batch["attention_mask"][row].sum().item())
            if max_seq_len is not None:
                if seq_len > max_seq_len:
                    skipped_long += 1
                    continue
            keep_rows.append(row)

        if not keep_rows:
            continue

        pending_groups = [keep_rows]
        while pending_groups:
            group_rows = pending_groups.pop()
            rows = torch.tensor(group_rows, dtype=torch.long, device=batch["input_ids"].device)
            batch_kept = {
                k: v.index_select(0, rows).to(accelerator.device)
                for k, v in batch.items()
                if isinstance(v, torch.Tensor) and k != "sample_index"
            }
            try:
                sketches = compute_lm_head_ce_gradient_sketches_no_backward(
                    model=model,
                    batch=batch_kept,
                    device=accelerator.device,
                    ignored_token_ids=empty_ignored,
                    sketch_dim=sketch_dim,
                    sketch_seed=sketch_seed,
                ).detach().cpu().to(torch.float16)
            except (torch.OutOfMemoryError, RuntimeError) as exc:
                if not _is_cuda_alloc_error(exc):
                    raise
                del batch_kept, rows
                _clear_cuda_after_oom()
                if len(group_rows) > 1:
                    pending_groups.extend([[row] for row in reversed(group_rows)])
                    print(
                        "[WARN] Build Prescreen Sketch Cache: OOM on batch of "
                        f"{len(group_rows)} samples; retrying one sample at a time.",
                        flush=True,
                    )
                    continue

                skipped_oom += 1
                row = group_rows[0]
                print(
                    "[WARN] Build Prescreen Sketch Cache: skipping train sample after OOM: "
                    f"{_format_prescreen_skip([train_indices[row]], [int(batch['attention_mask'][row].sum().item())])}",
                    flush=True,
                )
                continue

            sample_ids.extend(int(train_indices[row]) for row in group_rows)
            sketch_chunks.append(sketches)
            del batch_kept, sketches, rows

    if not sketch_chunks:
        print("Prescreen sketch cache is empty; falling back to exact scanning.")
        return None

    cache = {
        "sample_ids": torch.tensor(sample_ids, dtype=torch.long),
        "sketches": torch.cat(sketch_chunks, dim=0).contiguous(),
        "sketch_dim": int(sketch_dim),
        "sketch_seed": int(sketch_seed),
        "max_seq_len": max_seq_len,
    }
    torch.save(cache, cache_path)
    if skipped_long:
        print(f"  Sketch cache skipped {skipped_long} samples longer than {max_seq_len} tokens.")
    if skipped_oom:
        print(f"  Sketch cache skipped {skipped_oom} samples after CUDA OOM.")
    print(f"  Saved {cache['sketches'].size(0)} train sketches.")
    return cache


def _score_prescreen_sketch_cache(
    query_sketch: torch.Tensor,
    cache,
    device,
    *,
    allowed_indices: set[int] | None = None,
    chunk_size: int = 4096,
) -> list[tuple[int, float]]:
    ids = cache["sample_ids"]
    sketches = cache["sketches"]
    if allowed_indices is not None:
        allowed = torch.tensor(sorted(int(x) for x in allowed_indices), dtype=torch.long)
        mask = torch.isin(ids, allowed)
        ids = ids[mask]
        sketches = sketches[mask]
    if ids.numel() == 0:
        return []

    q = F.normalize(query_sketch.to(device=device, dtype=torch.float32), dim=0, eps=1e-12)
    score_chunks = []
    chunk_size = max(1, int(chunk_size))
    start = 0
    while start < sketches.size(0):
        cur_chunk_size = min(chunk_size, sketches.size(0) - start)
        while True:
            try:
                sketch_chunk = sketches[start:start + cur_chunk_size].to(device=device, dtype=torch.float32)
                score_chunks.append(sketch_chunk.matmul(q).detach().cpu())
                del sketch_chunk
                break
            except (torch.OutOfMemoryError, RuntimeError) as exc:
                if not _is_cuda_alloc_error(exc) or cur_chunk_size == 1:
                    raise
                _clear_cuda_after_oom()
                cur_chunk_size = max(1, cur_chunk_size // 2)
                print(
                    "[WARN] Prescreen sketch scoring OOM; "
                    f"retrying with chunk_size={cur_chunk_size}.",
                    flush=True,
                )
        start += cur_chunk_size
    scores = torch.cat(score_chunks, dim=0)
    del q, score_chunks
    return list(zip(ids.cpu().tolist(), scores.tolist()))


def _project_flat_grad(g: torch.Tensor, sketch_dim: int, seed: int) -> torch.Tensor:
    """Deterministic count-sketch projection + L2 normalize (CPU float32). Same idea as viz."""
    g = g.detach().float().reshape(-1).cpu()
    n = int(g.numel())
    out = torch.zeros(sketch_dim, dtype=torch.float32)
    chunk = 2_000_000
    for offset in range(0, n, chunk):
        end = min(offset + chunk, n)
        idx = torch.arange(offset, end, dtype=torch.int64)
        buckets = ((idx * 2654435761 + seed) % sketch_dim).long()
        signs = (((idx * 40503 + seed * 9176) & 1).float() * 2.0 - 1.0)
        out.scatter_add_(0, buckets, g[offset:end] * signs)
    return F.normalize(out, dim=0, eps=1e-12)


def _filtered_params(model, param_filter_fn) -> list:
    return [p for n, p in model.named_parameters() if param_filter_fn is None or param_filter_fn(n, p)]


def _compute_ce_flat_grad_filtered(model, batch, param_filter_fn, device) -> torch.Tensor | None:
    """Flat ∇_θ L_bank on filtered params (CE or CE+saliency via compute_bank_loss)."""
    return _compute_bank_flat_grad_filtered(
        model, batch, param_filter_fn, device, cfg=None, edges=None, special_ids=None,
    )


def _compute_bank_flat_grad_filtered(
    model,
    batch,
    param_filter_fn,
    device,
    *,
    cfg,
    edges,
    special_ids,
) -> torch.Tensor | None:
    """Flat ∇_θ L_train on filtered params (same θ space as ALTI-gradient features)."""
    from src.bank_loss import BankLossConfig

    target_params = _filtered_params(model, param_filter_fn)
    if not target_params:
        raise RuntimeError("No parameters matched param_filter_fn for saliency train bank.")

    if cfg is None:
        cfg = BankLossConfig(loss_mode="ce_only")

    model.eval()
    model.zero_grad(set_to_none=True)
    original_flags = [(p, p.requires_grad) for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    for p in target_params:
        p.requires_grad_(True)

    try:
        with torch.enable_grad():
            loss, _used = compute_bank_loss(
                model,
                batch,
                device=device,
                cfg=cfg,
                edges=edges,
                special_ids=special_ids,
            )
            if loss is None or not torch.isfinite(loss):
                return None
            grads = torch.autograd.grad(
                loss,
                target_params,
                create_graph=False,
                retain_graph=False,
                allow_unused=True,
            )
        flat = torch.cat([
            (g.reshape(-1).detach().cpu().float()
             if g is not None else torch.zeros(p.numel(), dtype=torch.float32))
            for g, p in zip(grads, target_params)
        ])
        return flat
    finally:
        for p, flag in original_flags:
            p.requires_grad_(flag)
        model.zero_grad(set_to_none=True)


def _saliency_train_bank_cache_path(
    model_path: str | None,
    model,
    train_ds,
    *,
    max_seq_len: int | None,
    sketch_dim: int,
    sketch_seed: int,
    fine_match_proj: str,
    last_n_layers: int,
    bank_cache_tag: str,
    cache_dir: str,
    bank_left_truncate: bool = True,
) -> str:
    # Prefer the actual checkpoint path so ce_only / ce_saliency do not collide.
    model_key = str(model_path or getattr(getattr(model, "config", None), "_name_or_path", "model"))
    model_hash = hashlib.sha1(model_key.encode()).hexdigest()[:12]
    data_hash = _dataset_fingerprint(
        train_ds,
        max_seq_len=max_seq_len,
        sketch_dim=sketch_dim,
        sketch_seed=sketch_seed,
    )
    lt = "lt1" if bank_left_truncate else "lt0"
    tag = (
        f"bank_{bank_cache_tag}_{fine_match_proj}_L{last_n_layers}"
        f"_{model_hash}_{data_hash}_d{sketch_dim}_s{sketch_seed}_{lt}.pt"
    )
    return os.path.join(cache_dir, tag)


def _left_truncate_train_row_for_bank(
    row_batch: dict,
    edges,
    max_seq_len: int,
) -> tuple[dict, list | None, int]:
    """Left-truncate a single train row so real seq_len <= max_seq_len; remap edges.

    ``seq_len`` is the full tokenized sequence (ChatML prompt + response), not
    prompt-only. ``attention_edges`` indices are into that same sequence, so
    after dropping ``D`` leading tokens we keep edges with
    ``src' = src - D``, ``dst' = dst - D`` only when both land in ``[0, new_len)``.
    """
    mask = row_batch["attention_mask"][0]
    real_len = int(mask.sum().item())
    # Content is left-aligned; ignore right padding when measuring length.
    if real_len <= max_seq_len:
        # Still squeeze away unused pad so bank forward sees the true length.
        if int(row_batch["input_ids"].size(1)) != real_len:
            out = {}
            for k, v in row_batch.items():
                if isinstance(v, torch.Tensor) and v.dim() >= 2 and v.size(1) >= real_len:
                    out[k] = v[:, :real_len].contiguous()
                else:
                    out[k] = v
            return out, edges, 0
        return row_batch, edges, 0
    drop = real_len - int(max_seq_len)
    new_len = int(max_seq_len)
    out = {}
    for k, v in row_batch.items():
        if isinstance(v, torch.Tensor) and v.dim() >= 2 and v.size(1) >= real_len:
            out[k] = v[:, drop:real_len].contiguous()
        else:
            out[k] = v
    remapped = None
    if edges:
        remapped = []
        for e in edges:
            try:
                src = int(e.get("src", e.get("token_i_idx", -1)))
                dst = int(e.get("dst", e.get("token_j_idx", -1)))
            except (TypeError, ValueError):
                continue
            src_n, dst_n = src - drop, dst - drop
            if 0 <= src_n < new_len and 0 <= dst_n < new_len:
                ne = dict(e)
                ne["src"] = src_n
                ne["dst"] = dst_n
                remapped.append(ne)
    return out, remapped, drop


def _load_or_build_saliency_train_bank(
    model,
    train_ds,
    train_loader,
    accelerator,
    *,
    model_path: str | None,
    param_filter_fn,
    max_seq_len: int | None,
    sketch_dim: int,
    sketch_seed: int,
    fine_match_proj: str,
    last_n_layers: int,
    cache_dir: str,
    bank_cfg,
    train_samples: list[dict],
    special_ids: set[int],
    bank_cache_tag_override: str | None = None,
    bank_left_truncate: bool = True,
):
    """
    Build/load sketched train gradients on selected params (LoRA or fine-attn).

    Bank objective matches viz:
      ce_only     -> CE
      ce_saliency -> CE + λ * saliency loss (needs attention_edges on samples)

    Retrieval query is sketched g_probe = ∇(-log C) (viz L_probe).

    If ``bank_left_truncate`` and a row is longer than ``max_seq_len``, keep the
    rightmost ``max_seq_len`` tokens and remap ``attention_edges`` (do not skip).
    """
    if sketch_dim <= 0:
        print("[WARN] sketch_dim<=0: saliency train bank disabled.", flush=True)
        return None
    if accelerator.num_processes != 1:
        print("[WARN] Saliency train bank disabled for multi-process Accelerator runs.", flush=True)
        return None

    cache_tag = bank_cache_tag_override or bank_cfg.cache_tag
    cache_path = _saliency_train_bank_cache_path(
        model_path,
        model,
        train_ds,
        max_seq_len=max_seq_len,
        sketch_dim=sketch_dim,
        sketch_seed=sketch_seed,
        fine_match_proj=fine_match_proj,
        last_n_layers=last_n_layers,
        bank_cache_tag=cache_tag,
        cache_dir=cache_dir,
        bank_left_truncate=bank_left_truncate,
    )
    if os.path.exists(cache_path):
        print(f"Loading saliency train bank: {cache_path}", flush=True)
        blob = torch.load(cache_path, map_location="cpu")
        if int(blob.get("sketch_dim", -1)) != int(sketch_dim) or int(blob.get("sketch_seed", -1)) != int(sketch_seed):
            raise RuntimeError(
                f"Saliency train bank sketch mismatch. Delete {cache_path} and rebuild."
            )
        if blob.get("fine_match_proj") not in (None, fine_match_proj):
            raise RuntimeError(
                f"Saliency train bank proj mismatch (cache={blob.get('fine_match_proj')}, "
                f"need={fine_match_proj}). Delete {cache_path} and rebuild."
            )
        if blob.get("bank_loss_tag") not in (None, bank_cfg.cache_tag, cache_tag):
            raise RuntimeError(
                f"Saliency train bank loss-tag mismatch "
                f"(cache={blob.get('bank_loss_tag')}, need={cache_tag}). "
                f"Delete {cache_path} and rebuild."
            )
        return blob

    os.makedirs(cache_dir, exist_ok=True)
    print(f"Building saliency train bank: {cache_path}", flush=True)
    print(
        f"  Bank objective: {bank_cfg.loss_mode} (tag={bank_cfg.cache_tag}). "
        "First run is slow; later runs reuse this file. "
        "Legacy .cache/prescreen_sketch is NOT used.",
        flush=True,
    )
    print(
        f"  Bank length: max_seq_len={max_seq_len} "
        f"left_truncate={bank_left_truncate} "
        "(full ChatML prompt+response; edges remapped on truncate).",
        flush=True,
    )
    if bank_cfg.loss_mode == "ce_saliency":
        n_edges = sum(1 for s in train_samples if s.get("attention_edges"))
        print(
            f"  Samples with attention_edges: {n_edges}/{len(train_samples)}. "
            "Samples without edges fall back to CE-only for that row.",
            flush=True,
        )
        if n_edges == 0:
            print(
                "[WARN] ce_saliency bank requested but no attention_edges found in train JSONL. "
                "Use compact data that includes attention_edges, or the bank "
                "will effectively be CE-only.",
                flush=True,
            )

    sample_ids = []
    sketch_chunks = []
    skipped_long = 0
    truncated_rows = 0
    truncated_edges_kept = 0
    skipped_bad = 0
    used_cesal = 0
    used_ce = 0
    # OOM retry ladder (full-seq CE+saliency grads are much heavier than training forward).
    _oom_retry_caps = (2048, 1536, 1280, 1024, 768)

    for batch in tqdm(train_loader, desc="Build Saliency Train Bank", leave=False):
        train_indices = batch["sample_index"].view(-1).tolist()
        for row, train_idx in enumerate(train_indices):
            seq_len = int(batch["attention_mask"][row].sum().item())
            row_batch_full = {
                k: v[row:row + 1]
                for k, v in batch.items()
                if isinstance(v, torch.Tensor) and k != "sample_index"
            }
            edges_full = None
            if 0 <= int(train_idx) < len(train_samples):
                edges_full = train_samples[int(train_idx)].get("attention_edges")

            if max_seq_len is not None and seq_len > max_seq_len and not bank_left_truncate:
                skipped_long += 1
                continue

            target_cap = seq_len
            if max_seq_len is not None:
                target_cap = min(seq_len, int(max_seq_len))
            attempt_caps = [int(target_cap)]
            if bank_left_truncate:
                for cand in _oom_retry_caps:
                    if cand < target_cap and cand not in attempt_caps:
                        attempt_caps.append(int(cand))

            flat = None
            edges_used = edges_full
            seq_used = seq_len
            did_truncate = False
            for try_cap in attempt_caps:
                row_batch, edges_used, dropped = _left_truncate_train_row_for_bank(
                    row_batch_full, edges_full, int(try_cap)
                )
                seq_used = int(row_batch["input_ids"].size(1))
                if dropped:
                    did_truncate = True
                    n_e = 0 if not edges_used else len(edges_used)
                    print(
                        f"  [bank] left-truncate train_idx={train_idx}: "
                        f"{seq_len}→{seq_used} (drop {dropped}; edges kept={n_e})",
                        flush=True,
                    )
                try:
                    flat = _compute_bank_flat_grad_filtered(
                        model,
                        row_batch,
                        param_filter_fn,
                        accelerator.device,
                        cfg=bank_cfg,
                        edges=edges_used,
                        special_ids=special_ids,
                    )
                    break
                except ImportError:
                    raise
                except (torch.OutOfMemoryError, RuntimeError) as exc:
                    if isinstance(exc, RuntimeError) and not _is_cuda_alloc_error(exc):
                        raise
                    _clear_cuda_after_oom()
                    flat = None
                    if try_cap == attempt_caps[-1] or not bank_left_truncate:
                        skipped_bad += 1
                        print(
                            f"[WARN] Saliency bank OOM/skip train_idx={train_idx} "
                            f"seq_len={seq_used}",
                            flush=True,
                        )
                    else:
                        print(
                            f"[WARN] Saliency bank OOM train_idx={train_idx} "
                            f"seq_len={seq_used}; retry shorter left-truncate…",
                            flush=True,
                        )
                    continue
                finally:
                    del row_batch

            if flat is None:
                continue
            if flat.numel() == 0:
                skipped_bad += 1
                continue
            if did_truncate:
                truncated_rows += 1
                truncated_edges_kept += 0 if not edges_used else len(edges_used)
            if bank_cfg.loss_mode == "ce_saliency" and edges_used:
                used_cesal += 1
            else:
                used_ce += 1
            sketch = _project_flat_grad(flat, sketch_dim, sketch_seed).to(torch.float16)
            sample_ids.append(int(train_idx))
            sketch_chunks.append(sketch.unsqueeze(0))
            del flat, sketch, row_batch_full
            torch.cuda.empty_cache()

    if not sketch_chunks:
        print("[WARN] Saliency train bank empty; cannot retrieve trains by saliency probe.", flush=True)
        return None

    cache = {
        "sample_ids": torch.tensor(sample_ids, dtype=torch.long),
        "sketches": torch.cat(sketch_chunks, dim=0).contiguous(),
        "sketch_dim": int(sketch_dim),
        "sketch_seed": int(sketch_seed),
        "max_seq_len": max_seq_len,
        "bank_left_truncate": bool(bank_left_truncate),
        "fine_match_proj": fine_match_proj,
        "last_n_layers": int(last_n_layers),
        "bank_type": "train_obj_on_fine_attn",
        "bank_loss_tag": cache_tag,
        "bank_loss_mode": bank_cfg.loss_mode,
        "model_path": str(model_path) if model_path else None,
        "retrieval": TRAIN_RETRIEVAL_METHOD,
        "used_ce_rows": used_ce,
        "used_cesal_rows": used_cesal,
    }
    torch.save(cache, cache_path)
    if truncated_rows:
        print(
            f"  Bank left-truncated {truncated_rows} long samples "
            f"(edges remapped; kept≈{truncated_edges_kept} edges total).",
            flush=True,
        )
    if skipped_long:
        print(f"  Bank skipped {skipped_long} samples longer than {max_seq_len} tokens.", flush=True)
    if skipped_bad:
        print(f"  Bank skipped {skipped_bad} samples after OOM/bad loss.", flush=True)
    print(
        f"  Saved {cache['sketches'].size(0)} train sketches "
        f"(ce_rows={used_ce}, cesal_rows={used_cesal}) → {cache_path}",
        flush=True,
    )
    return cache


def run_causal_intervention_experiment(
    model_path: str | None = None,
    train_data: str = "sft_train.jsonl",
    test_data: str = "sft_test.jsonl",
    train_limit: int | None = None,
    attn_implementation: str | None = None,
    prescreen_max_seq_len: int | None = SEQUENCE_LENGTH_LIMIT,
    max_gpu_memory: str | None = None,
    corr_feature_mode: str = "auto",
    prescreen_batch_size: int = 1,
    alti_grad_chunk_size: int = ALTI_GRAD_CHUNK_SIZE,
    alti_grad_max_seq_len: int | None = ALTI_GRAD_MAX_SEQ_LEN,
    fine_match_proj: str = FINE_MATCH_PROJ,
    prescreen_sketch_dim: int = PRESCREEN_SKETCH_DIM,
    prescreen_sketch_seed: int = PRESCREEN_SKETCH_SEED,
    prescreen_sketch_cache_dir: str = PRESCREEN_SKETCH_CACHE_DIR,
    saliency_train_bank_cache_dir: str = SALIENCY_TRAIN_BANK_CACHE_DIR,
    prescreen_length_sweep: tuple[int, ...] | list[int] | None = PRESCREEN_LENGTH_SWEEP,
    base_model_path: str | None = None,
    completion_source: str = "auto",
    bank_loss_mode: str | None = None,
    live_generate_compare: bool = True,
    saliency_mode: str | None = None,
    attr_max_seq_len: int | None = ATTR_MAX_SEQ_LEN,
    bank_left_truncate: bool = True,
    bank_max_seq_len: int | None = None,
    resume: bool = True,
):
    import sys; sys.stdout.reconfigure(line_buffering=True)
    print("[DEBUG] Initializing Accelerator...", flush=True)
    accelerator = Accelerator()
    print(f"[DEBUG] Accelerator ready. num_processes={accelerator.num_processes}, device={accelerator.device}", flush=True)
    set_seed(SEED)

    if attn_implementation is None:
        # ALTI needs materialized attention probabilities. SDPA/flash attention
        # often returns None for output_attentions in Qwen-family models.
        attn_implementation = "eager"
    if prescreen_max_seq_len is not None and prescreen_max_seq_len <= 0:
        prescreen_max_seq_len = None
    prescreen_batch_size = max(1, int(prescreen_batch_size))
    alti_grad_chunk_size = max(1, int(alti_grad_chunk_size))
    if alti_grad_max_seq_len is not None and alti_grad_max_seq_len <= 0:
        alti_grad_max_seq_len = None
    _saliency_mode = (saliency_mode or SALIENCY_MODE or "last_layer").strip().lower()
    if _saliency_mode not in {"last_layer", "full_alti"}:
        raise ValueError(
            f"Unknown saliency_mode={saliency_mode!r}; use last_layer|full_alti"
        )
    if _saliency_mode == "last_layer":
        _saliency_fn = compute_last_layer_saliency_vector
        _match_probe_fn = compute_last_layer_match_and_probe_gradients
        _corr_grad_fn = compute_last_layer_correlation_gradient
    else:
        _saliency_fn = compute_alti_saliency_vector
        _match_probe_fn = compute_alti_match_and_probe_gradients
        _corr_grad_fn = compute_alti_correlation_gradient
    print(
        f"[DEBUG] saliency_mode={_saliency_mode}  "
        f"train_scan≈{TOP_TARGETS or 'all'}×{TOP_K_SOURCE_PER_TARGET}",
        flush=True,
    )
    prescreen_sketch_dim = int(prescreen_sketch_dim or 0)
    prescreen_limit = PRESCREEN_SAMPLE_LIMIT
    fine_match_proj = (fine_match_proj or FINE_MATCH_PROJ).lower()
    fine_match_projection_names = _fine_match_projection_names(fine_match_proj)
    model, tokenizer = load_model_and_tokenizer(
        model_path,
        attn_implementation=attn_implementation,
        max_gpu_memory=max_gpu_memory,
        base_model_path=base_model_path,
    )
    if _saliency_mode == "last_layer":
        # Match viz/data_attribution: checkpointing cuts activation memory on backward.
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()
        model.train()
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.eval()
        print(
            "[DEBUG] gradient checkpointing ON for last_layer probe "
            "(dropout frozen; viz-aligned)",
            flush=True,
        )
    model_tag = model_tag_from_path(model_path)
    bank_cfg = load_bank_loss_config(
        model_path,
        model_tag,
        loss_mode_override=bank_loss_mode,
    )
    grad_space = getattr(model, "_eif_grad_space", "fine_attn")
    print(f"[DEBUG] model_tag={model_tag}  grad_space={grad_space}", flush=True)
    _bank_mode_src = (
        f"cli:{bank_loss_mode}"
        if bank_loss_mode and str(bank_loss_mode).strip().lower() not in ("", "auto", "none")
        else "auto"
    )
    print(
        f"[intervention] train_retrieval={TRAIN_RETRIEVAL_METHOD}  "
        f"bank_loss={bank_cfg.loss_mode}({bank_cfg.cache_tag}, src={_bank_mode_src})  "
        f"grad_space={grad_space}  "
        f"report_prefix=correlation_matching_results_{model_tag}_<task_id>_all_tokens.json",
        flush=True,
    )
    if corr_feature_mode != "auto":
        print(
            f"[DEBUG] --corr-feature-mode={corr_feature_mode} is ignored; "
            "fine matching now always uses ALTI-gradient features.",
            flush=True,
        )
    if grad_space == "lora":
        fine_param_filter = make_lora_param_filter(
            model, last_n_layers=FINE_MATCH_LAST_N_LAYERS
        )
        print(
            "[DEBUG] correlation / probe / train-bank gradients: LoRA params "
            f"(last {FINE_MATCH_LAST_N_LAYERS} layer(s); aligned bank↔probe↔∇C)",
            flush=True,
        )
        bank_grad_tag = f"lora_L{FINE_MATCH_LAST_N_LAYERS}_{bank_cfg.cache_tag}"
        match_desc = f"alti_gradient_lora_L{FINE_MATCH_LAST_N_LAYERS}"
    else:
        fine_param_filter = make_attention_projection_filter(
            model, FINE_MATCH_LAST_N_LAYERS, fine_match_proj
        )
        print(
            "[DEBUG] correlation feature mode: "
            f"alti_gradient_{fine_match_proj} over last {FINE_MATCH_LAST_N_LAYERS} layer(s) "
            f"({', '.join(fine_match_projection_names)})",
            flush=True,
        )
        bank_grad_tag = f"fineattn_{fine_match_proj}_L{FINE_MATCH_LAST_N_LAYERS}_{bank_cfg.cache_tag}"
        match_desc = f"alti_gradient_{fine_match_proj}"
    _stage3_tgt = "all valid pred tokens" if TOP_TARGETS is None else f"first {TOP_TARGETS} valid pred tokens"
    print(
        f"[DEBUG] Stage3 train scan: {_stage3_tgt} × saliency top-{TOP_K_SOURCE_PER_TARGET} sources "
        f"(default: all×{TOP_K_SOURCE_PER_TARGET}; quick 3×3: --train-scan-3x3)",
        flush=True,
    )
    print("[DEBUG] Model and tokenizer loaded.", flush=True)
    param_filter = lm_head_filter

    convert_to_chatml = partial(process_func_chatml, tokenizer=tokenizer)

    print("[DEBUG] Loading train/test samples...", flush=True)
    train_samples = load_train_samples(train_data)
    test_samples = load_samples(test_data)
    if train_limit is not None and train_limit < len(train_samples):
        print(f"[DEBUG] Limiting train samples: {len(train_samples)} -> {train_limit}", flush=True)
        train_samples = train_samples[:train_limit]
    print(f"[DEBUG] Loaded {len(train_samples)} train, {len(test_samples)} test samples.", flush=True)
    if not test_samples:
        raise ValueError(
            f"No test samples loaded from {test_data!r}. "
            "Need compact input_ids+labels, or messages[] / prompt+label|response|predict."
        )

    base_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model,
        padding=True, label_pad_token_id=-100, return_tensors="pt",
    )
    collator = CustomCollator(base_collator)

    print("[DEBUG] Building train dataset (compact, no re-encode)...", flush=True)
    train_ds = build_train_dataset(train_samples)
    print(f"[DEBUG] Train dataset built: {len(train_ds)} samples.", flush=True)
    prescreen_length_sweep_stats = _prescreen_length_sweep_report(
        train_ds,
        prescreen_length_sweep,
        prescreen_max_seq_len,
        accelerator,
    )
    train_loader = torch.utils.data.DataLoader(
        DatasetWrapper(train_ds), batch_size=prescreen_batch_size, collate_fn=collator,
    )
    print("[DEBUG] Calling accelerator.prepare(train_loader)...", flush=True)
    train_loader = accelerator.prepare(train_loader)
    print("[DEBUG] train_loader prepared.", flush=True)
    # Legacy CE lm_head sketch (.cache/prescreen_sketch) is no longer used for Top-K
    # train retrieval. Build/load the saliency influence bank instead.
    if prescreen_sketch_cache_dir and os.path.isdir(prescreen_sketch_cache_dir):
        print(
            f"[DEBUG] Note: legacy CE sketch dir {prescreen_sketch_cache_dir} is unused by "
            f"{TRAIN_RETRIEVAL_METHOD}. Safe to delete if you do not need old runs.",
            flush=True,
        )
    # Bank length guard: full ChatML seq (prompt+response), not prompt-only.
    # Default to prescreen_max_seq_len; override with --bank-max-seq-len for tighter VRAM.
    if bank_max_seq_len is None:
        bank_max_seq_len = prescreen_max_seq_len
    if bank_max_seq_len is not None and bank_max_seq_len <= 0:
        bank_max_seq_len = None

    saliency_train_bank = _load_or_build_saliency_train_bank(
        model,
        train_ds,
        train_loader,
        accelerator,
        model_path=model_path,
        param_filter_fn=fine_param_filter,
        max_seq_len=bank_max_seq_len,
        sketch_dim=prescreen_sketch_dim,
        sketch_seed=prescreen_sketch_seed,
        fine_match_proj=fine_match_proj if grad_space != "lora" else f"lora_L{FINE_MATCH_LAST_N_LAYERS}",
        last_n_layers=FINE_MATCH_LAST_N_LAYERS,
        cache_dir=saliency_train_bank_cache_dir,
        bank_cfg=bank_cfg,
        train_samples=train_samples,
        special_ids=set(getattr(tokenizer, "all_special_ids", []) or []),
        bank_cache_tag_override=bank_grad_tag,
        bank_left_truncate=bank_left_truncate,
    )
    if saliency_train_bank is None:
        raise RuntimeError(
            "Saliency train bank is required for Top-K train retrieval. "
            "Check --prescreen-sketch-dim (>0) and GPU memory, then retry."
        )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if _saliency_mode == "last_layer":
        # Bank grads leave the model in eval(), which disables checkpointing.
        from src.loss import prepare_last_layer_grad_checkpointing
        prepare_last_layer_grad_checkpointing(model)
        print(
            "[DEBUG] Restored train+grad-checkpointing after bank "
            "(needed for long last_layer probe).",
            flush=True,
        )

    infer_fw = NewInferenceFunction(
        model=model, tokenizer=tokenizer,
        train_loader=train_loader, accelerator=accelerator,
        param_filter_fn=param_filter, top_k=20,
    )
    print("[DEBUG] NewInferenceFunction created.", flush=True)

    # lm_head might live on a different device under device_map="auto"
    filtered_params = [p for n, p in model.named_parameters() if param_filter(n, p)]
    lm_head_device  = filtered_params[0].device if filtered_params else accelerator.device
    print(f"[DEBUG] lm_head_device={lm_head_device}", flush=True)

    marker_ids = tuple(tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False))

    # ── Build full test sequence (prompt + generated response) ──────────────
    # Prefer compact input_ids/label like viz/precompute._free_run so the prompt
    # is exactly ids[:first_label!=-100]. Fall back to ChatML re-encode otherwise.
    _cur_test     = test_samples[SELECTED_TEST_SAMPLE_INDEX]
    # Use task_id for output file naming; fall back to index if absent.
    _task_id = _safe_filename_token(
        _cur_test.get("task_id") or f"test{SELECTED_TEST_SAMPLE_INDEX}",
        fallback=f"test{SELECTED_TEST_SAMPLE_INDEX}",
    )
    print(
        f"[DEBUG] report files → correlation_matching_results_{model_tag}_{_task_id}_all_tokens"
        f"{saliency_rank_filename_tag()}*.json "
        f"(saliency ranks {TOP_K_PROMPT_OFFSET + 1}-"
        f"{TOP_K_PROMPT_OFFSET + TOP_K_PROMPT_TOKENS})",
        flush=True,
    )

    # Keep last_layer attribution in train+GC mode; do not flip to eval() here
    # (eval disables checkpointing and OOMs near attr-max-seq-len≈3000).
    if _saliency_mode != "last_layer":
        infer_fw.model.eval()
    _compact_ids = _cur_test.get("input_ids")
    _compact_lbl = _cur_test.get("labels")
    _fixed_predict = _cur_test.get("predict")
    _completion_source = (completion_source or "auto").strip().lower()
    if _completion_source not in {"auto", "predict", "generate"}:
        raise ValueError(
            f"Unknown completion_source={completion_source!r}; "
            "use auto|predict|generate"
        )
    _use_fixed_predict = (
        _completion_source == "predict"
        or (_completion_source == "auto" and isinstance(_fixed_predict, str) and bool(_fixed_predict))
    )
    if _completion_source == "predict" and not (
        isinstance(_fixed_predict, str) and _fixed_predict
    ):
        raise ValueError(
            "--completion-source=predict requires a non-empty 'predict' field on the test sample"
        )

    live_gen_text = None  # optional diagnostic generate; never used for attribution when fixed
    _attr_cap = None if not attr_max_seq_len or int(attr_max_seq_len) <= 0 else int(attr_max_seq_len)
    _gen_max_new = max(128, int(MAX_OUTPUT_TOKENS))

    if _use_fixed_predict:
        # Teacher-force the eval completion for attribution; optionally also
        # generate once for log comparison (same eval-aligned prompt).
        device = accelerator.device
        prompt_text = _render_qwen_eval_prompt(
            tokenizer,
            _cur_test["input"],
            system=_cur_test.get("system") or "",
            enable_thinking=False,
        )
        prompt_ids_full = tokenizer(
            prompt_text, add_special_tokens=False
        )["input_ids"]
        pred_ids_list = tokenizer(
            _fixed_predict, add_special_tokens=False
        )["input_ids"]
        if not pred_ids_list:
            raise ValueError("Tokenized predict is empty; cannot teacher-force attribution")

        if live_generate_compare:
            # Compare on the *full* eval prompt (not the attr-truncated one).
            print(
                "[DEBUG] Running live generate for log comparison "
                f"(max_new_tokens={_gen_max_new}, AI4Go-style, not used for attribution)...",
                flush=True,
            )
            _full_prompt_t = torch.tensor(prompt_ids_full, device=device, dtype=torch.long)
            _live_ids, live_gen_text = _greedy_generate_on_prompt(
                model, tokenizer, _full_prompt_t, max_new_tokens=_gen_max_new,
            )
            del _full_prompt_t
            print(
                f"[DEBUG] Live generate done: n_tokens={len(_live_ids)} "
                f"(attribution still uses JSONL predict).",
                flush=True,
            )
        else:
            print(
                "[DEBUG] Live generate compare disabled "
                "(--no-live-generate-compare); attribution uses JSONL predict only.",
                flush=True,
            )

        prompt_ids_list = _left_truncate_prompt_for_attr(
            prompt_ids_full, pred_ids_list, _attr_cap,
        )
        gold_text = _cur_test.get("output") or ""
        gold_ids_list = tokenizer(gold_text, add_special_tokens=False)["input_ids"] if gold_text else []
        prompt_len = len(prompt_ids_list)
        prompt_ids = torch.tensor(prompt_ids_list, device=device, dtype=torch.long)
        pred_ids = torch.tensor(pred_ids_list, device=device, dtype=torch.long)
        model_out_text = _fixed_predict
        gen_source = "fixed_predict"
        print(
            f"[DEBUG] Teacher-forcing eval predict "
            f"(completion_source={_completion_source}, enable_thinking=False); "
            f"prompt_len={prompt_len} predict_tokens={len(pred_ids_list)} "
            f"total={prompt_len + len(pred_ids_list)} "
            f"attr_max_seq_len={_attr_cap}",
            flush=True,
        )
        print(
            "[DEBUG] Tip: pass --completion-source generate to attribute the "
            "current adapter's live greedy decode (AI4Go eval prompt) instead.",
            flush=True,
        )
    elif _completion_source == "generate":
        # AI4Go / run_sample44_local: test prompt → greedy generate → attribute.
        device = accelerator.device
        prompt_text = _render_qwen_eval_prompt(
            tokenizer,
            _cur_test["input"],
            system=_cur_test.get("system") or "",
            enable_thinking=False,
        )
        prompt_ids_full = tokenizer(
            prompt_text, add_special_tokens=False
        )["input_ids"]
        print(
            f"[DEBUG] AI4Go-style generate for attribution "
            f"(completion_source=generate, enable_thinking=False, "
            f"max_new_tokens={_gen_max_new}); "
            f"prompt_tokens={len(prompt_ids_full)}",
            flush=True,
        )
        _prompt_t = torch.tensor(prompt_ids_full, device=device, dtype=torch.long)
        pred_ids_list, model_out_text = _greedy_generate_on_prompt(
            model, tokenizer, _prompt_t, max_new_tokens=_gen_max_new,
        )
        del _prompt_t
        if not pred_ids_list:
            raise ValueError("Live generate produced an empty completion; cannot attribute.")
        prompt_ids_list = _left_truncate_prompt_for_attr(
            prompt_ids_full, pred_ids_list, _attr_cap,
        )
        gold_text = _cur_test.get("output") or ""
        gold_ids_list = tokenizer(gold_text, add_special_tokens=False)["input_ids"] if gold_text else []
        prompt_len = len(prompt_ids_list)
        prompt_ids = torch.tensor(prompt_ids_list, device=device, dtype=torch.long)
        pred_ids = torch.tensor(pred_ids_list, device=device, dtype=torch.long)
        gen_source = "live_generate"
        print(
            f"[DEBUG] Live generate used for attribution: "
            f"prompt_len={prompt_len} gen_tokens={len(pred_ids_list)} "
            f"total={prompt_len + len(pred_ids_list)} "
            f"attr_max_seq_len={_attr_cap}",
            flush=True,
        )
    elif (
        isinstance(_compact_ids, list)
        and isinstance(_compact_lbl, list)
        and len(_compact_ids) == len(_compact_lbl)
        and any(int(x) != -100 for x in _compact_lbl)
    ):
        print(
            "[DEBUG] Using compact input_ids/label for free-run (viz-aligned prompt cut).",
            flush=True,
        )
        p0 = next(i for i, lab in enumerate(_compact_lbl) if int(lab) != -100)
        prompt_ids_list = [int(x) for x in _compact_ids[:p0]]
        gold_ans_ids = [int(tid) for tid, lab in zip(_compact_ids, _compact_lbl) if int(lab) != -100]
        device = accelerator.device
        prompt = torch.tensor([prompt_ids_list], device=device)
        print("[DEBUG] Starting viz-style model.generate(prompt)...", flush=True)
        pred_ids_list, model_out_text = _greedy_generate_on_prompt(
            model, tokenizer, prompt[0], max_new_tokens=128,
        )
        print("[DEBUG] Inference done.", flush=True)
        prompt_len = p0
        pred_ids = torch.tensor(pred_ids_list, device=device, dtype=torch.long)
        prompt_ids = torch.tensor(prompt_ids_list, device=device, dtype=torch.long)
        gold_text = tokenizer.decode(gold_ans_ids, skip_special_tokens=False)
        gen_source = "compact_ids"
        gold_ids_list = gold_ans_ids
    else:
        print(
            "[DEBUG] No compact input_ids/label on sample; falling back to ChatML re-encode + NIF.infer.",
            flush=True,
        )
        test_ds       = build_single_sample_dataset(_cur_test, convert_to_chatml)
        raw_test_batch = base_collator([test_ds[0]])
        print(f"[DEBUG] Moving test batch to device {accelerator.device}...", flush=True)
        raw_test_batch = {k: v.to(accelerator.device) for k, v in raw_test_batch.items()}
        print(f"[DEBUG] Test batch on device. input_ids shape={raw_test_batch['input_ids'].shape}", flush=True)
        print("[DEBUG] Starting infer_fw.infer(raw_test_batch)...", flush=True)
        gen_result = infer_fw.infer(raw_test_batch, skip_saliency=True)
        print("[DEBUG] Inference done.", flush=True)
        prompt_len = int(gen_result["target_idx"][0])
        pred_ids   = torch.tensor(
            gen_result["pred_ids"][0],
            device=raw_test_batch["input_ids"].device,
            dtype=raw_test_batch["input_ids"].dtype,
        )
        model_out_text = gen_result.get("pred_text", [None])[0]
        if model_out_text is None:
            model_out_text = tokenizer.decode(pred_ids.tolist(), skip_special_tokens=False)
        gold_text = gen_result.get("answer_text", [None])[0]
        if gold_text is None:
            gold_text = _cur_test.get("output", "")
        prompt_ids = raw_test_batch["input_ids"][0, :prompt_len]
        gen_source = "chatml_reencode"
        gold_ids_list = None

    model_out_clean = str(model_out_text).split("<|im_end|>")[0]
    gold_clean = str(gold_text).split("<|im_end|>")[0]
    _prompt_sha = hashlib.sha1(
        ",".join(str(int(x)) for x in prompt_ids.tolist()).encode()
    ).hexdigest()[:12]
    _gen_sha = hashlib.sha1(
        ",".join(str(int(x)) for x in pred_ids.tolist()).encode()
    ).hexdigest()[:12]
    print("=" * 64, flush=True)
    print(
        f"[DEBUG] attribution completion  task_id={_task_id}  source={gen_source}  "
        f"prompt_len={prompt_len}  prompt_sha1={_prompt_sha}  "
        f"n_attr_tokens={int(pred_ids.numel())}  attr_sha1={_gen_sha}",
        flush=True,
    )
    if gen_source == "fixed_predict":
        print("--- FIXED PREDICT (used for attribution) ---", flush=True)
    else:
        print("--- MODEL OUTPUT (this run, used for attribution) ---", flush=True)
    print(model_out_clean, flush=True)
    if live_gen_text is not None:
        live_clean = str(live_gen_text).split("<|im_end|>")[0]
        print("--- LIVE GENERATE (log only, NOT used for attribution) ---", flush=True)
        print(live_clean, flush=True)
        _live_match = live_clean.strip() == model_out_clean.strip()
        print(
            f"[DEBUG] live_generate == fixed_predict ? {_live_match}",
            flush=True,
        )
    print("--- GOLD (from label / JSONL) ---", flush=True)
    print(gold_clean, flush=True)
    print(
        f"[DEBUG] attr token ids (first 64): {pred_ids.tolist()[:64]}",
        flush=True,
    )
    print("=" * 64, flush=True)

    # Token lists for report UI (always available, both free-run paths).
    _prompt_id_list = [int(x) for x in prompt_ids.tolist()]
    _pred_id_list = [int(x) for x in pred_ids.tolist()]
    pred_full_tokens = [tokenizer.decode([i]) for i in (_prompt_id_list + _pred_id_list)]
    if gen_source in {"compact_ids", "fixed_predict"}:
        if gold_ids_list is None:
            _gold_id_list = []
        else:
            _gold_id_list = [int(x) for x in gold_ids_list]
        correct_full_tokens = [
            tokenizer.decode([i]) for i in (_prompt_id_list + _gold_id_list)
        ]
        correct_full_token_ids = _prompt_id_list + _gold_id_list
    else:
        correct_full_tokens = gen_result.get("full_tokens", [pred_full_tokens])[0]
        # Prefer ids if the generate path carried them; else rebuild from surfaces later.
        _maybe_ids = gen_result.get("full_token_ids")
        if isinstance(_maybe_ids, list) and _maybe_ids and len(_maybe_ids[0]) == len(correct_full_tokens):
            correct_full_token_ids = [int(x) for x in _maybe_ids[0]]
        else:
            # Same prompt + gold completion ids when available on gen_result.
            _g = gen_result.get("gold_token_ids")
            if isinstance(_g, list) and _g:
                correct_full_token_ids = _prompt_id_list + [int(x) for x in _g]
            else:
                correct_full_token_ids = _prompt_id_list + [
                    int(x) for x in tokenizer.encode(
                        "".join(correct_full_tokens[prompt_len:]),
                        add_special_tokens=False,
                    )
                ]
                if len(correct_full_token_ids) != len(correct_full_tokens):
                    # Fall back: keep surfaces; live gold will re-anchor via full_token_ids.
                    correct_full_token_ids = []
    gen_result = {
        "pred_full_tokens": [pred_full_tokens],
        "pred_full_token_ids": [_prompt_id_list + _pred_id_list],
        "full_tokens": [correct_full_tokens],
        "correct_full_token_ids": [correct_full_token_ids] if correct_full_token_ids else [[]],
    }

    new_input_ids      = torch.cat([prompt_ids, pred_ids], dim=0).unsqueeze(0)
    new_attention_mask = torch.ones_like(new_input_ids)
    new_labels         = new_input_ids.clone()
    new_labels[:, :prompt_len] = -100

    test_batch = {
        "input_ids":      new_input_ids,
        "attention_mask": new_attention_mask,
        "labels":         new_labels,
    }
    seq_len = test_batch["input_ids"].size(1)

    def _alti_grad_chunks() -> list[int]:
        chunks = []
        chunk = alti_grad_chunk_size
        while chunk >= 1:
            if chunk not in chunks:
                chunks.append(chunk)
            chunk //= 2
        if 1 not in chunks:
            chunks.append(1)
        return chunks

    def _compute_alti_correlation_gradient_retry(**kwargs):
        target_idx = int(kwargs["target_idx_in_seq"])
        if alti_grad_max_seq_len is not None and target_idx > alti_grad_max_seq_len:
            print(
                f"  Skipping ALTI-gradient target={target_idx}: "
                f"exceeds --alti-grad-max-seq-len={alti_grad_max_seq_len}"
            )
            return None

        chunks = _alti_grad_chunks()
        last_error_message = None
        for chunk in chunks:
            try:
                return _corr_grad_fn(
                    **kwargs,
                    chunk_size=chunk,
                )
            except torch.OutOfMemoryError as exc:
                last_error_message = str(exc)
                print(f"  OOM in ALTI-gradient chunk={chunk}; retrying smaller chunk...")
                torch.cuda.empty_cache()
            except RuntimeError as exc:
                if not _is_cuda_alloc_error(exc):
                    raise
                last_error_message = str(exc)
                print(f"  OOM in ALTI-gradient chunk={chunk}; retrying smaller chunk...")
                torch.cuda.empty_cache()
        raise RuntimeError(
            f"ALTI-gradient OOM after trying chunks {chunks}: {last_error_message}"
        )

    def _compute_alti_match_and_probe_retry(**kwargs):
        target_idx = int(kwargs["target_idx_in_seq"])
        if alti_grad_max_seq_len is not None and target_idx > alti_grad_max_seq_len:
            print(
                f"  Skipping ALTI match/probe target={target_idx}: "
                f"exceeds --alti-grad-max-seq-len={alti_grad_max_seq_len}"
            )
            return None
        chunks = _alti_grad_chunks()
        last_error_message = None
        for chunk in chunks:
            try:
                return _match_probe_fn(
                    **kwargs,
                    chunk_size=chunk,
                )
            except torch.OutOfMemoryError as exc:
                last_error_message = str(exc)
                print(f"  OOM in ALTI match/probe chunk={chunk}; retrying smaller chunk...")
                torch.cuda.empty_cache()
            except RuntimeError as exc:
                if not _is_cuda_alloc_error(exc):
                    raise
                last_error_message = str(exc)
                print(f"  OOM in ALTI match/probe chunk={chunk}; retrying smaller chunk...")
                torch.cuda.empty_cache()
        raise RuntimeError(
            f"ALTI match/probe OOM after trying chunks {chunks}: {last_error_message}"
        )

    # ════════════════════════════════════════════════════════════════════════════
    # Helper: Stage 3 processing for one train sample
    # Returns (train_sample_detail_dict, pair_records_list)
    # detail is None if the sample must be skipped (too long / no marker).
    # If the sample was already computed (cache hit), pass existing detail + candidate_pairs.
    # ════════════════════════════════════════════════════════════════════════════
    def _process_train_sample_stage3(
        train_idx: int,
        coarse_score: float,
        test_corr_features: dict,
        target_tok_text_for_record: str,
        target_tok_idx_for_record: int,
        pair_id_prefix: str,
        pair_id_start: int,
        cached_detail: dict | None = None,
    ) -> tuple[dict | None, list, int]:
        """Returns (detail_dict_or_None, pair_records, next_pair_id)."""
        nonlocal model, tokenizer, base_collator, accelerator, fine_param_filter

        if cached_detail is None:
            tr_ds   = build_single_sample_dataset(train_samples[train_idx])
            tr_batch = base_collator([tr_ds[0]])
            tr_batch = {k: v.to(accelerator.device) for k, v in tr_batch.items()}

            if tr_batch["input_ids"].size(1) > SEQUENCE_LENGTH_LIMIT:
                print(f"  Skipping train {train_idx}: sequence too long")
                return None, [], pair_id_start

            try:
                start_sys = _find_subseq_start(tr_batch["input_ids"][0], marker_ids) + len(marker_ids)
            except ValueError:
                print(f"  Skipping train {train_idx}: assistant marker not found")
                return None, [], pair_id_start

            response_start = find_first_valid_token_index(tokenizer, tr_batch["input_ids"], start_sys)
            tr_seq_len     = tr_batch["input_ids"].size(1)
            full_token_ids = [int(i) for i in tr_batch["input_ids"][0].tolist()]
            full_tokens    = [tokenizer.decode([i]) for i in full_token_ids]
            target_saliencies: dict[int, list[float]] = {}
            candidate_pairs: list[tuple[float, int, int]] = []

            # All non-trivial train answer tokens as targets (optional --top-targets cap);
            # each keeps saliency top-K sources (default 3) → ~n_valid × 3 candidate pairs.
            max_train_targets = TOP_TARGETS  # None => no cap
            with torch.inference_mode(False):
                t_tr = response_start
                kept_targets = 0
                while t_tr < tr_seq_len:
                    if max_train_targets is not None and kept_targets >= max_train_targets:
                        break
                    if is_trivial_token(tokenizer, int(tr_batch["input_ids"][0, t_tr].item())):
                        t_tr += 1
                        continue
                    sal_vec = _saliency_fn(
                        model,
                        tr_batch,
                        t_tr,
                        chunk_size=ALTI_CHUNK_SIZE,
                    )
                    target_saliencies[t_tr] = [round(float(s), 6) for s in sal_vec]
                    for s_idx, s_score in top_nontrivial_saliency_sources(
                        tokenizer,
                        tr_batch["input_ids"][0],
                        sal_vec,
                        TOP_K_SOURCE_PER_TARGET,
                    ):
                        candidate_pairs.append((float(s_score), t_tr, int(s_idx)))
                    kept_targets += 1
                    t_tr += 1

            print(
                f"  Step A: {len(candidate_pairs)} candidate pairs "
                f"({kept_targets} valid pred tokens × {TOP_K_SOURCE_PER_TARGET}s each)",
                flush=True,
            )

            # Collect annotation edges for this train sample, grouped by dst (target).
            # Filter out edges whose source is a ChatML structural token (noise).
            edges = train_samples[train_idx].get("attention_edges") or []
            annotations_by_target: dict[str, list[dict]] = {}
            for e in edges:
                src = int(e["src"])
                dst = int(e["dst"])
                # Skip edges where source is a ChatML template marker
                if src < len(full_tokens):
                    src_tok = full_tokens[src].strip()
                    if (src_tok in {"<|im_start|>", "<|im_end|>", "system", "user", "assistant",
                                    "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>",
                                    "<|repo_name|>", "<|file_sep|>", "<|endoftext|>"}
                            or "<|im_start|>" in src_tok or "<|im_end|>" in src_tok
                            or src_tok == "\n"):
                        continue
                dst_key = str(dst)
                entry = {"src": src, "subtype": str(e.get("subtype", ""))}
                annotations_by_target.setdefault(dst_key, []).append(entry)

            cached_detail = {
                "full_tokens":             full_tokens,
                "full_token_ids":          full_token_ids,
                "answer_start_index":      response_start,
                "coarse_cos_sim":          float(coarse_score),
                "saliencies_by_token":     {str(k): v for k, v in target_saliencies.items()},
                "annotations_by_target":   annotations_by_target,
                "_candidate_pairs":        candidate_pairs,
                "_tr_batch_cpu":           {k: v.cpu() for k, v in tr_batch.items()},
                "_feature_cache":          {},
            }
        else:
            # Update coarse score to the maximum seen across test tokens
            if coarse_score > cached_detail["coarse_cos_sim"]:
                cached_detail["coarse_cos_sim"] = float(coarse_score)
            candidate_pairs = cached_detail["_candidate_pairs"]
            cached_detail.setdefault("_feature_cache", {})

        # Move tr_batch to device for ALTI-gradient correlation matching
        tr_batch_gpu = {k: v.to(accelerator.device) for k, v in cached_detail["_tr_batch_cpu"].items()}
        ids_1d       = tr_batch_gpu["input_ids"][0]
        response_start = cached_detail["answer_start_index"]
        pair_counter   = pair_id_start
        pair_records   = []

        with torch.inference_mode(False):
            for saliency_score, t_tr, s_idx in candidate_pairs:
                train_target_tok   = tokenizer.decode([ids_1d[t_tr].item()])
                train_source_tok   = tokenizer.decode([ids_1d[s_idx].item()])
                response_tok_offset = t_tr - response_start

                print(f"  Step B: '{train_source_tok}' -> '{train_target_tok}' "
                      f"(offset={response_tok_offset}, sal={saliency_score:.4f})")

                feature_key = (int(t_tr), int(s_idx))
                if feature_key in cached_detail["_feature_cache"]:
                    train_feat = cached_detail["_feature_cache"][feature_key]
                    if train_feat is None:
                        continue
                    print("    feature cache hit")
                else:
                    train_feat = _compute_alti_correlation_gradient_retry(
                        model=model,
                        batch=tr_batch_gpu,
                        target_idx_in_seq=t_tr,
                        source_idx_in_seq=s_idx,
                        param_filter_fn=fine_param_filter,
                        device=accelerator.device,
                    )
                    cached_detail["_feature_cache"][feature_key] = train_feat
                    if train_feat is None:
                        continue
                source_ctx = get_context_window(tokenizer, ids_1d, s_idx)
                target_ctx = get_context_window(tokenizer, ids_1d, t_tr)

                for test_p_idx, (test_feat, test_src_text, test_saliency) in test_corr_features.items():
                    cos_sim = F.cosine_similarity(
                        test_feat,
                        train_feat,
                        dim=0,
                    ).item()

                    pair_records.append({
                        "id":            f"{pair_id_prefix}_{pair_counter:04d}",
                        "cos_sim":       float(cos_sim),
                        "coarse_cos_sim": float(coarse_score),
                        "train_sample_id": train_idx,
                        "test_correlation": {
                            "source_token":       test_src_text,
                            "source_token_index": test_p_idx,
                            "target_token":       target_tok_text_for_record,
                            "target_token_index": target_tok_idx_for_record,
                            "saliency_score":     float(test_saliency),
                        },
                        "train_correlation": {
                            "source_token":         train_source_tok,
                            "source_token_index":   s_idx,
                            "target_token":         train_target_tok,
                            "target_token_index":   t_tr,
                            "saliency_score":       saliency_score,
                            "response_token_offset": response_tok_offset,
                        },
                        "train_context": {
                            "source_context": source_ctx,
                            "target_context": target_ctx,
                        },
                        "annotation": None,
                    })
                    pair_counter += 1

        del tr_batch_gpu
        torch.cuda.empty_cache()
        return cached_detail, pair_records, pair_counter

    # ════════════════════════════════════════════════════════════════════════════
    # Helper: compute test-side CE gradient for one token position
    # ════════════════════════════════════════════════════════════════════════════
    def _test_ce_grad_for_token(tok_idx: int) -> torch.Tensor:
        ce_labels = torch.full_like(test_batch["input_ids"], -100)
        ce_labels[0, tok_idx] = test_batch["input_ids"][0, tok_idx]
        single_tok_batch = {
            "input_ids":      test_batch["input_ids"],
            "attention_mask": test_batch["attention_mask"],
            "labels":         ce_labels,
        }
        grad = compute_lm_head_ce_gradient_no_backward(
            model=model,
            batch=single_tok_batch,
            device=accelerator.device,
            ignored_token_ids=torch.tensor([], device=accelerator.device),
        )
        return grad.reshape(-1).to(lm_head_device).detach()

    def _test_ce_sketch_for_token(tok_idx: int) -> torch.Tensor:
        ce_labels = torch.full_like(test_batch["input_ids"], -100)
        ce_labels[0, tok_idx] = test_batch["input_ids"][0, tok_idx]
        single_tok_batch = {
            "input_ids":      test_batch["input_ids"],
            "attention_mask": test_batch["attention_mask"],
            "labels":         ce_labels,
        }
        return compute_lm_head_ce_gradient_sketches_no_backward(
            model=model,
            batch=single_tok_batch,
            device=accelerator.device,
            ignored_token_ids=torch.tensor([], device=accelerator.device),
            sketch_dim=prescreen_sketch_dim,
            sketch_seed=prescreen_sketch_seed,
        )[0].detach()

    # ════════════════════════════════════════════════════════════════════════════
    # Helper: compute test-side CE gradient aggregated over ALL response tokens
    # Used for the global one-shot coarse pre-screen in all_tokens mode.
    # test_batch["labels"] already masks prompt positions with -100, so
    # The analytic lm_head gradient naturally aggregates CE loss across all
    # response tokens.
    # ════════════════════════════════════════════════════════════════════════════
    def _test_ce_grad_full_response() -> torch.Tensor:
        grad = compute_lm_head_ce_gradient_no_backward(
            model=model,
            batch=test_batch,
            device=accelerator.device,
            ignored_token_ids=torch.tensor([], device=accelerator.device),
        )
        return grad.reshape(-1).to(lm_head_device).detach()

    def _test_ce_sketch_full_response() -> torch.Tensor:
        return compute_lm_head_ce_gradient_sketches_no_backward(
            model=model,
            batch=test_batch,
            device=accelerator.device,
            ignored_token_ids=torch.tensor([], device=accelerator.device),
            sketch_dim=prescreen_sketch_dim,
            sketch_seed=prescreen_sketch_seed,
        )[0].detach()

    # ════════════════════════════════════════════════════════════════════════════
    # ── ALL-TOKENS MODE ──────────────────────────────────────────────────────
    # ════════════════════════════════════════════════════════════════════════════
    max_end = min(prompt_len + MAX_OUTPUT_TOKENS, seq_len)
    # Every completion token in the window is a target (including `}`, `)`,
    # whitespace, etc.). Trivial filtering still applies to *source* ranking
    # via top_nontrivial_saliency_sources, and to Stage3 train-answer scan.
    valid_test_tokens = list(range(prompt_len, max_end))
    print(
        f"\nAll-tokens mode: {len(valid_test_tokens)} target tokens "
        f"(all response tokens in window; trivial NOT skipped as targets)",
        flush=True,
    )

    # ── Bank checkpoint (replaces CE global pre-screen) ──────────────────────
    # Top-K trains are retrieved per test edge (s→t) via cos(sketch(f_test), bank).
    print(
        f"\n=== Saliency train bank ready: {int(saliency_train_bank['sample_ids'].numel())} sketches "
        f"(method={TRAIN_RETRIEVAL_METHOD}) ===",
        flush=True,
    )
    bank_n = int(saliency_train_bank["sample_ids"].numel())
    prescreen_report_json = {
        "experiment_meta": {
            "test_sample_index": SELECTED_TEST_SAMPLE_INDEX,
            "task_id": _task_id,
            "model_name": model_tag,
            "model_path": model_path,
            "mode": "all_tokens",
            "stage": "saliency_train_bank",
            "is_checkpoint": True,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "tokens_analyzed": len(valid_test_tokens),
            "screening": TRAIN_RETRIEVAL_METHOD,
            "config": {
                "TOP_K_TRAIN_SAMPLES": TOP_K_TRAIN_SAMPLES,
                "TOP_K_PROMPT_TOKENS": TOP_K_PROMPT_TOKENS,
                "TOP_K_PROMPT_OFFSET": TOP_K_PROMPT_OFFSET,
                "SALIENCY_RANKS": (
                    f"{TOP_K_PROMPT_OFFSET + 1}-{TOP_K_PROMPT_OFFSET + TOP_K_PROMPT_TOKENS}"
                ),
                "prescreen_max_seq_len": prescreen_max_seq_len,
                "prescreen_batch_size": prescreen_batch_size,
                "PRESCREEN_SKETCH_DIM": prescreen_sketch_dim,
                "PRESCREEN_SKETCH_SEED": prescreen_sketch_seed,
                "SALIENCY_TRAIN_BANK_SIZE": bank_n,
                "SALIENCY_TRAIN_BANK_CACHE": True,
                "SALIENCY_METHOD": (
                    "alti_last_layer" if _saliency_mode == "last_layer" else "alti_full"
                ),
                "SALIENCY_MODE": _saliency_mode,
                "MATCHING_METHOD": match_desc,
                "GRAD_SPACE": grad_space,
                "TRAIN_RETRIEVAL_METHOD": TRAIN_RETRIEVAL_METHOD,
                "BANK_LOSS_MODE": bank_cfg.loss_mode,
                "BANK_LOSS_TAG": bank_cfg.cache_tag,
                "PROBE": "L_probe=-log(C+eps)",
                "FINE_MATCH_LAST_N_LAYERS": FINE_MATCH_LAST_N_LAYERS,
                "FINE_MATCH_PROJ": fine_match_proj,
                "ALTI_CHUNK_SIZE": ALTI_CHUNK_SIZE,
                "ALTI_GRAD_CHUNK_SIZE": alti_grad_chunk_size,
                "ALTI_GRAD_MAX_SEQ_LEN": alti_grad_max_seq_len,
                "PRESCREEN_LENGTH_SWEEP": prescreen_length_sweep_stats,
            },
        },
        "test_sample_baseline": {
            "full_tokens": gen_result["pred_full_tokens"][0],
            "full_token_ids": gen_result["pred_full_token_ids"][0],
            "correct_full_tokens": gen_result["full_tokens"][0],
            "correct_full_token_ids": (gen_result.get("correct_full_token_ids") or [[]])[0],
            "full_tokens_display": token_surfaces_for_display(
                tokenizer, gen_result["pred_full_token_ids"][0]
            ),
            "correct_full_tokens_display": (
                token_surfaces_for_display(
                    tokenizer, (gen_result.get("correct_full_token_ids") or [[]])[0]
                )
                if (gen_result.get("correct_full_token_ids") or [[]])[0]
                else []
            ),
            "prompt_len": prompt_len,
        },
        "per_token_results": [],
        "train_sample_details": {},
    }
    _sal_tag = saliency_rank_filename_tag()
    prescreen_filename = (
        f"correlation_matching_results_{model_tag}_{_task_id}_all_tokens{_sal_tag}_prescreen.json"
    )
    report_filename = (
        f"correlation_matching_results_{model_tag}_{_task_id}_all_tokens"
        f"{saliency_rank_filename_tag()}.json"
    )
    prescreen_path = _write_json_report(prescreen_report_json, prescreen_filename, accelerator)
    if prescreen_path is not None:
        print(f"  Bank checkpoint saved → {prescreen_path}", flush=True)

    # ── Per-token loop: ALTI → per-edge bank Top-K → Stage 3 pair matching ──
    per_token_results: list = []
    train_sample_cache: dict = {}   # str(train_idx) → detail dict (with _candidate_pairs, _tr_batch_cpu)
    prior_train_details: dict = {}  # public details restored from a partial report
    pair_id_counter = 0
    done_target_indices: set[int] = set()

    resume_fp = _resume_fingerprint(
        task_id=_task_id,
        model_path=model_path,
        model_tag=model_tag,
        test_index=SELECTED_TEST_SAMPLE_INDEX,
        prompt_len=prompt_len,
        full_token_ids=gen_result["pred_full_token_ids"][0],
        max_output_tokens=MAX_OUTPUT_TOKENS,
        saliency_mode=_saliency_mode,
        bank_loss_mode=bank_cfg.loss_mode,
        top_k_prompt=TOP_K_PROMPT_TOKENS,
        top_k_prompt_offset=TOP_K_PROMPT_OFFSET,
        top_k_train=TOP_K_TRAIN_SAMPLES,
    )
    if resume:
        loaded = _try_load_all_tokens_progress(report_filename, resume_fp)
        if loaded is not None:
            per_token_results, prior_train_details, done_target_indices = loaded
            pending = [t for t in valid_test_tokens if int(t) not in done_target_indices]
            print(
                f"[resume] skip {len(done_target_indices)} done; "
                f"{len(pending)}/{len(valid_test_tokens)} target token(s) remaining "
                f"→ {report_filename}",
                flush=True,
            )
            if not pending:
                print("[resume] all target tokens already present; rewriting final report.", flush=True)

    def _flush_all_tokens_report(*, partial: bool) -> str | None:
        """Persist current per-token progress into the main report JSON."""
        per_token_results.sort(
            key=lambda r: int(r.get("target_token_index", -1))
            if isinstance(r, dict) else -1
        )
        merged_details = {
            **prior_train_details,
            **_public_train_sample_details(train_sample_cache),
        }
        payload = {
            "experiment_meta": {
                "test_sample_index": SELECTED_TEST_SAMPLE_INDEX,
                "task_id": _task_id,
                "model_name": model_tag,
                "model_path": model_path,
                "mode": "all_tokens",
                "stage": "all_tokens_partial" if partial else "all_tokens",
                "is_partial": bool(partial),
                "is_checkpoint": bool(partial),
                "max_output_tokens": MAX_OUTPUT_TOKENS,
                "tokens_analyzed": len(per_token_results),
                "tokens_expected": len(valid_test_tokens),
                "screening": TRAIN_RETRIEVAL_METHOD,
                "config": {
                    "TOP_K_PROMPT_TOKENS": TOP_K_PROMPT_TOKENS,
                    "TOP_K_PROMPT_OFFSET": TOP_K_PROMPT_OFFSET,
                    "SALIENCY_RANKS": (
                        f"{TOP_K_PROMPT_OFFSET + 1}-"
                        f"{TOP_K_PROMPT_OFFSET + TOP_K_PROMPT_TOKENS}"
                    ),
                    "TOP_K_TRAIN_SAMPLES": TOP_K_TRAIN_SAMPLES,
                    "TOP_TARGETS": TOP_TARGETS,
                    "TOP_K_SOURCE_PER_TARGET": TOP_K_SOURCE_PER_TARGET,
                    "SALIENCY_MODE": _saliency_mode,
                    "CONTEXT_WINDOW_SIZE": CONTEXT_WINDOW_SIZE,
                    "SALIENCY_METHOD": (
                        "alti_last_layer" if _saliency_mode == "last_layer" else "alti_full"
                    ),
                    "MATCHING_METHOD": match_desc,
                    "GRAD_SPACE": grad_space,
                    "TRAIN_RETRIEVAL_METHOD": TRAIN_RETRIEVAL_METHOD,
                    "BANK_LOSS_MODE": bank_cfg.loss_mode,
                    "BANK_LOSS_TAG": bank_cfg.cache_tag,
                    "PROBE": "L_probe=-log(C+eps)",
                    "FINE_MATCH_LAST_N_LAYERS": FINE_MATCH_LAST_N_LAYERS,
                    "FINE_MATCH_PROJ": fine_match_proj,
                    "ALTI_CHUNK_SIZE": ALTI_CHUNK_SIZE,
                    "ALTI_GRAD_CHUNK_SIZE": alti_grad_chunk_size,
                    "ALTI_GRAD_MAX_SEQ_LEN": alti_grad_max_seq_len,
                    "prescreen_max_seq_len": prescreen_max_seq_len,
                    "prescreen_batch_size": prescreen_batch_size,
                    "PRESCREEN_BATCH_SIZE": prescreen_batch_size,
                    "PRESCREEN_SAMPLE_LIMIT": prescreen_limit,
                    "PRESCREEN_MAX_SEQ_LEN": prescreen_max_seq_len,
                    "PRESCREEN_LENGTH_SWEEP": prescreen_length_sweep_stats,
                    "PRESCREEN_SKETCH_DIM": prescreen_sketch_dim,
                    "PRESCREEN_SKETCH_SEED": prescreen_sketch_seed,
                    "SALIENCY_TRAIN_BANK_SIZE": bank_n,
                    "SALIENCY_TRAIN_BANK_CACHE": True,
                },
            },
            "test_sample_baseline": {
                "full_tokens": gen_result["pred_full_tokens"][0],
                "full_token_ids": gen_result["pred_full_token_ids"][0],
                "correct_full_tokens": gen_result["full_tokens"][0],
                "correct_full_token_ids": (gen_result.get("correct_full_token_ids") or [[]])[0],
                "full_tokens_display": token_surfaces_for_display(
                    tokenizer, gen_result["pred_full_token_ids"][0]
                ),
                "correct_full_tokens_display": (
                    token_surfaces_for_display(
                        tokenizer, (gen_result.get("correct_full_token_ids") or [[]])[0]
                    )
                    if (gen_result.get("correct_full_token_ids") or [[]])[0]
                    else []
                ),
                "prompt_len": prompt_len,
            },
            "per_token_results": per_token_results,
            "train_sample_details": merged_details,
        }
        return _write_json_report(payload, report_filename, accelerator)

    tokens_to_run = [
        t for t in valid_test_tokens if int(t) not in done_target_indices
    ]

    try:
        for t in tqdm(tokens_to_run, desc="Test tokens",
                      disable=not accelerator.is_local_main_process):
            target_tok_id   = int(test_batch["input_ids"][0, t].item())
            target_tok_text = tokenizer.decode([target_tok_id])
            print(f"\n=== Token {t}: '{target_tok_text}' ===")

            # Stage 1: saliency sources + probe features
            test_corr_features: dict = {}
            with torch.inference_mode(False):
                if _saliency_mode == "last_layer":
                    # Ranking uses a cheap no-grad forward (hidden_states only, no all-layer attn).
                    sal_vec = _saliency_fn(
                        model,
                        test_batch,
                        t,
                        chunk_size=ALTI_CHUNK_SIZE,
                    )
                    top_test_corr = top_nontrivial_saliency_sources(
                        tokenizer,
                        test_batch["input_ids"][0],
                        sal_vec,
                        TOP_K_PROMPT_TOKENS,
                        offset=TOP_K_PROMPT_OFFSET,
                    )
                    top_test_correlations = [
                        {
                            "source_token_index": idx,
                            "source_token": tokenizer.decode(
                                [int(test_batch["input_ids"][0, idx].item())]
                            ),
                            "target_token_index": t,
                            "target_token": target_tok_text,
                            "saliency_score": float(score),
                            "saliency_rank": TOP_K_PROMPT_OFFSET + rank_i,
                        }
                        for rank_i, (idx, score) in enumerate(top_test_corr, start=1)
                    ]
                    if not top_test_correlations:
                        per_token_results.append({
                            "target_token_index": t,
                            "target_token": target_tok_text,
                            "top_correlations": [],
                            "correlation_pairs": [],
                        })
                        path = _flush_all_tokens_report(partial=True)
                        if path is not None:
                            print(
                                f"  [checkpoint] {len(per_token_results)}/"
                                f"{len(valid_test_tokens)} tokens → {path}",
                                flush=True,
                            )
                        torch.cuda.empty_cache()
                        continue
                    # Per-edge ∇C / L_probe (same as full_alti). A previous
                    # optimization reused one top-k-mean probe for every source,
                    # which made UI edge switches show identical trains + cos_sim.
                    print(
                        f"  Computing {len(top_test_correlations)} test "
                        f"last_layer match/probe features (per source→target edge)...",
                        flush=True,
                    )
                    for item in top_test_correlations:
                        p_idx = item["source_token_index"]
                        pair = _compute_alti_match_and_probe_retry(
                            model=model,
                            batch=test_batch,
                            target_idx_in_seq=t,
                            source_idx_in_seq=p_idx,
                            param_filter_fn=fine_param_filter,
                            device=accelerator.device,
                        )
                        if pair is None:
                            continue
                        feat_match, feat_probe = pair
                        test_corr_features[p_idx] = (
                            feat_match, feat_probe, item["source_token"], item["saliency_score"],
                        )
                else:
                    sal_vec = _saliency_fn(
                        model,
                        test_batch,
                        t,
                        chunk_size=ALTI_CHUNK_SIZE,
                    )
                    top_test_corr = top_nontrivial_saliency_sources(
                        tokenizer,
                        test_batch["input_ids"][0],
                        sal_vec,
                        TOP_K_PROMPT_TOKENS,
                        offset=TOP_K_PROMPT_OFFSET,
                    )
                    top_test_correlations = [
                        {
                            "source_token_index": idx,
                            "source_token": tokenizer.decode(
                                [int(test_batch["input_ids"][0, idx].item())]
                            ),
                            "target_token_index": t,
                            "target_token": target_tok_text,
                            "saliency_score": float(score),
                            "saliency_rank": TOP_K_PROMPT_OFFSET + rank_i,
                        }
                        for rank_i, (idx, score) in enumerate(top_test_corr, start=1)
                    ]
                    print(
                        f"  Computing {len(top_test_correlations)} test "
                        f"{_saliency_mode} match/probe features..."
                    )
                    for item in top_test_correlations:
                        p_idx = item["source_token_index"]
                        pair = _compute_alti_match_and_probe_retry(
                            model=model,
                            batch=test_batch,
                            target_idx_in_seq=t,
                            source_idx_in_seq=p_idx,
                            param_filter_fn=fine_param_filter,
                            device=accelerator.device,
                        )
                        if pair is None:
                            continue
                        feat_match, feat_probe = pair
                        test_corr_features[p_idx] = (
                            feat_match, feat_probe, item["source_token"], item["saliency_score"],
                        )

            if not test_corr_features:
                print(f"  No ALTI-gradient test features survived for token {t}; skipping retrieval/Stage 3.")
                per_token_results.append({
                    "target_token_index": t,
                    "target_token":       target_tok_text,
                    "top_correlations":   top_test_correlations,
                    "correlation_pairs":  [],
                })
                path = _flush_all_tokens_report(partial=True)
                if path is not None:
                    print(
                        f"  [checkpoint] {len(per_token_results)}/"
                        f"{len(valid_test_tokens)} tokens → {path}",
                        flush=True,
                    )
                torch.cuda.empty_cache()
                continue

            # Stage 2: per-edge bank Top-K (L_probe) → Stage 3 pair matching (∇C)
            token_pair_records: list = []
            for p_idx, (feat_match, feat_probe, src_text, sal_score) in test_corr_features.items():
                probe_sketch = _project_flat_grad(feat_probe, prescreen_sketch_dim, prescreen_sketch_seed)
                edge_scores = _score_prescreen_sketch_cache(
                    probe_sketch,
                    saliency_train_bank,
                    accelerator.device,
                )
                related_samples = nlargest(TOP_K_TRAIN_SAMPLES, edge_scores, key=lambda x: x[1])
                print(
                    f"  Edge '{src_text}'→'{target_tok_text}' Top-{TOP_K_TRAIN_SAMPLES} (L_probe): "
                    f"{[(i, round(s, 4)) for i, s in related_samples]}",
                    flush=True,
                )
                single_edge_features = {p_idx: (feat_match, src_text, sal_score)}
                for rank, (train_idx, probe_score) in enumerate(related_samples):
                    print(
                        f"\n  --- Train {train_idx} (edge_rank={rank + 1}, "
                        f"probe_cos={probe_score:.4f}, src={src_text!r}) ---"
                    )
                    cached = train_sample_cache.get(str(train_idx))
                    detail, pairs, pair_id_counter = _process_train_sample_stage3(
                        train_idx, probe_score, single_edge_features,
                        target_tok_text, t,
                        f"t{t}_s{p_idx}", pair_id_counter,
                        cached_detail=cached,
                    )
                    if detail is not None:
                        train_sample_cache[str(train_idx)] = detail
                        token_pair_records.extend(pairs)
                del probe_sketch, edge_scores

            token_pair_records.sort(key=lambda x: x["cos_sim"], reverse=True)
            per_token_results.append({
                "target_token_index": t,
                "target_token":       target_tok_text,
                "top_correlations":   top_test_correlations,
                "correlation_pairs":  token_pair_records,
            })
            path = _flush_all_tokens_report(partial=True)
            if path is not None:
                print(
                    f"  [checkpoint] {len(per_token_results)}/"
                    f"{len(valid_test_tokens)} tokens → {path}",
                    flush=True,
                )

            # Free test-side features before next token
            del test_corr_features
            torch.cuda.empty_cache()
    except BaseException as exc:
        if accelerator.is_main_process and per_token_results:
            print(
                f"\n[checkpoint] interrupted ({type(exc).__name__}) after "
                f"{len(per_token_results)}/{len(valid_test_tokens)} tokens; "
                f"partial report kept at {report_filename}. "
                f"Re-run the same command to resume (default --resume).",
                flush=True,
            )
        raise

    # ── Final save (clears is_partial) ───────────────────────────────────────
    report_path = _flush_all_tokens_report(partial=False)
    if report_path is not None:
        print(f"\nExperiment completed. Results → {report_path}")
        print(
            f"[intervention] model_name={model_tag} task_id={_task_id} "
            f"screening={TRAIN_RETRIEVAL_METHOD}",
            flush=True,
        )



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run causal intervention experiment for all semantic output tokens."
    )
    parser.add_argument(
        "--test-index", type=int, default=None,
        help="Index of the test sample to analyse (overrides SELECTED_TEST_SAMPLE_INDEX).",
    )
    parser.add_argument(
        "--model-path", type=str, default=None,
        help=(
            "Path to a full CausalLM checkpoint OR a PEFT LoRA adapter directory "
            "(containing adapter_config.json). For adapters, also pass --base-model-path "
            "if auto-resolve fails."
        ),
    )
    parser.add_argument(
        "--base-model-path", type=str, default=None,
        help=(
            "Base CausalLM path when --model-path is a LoRA adapter. "
            "Example: .../code-corr-annotation/models/Qwen2.5-Coder-7B-Instruct"
        ),
    )
    parser.add_argument(
        "--train-data", type=str, default="sft_train.jsonl",
        help="Path to the training JSONL file (default: sft_train.jsonl).",
    )
    parser.add_argument(
        "--test-data", type=str, default="sft_test.jsonl",
        help="Path to the test JSONL file containing error samples (default: sft_test.jsonl).",
    )
    parser.add_argument(
        "--train-limit", type=int, default=None,
        help="Limit the number of training samples to process (default: all). Useful for debugging.",
    )
    parser.add_argument(
        "--completion-source",
        type=str,
        default="auto",
        choices=["auto", "predict", "generate"],
        help=(
            "Where the attributed completion comes from. "
            "auto: use JSONL 'predict' when present, else generate; "
            "predict: teacher-force JSONL predict (AI4Go eval-aligned prompt, thinking off); "
            "generate: AI4Go-style greedy decode on the test prompt "
            "(same as hw_test_data/run_sample44_local.py), then attribute that completion."
        ),
    )
    parser.add_argument(
        "--live-generate-compare",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When teacher-forcing JSONL predict, also run one greedy generate for log "
            "comparison only. Use --no-live-generate-compare to skip (saves time/VRAM)."
        ),
    )
    parser.add_argument(
        "--bank-loss-mode",
        type=str,
        default="auto",
        choices=["auto", "ce_only", "ce_saliency"],
        help=(
            "Train-bank gradient objective when retrieving training samples. "
            "auto: infer from saliency_training_config.json / model_tag path "
            "(checkpoint-N often falls back to ce_only); "
            "ce_only: CE only; ce_saliency: CE + λ·saliency (needs attention_edges)."
        ),
    )
    parser.add_argument(
        "--bank-left-truncate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When building the saliency train bank, left-truncate long sequences "
            "(and remap attention_edges) instead of skipping. On OOM, retry with "
            "shorter caps (2048→768). Use --no-bank-left-truncate to skip long rows."
        ),
    )
    parser.add_argument(
        "--bank-max-seq-len",
        type=int,
        default=None,
        help=(
            "Max full-sequence length (prompt+response tokens) when building the "
            "train bank. Default: same as --prescreen-max-seq-len. Bank grads are "
            "heavier than train forward; try 1500–2000 if bank still OOMs under 3000."
        ),
    )
    parser.add_argument(
        "--attn-implementation", type=str, default=None,
        choices=["eager", "sdpa", "flash_attention_2"],
        help=(
            "Attention backend for model loading. Defaults to eager because ALTI "
            "requires materialized attention probabilities."
        ),
    )
    parser.add_argument(
        "--prescreen-max-seq-len", type=int, default=SEQUENCE_LENGTH_LIMIT,
        help=(
            "Skip training samples longer than this during prescreen/rerank. "
            "Use 0 to disable this guard."
        ),
    )
    parser.add_argument(
        "--prescreen-length-sweep", type=str, default=",".join(str(x) for x in PRESCREEN_LENGTH_SWEEP),
        help=(
            "Comma-separated length limits to report before prescreen, e.g. 3000,2500,2000. "
            "Use an empty string to disable the report."
        ),
    )
    parser.add_argument(
        "--max-gpu-memory", type=str, default=None,
        help=(
            "Per-GPU max_memory passed to from_pretrained, e.g. 26GiB. "
            "Useful for leaving activation headroom with device_map=auto."
        ),
    )
    parser.add_argument(
        "--corr-feature-mode", type=str, default="auto",
        choices=["auto", "second_order", "saliency_proxy"],
        help=(
            "Deprecated compatibility flag. Fine matching now always uses ALTI-gradient features."
        ),
    )
    parser.add_argument(
        "--prescreen-batch-size", type=int, default=1,
        help="Batch size for global prescreen and pool rerank forwards.",
    )
    parser.add_argument(
        "--prescreen-sketch-dim", type=int, default=PRESCREEN_SKETCH_DIM,
        help=(
            "Sketch dimension for the saliency train-bank (and legacy CE cache). "
            "Use <=0 to disable (not supported for saliency retrieval)."
        ),
    )
    parser.add_argument(
        "--prescreen-sketch-seed", type=int, default=PRESCREEN_SKETCH_SEED,
        help="Random seed for deterministic count-sketch hashes.",
    )
    parser.add_argument(
        "--prescreen-sketch-cache-dir", type=str, default=PRESCREEN_SKETCH_CACHE_DIR,
        help="Legacy CE lm_head sketch dir (unused by saliency_probe_bank retrieval).",
    )
    parser.add_argument(
        "--saliency-train-bank-cache-dir", type=str, default=SALIENCY_TRAIN_BANK_CACHE_DIR,
        help="Directory used to store/reuse saliency train-bank sketches.",
    )
    parser.add_argument(
        "--alti-grad-chunk-size", type=int, default=ALTI_GRAD_CHUNK_SIZE,
        help="Initial query chunk size for ALTI-gradient matching; OOM retries use smaller chunks.",
    )
    parser.add_argument(
        "--alti-grad-max-seq-len", type=int, default=ALTI_GRAD_MAX_SEQ_LEN,
        help="Skip ALTI-gradient pairs whose target prefix is longer than this. Use <=0 to disable.",
    )
    parser.add_argument(
        "--fine-match-proj", type=str, default=FINE_MATCH_PROJ,
        help=(
            "Attention projections used for ALTI-gradient fine matching. "
            "Default: qk. Use qkvo/all for the previous behavior, or vo/q/k/v/o for ablations."
        ),
    )
    parser.add_argument(
        "--top-k-prompt-tokens", type=int, default=None,
        help=(
            "How many test saliency sources to keep per target token (default 4). "
            "Together with --top-k-prompt-offset: offset=0,k=4 → ranks 1–4; "
            "offset=4,k=4 → ranks 5–8."
        ),
    )
    parser.add_argument(
        "--top-k-prompt-offset", type=int, default=None,
        help=(
            "Skip this many higher-ranked nontrivial saliency sources before taking "
            "--top-k-prompt-tokens (default 0). Example for ranks 5–8: "
            "--top-k-prompt-offset 4 --top-k-prompt-tokens 4."
        ),
    )
    parser.add_argument(
        "--top-targets", type=int, default=None,
        help=(
            "Cap Stage3 to the first K non-trivial answer tokens per train sample "
            "(each still keeps saliency top sources). Default: no cap (all valid "
            "tokens × --top-k-source-per-target). Pass K>0 to cap; <=0 also means all."
        ),
    )
    parser.add_argument(
        "--top-k-source-per-target", type=int, default=None,
        help=(
            "Saliency top-K sources kept per train answer token in Stage3 "
            "(default 3). Default scan is all valid answer tokens × this K; "
            "use --train-scan-3x3 for a quick 3×3 smoke."
        ),
    )
    parser.add_argument(
        "--train-scan-3x3",
        action="store_true",
        help=(
            "Quick smoke: --top-targets 3 and --top-k-source-per-target 3. "
            "Without this flag, Stage3 defaults to all valid answer tokens × top-3."
        ),
    )
    parser.add_argument(
        "--saliency-mode",
        type=str,
        default="last_layer",
        choices=["last_layer", "full_alti"],
        help=(
            "Saliency / pair-gradient backend. "
            "last_layer: viz/precompute single-layer ALTI C[q,s] (default, cheaper); "
            "full_alti: multi-layer ALTI rollout + match/probe (legacy, VRAM-heavy)."
        ),
    )
    parser.add_argument(
        "--attr-max-seq-len",
        type=int,
        default=ATTR_MAX_SEQ_LEN,
        help=(
            "Max prompt+predict tokens for attribution (default 3000, align with "
            "training max_len). Left-truncates the prompt if longer. Use 0 to disable. "
            "Eval chat prompts are often > train compact length and OOM for that reason."
        ),
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help=(
            "How many response tokens (from prompt_len) to analyze in all-tokens mode "
            f"(default {MAX_OUTPUT_TOKENS}). Every token in the window is a target "
            "(punctuation like }} / ) included). "
            "Use e.g. 250 to cover a longer predict; runtime scales roughly with this."
        ),
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After each finished target token, atomically update the main report JSON "
            "(is_partial=true). On restart with the same fingerprint, skip done tokens. "
            "Use --no-resume to ignore an existing report and overwrite from scratch."
        ),
    )
    args = parser.parse_args()

    if args.test_index is not None:
        SELECTED_TEST_SAMPLE_INDEX = args.test_index
    if args.top_k_prompt_tokens is not None:
        TOP_K_PROMPT_TOKENS = max(1, int(args.top_k_prompt_tokens))
    if args.top_k_prompt_offset is not None:
        TOP_K_PROMPT_OFFSET = max(0, int(args.top_k_prompt_offset))
    if args.max_output_tokens is not None:
        MAX_OUTPUT_TOKENS = max(1, int(args.max_output_tokens))
    if args.train_scan_3x3:
        TOP_TARGETS = 3
        TOP_K_SOURCE_PER_TARGET = 3
    if args.top_targets is not None:
        TOP_TARGETS = None if int(args.top_targets) <= 0 else max(1, int(args.top_targets))
    if args.top_k_source_per_target is not None:
        TOP_K_SOURCE_PER_TARGET = max(1, int(args.top_k_source_per_target))

    prescreen_length_sweep = []
    if args.prescreen_length_sweep.strip():
        prescreen_length_sweep = [
            int(x.strip())
            for x in args.prescreen_length_sweep.split(",")
            if x.strip()
        ]

    print(
        f"[intervention] test_index={SELECTED_TEST_SAMPLE_INDEX}  mode=all_tokens  "
        f"saliency_mode={args.saliency_mode}  "
        f"train_scan≈{TOP_TARGETS or 'all'}×{TOP_K_SOURCE_PER_TARGET}  "
        f"max_output_tokens={MAX_OUTPUT_TOKENS}  "
        f"saliency_ranks={TOP_K_PROMPT_OFFSET + 1}-"
        f"{TOP_K_PROMPT_OFFSET + TOP_K_PROMPT_TOKENS}"
    )
    run_causal_intervention_experiment(
        model_path=args.model_path,
        train_data=args.train_data,
        test_data=args.test_data,
        train_limit=args.train_limit,
        attn_implementation=args.attn_implementation,
        prescreen_max_seq_len=args.prescreen_max_seq_len,
        max_gpu_memory=args.max_gpu_memory,
        corr_feature_mode=args.corr_feature_mode,
        prescreen_batch_size=args.prescreen_batch_size,
        alti_grad_chunk_size=args.alti_grad_chunk_size,
        alti_grad_max_seq_len=args.alti_grad_max_seq_len,
        fine_match_proj=args.fine_match_proj,
        prescreen_sketch_dim=args.prescreen_sketch_dim,
        prescreen_sketch_seed=args.prescreen_sketch_seed,
        prescreen_sketch_cache_dir=args.prescreen_sketch_cache_dir,
        saliency_train_bank_cache_dir=args.saliency_train_bank_cache_dir,
        prescreen_length_sweep=prescreen_length_sweep,
        base_model_path=args.base_model_path,
        completion_source=args.completion_source,
        bank_loss_mode=args.bank_loss_mode,
        live_generate_compare=args.live_generate_compare,
        saliency_mode=args.saliency_mode,
        attr_max_seq_len=args.attr_max_seq_len,
        bank_left_truncate=args.bank_left_truncate,
        bank_max_seq_len=args.bank_max_seq_len,
        resume=args.resume,
    )
