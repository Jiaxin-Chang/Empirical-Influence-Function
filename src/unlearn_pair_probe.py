"""Learn/unlearn probe for a single train↔test saliency correlation pair.

Uses the same model loading path as ``intervention_experiment`` (full checkpoint
OR base + LoRA adapter). The update uses the **same parameter subspace as pair
matching** (default: last-1-layer LoRA).

Train objective (aligned with ce_saliency training, but **one selected edge**)::

    L = CE(train_target) + λ · L_contrastive(pos = selected src→tgt)

Same scalar L for both directions ("整段 L 同号"):

    unlearn: θ ← θ + η · normalize(∇_θ L)   # ascent on L
    learn:   θ ← θ − η · normalize(∇_θ L)   # descent on L

i.e. we do **not** flip only the saliency term or only CE — Learn and Unlearn
are opposite steps on the identical combined objective.

Default η≈0.05 (normalized step L2). With ``persist=True``, clicks accumulate;
``recover_pair_intervention`` undoes the stack.

CE labels: only the pair's train *target* token (others -100).
Saliency: that edge is the sole positive; other causal sources at the target
are negatives (contrastive + neg_sample_k, same hypers as bank/train).
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

from src.attribution_evaluation import _target_token_losses
from src.bank_loss import (
    BankLossConfig,
    _annot_pairs_from_edges,
    _import_saliency_loss_from_outputs,
    load_bank_loss_config,
)
from src.export_real_ttav_bundle import (
    convert_report_tokens_to_ids,
    validate_roundtrip_tokens,
)
from src.export_ttav_bundle import infer_sample_id
from src.intervention_experiment import (
    load_model_and_tokenizer,
    make_attention_projection_filter,
    make_lora_param_filter,
)
from src.loss import (
    compute_alti_saliency_vector,
    compute_last_layer_saliency_vector,
)


REPO_ROOT = Path(__file__).resolve().parent.parent

# Reuse one loaded model across Unlearn/Learn clicks in the same API process.
_MODEL_CACHE: dict[tuple[str, str], tuple[object, object]] = {}

# Persistent multi-step intervention stack (until recover).
_INTERVENTION: dict[str, Any] | None = None

# Normalized LoRA step size: ||Δθ||₂ = |η|. ~one mild CE step on the train target.
DEFAULT_PAIR_INTERVENE_LR = 0.05


def _default_pair_intervene_lr() -> float:
    raw = (os.environ.get("EIF_PAIR_INTERVENE_LR") or "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return float(DEFAULT_PAIR_INTERVENE_LR)


def _device_of(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_paths(
    report: dict[str, Any],
    model_path: str | None,
    base_model_path: str | None,
) -> tuple[str, str | None]:
    from src.eif_adapter_env import (
        adapter_path_for_family,
        base_model_path_from_env,
        infer_report_family,
    )

    meta = report.get("experiment_meta") or {}
    family = infer_report_family(
        str(meta.get("report_file") or meta.get("fileName") or ""),
        report,
    )
    family_adapter = adapter_path_for_family(family)
    resolved_model = (
        str(model_path or "").strip()
        or family_adapter
        or str(os.environ.get("EIF_ADAPTER_PATH") or os.environ.get("EIF_MODEL_PATH") or "").strip()
        or str(meta.get("model_path") or meta.get("adapter_path") or "").strip()
    )
    if not resolved_model:
        raise ValueError(
            "No adapter/model path. Set EIF_ADAPTER_PATH_CE / EIF_ADAPTER_PATH_SALIENCY "
            "(or legacy EIF_ADAPTER_PATH), pass modelPath, or put model_path in the report meta."
        )

    resolved_base = (
        str(base_model_path or "").strip()
        or base_model_path_from_env()
        or str(os.environ.get("EIF_BASE_MODEL_PATH") or "").strip()
        or str(meta.get("base_model_path") or "").strip()
        or None
    )
    return resolved_model, resolved_base


def _model_has_inference_params(model) -> bool:
    return any(torch.is_inference(p) for p in model.parameters())


def _get_model(model_path: str, base_model_path: str | None):
    """Load/reuse model. Never construct weights under ``torch.inference_mode``."""
    key = (
        os.path.abspath(model_path),
        os.path.abspath(base_model_path) if base_model_path else "",
    )

    def _store(model, tokenizer):
        _MODEL_CACHE[key] = (model, tokenizer)
        try:
            from src.gold_live_attribution import _SESSION as _GOLD_SESSION
            if _GOLD_SESSION is not None:
                g_model = str(_GOLD_SESSION.get("model_path") or "")
                g_base = str(_GOLD_SESSION.get("base_path") or "")
                if (
                    os.path.abspath(g_model) == key[0]
                    and os.path.abspath(g_base or "") == key[1]
                ):
                    # Never replace a live gold model with a second load — that
                    # raced with stage2/3 and left float32 LoRA vs bf16 base.
                    if _GOLD_SESSION.get("model") is None:
                        _GOLD_SESSION["model"] = model
                        _GOLD_SESSION["tokenizer"] = tokenizer
        except Exception:
            pass
        return model, tokenizer
    with torch.inference_mode(False):
        # Prefer an already-loaded gold-live session when paths match (save VRAM).
        try:
            from src.gold_live_attribution import _SESSION as _GOLD_SESSION
            if _GOLD_SESSION is not None:
                g_model = str(_GOLD_SESSION.get("model_path") or "")
                g_base = str(_GOLD_SESSION.get("base_path") or "")
                if (
                    os.path.abspath(g_model) == key[0]
                    and os.path.abspath(g_base or "") == key[1]
                ):
                    model = _GOLD_SESSION.get("model")
                    if model is not None and not _model_has_inference_params(model):
                        return model, _GOLD_SESSION["tokenizer"]
                    if model is not None and _model_has_inference_params(model):
                        print(
                            "[unlearn][WARN] shared gold model has InferenceMode-tainted "
                            "weights; reloading outside inference_mode…",
                            flush=True,
                        )
                        _GOLD_SESSION["model"] = None
                        _release_cuda_memory(model, reason="reload_tainted_gold")
                        model, tokenizer = load_model_and_tokenizer(
                            model_path=model_path,
                            base_model_path=base_model_path,
                        )
                        _GOLD_SESSION["model"] = model
                        _GOLD_SESSION["tokenizer"] = tokenizer
                        return _store(model, tokenizer)
        except Exception:
            pass

        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            model, tokenizer = cached
            if not _model_has_inference_params(model):
                return model, tokenizer
            print(
                "[unlearn][WARN] cached model has InferenceMode-tainted weights; "
                "reloading outside inference_mode…",
                flush=True,
            )
            del _MODEL_CACHE[key]
            _release_cuda_memory(model, reason="reload_tainted_cache")

        model, tokenizer = load_model_and_tokenizer(
            model_path=model_path,
            base_model_path=base_model_path,
        )
        return _store(model, tokenizer)


def intervention_status() -> dict[str, Any]:
    if _INTERVENTION is None:
        return {"active": False}
    return {
        "active": True,
        "direction": _INTERVENTION.get("direction"),
        "pairId": _INTERVENTION.get("pair_id"),
        "trainSampleId": _INTERVENTION.get("train_sample_id"),
        "steps": int(_INTERVENTION.get("steps") or 1),
    }


def _accumulate_deltas(
    old_deltas: list[torch.Tensor],
    new_deltas: list[torch.Tensor],
) -> list[torch.Tensor]:
    if len(old_deltas) != len(new_deltas):
        raise RuntimeError(
            f"Cannot accumulate intervention deltas: "
            f"len {len(old_deltas)} vs {len(new_deltas)}"
        )
    return [o + n for o, n in zip(old_deltas, new_deltas)]


def recover_pair_intervention() -> dict[str, Any]:
    """Undo the full persisted learn/unlearn stack (if any)."""
    global _INTERVENTION
    state = _INTERVENTION
    if state is None:
        return {"status": "success", "recovered": False, "message": "No active intervention."}
    params = state.get("params") or []
    deltas = state.get("deltas") or []
    steps = int(state.get("steps") or 1)
    if params and deltas:
        _restore_filtered_ascent(params, deltas)
    direction = state.get("direction")
    pair_id = state.get("pair_id")
    model = state.get("model")
    _INTERVENTION = None
    _release_cuda_memory(model, reason="intervene_recover")
    return {
        "status": "success",
        "recovered": True,
        "direction": direction,
        "pairId": pair_id,
        "steps": steps,
        "intervention": intervention_status(),
    }


def _hydrate_annotation_viewer_env() -> None:
    """Ensure repo-root eif_api.env is loaded (shared with ttav / annotation-viewer)."""
    from src.gold_live_attribution import _hydrate_eif_env

    _hydrate_eif_env()

def _resolve_existing_path(raw: str) -> Path | None:
    p = Path(raw).expanduser()
    candidates = [p] if p.is_absolute() else [
        Path.cwd() / p,
        REPO_ROOT / p,
        REPO_ROOT / "tools" / "annotation-viewer" / p,
    ]
    for cand in candidates:
        try:
            resolved = cand.resolve()
        except OSError:
            continue
        if resolved.is_file():
            return resolved
    return None


def _train_jsonl_path() -> Path | None:
    _hydrate_annotation_viewer_env()
    for key in ("EIF_TRAIN_DATA", "ANNOTATION_TRAIN_DATA"):
        raw = (os.environ.get(key) or "").strip()
        if not raw:
            continue
        found = _resolve_existing_path(raw)
        if found is not None:
            return found
    smoke = REPO_ROOT / "smoke_train_data.jsonl"
    return smoke if smoke.is_file() else None


def _read_jsonl_input_ids(path: Path, index: int) -> list[int] | None:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            if i != index:
                continue
            obj = json.loads(line)
            ids = obj.get("input_ids")
            if isinstance(ids, list) and ids:
                return [int(x) for x in ids]
            return None
    return None


def _token_matches_id(tokenizer, token: str, token_id: int) -> bool:
    decoded = tokenizer.decode([int(token_id)])
    if token == decoded:
        return True
    vocab_tok = tokenizer.convert_ids_to_tokens(int(token_id))
    if token == vocab_tok:
        return True
    # Report JSON sometimes shows U+FFFD for byte-fallback pieces; allow soft match.
    if "\ufffd" in token and len(token) == len(decoded):
        return all(
            token[j] == decoded[j]
            for j in range(len(token))
            if token[j] != "\ufffd"
        )
    return False


def _ids_match_tokens(
    tokenizer,
    ids: list[int],
    tokens: list[str],
    *,
    check_n: int | None = None,
) -> bool:
    if len(ids) != len(tokens):
        return False
    limit = len(tokens) if check_n is None else min(check_n, len(tokens))
    return all(_token_matches_id(tokenizer, tokens[i], ids[i]) for i in range(limit))


def _align_pt_ids_to_tokens(tokenizer, ids: list[int], tokens: list[str]) -> list[int] | None:
    """Find a contiguous window in ``ids`` whose decode matches report tokens."""
    n = len(tokens)
    if n == 0:
        return []
    if len(ids) == n and _ids_match_tokens(tokenizer, ids, tokens):
        return ids
    if len(ids) < n:
        return None
    for start in range(len(ids) - n + 1):
        window = ids[start : start + n]
        if not _ids_match_tokens(tokenizer, window, tokens, check_n=min(32, n)):
            continue
        if _ids_match_tokens(tokenizer, window, tokens):
            return window
    return None


def _load_ids_from_embedding_pt(
    tokenizer,
    pt_path: Path,
    tokens: list[str],
) -> list[int] | None:
    try:
        blob = torch.load(pt_path, map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(pt_path, map_location="cpu")
    except Exception:
        return None
    if not isinstance(blob, dict) or "input_ids" not in blob:
        return None
    raw = blob["input_ids"]
    if torch.is_tensor(raw):
        ids = [int(x) for x in raw.tolist()]
    else:
        ids = [int(x) for x in raw]
    return _align_pt_ids_to_tokens(tokenizer, ids, tokens)


def _embedding_pt_paths(sample_id: str | None, train_sample_id: int) -> tuple[Path | None, Path | None]:
    if not sample_id:
        return None, None
    root = Path(
        (os.environ.get("EIF_TOKEN_EMBEDDING_ROOT") or "").strip()
        or (REPO_ROOT / "token_embeddings")
    )
    test_pt = root / sample_id / "test.pt"
    train_pt = root / sample_id / f"train_{train_sample_id}.pt"
    return (
        test_pt if test_pt.is_file() else None,
        train_pt if train_pt.is_file() else None,
    )


def _resolve_token_ids(
    tokenizer,
    tokens: list[str],
    *,
    stored_ids: list[int] | None = None,
    jsonl_ids: list[int] | None = None,
    pt_ids: list[int] | None = None,
    side: str = "tokens",
) -> list[int]:
    """Resolve ids: report → embedding .pt → train JSONL → surface remap."""
    if stored_ids is not None and len(stored_ids) == len(tokens) and stored_ids:
        return [int(x) for x in stored_ids]
    if pt_ids is not None and len(pt_ids) == len(tokens) and pt_ids:
        return [int(x) for x in pt_ids]
    if jsonl_ids is not None and len(jsonl_ids) == len(tokens) and jsonl_ids:
        if _ids_match_tokens(tokenizer, jsonl_ids, tokens):
            return [int(x) for x in jsonl_ids]
    try:
        token_ids = convert_report_tokens_to_ids(tokenizer, tokens)
        validate_roundtrip_tokens(tokenizer, token_ids, tokens)
        return token_ids
    except ValueError as exc:
        raise ValueError(
            f"Failed to resolve {side} token ids ({exc}). "
            "Set EIF_TRAIN_DATA / ANNOTATION_TRAIN_DATA to the original train JSONL, "
            "or ensure token_embeddings/<sample_id>/{test,train_N}.pt exist."
        ) from exc


def _batch_from_tokens(
    tokenizer,
    tokens: list[str],
    *,
    answer_start_index: int,
    labeled_positions: set[int] | None = None,
    stored_ids: list[int] | None = None,
    jsonl_ids: list[int] | None = None,
    pt_ids: list[int] | None = None,
    side: str = "tokens",
) -> dict[str, torch.Tensor]:
    """Build a single-row causal-LM batch from report token surfaces / ids."""
    token_ids = _resolve_token_ids(
        tokenizer,
        tokens,
        stored_ids=stored_ids,
        jsonl_ids=jsonl_ids,
        pt_ids=pt_ids,
        side=side,
    )
    input_ids = torch.tensor([token_ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    labels = input_ids.clone()
    if answer_start_index > 0:
        labels[0, :answer_start_index] = -100
    if labeled_positions is not None:
        keep = {int(p) for p in labeled_positions}
        for i in range(labels.size(1)):
            if i not in keep:
                labels[0, i] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def _logprob_from_ce(ce: float) -> float:
    return float(-ce)


def _resolve_saliency_fn(report: dict[str, Any]):
    """Match intervention_experiment saliency (default last_layer, not full ALTI)."""
    meta = report.get("experiment_meta") or {}
    cfg = meta.get("config") or {}
    mode = (
        str(cfg.get("SALIENCY_MODE") or meta.get("saliency_mode") or "last_layer")
        .strip()
        .lower()
    )
    if mode in {"full_alti", "alti_full", "full"}:
        return compute_alti_saliency_vector, "full_alti"
    return compute_last_layer_saliency_vector, "last_layer"


def _prepare_model_for_intervention_saliency(model, saliency_mode: str) -> None:
    """Mirror intervention_experiment last_layer setup before scoring saliency.

    Report generation puts the model in ``train()`` with Dropout frozen to
    ``eval()`` (and grad-checkpointing enabled). Plain ``model.eval()`` can
    change Peft / checkpointing behavior enough to shift ALTI magnitudes.
    """
    if saliency_mode != "last_layer":
        model.eval()
        return
    if hasattr(model, "enable_input_require_grads"):
        try:
            model.enable_input_require_grads()
        except Exception:
            pass
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except TypeError:
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass
    except Exception:
        pass
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.eval()


def _release_cuda_memory(model=None, *, reason: str = "") -> None:
    """Drop transient graphs / allocator cache after each probe.

    Unlearn does not write adapter junk to disk; OOM after a few clicks is
    almost always VRAM fragmentation from long-seq last-layer attn (HxTxT)
    plus the process-cached model. Call this in a ``finally`` every time.
    """
    if model is not None:
        try:
            model.eval()
            model.zero_grad(set_to_none=True)
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        if reason:
            try:
                free_b, total_b = torch.cuda.mem_get_info()
                print(
                    f"[unlearn] cuda after {reason}: "
                    f"free={free_b / 1e9:.2f}G / total={total_b / 1e9:.2f}G",
                    flush=True,
                )
            except Exception:
                pass


def _score_edge_saliency(
    model,
    batch: dict[str, torch.Tensor],
    *,
    saliency_fn,
    saliency_mode: str,
    target_index: int,
    source_index: int,
) -> float | None:
    _prepare_model_for_intervention_saliency(model, saliency_mode)
    # Use no_grad (not inference_mode): IM can taint shared model weights if any
    # param is (re)allocated / .data-assigned during the forward.
    with torch.no_grad():
        sal_vec = saliency_fn(model, batch, target_index)
    score = None
    if 0 <= source_index < len(sal_vec):
        score = float(sal_vec[source_index])
    del sal_vec
    _release_cuda_memory(model, reason="saliency")
    return score


def _resolve_match_param_filter(
    model,
    report: dict[str, Any],
) -> tuple[Callable[[str, Any], bool], str, int]:
    """Same θ filter as intervention Stage-3 matching (LoRA / fine-attn last-N)."""
    meta = report.get("experiment_meta") or {}
    cfg = meta.get("config") or {}
    last_n = int(cfg.get("FINE_MATCH_LAST_N_LAYERS") or 1)
    if last_n <= 0:
        last_n = 1
    proj = str(cfg.get("FINE_MATCH_PROJ") or "qk").strip().lower() or "qk"

    model_space = str(getattr(model, "_eif_grad_space", "") or "").strip().lower()
    report_space = str(cfg.get("GRAD_SPACE") or meta.get("grad_space") or "").strip().lower()
    space = model_space or report_space or "lora"

    if space == "lora" or "lora" in space:
        filt = make_lora_param_filter(model, last_n_layers=last_n)
        tag = f"lora_L{last_n}"
    else:
        filt = make_attention_projection_filter(model, last_n, proj)
        tag = f"fineattn_{proj}_L{last_n}"
    return filt, tag, last_n


def _batch_to_device(batch: dict[str, torch.Tensor], device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def _intervene_loss_config(model_path: str | None) -> BankLossConfig:
    """CE + single-edge contrastive hypers (force ce_saliency; reuse bank/train knobs)."""
    base = load_bank_loss_config(model_path, model_tag="")
    return BankLossConfig(
        loss_mode="ce_saliency",
        saliency_loss_type="contrastive",
        saliency_lambda=float(base.saliency_lambda),
        alpha=float(base.alpha),
        eps=float(base.eps),
        margin_plus=float(base.margin_plus),
        neg_sample_k=int(base.neg_sample_k),
        saliency_layer=int(base.saliency_layer),
        exclude_sink_prefix=int(base.exclude_sink_prefix),
        exclude_special_tokens=bool(base.exclude_special_tokens),
    )


def _compute_filtered_train_grads(
    model,
    batch: dict[str, torch.Tensor],
    param_filter: Callable[[str, Any], bool],
    device,
    *,
    train_source_index: int,
    train_target_index: int,
    loss_cfg: BankLossConfig,
) -> tuple[list[torch.nn.Parameter], list[torch.Tensor], dict[str, float]]:
    """∇_θ L on filtered params, L = CE(tgt) + λ · contrastive(selected edge)."""
    if _model_has_inference_params(model):
        raise RuntimeError(
            "Model parameters are InferenceMode-tainted (often from loading under "
            "@torch.inference_mode, e.g. next-token probs). Reload the API process "
            "or call _get_model again so weights are reconstructed outside inference_mode."
        )

    with torch.inference_mode(False):
        named = [(n, p) for n, p in model.named_parameters() if param_filter(n, p)]
        if not named:
            raise RuntimeError(
                "No parameters matched the match-space filter "
                "(expected last-N LoRA or fine-attn projections)."
            )
        params = [p for _, p in named]
        n_elems = sum(p.numel() for p in params)
        print(
            f"[unlearn] match-space grad: {len(params)} tensors, {n_elems / 1e6:.3f}M elems "
            f"(e.g. {named[0][0]})",
            flush=True,
        )

        local = _batch_to_device(batch, device)
        model.eval()
        model.zero_grad(set_to_none=True)
        original_flags = [(p, p.requires_grad) for p in model.parameters()]
        for p in model.parameters():
            p.requires_grad_(False)
        for p in params:
            p.requires_grad_(True)

        try:
            with torch.enable_grad():
                outputs = model(
                    input_ids=local["input_ids"],
                    attention_mask=local.get("attention_mask"),
                    labels=local["labels"],
                    output_attentions=True,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
                ce = outputs.loss
                if ce is None or not torch.isfinite(ce):
                    raise RuntimeError(f"Train CE loss invalid: {ce}")
                if ce.dim() > 0:
                    ce = ce.mean()

                saliency_loss_from_outputs = _import_saliency_loss_from_outputs()
                n_tokens = int(local["input_ids"].size(1))
                edges = [{"src": int(train_source_index), "dst": int(train_target_index)}]
                annot_pairs = _annot_pairs_from_edges(edges, n_tokens, device)
                exclude = None
                if loss_cfg.exclude_sink_prefix > 0:
                    em = torch.zeros_like(local["input_ids"], dtype=torch.bool)
                    em[:, : loss_cfg.exclude_sink_prefix] = True
                    exclude = em

                diag = saliency_loss_from_outputs(
                    model,
                    outputs,
                    annot_pairs,
                    saliency_layer=loss_cfg.saliency_layer,
                    exclude_source_mask=exclude,
                    alpha=loss_cfg.alpha,
                    eps=loss_cfg.eps,
                    floor_eps=0.0,
                    floor_eps_mode="fixed",
                    floor_eps_step=0,
                    floor_eps_warmup_steps=0,
                    floor_logit_eps=None,
                    loss_type=loss_cfg.saliency_loss_type,
                    margin_plus=loss_cfg.margin_plus,
                    neg_sample_k=loss_cfg.neg_sample_k,
                )
                sal = diag.loss
                if sal is None or not torch.isfinite(sal):
                    raise RuntimeError(f"Train saliency loss invalid: {sal}")
                loss = ce + float(loss_cfg.saliency_lambda) * sal
                print(
                    f"[unlearn] train L=CE+λ·con: ce={float(ce.detach()):.6g} "
                    f"sal={float(sal.detach()):.6g} λ={loss_cfg.saliency_lambda:g} "
                    f"L={float(loss.detach()):.6g} "
                    f"edge={train_source_index}->{train_target_index} "
                    f"k={loss_cfg.neg_sample_k} m={loss_cfg.margin_plus:g}",
                    flush=True,
                )
                grads = torch.autograd.grad(
                    loss,
                    params,
                    create_graph=False,
                    retain_graph=False,
                    allow_unused=True,
                )
            out_grads: list[torch.Tensor] = []
            for p, g in zip(params, grads):
                if g is None:
                    out_grads.append(torch.zeros_like(p, dtype=torch.float32))
                else:
                    out_grads.append(g.detach().to(dtype=torch.float32))
            stats = {
                "ce": float(ce.detach().cpu()),
                "saliency": float(sal.detach().cpu()),
                "total": float(loss.detach().cpu()),
                "saliencyLambda": float(loss_cfg.saliency_lambda),
            }
            return params, out_grads, stats
        finally:
            for p, flag in original_flags:
                p.requires_grad_(flag)
            model.zero_grad(set_to_none=True)
            del local


def _apply_filtered_ascent(
    params: list[torch.nn.Parameter],
    grads: list[torch.Tensor],
    *,
    lr: float,
    normalize: bool,
) -> list[torch.Tensor]:
    flat = torch.cat([g.reshape(-1) for g in grads]).float()
    if flat.numel() == 0:
        raise RuntimeError("Empty gradient for ascent update.")
    if normalize:
        flat = flat / flat.norm().clamp_min(1e-12)
    flat = flat * float(lr)

    deltas: list[torch.Tensor] = []
    offset = 0
    for p, g in zip(params, grads):
        n = int(g.numel())
        chunk = flat[offset : offset + n].reshape_as(p).to(device=p.device, dtype=p.dtype)
        p.data.add_(chunk)
        deltas.append(chunk)
        offset += n
    return deltas


def _restore_filtered_ascent(
    params: list[torch.nn.Parameter],
    deltas: list[torch.Tensor],
) -> None:
    for p, d in zip(params, deltas):
        p.data.sub_(d.to(device=p.device, dtype=p.dtype))


def run_unlearn_pair_probe(
    report: dict[str, Any],
    *,
    train_sample_id: int,
    test_source_index: int,
    test_target_index: int,
    train_source_index: int,
    train_target_index: int,
    pair_id: str | None = None,
    model_path: str | None = None,
    base_model_path: str | None = None,
    unlearn_lr: float | None = None,
    normalize_grad: bool = True,
    recompute_saliency: bool = True,
    sample_id: str | None = None,
    report_json_path: str | None = None,
    direction: str = "unlearn",
    persist: bool = False,
    completion_mode: str = "predict",
    train_sample_detail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one learn/unlearn step; with persist, stack until ``recover_pair_intervention``."""
    global _INTERVENTION

    direction_norm = (direction or "unlearn").strip().lower()
    if direction_norm not in {"unlearn", "learn"}:
        raise ValueError(f"direction must be 'unlearn' or 'learn', got {direction!r}")
    completion_norm = (completion_mode or "predict").strip().lower()
    if completion_norm not in {"predict", "gold"}:
        raise ValueError(f"completion_mode must be predict|gold, got {completion_mode!r}")
    lr_mag = float(unlearn_lr) if unlearn_lr is not None else _default_pair_intervene_lr()
    # Unlearn = ascent (+lr); Learn = descent (-lr).
    signed_lr = lr_mag if direction_norm == "unlearn" else -lr_mag

    baseline = report.get("test_sample_baseline") or {}
    if completion_norm == "gold":
        test_tokens = list(baseline.get("correct_full_tokens") or [])
        test_ids_key = "correct_full_token_ids"
    else:
        test_tokens = list(baseline.get("full_tokens") or [])
        test_ids_key = "full_token_ids"
    if not test_tokens:
        raise ValueError(f"Report is missing test tokens for completion_mode={completion_norm}.")

    if train_sample_detail is not None:
        details = dict(report.get("train_sample_details") or {})
        details[str(train_sample_id)] = train_sample_detail
        report = {**report, "train_sample_details": details}

    train_detail = (report.get("train_sample_details") or {}).get(str(train_sample_id))
    if not isinstance(train_detail, dict):
        raise ValueError(f"Report has no train_sample_details[{train_sample_id}].")
    train_tokens = list(train_detail.get("full_tokens") or [])
    if not train_tokens:
        raise ValueError(f"Train sample {train_sample_id} has empty full_tokens.")

    answer_start = int(train_detail.get("answer_start_index") or 0)
    test_prompt_len = int(baseline.get("prompt_len") or 0)

    if not (0 <= test_source_index < test_target_index < len(test_tokens)):
        raise ValueError(
            f"Invalid test edge indices: source={test_source_index}, "
            f"target={test_target_index}, seq_len={len(test_tokens)}."
        )
    if not (0 <= train_source_index < train_target_index < len(train_tokens)):
        raise ValueError(
            f"Invalid train edge indices: source={train_source_index}, "
            f"target={train_target_index}, seq_len={len(train_tokens)}."
        )

    # Non-persist probes must start from clean weights. Persist stacks accumulate.
    if _INTERVENTION is not None and not persist:
        recover_pair_intervention()

    resolved_model, resolved_base = _resolve_paths(report, model_path, base_model_path)
    meta_model = str((report.get("experiment_meta") or {}).get("model_path") or "").strip()
    if meta_model and os.path.abspath(resolved_model) != os.path.abspath(meta_model):
        print(
            f"[unlearn][WARN] loaded model differs from report meta:\n"
            f"  loaded: {resolved_model}\n"
            f"  report: {meta_model}",
            flush=True,
        )
    model, tokenizer = _get_model(resolved_model, resolved_base)
    device = _device_of(model)
    grad_space = getattr(model, "_eif_grad_space", "unknown")
    resolved_base_used = getattr(model, "_eif_base_model_path", resolved_base)

    test_ids_stored = baseline.get(test_ids_key)
    if not isinstance(test_ids_stored, list):
        test_ids_stored = None
    train_ids_stored = train_detail.get("full_token_ids")
    if not isinstance(train_ids_stored, list):
        train_ids_stored = None

    resolved_sample_id = (sample_id or "").strip() or None
    if not resolved_sample_id and report_json_path:
        resolved_sample_id = infer_sample_id(str(report_json_path))

    test_pt, train_pt = _embedding_pt_paths(resolved_sample_id, train_sample_id)
    test_pt_ids = (
        _load_ids_from_embedding_pt(tokenizer, test_pt, test_tokens) if test_pt else None
    )
    train_pt_ids = (
        _load_ids_from_embedding_pt(tokenizer, train_pt, train_tokens) if train_pt else None
    )

    train_jsonl = _train_jsonl_path()
    train_jsonl_ids = (
        _read_jsonl_input_ids(train_jsonl, train_sample_id) if train_jsonl else None
    )
    if train_jsonl_ids is not None and len(train_jsonl_ids) != len(train_tokens):
        train_jsonl_ids = None

    print(
        f"[{direction_norm}] id sources: sample={resolved_sample_id or '-'} "
        f"test_pt={'yes' if test_pt_ids else 'no'} "
        f"train_pt={'yes' if train_pt_ids else 'no'} "
        f"train_jsonl={train_jsonl or '-'} "
        f"({'aligned' if train_jsonl_ids else 'skip'})",
        flush=True,
    )

    test_batch = _batch_from_tokens(
        tokenizer,
        test_tokens,
        answer_start_index=test_prompt_len,
        labeled_positions=None,
        stored_ids=test_ids_stored,
        pt_ids=test_pt_ids,
        side="test",
    )
    train_batch = _batch_from_tokens(
        tokenizer,
        train_tokens,
        answer_start_index=answer_start,
        labeled_positions={train_target_index},
        stored_ids=train_ids_stored,
        pt_ids=train_pt_ids,
        jsonl_ids=train_jsonl_ids,
        side="train",
    )

    reported_saliency = None
    for corr in baseline.get("top_correlations") or []:
        if int(corr.get("source_token_index", -1)) == int(test_source_index) and int(
            corr.get("target_token_index", test_target_index)
        ) == int(test_target_index):
            reported_saliency = float(corr.get("saliency_score"))
            break
    if reported_saliency is None:
        for row in report.get("per_token_results") or []:
            if int(row.get("target_token_index", -1)) != int(test_target_index):
                continue
            for corr in row.get("top_correlations") or []:
                if int(corr.get("source_token_index", -1)) == int(test_source_index):
                    reported_saliency = float(corr.get("saliency_score"))
                    break
            break

    base_losses = _target_token_losses(model, test_batch, [test_target_index], device)
    if base_losses.numel() == 0:
        raise RuntimeError("Could not score CE at the test target position.")
    base_ce = float(base_losses[0].item())
    base_logprob = _logprob_from_ce(base_ce)

    saliency_fn, saliency_mode = _resolve_saliency_fn(report)
    print(f"[{direction_norm}] saliency_mode={saliency_mode}", flush=True)
    print(
        f"[{direction_norm}] test edge src={test_source_index} "
        f"{test_tokens[test_source_index]!r} -> tgt={test_target_index} "
        f"{test_tokens[test_target_index]!r} "
        f"id_src={int(test_batch['input_ids'][0, test_source_index])} "
        f"id_tgt={int(test_batch['input_ids'][0, test_target_index])}",
        flush=True,
    )

    base_saliency = None
    if recompute_saliency:
        base_saliency = _score_edge_saliency(
            model,
            test_batch,
            saliency_fn=saliency_fn,
            saliency_mode=saliency_mode,
            target_index=test_target_index,
            source_index=test_source_index,
        )
        if (
            reported_saliency is not None
            and base_saliency is not None
            and abs(base_saliency - reported_saliency) > max(1e-3, 0.05 * abs(reported_saliency))
        ):
            print(
                f"[{direction_norm}][WARN] recomputed saliency {base_saliency:.6g} differs from "
                f"report {reported_saliency:.6g} (mode={saliency_mode}). "
                f"Check model_path/base_path match the report run.",
                flush=True,
            )
        elif reported_saliency is not None and base_saliency is not None:
            print(
                f"[{direction_norm}] saliency OK vs report: {base_saliency:.6g} ≈ {reported_saliency:.6g}",
                flush=True,
            )

    param_filter, param_space_tag, last_n_layers = _resolve_match_param_filter(model, report)
    loss_cfg = _intervene_loss_config(resolved_model)
    print(
        f"[{direction_norm}] param_space={param_space_tag} (aligned with pair matching; "
        f"model_grad_space={grad_space})",
        flush=True,
    )
    print(
        f"[{direction_norm}] train_obj=CE+λ·contrastive(single_edge) "
        f"λ={loss_cfg.saliency_lambda:g} α={loss_cfg.alpha:g} "
        f"m={loss_cfg.margin_plus:g} k={loss_cfg.neg_sample_k} "
        f"layer={loss_cfg.saliency_layer}",
        flush=True,
    )

    params = None
    deltas = None
    restored = True
    train_loss_stats: dict[str, float] | None = None
    try:
        params, grads, train_loss_stats = _compute_filtered_train_grads(
            model,
            train_batch,
            param_filter,
            device,
            train_source_index=train_source_index,
            train_target_index=train_target_index,
            loss_cfg=loss_cfg,
        )
        grad_norm = float(torch.cat([g.reshape(-1) for g in grads]).float().norm().item())
        deltas = _apply_filtered_ascent(
            params,
            grads,
            lr=signed_lr,
            normalize=bool(normalize_grad),
        )
        delta_norm = float(torch.cat([d.reshape(-1).float() for d in deltas]).norm().item())
        print(
            f"[{direction_norm}] grad_norm={grad_norm:.6g} step_norm={delta_norm:.6g} "
            f"lr={signed_lr} persist={persist}",
            flush=True,
        )
        del grads

        after_losses = _target_token_losses(model, test_batch, [test_target_index], device)
        if after_losses.numel() == 0:
            raise RuntimeError(f"Could not score post-{direction_norm} CE at the test target.")
        after_ce = float(after_losses[0].item())
        after_logprob = _logprob_from_ce(after_ce)
        print(
            f"[{direction_norm}] CE before={base_ce:.8g} after={after_ce:.8g} "
            f"dCE={after_ce - base_ce:.8g}",
            flush=True,
        )

        after_saliency = None
        if recompute_saliency:
            after_saliency = _score_edge_saliency(
                model,
                test_batch,
                saliency_fn=saliency_fn,
                saliency_mode=saliency_mode,
                target_index=test_target_index,
                source_index=test_source_index,
            )

        stacked_steps = 1
        if persist:
            if _INTERVENTION is not None:
                prev_params = _INTERVENTION.get("params") or []
                prev_deltas = _INTERVENTION.get("deltas") or []
                if (
                    prev_params
                    and prev_deltas
                    and len(prev_params) == len(params)
                    and all(a is b for a, b in zip(prev_params, params))
                ):
                    deltas = _accumulate_deltas(prev_deltas, deltas)
                    stacked_steps = int(_INTERVENTION.get("steps") or 1) + 1
                else:
                    # Param set changed — drop the old stack and keep only this step.
                    print(
                        f"[{direction_norm}][WARN] intervention param set changed; "
                        "starting a fresh persist stack on current weights.",
                        flush=True,
                    )
                    stacked_steps = 1
            _INTERVENTION = {
                "direction": direction_norm,
                "pair_id": pair_id,
                "train_sample_id": int(train_sample_id),
                "params": params,
                "deltas": deltas,
                "model": model,
                "steps": stacked_steps,
            }
            restored = False
            params = None
            deltas = None
    finally:
        if restored and params is not None and deltas is not None:
            _restore_filtered_ascent(params, deltas)
        if restored:
            params = None
            deltas = None
            _release_cuda_memory(model, reason=f"{direction_norm}_finally")

    delta_ce = after_ce - base_ce
    delta_logprob = after_logprob - base_logprob
    delta_saliency = (
        None if base_saliency is None or after_saliency is None
        else after_saliency - base_saliency
    )

    if direction_norm == "unlearn":
        verdict = "inconclusive"
        if delta_ce > 1e-4 and (delta_saliency is None or delta_saliency < -1e-8):
            verdict = "supports_causal"
        elif abs(delta_ce) <= 1e-4 and delta_saliency is not None and delta_saliency < -1e-4:
            verdict = "saliency_only"
        elif abs(delta_ce) <= 1e-4:
            verdict = "no_effect"
        elif delta_ce < -1e-4:
            verdict = "opposite_effect"
    else:
        # Learn: expect test CE ↓ / logP ↑ if the train edge helps the target.
        verdict = "inconclusive"
        if delta_ce < -1e-4 and (delta_saliency is None or delta_saliency > 1e-8):
            verdict = "supports_causal"
        elif abs(delta_ce) <= 1e-4 and delta_saliency is not None and delta_saliency > 1e-4:
            verdict = "saliency_only"
        elif abs(delta_ce) <= 1e-4:
            verdict = "no_effect"
        elif delta_ce > 1e-4:
            verdict = "opposite_effect"

    rule = (
        "theta <- theta + eta * normalized(grad_train_L)"
        if direction_norm == "unlearn"
        else "theta <- theta - eta * normalized(grad_train_L)"
    )
    if not normalize_grad:
        rule = rule.replace("normalized(", "").replace(")", "", 1)

    stacked_steps_out = int((_INTERVENTION or {}).get("steps") or 1) if not restored else 1

    result = {
        "status": "success",
        "pairId": pair_id,
        "direction": direction_norm,
        "trainSampleId": int(train_sample_id),
        "modelPath": resolved_model,
        "baseModelPath": resolved_base_used,
        "gradSpace": grad_space,
        "update": {
            "paramSpace": param_space_tag,
            "lastNLayers": int(last_n_layers),
            "rule": rule,
            "direction": direction_norm,
            "unlearnLr": float(lr_mag),
            "signedLr": float(signed_lr),
            "normalizeGrad": bool(normalize_grad),
            "steps": int(stacked_steps_out),
            "persist": bool(persist),
            "persistsToDisk": False,
            "trainLossScope": "ce_plus_single_edge_contrastive",
            "trainObjective": "CE(train_target) + lambda * contrastive(selected_edge)",
            "sameSignObjective": True,
            "trainTargetIndex": int(train_target_index),
            "trainSourceIndex": int(train_source_index),
            "saliencyLossType": loss_cfg.saliency_loss_type,
            "saliencyLambda": float(loss_cfg.saliency_lambda),
            "saliencyAlpha": float(loss_cfg.alpha),
            "saliencyMarginPlus": float(loss_cfg.margin_plus),
            "saliencyNegSampleK": int(loss_cfg.neg_sample_k),
            "saliencyLayer": int(loss_cfg.saliency_layer),
            "trainLoss": train_loss_stats,
        },
        "testEdge": {
            "sourceIndex": int(test_source_index),
            "targetIndex": int(test_target_index),
            "sourceToken": test_tokens[test_source_index],
            "targetToken": test_tokens[test_target_index],
            "reportedSaliency": reported_saliency,
            "saliencyMode": saliency_mode,
        },
        "before": {
            "ce": base_ce,
            "logprob": base_logprob,
            "saliency": base_saliency,
        },
        "after": {
            "ce": after_ce,
            "logprob": after_logprob,
            "saliency": after_saliency,
        },
        "delta": {
            "ce": delta_ce,
            "logprob": delta_logprob,
            "saliency": delta_saliency,
        },
        "verdict": verdict,
        "restored": restored,
        "intervention": intervention_status(),
    }
    del test_batch, train_batch
    if restored:
        _release_cuda_memory(model, reason=f"{direction_norm}_return")
    else:
        _release_cuda_memory(model, reason=f"{direction_norm}_persist")
    return result


def compute_next_token_probs(
    report: dict[str, Any],
    *,
    mode: str,
    target_index: int,
    top_k: int = 10,
    model_path: str | None = None,
    base_model_path: str | None = None,
) -> dict[str, Any]:
    """Top-k next-token distribution that produces ``tokens[target_index]``.

    ``mode=predict`` uses model completion tokens; ``mode=gold`` uses teacher-forced
    gold tokens. Prefix is ``ids[:target_index]`` (causal LM predicts position t
    from tokens 0..t-1).

    Model load must NOT run under ``torch.inference_mode`` — that permanently marks
    Parameters as inference tensors and breaks later learn/unlearn grads.
    """
    mode_norm = (mode or "predict").strip().lower()
    if mode_norm not in {"predict", "gold"}:
        raise ValueError(f"mode must be 'predict' or 'gold', got {mode!r}")

    baseline = report.get("test_sample_baseline") or {}
    if mode_norm == "gold":
        tokens = list(baseline.get("correct_full_tokens") or [])
        stored_ids = baseline.get("correct_full_token_ids")
    else:
        tokens = list(baseline.get("full_tokens") or [])
        stored_ids = baseline.get("full_token_ids")
    if not tokens:
        raise ValueError(f"Report missing tokens for mode={mode_norm}.")

    resolved_model, resolved_base = _resolve_paths(report, model_path, base_model_path)
    model, tokenizer = _get_model(resolved_model, resolved_base)
    try:
        from src.intervention_experiment import ensure_peft_lora_dtype
        ensure_peft_lora_dtype(model, torch.bfloat16)
    except Exception:
        pass
    device = _device_of(model)

    if isinstance(stored_ids, list) and len(stored_ids) == len(tokens):
        ids = [int(x) for x in stored_ids]
    else:
        hint = None
        hint_until = None
        if mode_norm == "gold":
            pred_ids = baseline.get("full_token_ids")
            prompt_len = int(baseline.get("prompt_len") or 0)
            if isinstance(pred_ids, list) and prompt_len > 0:
                hint = [int(x) for x in pred_ids]
                hint_until = prompt_len
        try:
            ids = convert_report_tokens_to_ids(
                tokenizer, tokens, hint_ids=hint, hint_until=hint_until,
            )
        except ValueError:
            if mode_norm != "gold" or hint is None or hint_until is None:
                raise
            prompt_ids = hint[:hint_until]
            answer_surf = "".join(tokens[hint_until:])
            answer_ids = [int(x) for x in tokenizer.encode(answer_surf, add_special_tokens=False)]
            ids = prompt_ids + answer_ids
            if len(ids) != len(tokens):
                tokens = list(tokens[:hint_until]) + [
                    tokenizer.decode([i]) for i in answer_ids
                ]
        if len(ids) != len(tokens):
            raise ValueError(
                f"Could not align token ids ({len(ids)}) to surfaces ({len(tokens)})."
            )

    t = int(target_index)
    if not (0 < t < len(ids)):
        raise ValueError(
            f"target_index={t} out of range for next-token probs "
            f"(need 0 < t < {len(ids)})."
        )

    # Gold/unlearn leave the shared model in train()+GC with input-require-grads.
    # A plain forward then often hits float32 activations vs bf16 LoRA/base weights.
    was_training = bool(model.training)
    model.eval()
    disabled_input_grads = False
    if hasattr(model, "disable_input_require_grads"):
        try:
            model.disable_input_require_grads()
            disabled_input_grads = True
        except Exception:
            pass
    if hasattr(model, "gradient_checkpointing_disable"):
        try:
            model.gradient_checkpointing_disable()
        except Exception:
            pass

    prefix = torch.tensor([ids[:t]], dtype=torch.long, device=device)
    attn = torch.ones_like(prefix)
    outputs = None
    logits = None
    probs = None
    actual_id = int(ids[t])
    actual_prob = 0.0
    top_rows: list[dict[str, Any]] = []
    try:
        with torch.no_grad():
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(
                        input_ids=prefix,
                        attention_mask=attn,
                        use_cache=False,
                    )
            else:
                outputs = model(
                    input_ids=prefix,
                    attention_mask=attn,
                    use_cache=False,
                )
            logits = outputs.logits[0, -1].float()
            probs = F.softmax(logits, dim=-1)
            k = max(1, min(int(top_k), int(probs.numel())))
            values, indices = torch.topk(probs, k=k)

            actual_prob = float(probs[actual_id].item())
            top_rows = []
            for p, tid in zip(values.tolist(), indices.tolist()):
                tid_i = int(tid)
                top_rows.append({
                    "token": tokenizer.decode([tid_i]),
                    "tokenId": tid_i,
                    "prob": float(p),
                    "isActual": tid_i == actual_id,
                })
    finally:
        del outputs, logits, probs, prefix, attn
        if disabled_input_grads and hasattr(model, "enable_input_require_grads"):
            try:
                model.enable_input_require_grads()
            except Exception:
                pass
        if was_training:
            # Restore gold/unlearn last-layer setup (GC + frozen dropout).
            try:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                try:
                    model.gradient_checkpointing_enable()
                except Exception:
                    pass
            except Exception:
                pass
            model.train()
            for m in model.modules():
                if isinstance(m, torch.nn.Dropout):
                    m.eval()
        _release_cuda_memory(model, reason="next_token_probs")

    return {
        "status": "success",
        "mode": mode_norm,
        "targetIndex": t,
        "actualToken": tokens[t],
        "actualTokenId": actual_id,
        "actualProb": actual_prob,
        "top": top_rows,
        "intervention": intervention_status(),
        "modelPath": resolved_model,
    }
