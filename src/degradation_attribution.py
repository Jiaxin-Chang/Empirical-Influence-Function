"""Decision-flip (degradation) attribution: compare adapters at one prefix.

Target quantity at teacher-forced / predict position t::

    Δ = (logit_gained − logit_lost)_live − (logit_gained − logit_lost)_compare

Default pair is live argmax vs compare-view argmax (gold mode: lost = gold token).
Who is responsible ≈ train-bank rows whose sketched ∇L_train aligns with
∇_θ (logit_gained − logit_lost) on the live adapter.
"""

from __future__ import annotations

import math
import os
from contextlib import contextmanager
from heapq import nlargest
from typing import Any, Iterator

import torch
import torch.nn.functional as F

from src.eif_adapter_env import (
    env_adapter_path_for_family,
    infer_report_family,
    list_compare_views,
    normalize_view_family,
)
from src.intervention_experiment import (
    PRESCREEN_SKETCH_DIM,
    PRESCREEN_SKETCH_SEED,
    _project_flat_grad,
    _score_prescreen_sketch_cache,
    ensure_peft_lora_dtype,
)
from src.loss import prepare_last_layer_grad_checkpointing


def _abspath(path: str | None) -> str:
    return os.path.abspath(path) if path else ""


def _active_adapter_name(model) -> str:
    if hasattr(model, "active_adapters"):
        ads = model.active_adapters
        if ads:
            return str(ads[0] if isinstance(ads, (list, tuple)) else ads)
    if hasattr(model, "active_adapter"):
        a = model.active_adapter
        if isinstance(a, (list, tuple)):
            return str(a[0]) if a else "default"
        return str(a or "default")
    return "default"


def _peft_config_names(model) -> set[str]:
    cfg = getattr(model, "peft_config", None) or {}
    try:
        return {str(k) for k in cfg.keys()}
    except Exception:
        return set()


def _needs_adapter_swap(view_family: str, live_adapter_path: str) -> bool:
    view = normalize_view_family(view_family)
    if view == "live":
        return False
    if view == "base":
        return True
    path = env_adapter_path_for_family("ce" if view == "ce" else "saliency")
    if not path:
        raise ValueError(
            f"No adapter path for viewFamily={view}. "
            "Set EIF_ADAPTER_PATH_CE / EIF_ADAPTER_PATH_SALIENCY in eif_api.env."
        )
    return _abspath(path) != _abspath(live_adapter_path)


@contextmanager
def apply_view_family(
    model,
    view_family: str,
    *,
    live_adapter_path: str,
) -> Iterator[str]:
    """Temporarily run ``model`` as CE / saliency / base. Always restores."""
    view = normalize_view_family(view_family)
    if not _needs_adapter_swap(view, live_adapter_path):
        yield view
        return

    from src.unlearn_pair_probe import intervention_status
    if intervention_status().get("active"):
        raise ValueError("先 Recover Learn/Unlearn，再切换 Base/CE/Saliency 对比。")

    if not hasattr(model, "set_adapter") and view != "base":
        raise ValueError("当前权重不是 PEFT adapter，无法切换 CE/Saliency。")

    prev = _active_adapter_name(model)
    loaded = _peft_config_names(model)

    if view == "base":
        if not hasattr(model, "disable_adapter"):
            raise ValueError("当前模型不支持 disable_adapter（需要 PEFT）。")
        with model.disable_adapter():
            yield "base"
        return

    adapter_name = view  # "ce" | "saliency"
    path = env_adapter_path_for_family("ce" if view == "ce" else "saliency")
    if adapter_name not in loaded:
        print(f"[degrade] load_adapter name={adapter_name} path={path}", flush=True)
        try:
            model.load_adapter(path, adapter_name=adapter_name, is_trainable=False)
        except TypeError:
            model.load_adapter(path, adapter_name=adapter_name)
        try:
            ensure_peft_lora_dtype(model, torch.bfloat16)
        except Exception:
            pass
        for n, p in model.named_parameters():
            if "lora_" in n and n.rsplit(".", 1)[-1] == adapter_name:
                p.requires_grad_(False)

    model.set_adapter(adapter_name)
    try:
        yield view
    finally:
        try:
            model.set_adapter(prev or "default")
        except Exception as exc:
            print(f"[degrade] WARN restore adapter {prev!r} failed: {exc}", flush=True)


def _forward_last_logits(model, prefix: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
    device = prefix.device
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
    del outputs
    return logits


def top_rows_from_logits(
    logits: torch.Tensor,
    tokenizer,
    *,
    actual_id: int,
    top_k: int,
) -> tuple[list[dict[str, Any]], float]:
    probs = F.softmax(logits, dim=-1)
    k = max(1, min(int(top_k), int(probs.numel())))
    values, indices = torch.topk(probs, k=k)
    actual_prob = float(probs[int(actual_id)].item())
    rows = []
    for p, tid in zip(values.tolist(), indices.tolist()):
        tid_i = int(tid)
        rows.append({
            "token": tokenizer.decode([tid_i]),
            "tokenId": tid_i,
            "prob": float(p),
            "isActual": tid_i == int(actual_id),
        })
    return rows, actual_prob


def _side_stats(
    logits: torch.Tensor,
    tokenizer,
    gained_id: int,
    lost_id: int,
) -> dict[str, Any]:
    probs = F.softmax(logits, dim=-1)
    gid, lid = int(gained_id), int(lost_id)
    lg = float(logits[gid].item())
    ll = float(logits[lid].item())
    pg = float(probs[gid].item())
    pl = float(probs[lid].item())
    argmax_id = int(logits.argmax().item())
    return {
        "argmaxId": argmax_id,
        "argmaxToken": tokenizer.decode([argmax_id]),
        "argmaxProb": float(probs[argmax_id].item()),
        "logitGained": lg,
        "logitLost": ll,
        "logitMargin": lg - ll,
        "pGained": pg,
        "pLost": pl,
        "nllLost": float(-math.log(max(pl, 1e-12))),
    }


def build_flip_stats(
    tokenizer,
    logits_live: torch.Tensor,
    logits_view: torch.Tensor,
    *,
    actual_id: int,
    actual_token: str,
    mode: str,
    view_family: str,
    live_family: str,
    gained_token_id: int | None = None,
    lost_token_id: int | None = None,
) -> dict[str, Any]:
    """Contrast live vs compare-view at the same prefix.

    gained = live argmax (SFT 现在选的)
    lost   = gold token (gold mode) else compare-view argmax
    """
    live_argmax = int(logits_live.argmax().item())
    view_argmax = int(logits_view.argmax().item())
    gained_id = int(gained_token_id) if gained_token_id is not None else live_argmax
    if lost_token_id is not None:
        lost_id = int(lost_token_id)
    elif (mode or "").strip().lower() == "gold":
        lost_id = int(actual_id)
    elif view_argmax != gained_id:
        lost_id = view_argmax
    elif int(actual_id) != gained_id:
        lost_id = int(actual_id)
    else:
        # Same argmax on both sides; pick live's 2nd-best as the contrast token.
        top2 = torch.topk(logits_live, k=min(2, logits_live.numel())).indices.tolist()
        lost_id = int(top2[1]) if len(top2) > 1 else int(actual_id)

    live_s = _side_stats(logits_live, tokenizer, gained_id, lost_id)
    view_s = _side_stats(logits_view, tokenizer, gained_id, lost_id)
    flipped = live_s["argmaxId"] != view_s["argmaxId"]
    return {
        "viewFamily": normalize_view_family(view_family),
        "liveFamily": live_family,
        "gainedToken": tokenizer.decode([gained_id]),
        "gainedTokenId": gained_id,
        "lostToken": tokenizer.decode([lost_id]),
        "lostTokenId": lost_id,
        "actualToken": actual_token,
        "actualTokenId": int(actual_id),
        "flipped": bool(flipped),
        "live": live_s,
        "compare": view_s,
        "deltaMargin": float(live_s["logitMargin"] - view_s["logitMargin"]),
        "deltaNllLost": float(live_s["nllLost"] - view_s["nllLost"]),
    }


def _live_adapter_filter(param_filter_fn, adapter_name: str):
    known = {"default", "ce", "saliency"}

    def _f(name, param):
        if param_filter_fn is not None and not param_filter_fn(name, param):
            return False
        if "lora_" not in name:
            return False
        last = name.rsplit(".", 1)[-1]
        if last in known:
            return last == adapter_name
        return True

    return _f


def _flat_logit_diff_grad(
    model,
    prefix: torch.Tensor,
    attn: torch.Tensor,
    gained_id: int,
    lost_id: int,
    param_filter_fn,
    device,
) -> torch.Tensor:
    """Flat ∇_θ (logit_gained − logit_lost) on filtered live-adapter LoRA."""
    target_params = [
        p for n, p in model.named_parameters()
        if param_filter_fn is None or param_filter_fn(n, p)
    ]
    if not target_params:
        raise RuntimeError("degradation query: no LoRA params matched the live-adapter filter.")

    prepare_last_layer_grad_checkpointing(model)
    model.zero_grad(set_to_none=True)
    original_flags = [(p, p.requires_grad) for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    for p in target_params:
        p.requires_grad_(True)

    try:
        with torch.enable_grad():
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
            objective = logits[int(gained_id)] - logits[int(lost_id)]
            grads = torch.autograd.grad(
                objective,
                target_params,
                create_graph=False,
                retain_graph=False,
                allow_unused=True,
            )
        flat = torch.cat([
            (
                g.reshape(-1).detach().cpu().float()
                if g is not None
                else torch.zeros(p.numel(), dtype=torch.float32)
            )
            for g, p in zip(grads, target_params)
        ])
        return flat
    finally:
        for p, flag in original_flags:
            p.requires_grad_(flag)
        model.zero_grad(set_to_none=True)
        try:
            del outputs, logits, objective
        except Exception:
            pass


def _train_snippet(tokenizer, sample: dict[str, Any], *, max_chars: int = 180) -> dict[str, Any]:
    ids = [int(x) for x in (sample.get("input_ids") or [])]
    labels = [int(x) for x in (sample.get("labels") or [])]
    edges = sample.get("attention_edges") or []
    start = 0
    for i, lab in enumerate(labels):
        if lab != -100:
            start = i
            break
    window = ids[max(0, start - 24): start + 48]
    text = tokenizer.decode(window) if window else ""
    text = " ".join(text.split())
    if len(text) > max_chars:
        text = text[: max_chars - 1] + "…"
    comp_ids = [tid for tid, lab in zip(ids, labels) if lab != -100]
    comp = tokenizer.decode(comp_ids[:64]) if comp_ids else ""
    low = comp.lower()
    return {
        "taskId": str(sample.get("task_id") or sample.get("uid") or ""),
        "snippet": text,
        "nEdges": len(edges) if isinstance(edges, list) else 0,
        "mentionsNew": (".new" in low) or ("new(" in low),
        "mentionsWrap": (".wrap" in low) or ("wrap(" in low),
    }


def retrieve_degradation(
    report: dict[str, Any],
    *,
    mode: str,
    target_index: int,
    compare_family: str = "ce",
    gained_token_id: int | None = None,
    lost_token_id: int | None = None,
    top_trains: int | None = None,
    top_k: int = 10,
) -> dict[str, Any]:
    """Score train bank against ∇(logit_gained − logit_lost) on the live adapter."""
    from src.gold_live_attribution import (
        _completion_tokens_and_ids,
        _ensure_session,
        _env_int,
        infer_sample_id_from_report,
    )

    compare = normalize_view_family(compare_family)
    if compare == "live":
        raise ValueError("degradation retrieve needs compareFamily=ce|base|saliency, not live.")

    session = _ensure_session(report)
    model = session["model"]
    tokenizer = session["tokenizer"]
    device = session["device"]
    bank = session["bank"]
    live_path = str(session.get("model_path") or "")
    live_family = infer_report_family(
        str((report.get("experiment_meta") or {}).get("report_file") or ""),
        report,
    )
    n_trains = max(1, int(top_trains if top_trains is not None else _env_int("EIF_GOLD_TOP_TRAINS", 5)))

    tokens, ids, _prompt_len = _completion_tokens_and_ids(report, tokenizer, mode=mode)
    t = int(target_index)
    if not (0 < t < len(ids)):
        raise ValueError(f"target_index={t} out of range for degradation retrieve (len={len(ids)}).")

    actual_id = int(ids[t])
    prefix = torch.tensor([ids[:t]], dtype=torch.long, device=device)
    attn = torch.ones_like(prefix)

    was_training = bool(model.training)
    model.eval()
    if hasattr(model, "disable_input_require_grads"):
        try:
            model.disable_input_require_grads()
        except Exception:
            pass
    if hasattr(model, "gradient_checkpointing_disable"):
        try:
            model.gradient_checkpointing_disable()
        except Exception:
            pass

    try:
        with torch.no_grad():
            logits_live = _forward_last_logits(model, prefix, attn)
            with apply_view_family(model, compare, live_adapter_path=live_path):
                logits_view = _forward_last_logits(model, prefix, attn)

        flip = build_flip_stats(
            tokenizer,
            logits_live,
            logits_view,
            actual_id=actual_id,
            actual_token=tokens[t],
            mode=mode,
            view_family=compare,
            live_family=live_family,
            gained_token_id=gained_token_id,
            lost_token_id=lost_token_id,
        )
        live_rows, live_p = top_rows_from_logits(
            logits_live, tokenizer, actual_id=actual_id, top_k=top_k,
        )
        view_rows, view_p = top_rows_from_logits(
            logits_view, tokenizer, actual_id=actual_id, top_k=top_k,
        )

        live_adapter = _active_adapter_name(model)
        query_filter = _live_adapter_filter(session["param_filter"], live_adapter)
        flat = _flat_logit_diff_grad(
            model,
            prefix,
            attn,
            int(flip["gainedTokenId"]),
            int(flip["lostTokenId"]),
            query_filter,
            device,
        )
        probe_sketch = _project_flat_grad(flat, PRESCREEN_SKETCH_DIM, PRESCREEN_SKETCH_SEED)
        edge_scores = _score_prescreen_sketch_cache(probe_sketch, bank, device)
        related = nlargest(n_trains, edge_scores, key=lambda x: x[1])
        train_samples = session["train_samples"]
        trains = []
        for idx, score in related:
            i = int(idx)
            row: dict[str, Any] = {
                "trainSampleId": i,
                "probeCos": float(score),
            }
            if 0 <= i < len(train_samples):
                row.update(_train_snippet(tokenizer, train_samples[i]))
            trains.append(row)
        print(
            f"[degrade] retrieve {flip['gainedToken']!r} vs {flip['lostToken']!r} "
            f"compare={compare} top={[(r['trainSampleId'], round(r['probeCos'], 4)) for r in trains]}",
            flush=True,
        )
    finally:
        if hasattr(model, "enable_input_require_grads"):
            try:
                model.enable_input_require_grads()
            except Exception:
                pass
        if was_training:
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
        from src.unlearn_pair_probe import _release_cuda_memory
        _release_cuda_memory(model, reason="degradation_retrieve")

    return {
        "status": "success",
        "mode": (mode or "predict").strip().lower(),
        "targetIndex": t,
        "compareFamily": compare,
        "liveFamily": live_family,
        "query": "grad(logit_gained - logit_lost) vs train bank",
        "flip": flip,
        "liveTop": live_rows,
        "compareTop": view_rows,
        "liveActualProb": live_p,
        "compareActualProb": view_p,
        "relatedTrains": trains,
        "filterTag": session.get("filter_tag"),
        "sampleIdHint": infer_sample_id_from_report(report),
        "availableViews": list_compare_views(live_family),
    }
