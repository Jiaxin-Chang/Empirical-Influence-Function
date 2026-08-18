"""Decision-flip (degradation) attribution: compare adapters at one prefix.

Target quantity::

    f = logit_gained − logit_lost
    Δ = f_live − f_compare

Pair-level score (descent) for train edge e::

    contrib(e) = −cos(∇f, ∇L_sal(e))

Positive contrib = this saliency edge pushes the flip (New over Wrap).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from contextlib import contextmanager
from pathlib import Path
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
    PRESCREEN_SKETCH_SEED,
    SEQUENCE_LENGTH_LIMIT,
    _compute_bank_flat_grad_filtered,
    _project_flat_grad,
    ensure_peft_lora_dtype,
    get_context_window,
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


def _parse_train_edges(edges) -> list[tuple[int, int, float, str]]:
    out: list[tuple[int, int, float, str]] = []
    for e in edges or []:
        try:
            if isinstance(e, (list, tuple)) and len(e) >= 2:
                src, dst = int(e[0]), int(e[1])
                w = float(e[2]) if len(e) >= 3 else 1.0
                subtype = ""
            else:
                src = int(e.get("src", e.get("source", -1)))
                dst = int(e.get("dst", e.get("target", -1)))
                try:
                    w = float(e.get("weight", 1.0))
                except (TypeError, ValueError):
                    w = 1.0
                subtype = str(e.get("subtype") or "")
        except (TypeError, ValueError, AttributeError):
            continue
        if src < 0 or dst < 0 or src == dst:
            continue
        out.append((src, dst, max(1.0, w), subtype))
    return out


def _env_flag(name: str, default: bool = True) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw not in ("0", "false", "no", "off")


def _edge_grad_cache_sketch_dim() -> int:
    """0 = store full last-layer LoRA grads (large). Default 8192 CountSketch."""
    raw = (os.environ.get("EIF_DEGRADE_EDGE_CACHE_SKETCH") or "8192").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 8192


def _edge_grad_cache_root() -> Path:
    raw = (os.environ.get("EIF_DEGRADE_EDGE_CACHE_DIR") or "").strip()
    if raw:
        return Path(raw)
    return Path(__file__).resolve().parents[1] / ".cache" / "degrade_sal_edge_grads"


def _edge_cache_key(src: int, dst: int) -> str:
    return f"{int(src)}_{int(dst)}"


def _edge_grad_cache_dir(
    *,
    live_path: str,
    train_path: str,
    bank_cfg,
    filter_tag: str,
    sketch_dim: int,
    n_trains: int,
    n_annot: int,
) -> Path | None:
    if not _env_flag("EIF_DEGRADE_EDGE_CACHE", True):
        return None
    train_p = Path(train_path) if train_path else None
    stamp = ""
    if train_p is not None and train_p.is_file():
        st = train_p.stat()
        stamp = f"{st.st_mtime_ns}:{st.st_size}"
    blob = {
        "adapter": _abspath(live_path),
        "train": _abspath(train_path),
        "trainStamp": stamp,
        "bank": getattr(bank_cfg, "cache_tag", ""),
        "lossType": getattr(bank_cfg, "saliency_loss_type", ""),
        "layer": int(getattr(bank_cfg, "saliency_layer", -1) or -1),
        "negK": int(getattr(bank_cfg, "neg_sample_k", 0) or 0),
        "margin": float(getattr(bank_cfg, "margin_plus", 0) or 0),
        "filter": str(filter_tag or ""),
        "sketch": int(sketch_dim),
        "nTrains": int(n_trains),
        "nAnnot": int(n_annot),
    }
    tag = hashlib.sha1(json.dumps(blob, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    d = _edge_grad_cache_root() / tag
    d.mkdir(parents=True, exist_ok=True)
    meta = d / "meta.json"
    if not meta.is_file():
        meta.write_text(json.dumps(blob, indent=2, ensure_ascii=False), encoding="utf-8")
    return d


def _load_train_edge_cache(cache_dir: Path | None, train_idx: int) -> dict[str, torch.Tensor]:
    if cache_dir is None:
        return {}
    path = cache_dir / f"train_{int(train_idx)}.pt"
    if not path.is_file():
        return {}
    try:
        blob = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(str(path), map_location="cpu")
    except Exception as exc:
        print(f"[degrade] cache load failed {path.name}: {exc}", flush=True)
        return {}
    edges = blob.get("edges") if isinstance(blob, dict) else None
    if not isinstance(edges, dict):
        return {}
    out: dict[str, torch.Tensor] = {}
    for k, v in edges.items():
        if torch.is_tensor(v):
            out[str(k)] = v.detach().cpu()
    return out


def _save_train_edge_cache(
    cache_dir: Path | None,
    train_idx: int,
    edges: dict[str, torch.Tensor],
    *,
    sketch_dim: int,
    sketch_seed: int,
) -> None:
    if cache_dir is None or not edges:
        return
    path = cache_dir / f"train_{int(train_idx)}.pt"
    tmp = path.with_suffix(".tmp")
    packed = {
        str(k): (v.detach().cpu().half() if v.dtype != torch.float16 else v.detach().cpu())
        for k, v in edges.items()
        if torch.is_tensor(v)
    }
    blob = {
        "trainIdx": int(train_idx),
        "sketchDim": int(sketch_dim),
        "sketchSeed": int(sketch_seed),
        "n": len(packed),
        "edges": packed,
    }
    torch.save(blob, str(tmp))
    tmp.replace(path)


def _seed_edge_grad(train_idx: int, src: int, dst: int) -> None:
    seed = (int(train_idx) * 1_000_003 + int(src) * 97 + int(dst) * 13 + 42) % (2**31)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_match_vec(g: torch.Tensor, sketch_dim: int, sketch_seed: int) -> torch.Tensor:
    if int(sketch_dim) > 0:
        return _project_flat_grad(g, int(sketch_dim), int(sketch_seed))
    return F.normalize(g.reshape(-1).float(), dim=0, eps=1e-12)


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
    progress_cb=None,
) -> dict[str, Any]:
    """Rank train saliency edges by contrib = −cos(∇f, ∇L_sal(edge))."""
    from src.gold_live_attribution import (
        _completion_tokens_and_ids,
        _ensure_session,
        infer_sample_id_from_report,
    )
    from src.NIF import build_single_sample_dataset

    compare = normalize_view_family(compare_family)
    if compare == "live":
        raise ValueError("degradation retrieve needs compareFamily=ce|base|saliency, not live.")

    session = _ensure_session(report)
    model = session["model"]
    tokenizer = session["tokenizer"]
    device = session["device"]
    live_path = str(session.get("model_path") or "")
    live_family = infer_report_family(
        str((report.get("experiment_meta") or {}).get("report_file") or ""),
        report,
    )
    n_keep = max(1, int(top_trains if top_trains is not None else 20))
    bank_cfg = session.get("bank_cfg")
    if bank_cfg is None:
        from src.bank_loss import load_bank_loss_config
        bank_cfg = load_bank_loss_config(live_path, live_family or "")
    collator = session["collator"]
    train_samples = session["train_samples"]
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])

    def _prog(**info):
        if progress_cb is None:
            return
        try:
            progress_cb(info)
        except Exception:
            pass

    n_trains = len(train_samples)
    n_annot = sum(
        len(_parse_train_edges(s.get("attention_edges") or s.get("edges")))
        for s in train_samples
    )
    print(
        f"[degrade] train_path={session.get('train_path')} "
        f"samples={n_trains} parsed_edges={n_annot}",
        flush=True,
    )
    sketch_dim = _edge_grad_cache_sketch_dim()
    sketch_seed = int(PRESCREEN_SKETCH_SEED)
    cache_dir = _edge_grad_cache_dir(
        live_path=live_path,
        train_path=str(session.get("train_path") or ""),
        bank_cfg=bank_cfg,
        filter_tag=str(session.get("filter_tag") or ""),
        sketch_dim=sketch_dim,
        n_trains=n_trains,
        n_annot=n_annot,
    )
    if cache_dir is not None:
        print(
            f"[degrade] edge-grad cache={cache_dir} "
            f"sketch={sketch_dim or 'full'} seed={sketch_seed} "
            f"(set EIF_DEGRADE_EDGE_CACHE=0 to disable)",
            flush=True,
        )
    _prog(
        stage="query",
        done=0,
        total=n_annot,
        nTrains=n_trains,
        message=(
            f"共 {n_annot} 条标注边，正在算 ∇f"
            + (f"；缓存 {cache_dir.name}" if cache_dir is not None else "")
            + "…"
        ),
    )

    tokens, ids, prompt_len = _completion_tokens_and_ids(report, tokenizer, mode=mode)
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
        g_f = _flat_logit_diff_grad(
            model,
            prefix,
            attn,
            int(flip["gainedTokenId"]),
            int(flip["lostTokenId"]),
            query_filter,
            device,
        )
        q = _to_match_vec(g_f, sketch_dim, sketch_seed)

        prepare_last_layer_grad_checkpointing(model)
        scored: list[dict[str, Any]] = []
        details: dict[str, Any] = {}
        n_edges_total = 0
        done = 0
        cache_hits = 0
        cache_misses = 0
        _prog(
            stage="edges",
            done=0,
            total=n_annot,
            nTrains=n_trains,
            message=f"0 / {n_annot} 标注边",
        )
        for train_idx, sample in enumerate(train_samples):
            edge_list = _parse_train_edges(sample.get("attention_edges") or sample.get("edges"))
            if not edge_list:
                continue
            tr_ds = build_single_sample_dataset(sample)
            tr_batch = collator([tr_ds[0]])
            tr_batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in tr_batch.items()}
            seq_len = int(tr_batch["input_ids"].size(1))
            if seq_len > SEQUENCE_LENGTH_LIMIT:
                print(
                    f"[degrade] skip train#{train_idx} seq={seq_len} > {SEQUENCE_LENGTH_LIMIT}",
                    flush=True,
                )
                done += len(edge_list)
                _prog(
                    stage="edges",
                    done=done,
                    total=n_annot,
                    nTrains=n_trains,
                    trainIdx=train_idx,
                    message=f"{done} / {n_annot}  · skip train#{train_idx} (seq too long)",
                )
                continue
            ids_1d = tr_batch["input_ids"][0]
            full_token_ids = [int(x) for x in ids_1d.tolist()]
            full_tokens = [tokenizer.decode([i]) for i in full_token_ids]
            labels_1d = tr_batch["labels"][0].tolist()
            answer_start = 0
            for li, lab in enumerate(labels_1d):
                if int(lab) != -100:
                    answer_start = li
                    break
            details[str(train_idx)] = {
                "full_tokens": full_tokens,
                "full_token_ids": full_token_ids,
                "answer_start_index": answer_start,
                "coarse_cos_sim": 0.0,
                "saliencies_by_token": {},
            }
            train_cache = _load_train_edge_cache(cache_dir, train_idx)
            train_dirty = 0
            for src, dst, weight, subtype in edge_list:
                src_tok = full_tokens[src] if 0 <= src < len(full_tokens) else ""
                dst_tok = full_tokens[dst] if 0 <= dst < len(full_tokens) else ""
                show = min(done + 1, n_annot) if n_annot else 0
                ck = _edge_cache_key(src, dst)
                hit = ck in train_cache
                _prog(
                    stage="edges",
                    done=show,
                    total=n_annot,
                    nTrains=n_trains,
                    trainIdx=train_idx,
                    src=src,
                    dst=dst,
                    srcTok=src_tok,
                    dstTok=dst_tok,
                    cacheHits=cache_hits,
                    cacheMisses=cache_misses,
                    message=(
                        f"{show} / {n_annot}  · train#{train_idx} "
                        f"{(src_tok or '').strip() or '·'}→{(dst_tok or '').strip() or '·'}"
                        f"{'  cache' if hit else '  compute'}"
                        f"  (hit {cache_hits} / miss {cache_misses})"
                    ),
                )
                try:
                    if not (0 <= src < seq_len and 0 <= dst < seq_len):
                        continue
                    n_edges_total += 1
                    e = None
                    if hit:
                        e = F.normalize(
                            train_cache[ck].reshape(-1).float(), dim=0, eps=1e-12,
                        )
                        cache_hits += 1
                    else:
                        _seed_edge_grad(train_idx, src, dst)
                        g_e = _compute_bank_flat_grad_filtered(
                            model,
                            tr_batch,
                            query_filter,
                            device,
                            cfg=bank_cfg,
                            edges=[{"src": src, "dst": dst, "weight": weight, "subtype": subtype}],
                            special_ids=special_ids,
                            saliency_only=True,
                        )
                        if g_e is None or not torch.isfinite(g_e).all():
                            continue
                        e = _to_match_vec(g_e, sketch_dim, sketch_seed)
                        train_cache[ck] = e.detach().cpu().half()
                        cache_misses += 1
                        train_dirty += 1
                        if train_dirty >= 8:
                            _save_train_edge_cache(
                                cache_dir, train_idx, train_cache,
                                sketch_dim=sketch_dim, sketch_seed=sketch_seed,
                            )
                            train_dirty = 0
                    if e is None or e.numel() != q.numel():
                        if e is not None:
                            print(
                                f"[degrade] skip train#{train_idx} {src}->{dst}: "
                                f"grad dim {e.numel()} vs query {q.numel()}",
                                flush=True,
                            )
                        continue
                    cos = float((q * e).sum().item())
                    contrib = -cos
                    src_ctx = get_context_window(tokenizer, ids_1d, src)
                    dst_ctx = get_context_window(tokenizer, ids_1d, dst)
                    scored.append({
                        "id": f"degrade_tr{train_idx}_s{src}_t{dst}",
                        "cos_sim": contrib,
                        "coarse_cos_sim": cos,
                        "score": contrib,
                        "retrieval": "degrade_sal_edge",
                        "train_sample_id": int(train_idx),
                        "test_correlation": {
                            "source_token": flip["gainedToken"],
                            "source_token_index": int(t),
                            "target_token": flip["lostToken"],
                            "target_token_index": int(t),
                            "saliency_score": float(flip.get("deltaMargin") or 0.0),
                        },
                        "train_correlation": {
                            "source_token": src_tok or tokenizer.decode([int(ids_1d[src])]),
                            "source_token_index": int(src),
                            "target_token": dst_tok or tokenizer.decode([int(ids_1d[dst])]),
                            "target_token_index": int(dst),
                            "saliency_score": contrib,
                            "response_token_offset": max(0, int(dst) - int(answer_start)),
                        },
                        "train_context": {
                            "source_context": src_ctx,
                            "target_context": dst_ctx,
                        },
                        "annotation": subtype or None,
                        "contrib": contrib,
                        "rawCos": cos,
                        "subtype": subtype,
                        "weight": weight,
                    })
                finally:
                    done += 1
            if train_dirty:
                _save_train_edge_cache(
                    cache_dir, train_idx, train_cache,
                    sketch_dim=sketch_dim, sketch_seed=sketch_seed,
                )
            from src.unlearn_pair_probe import _release_cuda_memory
            _release_cuda_memory(model, reason=f"degrade_train{train_idx}")

        _prog(
            stage="done",
            done=n_annot,
            total=n_annot,
            nTrains=n_trains,
            cacheHits=cache_hits,
            cacheMisses=cache_misses,
            message=(
                f"完成：有效 {n_edges_total} / {n_annot} 条"
                f"（cache hit {cache_hits} / miss {cache_misses}）"
            ),
        )

        scored.sort(key=lambda p: p["cos_sim"], reverse=True)
        top_pairs = scored[:n_keep]
        used = {str(p["train_sample_id"]) for p in top_pairs}
        details = {k: v for k, v in details.items() if k in used}
        for p in top_pairs:
            key = str(p["train_sample_id"])
            if key in details:
                details[key]["coarse_cos_sim"] = max(
                    float(details[key].get("coarse_cos_sim") or 0),
                    float(p["cos_sim"]),
                )
        print(
            f"[degrade] scored {n_edges_total} saliency edges; "
            f"cache hit={cache_hits} miss={cache_misses}; "
            f"top contrib={[(p['train_sample_id'], p['train_correlation']['source_token'], p['train_correlation']['target_token'], round(p['cos_sim'], 4)) for p in top_pairs[:8]]}",
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
        "promptLen": prompt_len,
        "compareFamily": compare,
        "liveFamily": live_family,
        "query": "contrib=-cos(grad(logit_gained-logit_lost), grad L_sal(edge))",
        "flip": flip,
        "liveTop": live_rows,
        "compareTop": view_rows,
        "liveActualProb": live_p,
        "compareActualProb": view_p,
        "correlationPairs": top_pairs,
        "trainSampleDetails": details,
        "nEdgesScored": n_edges_total,
        "cacheHits": cache_hits,
        "cacheMisses": cache_misses,
        "cacheDir": str(cache_dir) if cache_dir is not None else None,
        "cacheSketchDim": int(sketch_dim),
        "filterTag": session.get("filter_tag"),
        "sampleIdHint": infer_sample_id_from_report(report),
        "availableViews": list_compare_views(live_family),
    }
