"""One-step gradient check: does this annotation help the current test gold CE?

Saliency loss is **not** a sum over edges. Edges that share a query compete in
one softmax (and negatives are subsampled when ``neg_sample_k`` > 0), so an
edge score is the leave-one-out change in

    ∇L_test · ∇L_saliency

Predicted gold-CE change of one SGD step on the train objective is

    ΔL_test ≈ −η ∇L_test · ∇L_train

so a positive dot means the step would lower test gold CE.

    dot(∇test, ∇(CE + λ Sal)) > 0     CE+saliency helps
    dot(∇test, ∇Sal) > 0               that help beats CE-only
    dot_full − dot_without(e) < 0      edge e hurts; drop it
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import torch

from src.bank_loss import BankLossConfig, compute_bank_loss, load_bank_loss_config
from src.continue_train_eval import (
    _compact_batch,
    _device_of,
    _evict_cached_models,
    _pick_continue_attn_implementation,
    _render_eval_prompt,
    _resolve_path,
    resolve_continue_start_adapter,
)
from src.eif_adapter_env import base_model_path_from_env
from src.intervention_experiment import load_model_and_tokenizer


ProgressCb = Callable[[str, str], None]


def _noop(_stage: str, _message: str) -> None:
    return None


def _edge_src_dst(edge: Any) -> tuple[int, int] | None:
    if isinstance(edge, (list, tuple)) and len(edge) >= 2:
        try:
            return int(edge[0]), int(edge[1])
        except (TypeError, ValueError):
            return None
    if isinstance(edge, dict):
        try:
            return int(edge.get("src", edge.get("source"))), int(edge.get("dst", edge.get("target")))
        except (TypeError, ValueError):
            return None
    return None


def _row_edges(row: dict[str, Any]) -> list[Any]:
    edges = row.get("attention_edges")
    if edges is None:
        edges = row.get("edges")
    return list(edges) if isinstance(edges, list) else []


def _dedup_key(row: dict[str, Any], index: int) -> str:
    line = row.get("source_corpus_line")
    try:
        if line is not None and str(line).strip() != "":
            return f"line:{int(line)}"
    except (TypeError, ValueError):
        pass
    uid = str(row.get("duplicate_of") or row.get("uid") or row.get("task_id") or "")
    uid = uid.split("::dup")[0].split("::v")[0]
    if uid:
        return f"uid:{uid}"
    ids = row.get("input_ids")
    if isinstance(ids, list) and ids:
        return "ids:" + str(len(ids)) + ":" + str(ids[:8])
    return f"row:{index}"


def _load_continue_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            if isinstance(obj, dict) and isinstance(obj.get("input_ids"), list):
                rows.append(obj)
    return rows


def _latest_group(rows: list[dict[str, Any]]) -> tuple[str, list[int], dict[str, Any]]:
    if not rows:
        raise ValueError("continue JSONL has no compact rows")
    key = _dedup_key(rows[-1], len(rows) - 1)
    idxs = [i for i, row in enumerate(rows) if _dedup_key(row, i) == key]
    return key, idxs, rows[idxs[-1]]


def _lora_params(model) -> list[torch.nn.Parameter]:
    params = [p for n, p in model.named_parameters() if "lora_" in n]
    if not params:
        params = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        p.requires_grad_(True)
    return params


def _flat_grad(loss: torch.Tensor, params: list[torch.nn.Parameter]) -> torch.Tensor:
    grads = torch.autograd.grad(loss, params, allow_unused=True, retain_graph=False)
    parts: list[torch.Tensor] = []
    for grad, param in zip(grads, params):
        if grad is None:
            parts.append(torch.zeros(param.numel(), dtype=torch.float32))
        else:
            parts.append(grad.detach().float().reshape(-1).cpu())
    return torch.cat(parts) if parts else torch.zeros(1)


def _dot(a: torch.Tensor, b: torch.Tensor) -> float:
    n = min(int(a.numel()), int(b.numel()))
    if n <= 0:
        return 0.0
    return float(torch.dot(a[:n], b[:n]).item())


def _gold_ce_loss(model, tokenizer, prompt: str, label: str) -> torch.Tensor:
    prefix = _render_eval_prompt(tokenizer, prompt or "")
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    gold_ids = tokenizer(label or "", add_special_tokens=False)["input_ids"]
    if not isinstance(prefix_ids, list):
        prefix_ids = [int(x) for x in prefix_ids]
    if not isinstance(gold_ids, list):
        gold_ids = [int(x) for x in gold_ids]
    if not gold_ids:
        raise ValueError("current test gold is empty after tokenization")
    device = _device_of(model)
    input_ids = torch.tensor([prefix_ids + gold_ids], dtype=torch.long, device=device)
    labels = torch.tensor(
        [[-100] * len(prefix_ids) + gold_ids],
        dtype=torch.long,
        device=device,
    )
    out = model(input_ids=input_ids, labels=labels, use_cache=False)
    loss = out.loss
    if loss is None:
        raise RuntimeError("gold CE forward returned no loss")
    return loss.mean() if loss.dim() > 0 else loss


def _bank_cfg_for_probe(adapter_path: str, lam: float) -> BankLossConfig:
    cfg = load_bank_loss_config(adapter_path, Path(adapter_path).name, loss_mode_override="ce_saliency")
    # Leave-one-out must be deterministic. Random negative subsampling
    # makes "remove this edge" incomparable across passes.
    cfg.neg_sample_k = 0
    cfg.loss_mode = "ce_saliency"
    if lam > 0:
        cfg.saliency_lambda = float(lam)
    return cfg


def _continue_lambda(adapter_cfg: BankLossConfig) -> float:
    raw = (os.environ.get("EIF_CONTINUE_SALIENCY_LAMBDA") or "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return float(adapter_cfg.saliency_lambda)


def _forward_loss(
    model,
    batch: dict[str, torch.Tensor],
    edges: list[Any] | None,
    cfg: BankLossConfig,
    *,
    mode: str,
) -> torch.Tensor:
    device = _device_of(model)
    saliency_only = mode == "saliency"
    use_cfg = cfg
    if mode == "ce":
        use_cfg = deepcopy(cfg)
        use_cfg.loss_mode = "ce_only"
    loss, _kind = compute_bank_loss(
        model,
        batch,
        device=device,
        cfg=use_cfg,
        edges=edges if mode != "ce" else None,
        saliency_only=saliency_only,
    )
    if not torch.is_tensor(loss):
        raise RuntimeError(f"{mode} loss was not a tensor")
    return loss.mean() if loss.dim() > 0 else loss


def _clear(model) -> None:
    model.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _score_pair(
    model,
    tokenizer,
    params: list[torch.nn.Parameter],
    *,
    prompt: str,
    label: str,
    train_row: dict[str, Any],
    cfg: BankLossConfig,
    progress: ProgressCb,
    loo: bool,
) -> dict[str, Any]:
    device = _device_of(model)
    edges = _row_edges(train_row)
    batch = _compact_batch(train_row, device)
    model.eval()

    progress("test_grad", "测试 gold CE 梯度…")
    _clear(model)
    g_test = _flat_grad(_gold_ce_loss(model, tokenizer, prompt, label), params)
    _clear(model)

    progress("train_ce", "训练 CE 梯度…")
    g_ce = _flat_grad(_forward_loss(model, batch, edges, cfg, mode="ce"), params)
    _clear(model)

    if not edges:
        dot_ce = _dot(g_test, g_ce)
        return {
            "n_edges": 0,
            "dot_ce": round(dot_ce, 6),
            "dot_saliency": None,
            "dot_ce_saliency": round(dot_ce, 6),
            "both_positive": False,
            "dropped": [],
            "kept_edges": 0,
            "message": "这条样本没有 attention_edges，只能看纯 CE 对齐",
        }

    progress("train_sal", f"训练 saliency 梯度（{len(edges)} 边，neg_sample_k=0）…")
    g_sal = _flat_grad(_forward_loss(model, batch, edges, cfg, mode="saliency"), params)
    _clear(model)

    lam = float(cfg.saliency_lambda)
    dot_ce = _dot(g_test, g_ce)
    dot_sal = _dot(g_test, g_sal)
    dot_both = dot_ce + lam * dot_sal

    if not loo:
        return {
            "n_edges": len(edges),
            "lambda": lam,
            "dot_ce": round(dot_ce, 6),
            "dot_saliency": round(dot_sal, 6),
            "dot_ce_saliency": round(dot_both, 6),
            "both_positive": bool(dot_both > 0 and dot_sal > 0),
            "dropped": [],
            "kept_edges": len(edges),
        }

    dropped: list[dict[str, Any]] = []
    keep_idx: list[int] = []
    for i, edge in enumerate(edges):
        progress("loo", f"去掉第 {i + 1}/{len(edges)} 条边再算点积…")
        rest = [e for j, e in enumerate(edges) if j != i]
        if not rest:
            # Single edge: without it saliency grad is 0, so its contribution is dot_sal.
            marginal = dot_sal
            dot_wo = 0.0
        else:
            g_wo = _flat_grad(_forward_loss(model, batch, rest, cfg, mode="saliency"), params)
            _clear(model)
            dot_wo = _dot(g_test, g_wo)
            marginal = dot_sal - dot_wo
        pair = _edge_src_dst(edge)
        rec = {
            "index": i,
            "src": None if pair is None else pair[0],
            "dst": None if pair is None else pair[1],
            "marginal": round(marginal, 6),
            "dot_without": round(dot_wo, 6),
        }
        if marginal < 0:
            rec["action"] = "drop"
            dropped.append(rec)
        else:
            rec["action"] = "keep"
            keep_idx.append(i)

    return {
        "n_edges": len(edges),
        "lambda": lam,
        "dot_ce": round(dot_ce, 6),
        "dot_saliency": round(dot_sal, 6),
        "dot_ce_saliency": round(dot_both, 6),
        "both_positive": bool(dot_both > 0 and dot_sal > 0),
        "dropped": dropped,
        "kept_edges": len(keep_idx),
    }


def _drop_edge(edge: Any, banned: set[tuple[int, int]]) -> bool:
    pair = _edge_src_dst(edge)
    return pair is not None and pair in banned


def _filter_edges(edges: Any, banned: set[tuple[int, int]]) -> list[Any]:
    if not isinstance(edges, list):
        return []
    return [e for e in edges if not _drop_edge(e, banned)]


def _apply_drops(path: Path, row_indexes: set[int], banned: set[tuple[int, int]]) -> int:
    if not banned:
        return 0
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    compact_i = 0
    changed = 0
    out_lines: list[str] = []
    for raw in raw_lines:
        if not raw.strip():
            continue
        obj = json.loads(raw)
        if isinstance(obj, dict) and isinstance(obj.get("input_ids"), list):
            if compact_i in row_indexes:
                obj["attention_edges"] = _filter_edges(obj.get("attention_edges"), banned)
                if isinstance(obj.get("viz_attention_edges"), list):
                    obj["viz_attention_edges"] = _filter_edges(obj["viz_attention_edges"], banned)
                if isinstance(obj.get("annotations"), list):
                    kept_ann = []
                    for ann in obj["annotations"]:
                        if not isinstance(ann, dict):
                            kept_ann.append(ann)
                            continue
                        try:
                            pair = (int(ann.get("token_i_idx")), int(ann.get("token_j_idx")))
                        except (TypeError, ValueError):
                            kept_ann.append(ann)
                            continue
                        if pair not in banned and (pair[1], pair[0]) not in banned:
                            kept_ann.append(ann)
                    obj["annotations"] = kept_ann
                changed += 1
            compact_i += 1
        out_lines.append(json.dumps(obj, ensure_ascii=False))
    path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
    return changed


def _undo_path(train_path: Path) -> Path:
    return Path(str(train_path) + ".grad_align_undo.json")


def _save_undo(train_path: Path, key: str, idxs: list[int], rows: list[dict[str, Any]]) -> None:
    saved = []
    for i in idxs:
        row = rows[i]
        saved.append({
            "index": i,
            "attention_edges": row.get("attention_edges"),
            "viz_attention_edges": row.get("viz_attention_edges"),
            "annotations": row.get("annotations"),
        })
    _undo_path(train_path).write_text(
        json.dumps({"trainData": str(train_path), "group": key, "rows": saved}, ensure_ascii=False),
        encoding="utf-8",
    )


def undo_grad_align(train_data: str | None = None) -> dict[str, Any]:
    """Restore attention edges removed by the latest 梯度筛边."""
    train_path = _resolve_path(
        train_data or (os.environ.get("ANNOTATION_CONTINUE_TRAIN_DATA") or "")
    )
    if not train_path:
        raise FileNotFoundError("continue JSONL missing")
    path = Path(train_path)
    undo_file = _undo_path(path)
    if not undo_file.is_file():
        raise FileNotFoundError("没有可撤回的筛边记录")
    payload = json.loads(undo_file.read_text(encoding="utf-8"))
    by_index = {
        int(item["index"]): item
        for item in (payload.get("rows") or [])
        if isinstance(item, dict) and item.get("index") is not None
    }
    if not by_index:
        raise FileNotFoundError("筛边记录是空的")
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    compact_i = 0
    restored = 0
    out_lines: list[str] = []
    for raw in raw_lines:
        if not raw.strip():
            continue
        obj = json.loads(raw)
        if isinstance(obj, dict) and isinstance(obj.get("input_ids"), list):
            snap = by_index.get(compact_i)
            if snap is not None:
                for field in ("attention_edges", "viz_attention_edges", "annotations"):
                    if field in snap:
                        obj[field] = snap[field]
                restored += 1
            compact_i += 1
        out_lines.append(json.dumps(obj, ensure_ascii=False))
    path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
    undo_file.unlink(missing_ok=True)
    return {
        "status": "success",
        "restored_rows": restored,
        "group": payload.get("group"),
        "trainData": str(path),
        "summary": f"已撤回筛边，恢复 {restored} 行（{payload.get('group') or ''}）",
    }


def run_grad_align(
    *,
    prompt: str,
    label: str,
    train_data: str | None = None,
    adapter_path: str | None = None,
    report_family: str | None = None,
    report_file_name: str | None = None,
    mode: str = "verify",
    progress_cb: ProgressCb | None = None,
) -> dict[str, Any]:
    progress = progress_cb or _noop
    if not (prompt or "").strip() or not (label or "").strip():
        raise ValueError("current test prompt and label are required")

    adapter, family = resolve_continue_start_adapter(
        adapter_path=adapter_path,
        report_family=report_family,
        report_file_name=report_file_name,
    )
    if not adapter:
        raise ValueError("no CE/saliency adapter; set EIF_ADAPTER_PATH_CE")
    base = base_model_path_from_env()
    train_path = _resolve_path(
        train_data or (os.environ.get("ANNOTATION_CONTINUE_TRAIN_DATA") or "")
    )
    if not train_path or not Path(train_path).is_file():
        raise FileNotFoundError(
            "continue JSONL missing. Set ANNOTATION_CONTINUE_TRAIN_DATA or pass trainData."
        )

    rows = _load_continue_rows(Path(train_path))
    key, idxs, sample = _latest_group(rows)
    filtering = (mode or "verify").strip().lower() == "filter"
    progress(
        "load",
        f"加载 adapter {family}，{'筛边' if filtering else '验证'}最近一条标注 {key}（{len(idxs)} 份副本）…",
    )

    probe_cfg = _bank_cfg_for_probe(adapter, lam=0.0)
    probe_cfg.saliency_lambda = _continue_lambda(probe_cfg)

    _evict_cached_models()
    attn_impl = _pick_continue_attn_implementation()
    try:
        model, tokenizer = load_model_and_tokenizer(
            model_path=adapter,
            base_model_path=base,
            attn_implementation=attn_impl,
        )
    except Exception:
        if attn_impl == "sdpa":
            raise
        model, tokenizer = load_model_and_tokenizer(
            model_path=adapter,
            base_model_path=base,
            attn_implementation="sdpa",
        )
    try:
        params = _lora_params(model)
        scored = _score_pair(
            model,
            tokenizer,
            params,
            prompt=prompt,
            label=label,
            train_row=sample,
            cfg=probe_cfg,
            progress=progress,
            loo=filtering,
        )
    finally:
        del model
        _evict_cached_models()

    banned = {
        (int(e["src"]), int(e["dst"]))
        for e in scored.get("dropped") or []
        if e.get("src") is not None and e.get("dst") is not None
    }
    written = 0
    if filtering and banned:
        progress("write", f"从续训小集删掉 {len(banned)} 条有害边…")
        _save_undo(Path(train_path), key, idxs, rows)
        written = _apply_drops(Path(train_path), set(idxs), banned)

    both = bool(scored.get("both_positive"))
    if filtering:
        summary = (
            f"{key} 删边 {len(banned)}，写回 {written} 行。"
            "请再点验证看两个点积。"
        )
    else:
        summary = (
            f"{key} ∇(CE+λSal)={scored.get('dot_ce_saliency')} "
            f"∇Sal={scored.get('dot_saliency')} "
            + ("都 > 0" if both else "未都 > 0")
        )
    print(f"[grad-align] {summary}", flush=True)
    return {
        "status": "success",
        "mode": "filter" if filtering else "verify",
        "summary": summary,
        "group": key,
        "copies": len(idxs),
        "trainData": train_path,
        "adapter": adapter,
        "applied_rows": written,
        "can_undo": bool(filtering and written),
        "neg_sample_k": 0,
        **scored,
    }
