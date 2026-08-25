"""Build a synthetic all-tokens report from a raw eval JSONL row (prompt/label/predict)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.eif_adapter_env import base_model_path_from_env, env_adapter_path_for_family

RAW_DIR_NAME = "raw"


def resolve_raw_jsonl(corr_results_dir: Path, file_name: str) -> Path | None:
    """Resolve ``raw/foo.jsonl`` or ``foo.jsonl`` under correlation_matching_results/raw."""
    rel = (file_name or "").strip().replace("\\", "/").lstrip("/")
    if not rel or ".." in rel.split("/"):
        return None
    parts = [p for p in rel.split("/") if p]
    if not parts:
        return None
    if parts[0].lower() == RAW_DIR_NAME:
        parts = parts[1:]
    if len(parts) != 1:
        return None
    name = parts[0]
    if not name.lower().endswith(".jsonl"):
        return None
    cand = (corr_results_dir / RAW_DIR_NAME / name).resolve()
    try:
        cand.relative_to((corr_results_dir / RAW_DIR_NAME).resolve())
    except ValueError:
        return None
    return cand if cand.is_file() else None


def list_raw_jsonl_files(corr_results_dir: Path) -> list[dict[str, Any]]:
    raw_dir = corr_results_dir / RAW_DIR_NAME
    if not raw_dir.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for path in sorted(raw_dir.glob("*.jsonl")):
        n = 0
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        n += 1
        except OSError:
            continue
        out.append({
            "fileName": f"{RAW_DIR_NAME}/{path.name}",
            "label": path.stem,
            "nRows": n,
        })
    return out


def list_raw_jsonl_rows(path: Path, *, preview_chars: int = 96) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            task_id = str(obj.get("task_id") or f"row_{line_no}")
            predict = str(obj.get("predict") or obj.get("output") or "")
            label = str(obj.get("label") or obj.get("response") or obj.get("gold") or "")
            rows.append({
                "line": line_no,
                "task_id": task_id,
                "predict_preview": predict[:preview_chars],
                "label_preview": label[:preview_chars],
            })
    return rows


def read_raw_jsonl_row(path: Path, line_no: int) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        for i, line in enumerate(handle, start=1):
            if i != int(line_no):
                continue
            if not line.strip():
                raise ValueError(f"raw jsonl line {line_no} is empty")
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"raw jsonl line {line_no} is not an object")
            return obj
    raise ValueError(f"raw jsonl has no line {line_no}")


def _load_tokenizer():
    from transformers import AutoTokenizer

    path = (
        base_model_path_from_env()
        or env_adapter_path_for_family("saliency")
        or env_adapter_path_for_family("ce")
    )
    if not path:
        raise RuntimeError(
            "Set EIF_BASE_MODEL_PATH (or a family adapter path) in eif_api.env "
            "to tokenize raw eval JSONL."
        )
    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def _encode_prompt_completion(tokenizer, prompt: str, completion: str) -> tuple[list[str], list[int], int]:
    from src.continue_train_eval import _render_eval_prompt
    from src.export_real_ttav_bundle import token_surfaces_for_display

    rendered = _render_eval_prompt(tokenizer, prompt)
    prompt_ids = [int(x) for x in tokenizer.encode(rendered, add_special_tokens=False)]
    comp = completion or ""
    comp_ids = [int(x) for x in tokenizer.encode(comp, add_special_tokens=False)] if comp else []
    ids = prompt_ids + comp_ids
    tokens = token_surfaces_for_display(tokenizer, ids)
    if len(tokens) != len(ids):
        tokens = [tokenizer.decode([i], skip_special_tokens=False) for i in ids]
    return tokens, ids, len(prompt_ids)


def build_raw_eval_report(
    row: dict[str, Any],
    *,
    file_name: str,
    line_no: int,
    tokenizer=None,
) -> dict[str, Any]:
    prompt = str(row.get("prompt") or row.get("input") or "")
    label = str(row.get("label") or row.get("response") or row.get("gold") or "")
    predict = str(row.get("predict") or row.get("output") or "")
    if not prompt.strip():
        raise ValueError("raw row missing prompt/input")
    if not label.strip() and not predict.strip():
        raise ValueError("raw row missing both label and predict")
    if not predict.strip():
        predict = label
    if not label.strip():
        label = predict

    tok = tokenizer or _load_tokenizer()
    pred_tokens, pred_ids, prompt_len = _encode_prompt_completion(tok, prompt, predict)
    gold_tokens, gold_ids, gold_prompt_len = _encode_prompt_completion(tok, prompt, label)
    if gold_prompt_len != prompt_len:
        # Same prompt string should encode identically; if not, keep predict prompt_len
        # and re-slice gold (rare tokenizer non-determinism).
        prompt_len = min(prompt_len, gold_prompt_len, len(pred_ids) - 1, len(gold_ids) - 1)
        pred_tokens = pred_tokens[:prompt_len] + pred_tokens[prompt_len:]
        gold_tokens = gold_tokens[:prompt_len] + gold_tokens[prompt_len:]

    task_id = str(row.get("task_id") or f"row_{line_no}")
    rel = file_name if str(file_name).replace("\\", "/").startswith("raw/") else f"raw/{Path(file_name).name}"
    return {
        "experiment_meta": {
            "test_sample_index": int(line_no),
            "mode": "all_tokens",
            "tokens_analyzed": 0,
            "task_id": task_id,
            "report_file": rel,
            "report_family": "saliency",
            "raw_eval": True,
            "model_name": Path(rel).stem,
        },
        "test_sample_baseline": {
            "full_tokens": pred_tokens,
            "full_token_ids": pred_ids,
            "correct_full_tokens": gold_tokens,
            "correct_full_token_ids": gold_ids,
            "full_tokens_display": pred_tokens,
            "correct_full_tokens_display": gold_tokens,
            "prompt_len": int(prompt_len),
            "raw_prompt": prompt,
            "raw_label": label,
            "raw_predict": predict,
        },
        "per_token_results": [],
        "train_sample_details": {},
    }
