"""Build a synthetic all-tokens report from a raw eval JSONL row (prompt/label/predict).

Layout under ``correlation_matching_results``::

    raw_ce/*.jsonl    → report_family=ce        → EIF_ADAPTER_PATH_CE
    raw_sal/*.jsonl   → report_family=saliency  → EIF_ADAPTER_PATH_SALIENCY

``raw_sa/`` is still accepted as a legacy alias of ``raw_sal/``.
Legacy ``raw/`` is listed only for resolve (family unknown).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from src.eif_adapter_env import base_model_path_from_env, env_adapter_path_for_family

RawFamily = Literal["ce", "saliency"]

# folder name → adapter family
RAW_FAMILY_DIRS: dict[str, RawFamily] = {
    "raw_ce": "ce",
    "raw_sal": "saliency",
    "raw_sa": "saliency",  # legacy alias of raw_sal
}
# Legacy unscoped folder (prefer migrating files into raw_ce / raw_sal).
RAW_LEGACY_DIR = "raw"
ALL_RAW_DIRS: tuple[str, ...] = ("raw_ce", "raw_sal", "raw_sa", RAW_LEGACY_DIR)


def family_from_raw_relpath(file_name: str) -> RawFamily | None:
    """Return ce/saliency from ``raw_ce/foo.jsonl`` / ``raw_sal/foo.jsonl``."""
    rel = (file_name or "").strip().replace("\\", "/").lstrip("/")
    top = rel.split("/", 1)[0].lower() if rel else ""
    return RAW_FAMILY_DIRS.get(top)


def resolve_raw_jsonl(corr_results_dir: Path, file_name: str) -> Path | None:
    """Resolve ``raw_ce/foo.jsonl``, ``raw_sal/foo.jsonl``, or legacy ``raw/foo.jsonl``."""
    rel = (file_name or "").strip().replace("\\", "/").lstrip("/")
    if not rel or ".." in rel.split("/"):
        return None
    parts = [p for p in rel.split("/") if p]
    if len(parts) == 1:
        # Bare filename: search raw_ce → raw_sal → raw_sa → raw
        name = parts[0]
        if not name.lower().endswith(".jsonl"):
            return None
        for folder in ALL_RAW_DIRS:
            cand = (corr_results_dir / folder / name).resolve()
            try:
                cand.relative_to((corr_results_dir / folder).resolve())
            except ValueError:
                continue
            if cand.is_file():
                return cand
        return None
    if len(parts) != 2:
        return None
    folder, name = parts[0].lower(), parts[1]
    if folder not in ALL_RAW_DIRS:
        return None
    if not name.lower().endswith(".jsonl"):
        return None
    cand = (corr_results_dir / folder / name).resolve()
    try:
        cand.relative_to((corr_results_dir / folder).resolve())
    except ValueError:
        return None
    return cand if cand.is_file() else None


def list_raw_jsonl_files(corr_results_dir: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for folder, family in RAW_FAMILY_DIRS.items():
        raw_dir = corr_results_dir / folder
        if not raw_dir.is_dir():
            continue
        for path in sorted(raw_dir.glob("*.jsonl")):
            n = 0
            try:
                with path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        if line.strip():
                            n += 1
            except OSError:
                continue
            tag = "CE" if family == "ce" else "SAL"
            out.append({
                "fileName": f"{folder}/{path.name}",
                "label": f"[{tag}] {path.stem}",
                "nRows": n,
                "folder": folder,
                "reportFamily": family,
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


def _surfaces_for_ids(tokenizer, ids: list[int]) -> list[str]:
    """Per-id display strings. Never merge across ids (see raw encode note)."""
    from src.export_real_ttav_bundle import token_surfaces_for_display

    if not ids:
        return []
    # Region-local merge is OK (fixes U+FFFD chips) and cannot swallow the answer
    # into the prompt, because prompt/answer are surfaced in separate calls.
    tokens = token_surfaces_for_display(tokenizer, ids)
    if len(tokens) != len(ids):
        tokens = [tokenizer.decode([i], skip_special_tokens=False) for i in ids]
    return tokens


def _encode_prompt_once(tokenizer, prompt: str) -> tuple[list[str], list[int]]:
    from src.continue_train_eval import _render_eval_prompt

    rendered = _render_eval_prompt(tokenizer, prompt)
    prompt_ids = [int(x) for x in tokenizer.encode(rendered, add_special_tokens=False)]
    if not prompt_ids:
        raise ValueError("raw prompt encoded to empty token ids")
    return _surfaces_for_ids(tokenizer, prompt_ids), prompt_ids


def _encode_completion(tokenizer, completion: str) -> tuple[list[str], list[int]]:
    comp = completion or ""
    if not comp:
        return [], []
    comp_ids = [int(x) for x in tokenizer.encode(comp, add_special_tokens=False)]
    return _surfaces_for_ids(tokenizer, comp_ids), comp_ids


def build_raw_eval_report(
    row: dict[str, Any],
    *,
    file_name: str,
    line_no: int,
    tokenizer=None,
    report_family: RawFamily | None = None,
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

    rel = (file_name or "").strip().replace("\\", "/").lstrip("/")
    family = report_family or family_from_raw_relpath(rel)
    if family is None:
        raise ValueError(
            f"raw file must live under raw_ce/ or raw_sal/ (got {rel!r}). "
            "Move the JSONL out of legacy raw/."
        )
    if not rel.startswith(("raw_ce/", "raw_sal/", "raw_sa/")):
        rel = f"{'raw_ce' if family == 'ce' else 'raw_sal'}/{Path(rel).name}"

    tok = tokenizer or _load_tokenizer()
    # Shared prompt ids so Model/Gold share the same prompt_len boundary.
    # Critical: surface prompt and completion SEPARATELY. Running
    # token_surfaces_for_display on the full sequence lets a trailing U+FFFD
    # byte-fallback merge swallow the entire answer into the last prompt chip;
    # answer indices become "" → UI shows text that is not clickable, and Gold
    # looks empty while still rendering a GOLD header.
    prompt_tokens, prompt_ids = _encode_prompt_once(tok, prompt)
    prompt_len = len(prompt_ids)
    pred_ans_tokens, pred_ans_ids = _encode_completion(tok, predict)
    gold_ans_tokens, gold_ans_ids = _encode_completion(tok, label)
    if not pred_ans_ids and predict.strip():
        raise ValueError("raw predict encoded to empty token ids")
    if not gold_ans_ids and label.strip():
        raise ValueError("raw label encoded to empty token ids")

    pred_tokens = prompt_tokens + pred_ans_tokens
    pred_ids = prompt_ids + pred_ans_ids
    gold_tokens = prompt_tokens + gold_ans_tokens
    gold_ids = prompt_ids + gold_ans_ids

    task_id = str(row.get("task_id") or f"row_{line_no}")
    n_answer = len(pred_ans_ids)
    family_tag = "ce" if family == "ce" else "sal"
    return {
        "experiment_meta": {
            "test_sample_index": int(line_no),
            "mode": "all_tokens",
            "tokens_analyzed": int(n_answer),
            "task_id": task_id,
            "report_file": rel,
            "report_family": family,
            "raw_eval": True,
            "model_name": f"{Path(rel).stem}[{family_tag}]",
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
