import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from src.export_real_ttav_bundle import (
    DEFAULT_MODEL_PATH,
    compute_contextual_embeddings,
    compute_input_embeddings,
    convert_report_tokens_to_ids,
    load_model_and_tokenizer,
    load_report,
    validate_roundtrip_tokens,
)
from src.export_ttav_bundle import (
    compute_projection,
    decode_token,
    infer_sample_id,
    normalize_token_for_display,
)

ProgressCallback = Callable[[str, str], None]

PROBE_CLASSES = [
    "train_context",
    "train_source",
    "train_target",
    "test_context",
    "test_source",
    "test_target",
]
PROBE_CLASS_TO_ID = {name: idx for idx, name in enumerate(PROBE_CLASSES)}


def _compute_sequence_embeddings(
    model,
    tokenizer,
    tokens: list[str],
    embedding_type: str,
    hidden_layer: int,
) -> tuple[np.ndarray, list[int]]:
    token_ids = convert_report_tokens_to_ids(tokenizer, tokens)
    validate_roundtrip_tokens(tokenizer, token_ids, tokens)

    if getattr(model, "device", None) is not None:
        device = model.device
    else:
        device = next(model.parameters()).device
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    if embedding_type == "input":
        embeddings = compute_input_embeddings(model, input_ids)
    elif embedding_type == "contextual":
        embeddings = compute_contextual_embeddings(model, input_ids, hidden_layer)
    else:
        raise ValueError(f"Unsupported embedding_type: {embedding_type}")

    return embeddings.astype(np.float32), token_ids


def _expand_indices(indices: set[int], total: int, radius: int) -> list[int]:
    expanded: set[int] = set()
    for idx in indices:
        if idx < 0 or idx >= total:
            continue
        lo = max(0, idx - radius)
        hi = min(total - 1, idx + radius)
        for pos in range(lo, hi + 1):
            expanded.add(pos)
    return sorted(expanded)


def _build_probe_label(side: str, role: str, token: str, idx: int) -> str:
    prefix = {
        ("train", "context"): "TC",
        ("train", "source"): "TS",
        ("train", "target"): "TT",
        ("test", "context"): "XC",
        ("test", "source"): "XS",
        ("test", "target"): "XT",
    }[(side, role)]
    normalized = normalize_token_for_display(token)
    shortened = normalized[:15] + "…" if len(normalized) > 18 else normalized
    return f"{prefix}{idx}: {shortened}"


def _append_probe_points(
    side: str,
    tokens: list[str],
    embeddings: np.ndarray,
    token_ids: list[int],
    ordered_indices: list[int],
    source_indices: set[int],
    target_indices: set[int],
    point_records: list[dict],
    label_ids: list[int],
    text_list: list[str],
    text_data: list[str],
    token_list: list[str],
    selected_indices: list[int],
    embedding_rows: list[np.ndarray],
    focus_indices: set[int] | None = None,
    select_non_context: bool = True,
) -> tuple[int | None, dict[int, int]]:
    first_target_point: int | None = None
    point_index_by_token: dict[int, int] = {}
    focus_indices = focus_indices or set()

    for point_idx, token_idx in enumerate(ordered_indices, start=len(point_records)):
        role = "context"
        if token_idx in target_indices:
            role = "target"
        elif token_idx in source_indices:
            role = "source"

        class_name = f"{side}_{role}"
        label_ids.append(PROBE_CLASS_TO_ID[class_name])
        text_list.append(_build_probe_label(side, role, tokens[token_idx], token_idx))
        text_data.append(f"{side} {role} #{token_idx}: {decode_token(tokens[token_idx]).strip() or '·'}")
        token_list.append(normalize_token_for_display(tokens[token_idx]))
        embedding_rows.append(embeddings[token_idx])
        point_records.append({
            "point_index": point_idx,
            "side": side,
            "role": role,
            "token_index": token_idx,
            "token_id": token_ids[token_idx],
            "token": tokens[token_idx],
            "token_display": normalize_token_for_display(tokens[token_idx]),
            "is_focus": token_idx in focus_indices,
        })
        point_index_by_token[token_idx] = point_idx
        if (select_non_context and role != "context") or token_idx in focus_indices:
            selected_indices.append(point_idx)
        if role == "target" and first_target_point is None:
            first_target_point = point_idx

    return first_target_point, point_index_by_token


def _build_focus_comparison_summary(
    tokens: list[str],
    embeddings: np.ndarray,
    focus_indices: list[int],
    point_index_by_token: dict[int, int],
) -> dict:
    clean_indices = [idx for idx in sorted(dict.fromkeys(focus_indices)) if 0 <= idx < len(tokens)]
    token_entries = [
        {
            "tokenIndex": idx,
            "token": tokens[idx],
            "tokenDisplay": normalize_token_for_display(tokens[idx]),
            "pointIndex": point_index_by_token.get(idx),
        }
        for idx in clean_indices
    ]

    pair_entries: list[dict] = []
    for left_pos, left_idx in enumerate(clean_indices):
        left_embedding = embeddings[left_idx]
        left_norm = float(np.linalg.norm(left_embedding))
        for right_idx in clean_indices[left_pos + 1:]:
            right_embedding = embeddings[right_idx]
            right_norm = float(np.linalg.norm(right_embedding))
            denom = left_norm * right_norm
            cosine = 0.0 if denom == 0 else float(np.dot(left_embedding, right_embedding) / denom)
            pair_entries.append({
                "leftIndex": left_idx,
                "leftToken": tokens[left_idx],
                "leftTokenDisplay": normalize_token_for_display(tokens[left_idx]),
                "rightIndex": right_idx,
                "rightToken": tokens[right_idx],
                "rightTokenDisplay": normalize_token_for_display(tokens[right_idx]),
                "cosine": cosine,
            })

    return {
        "focusTokens": token_entries,
        "pairwiseCosine": pair_entries,
    }


def build_train_probe_bundle_payload(
    report_json_path: str,
    model_path: str,
    train_sample_id: int,
    probe_pairs: list[dict],
    sample_id: str | None = None,
    embedding_type: str = "contextual",
    hidden_layer: int = -1,
    vis_method: str = "TimeVis",
    vis_id: str = "1",
    dtype_name: str = "bfloat16",
    context_radius: int = 1,
    include_full_train: bool = False,
    focus_train_indices: list[int] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    if not probe_pairs:
        raise ValueError("probe_pairs is required for train probe export")

    if progress_callback:
        progress_callback("loading_report", "Loading EIF report JSON for train probe")
    report = load_report(report_json_path)
    base_sample_id = sample_id or infer_sample_id(report_json_path)

    train_details = report.get("train_sample_details", {})
    detail = train_details.get(str(train_sample_id))
    if not isinstance(detail, dict):
        raise ValueError(f"Train sample detail not found: {train_sample_id}")

    train_tokens = detail.get("full_tokens") or []
    test_tokens = report["test_sample_baseline"]["full_tokens"]
    if not train_tokens or not test_tokens:
        raise ValueError("Train/test token sequences are required to build a probe bundle")

    train_source_indices = {int(pair["trainSourceIndex"]) for pair in probe_pairs}
    train_target_indices = {int(pair["trainTargetIndex"]) for pair in probe_pairs}
    test_source_indices = {int(pair["testSourceIndex"]) for pair in probe_pairs}
    test_target_indices = {int(pair["testTargetIndex"]) for pair in probe_pairs}

    focus_train_indices = [
        int(idx) for idx in (focus_train_indices or [])
        if isinstance(idx, int) or (isinstance(idx, str) and idx.strip().isdigit())
    ]
    focus_train_index_set = {idx for idx in focus_train_indices if 0 <= idx < len(train_tokens)}

    context_radius = max(0, int(context_radius))
    ordered_train_indices = list(range(len(train_tokens))) if include_full_train else _expand_indices(
        train_source_indices | train_target_indices, len(train_tokens), context_radius
    )
    ordered_test_indices = _expand_indices(test_source_indices | test_target_indices, len(test_tokens), context_radius)

    if progress_callback:
        progress_callback("loading_model", "Loading model for train probe embeddings")
    model, tokenizer = load_model_and_tokenizer(
        model_path,
        dtype_name=dtype_name,
        progress_callback=progress_callback,
    )

    if progress_callback:
        progress_callback("computing_train_embeddings", "Computing train-sample token embeddings")
    train_embeddings_full, train_token_ids = _compute_sequence_embeddings(
        model, tokenizer, train_tokens, embedding_type, hidden_layer
    )

    if progress_callback:
        progress_callback("computing_test_embeddings", "Computing test-sample token embeddings")
    test_embeddings_full, test_token_ids = _compute_sequence_embeddings(
        model, tokenizer, test_tokens, embedding_type, hidden_layer
    )

    point_records: list[dict] = []
    label_ids: list[int] = []
    text_list: list[str] = []
    text_data: list[str] = []
    token_list: list[str] = []
    selected_indices: list[int] = []
    embedding_rows: list[np.ndarray] = []

    if progress_callback:
        progress_callback("assembling_probe", "Assembling train/test token probe bundle")
    default_focus_train_indices = sorted(train_source_indices | train_target_indices)
    explicit_focus_mode = include_full_train or bool(focus_train_index_set)
    if include_full_train:
        effective_focus_train_indices = sorted(focus_train_index_set)
    else:
        effective_focus_train_indices = sorted(focus_train_index_set) if focus_train_index_set else default_focus_train_indices

    first_target_index, train_point_index_by_token = _append_probe_points(
        "train",
        train_tokens,
        train_embeddings_full,
        train_token_ids,
        ordered_train_indices,
        train_source_indices,
        train_target_indices,
        point_records,
        label_ids,
        text_list,
        text_data,
        token_list,
        selected_indices,
        embedding_rows,
        focus_indices=set(effective_focus_train_indices),
        select_non_context=not explicit_focus_mode,
    )
    test_first_target_index, _ = _append_probe_points(
        "test",
        test_tokens,
        test_embeddings_full,
        test_token_ids,
        ordered_test_indices,
        test_source_indices,
        test_target_indices,
        point_records,
        label_ids,
        text_list,
        text_data,
        token_list,
        selected_indices,
        embedding_rows,
        focus_indices=set(),
        select_non_context=not explicit_focus_mode,
    )

    if explicit_focus_mode:
        if effective_focus_train_indices:
            first_focus = effective_focus_train_indices[0]
            first_target_index = train_point_index_by_token.get(first_focus, first_target_index)
        else:
            first_target_index = None
    elif test_first_target_index is not None:
        first_target_index = test_first_target_index

    selected_indices = sorted(dict.fromkeys(selected_indices))
    embeddings = np.vstack(embedding_rows).astype(np.float32) if embedding_rows else np.zeros((0, 2), dtype=np.float32)

    if progress_callback:
        progress_callback("projecting_embeddings", "Projecting train probe embeddings to 2D")
    projection = compute_projection(embeddings)

    pair_signature = [pair.get("id") or {
        "trainSourceIndex": pair["trainSourceIndex"],
        "trainTargetIndex": pair["trainTargetIndex"],
        "testSourceIndex": pair["testSourceIndex"],
        "testTargetIndex": pair["testTargetIndex"],
    } for pair in probe_pairs]
    probe_sample_id = f"{base_sample_id}_train{train_sample_id}_probe"
    train_count = len(ordered_train_indices)
    comparison_summary = _build_focus_comparison_summary(
        train_tokens,
        train_embeddings_full,
        effective_focus_train_indices,
        train_point_index_by_token,
    )

    return {
        "sample_id": probe_sample_id,
        "vis_method": vis_method,
        "vis_id": vis_id,
        "overwrite": True,
        "selected_indices": selected_indices,
        "target_index": first_target_index,
        "comparison_summary": comparison_summary,
        "bundle": {
            "model": Path(model_path).name,
            "checkpoint_path": model_path,
            "embedding_type": embedding_type,
            "embedding_note": f"train_probe_layer_{hidden_layer}" if embedding_type == "contextual" else "train_probe_input_embedding",
            "embedding_dtype": dtype_name,
            "classes": PROBE_CLASSES,
            "sample_index": int(report["experiment_meta"]["test_sample_index"]),
            "prompt_len": train_count,
            "labels": label_ids,
            "text_list": text_list,
            "text_data": text_data,
            "token_list": token_list,
            "index": {
                "train": list(range(train_count)),
                "test": list(range(train_count, len(point_records))),
            },
            "embeddings": embeddings.tolist(),
            "projection": projection.tolist(),
            "probe_metadata": {
                "kind": "train_correlation_probe",
                "base_sample_id": base_sample_id,
                "train_sample_id": train_sample_id,
                "context_radius": context_radius,
                "include_full_train": include_full_train,
                "pair_signature": pair_signature,
                "focus_train_indices": effective_focus_train_indices,
                "point_records": point_records,
            },
        },
    }
