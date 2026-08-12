"""Live gold-path attribution (teacher-force label) for the correlation-report UI.

Three stages (same math as intervention_experiment, completion = gold):

1. ``gold_saliency_top_k`` — last-layer ALTI top sources for a gold target token
2. ``gold_retrieve_trains`` — bank Top-K trains via sketched L_probe gradient
3. ``gold_stage3_pairs`` — Stage3 ∇C matching on those trains

Configure paths in repo-root ``eif_api.env`` (see ``eif_api.env.example``).
"""

from __future__ import annotations

import gc
import os
from functools import partial
from heapq import nlargest
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import DataCollatorForSeq2Seq

from src.NIF import (
    DatasetWrapper,
    CustomCollator,
    _find_subseq_start,
    build_single_sample_dataset,
    build_train_dataset,
)
from src.bank_loss import load_bank_loss_config
from src.export_real_ttav_bundle import convert_report_tokens_to_ids
from src.export_ttav_bundle import infer_sample_id
from src.intervention_experiment import (
    ALTI_CHUNK_SIZE,
    FINE_MATCH_LAST_N_LAYERS,
    FINE_MATCH_PROJ,
    PRESCREEN_SKETCH_DIM,
    PRESCREEN_SKETCH_SEED,
    SALIENCY_TRAIN_BANK_CACHE_DIR,
    SEQUENCE_LENGTH_LIMIT,
    _load_or_build_saliency_train_bank,
    _project_flat_grad,
    _score_prescreen_sketch_cache,
    find_first_valid_token_index,
    get_context_window,
    is_trivial_token,
    load_model_and_tokenizer,
    load_samples,
    make_attention_projection_filter,
    make_lora_param_filter,
    top_nontrivial_saliency_sources,
)
from src.loss import (
    compute_last_layer_correlation_gradient,
    compute_last_layer_match_and_probe_gradients,
    compute_last_layer_saliency_vector,
    prepare_last_layer_grad_checkpointing,
)
from src.process_data import process_func_chatml
from src.unlearn_pair_probe import _release_cuda_memory


REPO_ROOT = Path(__file__).resolve().parent.parent

_SESSION: dict[str, Any] | None = None


def _is_placeholder_path(value: str) -> bool:
    v = value.strip()
    if not v:
        return True
    low = v.lower()
    return (
        low.startswith("/path/to/")
        or low.startswith("d:/path")
        or low.startswith("c:/path")
        or "your/checkpoint" in low
        or "/path/to/" in low
    )


def _hydrate_eif_env(*, force_file: bool = True) -> Path | None:
    """Load repo-root ``eif_api.env`` into ``os.environ``.

    By default, ``EIF_*`` / path keys from the file **overwrite** stale shell
    values so editing the file + restart (or next gold call) actually sticks.
    Placeholder ``/path/to/...`` values are never applied.
    Set ``EIF_ENV_NO_OVERRIDE=1`` to keep the old "shell wins" behavior.
    """
    no_override = (
        not force_file
        or os.environ.get("EIF_ENV_NO_OVERRIDE", "").strip().lower() in ("1", "true", "yes")
    )

    def _apply(path: Path) -> None:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if not key:
                continue
            if key.endswith("_PATH") or key.endswith("_DATA") or key.endswith("_ROOT"):
                if _is_placeholder_path(value):
                    # Drop stale placeholder previously injected into the process.
                    if key in os.environ and _is_placeholder_path(os.environ.get(key, "")):
                        del os.environ[key]
                    continue
            if no_override and key in os.environ:
                continue
            os.environ[key] = value

    for name in ("eif_api.env", ".env"):
        path = REPO_ROOT / name
        if path.is_file():
            _apply(path)
            return path
    viewer_env = REPO_ROOT / "tools" / "annotation-viewer" / ".env"
    if viewer_env.is_file():
        _apply(viewer_env)
        return viewer_env
    return None


def _env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_optional_int(name: str, default: int | None = 3) -> int | None:
    """Parse optional int; empty / <=0 → None (no cap) when default is None;
    otherwise fall back to ``default`` when unset."""
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        v = int(raw)
    except ValueError:
        return default
    return None if v <= 0 else v


def _device_of(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_model_paths(report: dict[str, Any]) -> tuple[str, str | None]:
    """Prefer family-specific env adapter from report folder (ce/ vs saliency/)."""
    from src.eif_adapter_env import (
        adapter_path_for_family,
        base_model_path_from_env,
        infer_report_family,
    )

    _hydrate_eif_env(force_file=True)
    meta = report.get("experiment_meta") or {}
    family = infer_report_family(
        str(meta.get("report_file") or meta.get("fileName") or ""),
        report,
    )
    env_model = adapter_path_for_family(family).strip()
    report_model = str(meta.get("model_path") or meta.get("adapter_path") or "").strip()

    if env_model and not _is_placeholder_path(env_model):
        model = env_model
        src = f"EIF_ADAPTER_PATH_{family.upper()}" if family in ("ce", "saliency") else "EIF_ADAPTER_PATH"
    elif report_model and not _is_placeholder_path(report_model):
        model = report_model
        src = "report experiment_meta.model_path"
        print(
            f"[gold-live] WARNING: env adapter unset for family={family}; "
            f"falling back to {src}={report_model!r}.",
            flush=True,
        )
    else:
        raise ValueError(
            "No usable model path. Set EIF_ADAPTER_PATH_CE / EIF_ADAPTER_PATH_SALIENCY "
            f"(and EIF_BASE_MODEL_PATH) in {REPO_ROOT / 'eif_api.env'}. "
            f"family={family!r} env={env_model!r} report={report_model!r}."
        )

    if not os.path.isdir(model):
        hint = ""
        if model.startswith("/") and os.name == "nt":
            hint = (
                " (Linux path on Windows — run API on the Ubuntu host with /mnt/md124, "
                "or point the family adapter path at a local directory)"
            )
        raise FileNotFoundError(
            f"{src} is not a local directory: {model!r}.{hint} "
            "Gold live only loads local checkpoints (not HuggingFace hub repo ids)."
        )

    base = (
        base_model_path_from_env().strip()
        or str(meta.get("base_model_path") or "").strip()
        or None
    )
    if base and _is_placeholder_path(base):
        base = None
    adapter_cfg = os.path.join(model, "adapter_config.json")
    if os.path.isfile(adapter_cfg):
        if not base:
            raise ValueError(
                f"{model} looks like a LoRA adapter but EIF_BASE_MODEL_PATH is empty. "
                f"Set it in {REPO_ROOT / 'eif_api.env'}."
            )
        if not os.path.isdir(base):
            raise FileNotFoundError(
                f"EIF_BASE_MODEL_PATH is not a local directory: {base!r}"
            )
    print(
        f"[gold-live] family={family} using model from {src}: {model}",
        flush=True,
    )
    return os.path.abspath(model), (os.path.abspath(base) if base else None)


def _resolve_train_data() -> Path:
    raw = (
        (os.environ.get("EIF_TRAIN_DATA") or "").strip()
        or (os.environ.get("ANNOTATION_TRAIN_DATA") or "").strip()
    )
    if not raw:
        raise ValueError(
            "No train data. Set EIF_TRAIN_DATA in eif_api.env "
            "(chat JSONL used by intervention_experiment)."
        )
    p = Path(raw).expanduser()
    candidates = [p] if p.is_absolute() else [Path.cwd() / p, REPO_ROOT / p]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError(f"Train JSONL not found: {raw}")


def _gold_tokens_and_ids(report: dict[str, Any], tokenizer) -> tuple[list[str], list[int], int]:
    baseline = report.get("test_sample_baseline") or {}
    tokens = list(baseline.get("correct_full_tokens") or [])
    if not tokens:
        raise ValueError("Report missing test_sample_baseline.correct_full_tokens.")
    prompt_len = int(baseline.get("prompt_len") or 0)
    if not (0 <= prompt_len < len(tokens)):
        raise ValueError(f"Invalid prompt_len={prompt_len} for gold len={len(tokens)}.")

    stored = baseline.get("correct_full_token_ids")
    if isinstance(stored, list) and len(stored) == len(tokens):
        ids = [int(x) for x in stored]
        return tokens, ids, prompt_len

    # Prefer shared prompt ids from the predict path (always stored as full_token_ids).
    pred_ids = baseline.get("full_token_ids")
    hint = pred_ids if isinstance(pred_ids, list) else None

    try:
        ids = convert_report_tokens_to_ids(
            tokenizer,
            tokens,
            hint_ids=[int(x) for x in hint] if hint is not None else None,
            hint_until=prompt_len,
        )
    except ValueError:
        # Rebuild answer ids by encoding the joined gold answer; keep prompt ids.
        if hint is None or len(hint) < prompt_len:
            raise
        prompt_ids = [int(x) for x in hint[:prompt_len]]
        answer_surf = "".join(tokens[prompt_len:])
        answer_ids = [int(x) for x in tokenizer.encode(answer_surf, add_special_tokens=False)]
        ids = prompt_ids + answer_ids
        if len(ids) != len(tokens):
            # Surfaces were lossy (U+FFFD); rebuild answer token list to match ids.
            new_answer_tokens = [tokenizer.decode([i]) for i in answer_ids]
            tokens = list(tokens[:prompt_len]) + new_answer_tokens
            print(
                f"[gold-live] WARNING: rebuilt gold answer tokens "
                f"({len(new_answer_tokens)} vs report surfaces) after U+FFFD remap; "
                f"click indices in the answer may shift slightly.",
                flush=True,
            )
        if len(ids) != len(tokens):
            raise ValueError(
                f"Could not align gold token ids ({len(ids)}) to surfaces ({len(tokens)})."
            )
    return tokens, ids, prompt_len


def _batch_from_ids(ids: list[int], prompt_len: int, device) -> dict[str, torch.Tensor]:
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


class _DummyAccelerator:
    """Minimal stand-in so bank builder can call .device / .is_main_process."""

    def __init__(self, device: torch.device):
        self.device = device
        self.is_main_process = True
        self.is_local_main_process = True
        self.num_processes = 1

    def prepare(self, loader):
        return loader


def _ensure_session(report: dict[str, Any]) -> dict[str, Any]:
    global _SESSION
    _hydrate_eif_env()
    model_path, base_path = _resolve_model_paths(report)
    train_path = _resolve_train_data()
    key = (
        os.path.abspath(model_path),
        os.path.abspath(base_path) if base_path else "",
        str(train_path),
    )
    if _SESSION is not None and _SESSION.get("key") == key:
        return _SESSION

    print(f"[gold-live] loading model adapter={model_path} base={base_path or '-'}", flush=True)
    with torch.inference_mode(False):
        model, tokenizer = load_model_and_tokenizer(
            model_path=model_path,
            base_model_path=base_path,
        )
    device = _device_of(model)
    prepare_last_layer_grad_checkpointing(model)

    grad_space = str(getattr(model, "_eif_grad_space", "") or "lora").lower()
    last_n = FINE_MATCH_LAST_N_LAYERS
    if "lora" in grad_space:
        param_filter = make_lora_param_filter(model, last_n_layers=last_n)
        filter_tag = f"lora_L{last_n}"
    else:
        param_filter = make_attention_projection_filter(model, last_n, FINE_MATCH_PROJ)
        filter_tag = f"fineattn_{FINE_MATCH_PROJ}_L{last_n}"

    bank_override = (os.environ.get("EIF_BANK_LOSS_MODE") or "").strip() or None
    from src.eif_adapter_env import (
        bank_loss_mode_for_family,
        bank_path_for_family,
        infer_report_family,
        resolve_bank_file,
    )
    family = infer_report_family(
        str((report.get("experiment_meta") or {}).get("report_file") or ""),
        report,
    )
    if not bank_override or bank_override.lower() in ("auto", "none"):
        bank_override = bank_loss_mode_for_family(family)

    bank_cfg = load_bank_loss_config(
        model_path,
        Path(model_path).name,
        loss_mode_override=bank_override,
    )

    print(f"[gold-live] loading train data {train_path}", flush=True)
    train_samples = load_samples(str(train_path))
    if not train_samples:
        raise ValueError(f"No train samples loaded from {train_path}")

    convert_to_chatml = partial(process_func_chatml, tokenizer=tokenizer)
    train_ds = build_train_dataset(train_samples, convert_to_chatml)
    base_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, padding=True, return_tensors="pt",
    )
    collator = CustomCollator(base_collator)
    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        DatasetWrapper(train_ds),
        batch_size=1,
        collate_fn=collator,
    )
    accel = _DummyAccelerator(device)

    bank_env_key = (
        f"EIF_SALIENCY_BANK_PATH_{family.upper()}"
        if family in ("ce", "saliency")
        else "EIF_SALIENCY_BANK_PATH"
    )
    explicit_bank_raw = bank_path_for_family(family)
    explicit_bank = resolve_bank_file(explicit_bank_raw, repo_root=REPO_ROOT)
    if explicit_bank is not None:
        print(
            f"[gold-live] family={family} loading bank from {bank_env_key}={explicit_bank}",
            flush=True,
        )
        bank = torch.load(str(explicit_bank), map_location="cpu", weights_only=False)
    else:
        if explicit_bank_raw:
            print(
                f"[gold-live] WARNING: {bank_env_key}={explicit_bank_raw!r} not found; "
                f"falling back to cache/build under {SALIENCY_TRAIN_BANK_CACHE_DIR}",
                flush=True,
            )
        print("[gold-live] load/build saliency train bank (may be slow on first call)…", flush=True)
        bank = _load_or_build_saliency_train_bank(
            model,
            train_ds,
            train_loader,
            accel,
            model_path=model_path,
            param_filter_fn=param_filter,
            max_seq_len=SEQUENCE_LENGTH_LIMIT,
            sketch_dim=PRESCREEN_SKETCH_DIM,
            sketch_seed=PRESCREEN_SKETCH_SEED,
            fine_match_proj=FINE_MATCH_PROJ,
            last_n_layers=last_n,
            cache_dir=SALIENCY_TRAIN_BANK_CACHE_DIR,
            bank_cfg=bank_cfg,
            train_samples=train_samples,
            special_ids=set(getattr(tokenizer, "all_special_ids", []) or []),
            bank_left_truncate=True,
        )
        if bank is None:
            raise RuntimeError("Failed to load/build saliency train bank.")

    prepare_last_layer_grad_checkpointing(model)

    marker = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    _SESSION = {
        "key": key,
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
        "param_filter": param_filter,
        "filter_tag": filter_tag,
        "bank": bank,
        "bank_cfg": bank_cfg,
        "train_samples": train_samples,
        "convert_to_chatml": convert_to_chatml,
        "collator": base_collator,
        "marker_ids": marker,
        "model_path": model_path,
        "base_path": base_path,
        "train_path": str(train_path),
        "train_detail_cache": {},
    }
    print(
        f"[gold-live] ready trains={len(train_samples)} bank_rows="
        f"{int(bank['sample_ids'].numel())} filter={filter_tag}",
        flush=True,
    )
    return _SESSION


def gold_saliency_top_k(
    report: dict[str, Any],
    *,
    target_index: int,
    top_k: int | None = None,
) -> dict[str, Any]:
    """Stage 1: top-k saliency sources for a gold target index (absolute in correct_full_tokens)."""
    session = _ensure_session(report)
    model = session["model"]
    tokenizer = session["tokenizer"]
    device = session["device"]
    k = max(1, int(top_k if top_k is not None else _env_int("EIF_GOLD_TOP_SALIENCY", 4)))

    tokens, ids, prompt_len = _gold_tokens_and_ids(report, tokenizer)
    if not (prompt_len <= target_index < len(ids)):
        raise ValueError(
            f"gold target_index={target_index} out of range "
            f"[prompt_len={prompt_len}, len={len(ids)})."
        )

    batch = _batch_from_ids(ids, prompt_len, device)
    prepare_last_layer_grad_checkpointing(model)
    with torch.no_grad():
        sal_vec = compute_last_layer_saliency_vector(model, batch, target_index)

    ranked = top_nontrivial_saliency_sources(
        tokenizer,
        batch["input_ids"][0],
        sal_vec,
        k,
        offset=0,
    )
    target_tok = tokens[target_index]
    top = []
    for rank_i, (idx, score) in enumerate(ranked, start=1):
        top.append({
            "source_token_index": int(idx),
            "source_token": tokens[idx] if idx < len(tokens) else tokenizer.decode([ids[idx]]),
            "target_token_index": int(target_index),
            "target_token": target_tok,
            "saliency_score": float(score),
            "saliency_rank": rank_i,
        })

    _release_cuda_memory(model, reason="gold_saliency")
    return {
        "status": "success",
        "mode": "gold",
        "promptLen": prompt_len,
        "targetTokenIndex": int(target_index),
        "targetToken": target_tok,
        "topCorrelations": top,
        "filterTag": session["filter_tag"],
    }


def _compute_edge_features(
    session: dict[str, Any],
    batch: dict[str, torch.Tensor],
    source_index: int,
    target_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    model = session["model"]
    prepare_last_layer_grad_checkpointing(model)
    with torch.inference_mode(False):
        feat_match, feat_probe = compute_last_layer_match_and_probe_gradients(
            model=model,
            batch=batch,
            target_idx_in_seq=target_index,
            source_idx_in_seq=source_index,
            param_filter_fn=session["param_filter"],
            device=session["device"],
        )
    return feat_match, feat_probe


def _stage3_one_train(
    session: dict[str, Any],
    train_idx: int,
    coarse_score: float,
    test_feat_match: torch.Tensor,
    test_src_text: str,
    test_sal: float,
    test_src_idx: int,
    test_tgt_idx: int,
    test_tgt_text: str,
    pair_id_prefix: str,
    pair_id_start: int,
) -> tuple[dict[str, Any] | None, list[dict], int]:
    model = session["model"]
    tokenizer = session["tokenizer"]
    device = session["device"]
    train_samples = session["train_samples"]
    convert_to_chatml = session["convert_to_chatml"]
    collator = session["collator"]
    marker_ids = session["marker_ids"]
    cache: dict = session["train_detail_cache"]

    top_targets = _env_optional_int("EIF_GOLD_TOP_TARGETS", 3)
    top_sources = max(1, _env_int("EIF_GOLD_TOP_SOURCES", 3))

    cached = cache.get(str(train_idx))
    if cached is None:
        tr_ds = build_single_sample_dataset(train_samples[train_idx], convert_to_chatml)
        tr_batch = collator([tr_ds[0]])
        tr_batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in tr_batch.items()}
        if tr_batch["input_ids"].size(1) > SEQUENCE_LENGTH_LIMIT:
            return None, [], pair_id_start
        try:
            start_sys = _find_subseq_start(tr_batch["input_ids"][0], marker_ids) + len(marker_ids)
        except ValueError:
            return None, [], pair_id_start
        response_start = find_first_valid_token_index(tokenizer, tr_batch["input_ids"], start_sys)
        tr_seq_len = tr_batch["input_ids"].size(1)
        full_token_ids = [int(i) for i in tr_batch["input_ids"][0].tolist()]
        full_tokens = [tokenizer.decode([i]) for i in full_token_ids]
        candidate_pairs: list[tuple[float, int, int]] = []
        target_saliencies: dict[int, list[float]] = {}

        prepare_last_layer_grad_checkpointing(model)
        with torch.inference_mode(False):
            t_tr = response_start
            kept = 0
            while t_tr < tr_seq_len:
                if top_targets is not None and kept >= top_targets:
                    break
                if is_trivial_token(tokenizer, int(tr_batch["input_ids"][0, t_tr].item())):
                    t_tr += 1
                    continue
                sal_vec = compute_last_layer_saliency_vector(
                    model, tr_batch, t_tr, chunk_size=ALTI_CHUNK_SIZE,
                )
                target_saliencies[t_tr] = [round(float(s), 6) for s in sal_vec]
                for s_idx, s_score in top_nontrivial_saliency_sources(
                    tokenizer, tr_batch["input_ids"][0], sal_vec, top_sources,
                ):
                    candidate_pairs.append((float(s_score), t_tr, int(s_idx)))
                kept += 1
                t_tr += 1

        edges = train_samples[train_idx].get("attention_edges") or []
        annotations_by_target: dict[str, list[dict]] = {}
        for e in edges:
            src = int(e["src"])
            dst = int(e["dst"])
            annotations_by_target.setdefault(str(dst), []).append(
                {"src": src, "subtype": str(e.get("subtype", ""))}
            )

        cached = {
            "full_tokens": full_tokens,
            "full_token_ids": full_token_ids,
            "answer_start_index": response_start,
            "coarse_cos_sim": float(coarse_score),
            "saliencies_by_token": {str(k): v for k, v in target_saliencies.items()},
            "annotations_by_target": annotations_by_target,
            "_candidate_pairs": candidate_pairs,
            "_tr_batch_cpu": {k: v.cpu() for k, v in tr_batch.items() if torch.is_tensor(v)},
            "_feature_cache": {},
        }
        cache[str(train_idx)] = cached
    else:
        if coarse_score > float(cached.get("coarse_cos_sim") or 0):
            cached["coarse_cos_sim"] = float(coarse_score)
        cached.setdefault("_feature_cache", {})

    tr_batch_gpu = {
        k: v.to(device) for k, v in cached["_tr_batch_cpu"].items()
    }
    ids_1d = tr_batch_gpu["input_ids"][0]
    response_start = cached["answer_start_index"]
    pair_counter = pair_id_start
    pair_records: list[dict] = []

    prepare_last_layer_grad_checkpointing(model)
    with torch.inference_mode(False):
        for saliency_score, t_tr, s_idx in cached["_candidate_pairs"]:
            feature_key = (int(t_tr), int(s_idx))
            if feature_key in cached["_feature_cache"]:
                train_feat = cached["_feature_cache"][feature_key]
                if train_feat is None:
                    continue
            else:
                try:
                    train_feat = compute_last_layer_correlation_gradient(
                        model=model,
                        batch=tr_batch_gpu,
                        target_idx_in_seq=t_tr,
                        source_idx_in_seq=s_idx,
                        param_filter_fn=session["param_filter"],
                        device=device,
                    )
                except Exception as exc:
                    print(f"[gold-live] stage3 grad skip train={train_idx}: {exc}", flush=True)
                    train_feat = None
                cached["_feature_cache"][feature_key] = train_feat
                if train_feat is None:
                    continue

            cos_sim = float(F.cosine_similarity(test_feat_match, train_feat, dim=0).item())
            train_target_tok = tokenizer.decode([int(ids_1d[t_tr].item())])
            train_source_tok = tokenizer.decode([int(ids_1d[s_idx].item())])
            pair_records.append({
                "id": f"{pair_id_prefix}_{pair_counter:04d}",
                "cos_sim": cos_sim,
                "coarse_cos_sim": float(coarse_score),
                "train_sample_id": int(train_idx),
                "test_correlation": {
                    "source_token": test_src_text,
                    "source_token_index": int(test_src_idx),
                    "target_token": test_tgt_text,
                    "target_token_index": int(test_tgt_idx),
                    "saliency_score": float(test_sal),
                },
                "train_correlation": {
                    "source_token": train_source_tok,
                    "source_token_index": int(s_idx),
                    "target_token": train_target_tok,
                    "target_token_index": int(t_tr),
                    "saliency_score": float(saliency_score),
                    "response_token_offset": int(t_tr - response_start),
                },
                "train_context": {
                    "source_context": get_context_window(tokenizer, ids_1d, s_idx),
                    "target_context": get_context_window(tokenizer, ids_1d, t_tr),
                },
                "annotation": None,
            })
            pair_counter += 1

    del tr_batch_gpu
    public = {k: v for k, v in cached.items() if not k.startswith("_")}
    return public, pair_records, pair_counter


def gold_retrieve_and_stage3(
    report: dict[str, Any],
    *,
    source_index: int,
    target_index: int,
    top_trains: int | None = None,
) -> dict[str, Any]:
    """Stage 2+3 for one gold saliency edge."""
    session = _ensure_session(report)
    model = session["model"]
    tokenizer = session["tokenizer"]
    device = session["device"]
    bank = session["bank"]
    n_trains = max(1, int(top_trains if top_trains is not None else _env_int("EIF_GOLD_TOP_TRAINS", 10)))

    tokens, ids, prompt_len = _gold_tokens_and_ids(report, tokenizer)
    if not (0 <= source_index < target_index < len(ids)):
        raise ValueError(
            f"Invalid gold edge source={source_index} target={target_index} len={len(ids)}"
        )
    if target_index < prompt_len:
        raise ValueError("Gold target must be in the response (index >= prompt_len).")

    batch = _batch_from_ids(ids, prompt_len, device)
    test_src_text = tokens[source_index]
    test_tgt_text = tokens[target_index]

    # Refresh saliency score for this edge (cheap vs match/probe).
    with torch.no_grad():
        sal_vec = compute_last_layer_saliency_vector(model, batch, target_index)
    test_sal = float(sal_vec[source_index]) if source_index < len(sal_vec) else 0.0

    print(
        f"[gold-live] stage2/3 edge {source_index}:{test_src_text!r} → "
        f"{target_index}:{test_tgt_text!r}",
        flush=True,
    )
    feat_match, feat_probe = _compute_edge_features(
        session, batch, source_index, target_index,
    )
    probe_sketch = _project_flat_grad(feat_probe, PRESCREEN_SKETCH_DIM, PRESCREEN_SKETCH_SEED)
    edge_scores = _score_prescreen_sketch_cache(probe_sketch, bank, device)
    related = nlargest(n_trains, edge_scores, key=lambda x: x[1])
    print(f"[gold-live] top trains: {[(i, round(s, 4)) for i, s in related]}", flush=True)

    all_pairs: list[dict] = []
    details: dict[str, Any] = {}
    pair_id = 0
    for train_idx, score in related:
        detail, pairs, pair_id = _stage3_one_train(
            session,
            int(train_idx),
            float(score),
            feat_match,
            test_src_text,
            test_sal,
            source_index,
            target_index,
            test_tgt_text,
            f"gold_t{target_index}_s{source_index}",
            pair_id,
        )
        if detail is not None:
            details[str(train_idx)] = detail
            all_pairs.extend(pairs)
        _release_cuda_memory(model, reason=f"gold_stage3_train{train_idx}")

    all_pairs.sort(key=lambda p: p["cos_sim"], reverse=True)
    _release_cuda_memory(model, reason="gold_retrieve_done")
    return {
        "status": "success",
        "mode": "gold",
        "promptLen": prompt_len,
        "edge": {
            "source_token_index": int(source_index),
            "source_token": test_src_text,
            "target_token_index": int(target_index),
            "target_token": test_tgt_text,
            "saliency_score": test_sal,
        },
        "relatedTrains": [
            {"trainSampleId": int(i), "probeCos": float(s)} for i, s in related
        ],
        "correlationPairs": all_pairs,
        "trainSampleDetails": details,
        "filterTag": session["filter_tag"],
        "sampleIdHint": infer_sample_id_from_report(report),
    }


def infer_sample_id_from_report(report: dict[str, Any]) -> str | None:
    meta = report.get("experiment_meta") or {}
    name = meta.get("fileName") or meta.get("report_file") or ""
    if name:
        try:
            return infer_sample_id(str(name))
        except Exception:
            pass
    task = meta.get("task_id")
    model = meta.get("model_name")
    if task and model:
        return f"{model}_{task}"
    return str(task) if task else None


def clear_gold_session() -> None:
    global _SESSION
    if _SESSION is not None:
        _release_cuda_memory(_SESSION.get("model"), reason="gold_session_clear")
    _SESSION = None
    gc.collect()
