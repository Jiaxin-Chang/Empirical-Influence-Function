"""Continue LoRA training on a **small annotated subset**, then eval with AI4Go line_hit.

Interactive loop intent:
  1) put newly annotated samples into ANNOTATION_CONTINUE_TRAIN_DATA
     (NOT the full original train set)
  2) continue-train a few AdamW steps from the current saliency adapter
  3) re-score EIF_TEST_DATA with line_hit_pre / line_hit_rec (same math as
     AI4Go ``hw_test_data/eval_gold_strip_three_models.py``)

Train data resolution (strict — never silently use the full bank train file):
  - ``ANNOTATION_CONTINUE_TRAIN_DATA`` / request ``trainData``
    (compact: input_ids + label + attention_edges; no ChatML re-encode)
  - OR request ``trainSampleIds`` sliced from ``EIF_TRAIN_DATA`` into a temp JSONL
    (source must also be compact)

Eval data:
  - ``EIF_TEST_DATA`` — JSONL with ``prompt``+``label|response`` (or chat ``input``+``output``)

CLI:
  python -m src.continue_train_eval \\
    --adapter-path $EIF_ADAPTER_PATH_SALIENCY \\
    --continue-train-data path/to/small_annotated.jsonl \\
    --test-data path/to/test.jsonl \\
    --max-steps 20 --lr 2e-5
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from src.intervention_experiment import (
    _left_truncate_train_row_for_bank,
    load_model_and_tokenizer,
    load_train_samples,
)
from src.bank_loss import BankLossConfig, compute_bank_loss, load_bank_loss_config
from src.eif_adapter_env import base_model_path_from_env


REPO_ROOT = Path(__file__).resolve().parent.parent
CHAT_STOP = "<|im_end|>"
# Without flash-attn, sdpa MATH fallback stores HxTxT per layer under training.
# Cap seq so continue-train fits on ~96GB; override with EIF_CONTINUE_MAX_SEQ_LEN.
_DEFAULT_CONTINUE_MAX_SEQ_LEN = 1536


@dataclass
class ContinueTrainConfig:
    adapter_path: str
    base_model_path: str | None
    train_data: str  # small annotated subset only
    test_data: str
    output_dir: str
    max_steps: int = 20
    learning_rate: float = 2e-5
    loss_mode: str = "ce_saliency"  # ce_only | ce_saliency
    eval_before: bool = True
    # Full EIF_TEST_DATA line_hit after train (off when only predicting current test).
    eval_after_full: bool = True
    # Optional: predict this test row after train (prompt/label or task_id lookup).
    current_test_task_id: str | None = None
    current_test_prompt: str | None = None
    current_test_label: str | None = None
    # Precomputed baseline line_hit JSONL (skip GPU eval_before when set + file exists).
    eval_before_cache: str | None = None
    max_new_tokens: int = 1024
    seed: int = 42
    train_sample_ids: list[int] = field(default_factory=list)
    # Optional full train JSONL used only when slicing by train_sample_ids.
    source_train_data: str | None = None
    also_truncate_score: bool = True
    # Left-truncate ChatML for CE+saliency continue-train (VRAM). 0 = no cap.
    max_seq_len: int = _DEFAULT_CONTINUE_MAX_SEQ_LEN


def _resolve_path(raw: str | None) -> str | None:
    if not raw or not str(raw).strip():
        return None
    expanded = os.path.expanduser(str(raw).strip())
    p = Path(expanded)
    if not p.is_absolute():
        for root in (Path.cwd(), REPO_ROOT):
            cand = (root / p).resolve()
            if cand.exists():
                return str(cand)
        return str((REPO_ROOT / p).resolve())
    return str(p.resolve())


def default_paths_from_env() -> dict[str, Any]:
    from src.eif_adapter_env import adapter_path_for_family

    # Small subset only — do NOT fall back to full EIF_TRAIN_DATA.
    continue_train = (os.environ.get("ANNOTATION_CONTINUE_TRAIN_DATA") or "").strip()
    source_train = (
        (os.environ.get("EIF_TRAIN_DATA") or "").strip()
        or (os.environ.get("ANNOTATION_TRAIN_DATA") or "").strip()
    )
    test = (os.environ.get("EIF_TEST_DATA") or "").strip()
    eval_before_cache = (
        (os.environ.get("EIF_CONTINUE_EVAL_BEFORE_CACHE") or "").strip()
        or (os.environ.get("EIF_EVAL_BEFORE_RESULTS") or "").strip()
    )
    out = (os.environ.get("EIF_CONTINUE_OUTPUT_DIR") or "").strip() or str(
        REPO_ROOT / "outputs" / "continue_trial"
    )
    adapter = (
        (os.environ.get("EIF_ADAPTER_PATH_SALIENCY") or "").strip()
        or adapter_path_for_family("saliency")
        or (os.environ.get("EIF_ADAPTER_PATH") or "").strip()
    )
    base = base_model_path_from_env() or (os.environ.get("EIF_BASE_MODEL_PATH") or "").strip() or None

    def _max_steps_default() -> int:
        raw = (os.environ.get("EIF_CONTINUE_MAX_STEPS") or "").strip()
        if raw:
            try:
                return max(1, int(raw))
            except ValueError:
                pass
        return 20

    def _lr_default() -> float:
        raw = (os.environ.get("EIF_CONTINUE_LR") or "").strip()
        if raw:
            try:
                return float(raw)
            except ValueError:
                pass
        return 2e-5

    return {
        "adapter_path": _resolve_path(adapter),
        "base_model_path": _resolve_path(base) if base else None,
        "continue_train_data": _resolve_path(continue_train),
        "source_train_data": _resolve_path(source_train),
        "test_data": _resolve_path(test),
        "eval_before_cache": _resolve_path(eval_before_cache),
        "output_dir": _resolve_path(out) or str(REPO_ROOT / "outputs" / "continue_trial"),
        "max_steps": _max_steps_default(),
        "learning_rate": _lr_default(),
    }


def _release_cuda():
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _evict_cached_models():
    try:
        from src.gold_live_attribution import clear_gold_session

        clear_gold_session()
    except Exception as exc:
        print(f"[continue-train] gold session clear skipped: {exc}", flush=True)
    try:
        from src.unlearn_pair_probe import _MODEL_CACHE, recover_pair_intervention

        recover_pair_intervention()
        # Move cached models off GPU before dropping refs (empty_cache alone is not enough).
        for key in list(_MODEL_CACHE.keys()):
            try:
                model, _tok = _MODEL_CACHE.pop(key)
            except KeyError:
                continue
            try:
                model.to("cpu")
            except Exception:
                pass
            del model
        _MODEL_CACHE.clear()
    except Exception as exc:
        print(f"[continue-train] unlearn cache clear skipped: {exc}", flush=True)
    _release_cuda()
    if torch.cuda.is_available():
        try:
            free_b, total_b = torch.cuda.mem_get_info()
            print(
                f"[continue-train] after evict: "
                f"free={free_b / 1e9:.2f}G / total={total_b / 1e9:.2f}G",
                flush=True,
            )
        except Exception:
            pass


def _device_of(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── AI4Go line_hit (from eval_gold_strip_three_models.py) ─────────────────────

def del_spaces(txt: str) -> str:
    return re.sub(r"\s", "", txt or "")


def to_lines(text: str) -> list[str]:
    rows: list[str] = []
    for raw in text.split("\n"):
        item = del_spaces(raw)
        if item and item not in "{}":
            rows.append(item)
    return rows


def cal_intersection(expects: list[str], actuals: list[str]) -> int:
    pool = list(actuals)
    cnt = 0
    for item in expects:
        if item in pool:
            cnt += 1
            pool.remove(item)
    return cnt


def line_hit(expect: str, actual: str, method: str) -> float:
    expects = to_lines(expect)
    actuals = to_lines(actual)
    denominator = len(expects) if method == "recall" else len(actuals)
    if denominator == 0:
        return 0.0
    return min(1.0, cal_intersection(expects, actuals) / denominator) * 100.0


def truncate_predict_to_label_lines(label: str, predict: str) -> str:
    n = len(label.splitlines())
    return "\n".join(predict.splitlines()[:n])


def remove_leading_thinking(text: str) -> tuple[str, bool]:
    """Best-effort strip of leading <think>…</think> blocks (AI4Go-compatible)."""
    if not text:
        return "", False
    pattern = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
    new, n = pattern.subn("", text, count=1)
    if n:
        return new, True
    # Unclosed thinking at start
    m = re.match(r"^\s*<think>(.*)$", text, re.DOTALL | re.IGNORECASE)
    if m:
        return "", True
    return text, False


def _render_eval_prompt(tokenizer, prompt: str) -> str:
    """ChatML eval prompt (thinking off), aligned with AI4Go generate path."""
    from src.intervention_experiment import _render_qwen_eval_prompt

    return _render_qwen_eval_prompt(tokenizer, prompt, system="")


# ── Train data: small subset only ─────────────────────────────────────────────

def slice_train_jsonl_by_ids(
    source_path: str,
    ids: list[int],
    dest_path: Path,
) -> str:
    """Copy selected 0-based line indices into a small continue-train JSONL."""
    wanted = sorted({int(i) for i in ids if int(i) >= 0})
    if not wanted:
        raise ValueError("trainSampleIds is empty")
    want_set = set(wanted)
    max_id = max(wanted)
    kept = 0
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(source_path, encoding="utf-8") as src, dest_path.open("w", encoding="utf-8") as out:
        for i, line in enumerate(src):
            if not line.strip():
                continue
            if i in want_set:
                out.write(line if line.endswith("\n") else line + "\n")
                kept += 1
            if i >= max_id and kept == len(want_set):
                break
    if kept == 0:
        raise ValueError(
            f"No rows extracted for ids={wanted} from {source_path}"
        )
    if kept < len(want_set):
        print(
            f"[continue-train][WARN] requested {len(want_set)} ids, got {kept}",
            flush=True,
        )
    print(
        f"[continue-train] sliced {kept} samples → {dest_path}",
        flush=True,
    )
    return str(dest_path.resolve())


def resolve_continue_train_jsonl(cfg: ContinueTrainConfig) -> str:
    """Return path to the small continue-train JSONL (never the full bank by accident)."""
    src = (cfg.source_train_data or "").strip()
    train_path = (cfg.train_data or "").strip()

    # Explicit trainSampleIds always win: slice a small subset from source (or from
    # train_data if it is the only available JSONL).
    if cfg.train_sample_ids:
        slice_src = src or train_path
        if not slice_src or not Path(slice_src).is_file():
            raise ValueError(
                "trainSampleIds given but source train JSONL missing "
                "(set EIF_TRAIN_DATA / request sourceTrainData)."
            )
        dest = Path(cfg.output_dir) / "continue_train_subset.jsonl"
        return slice_train_jsonl_by_ids(slice_src, cfg.train_sample_ids, dest)

    if train_path and Path(train_path).is_file():
        # Guard: refuse silently training on the full bank train file.
        if (
            src
            and Path(src).is_file()
            and Path(train_path).resolve() == Path(src).resolve()
        ):
            raise ValueError(
                "continue-train refuses to use the full EIF_TRAIN_DATA as the "
                "training set. Set ANNOTATION_CONTINUE_TRAIN_DATA to a small annotated "
                "JSONL, or pass trainSampleIds to slice a subset."
            )
        return train_path

    raise ValueError(
        "No small continue-train set. Provide ANNOTATION_CONTINUE_TRAIN_DATA "
        "(annotated subset JSONL) or trainSampleIds."
    )


def _compact_batch(sample: dict, device: torch.device) -> dict[str, torch.Tensor]:
    ids = sample.get("input_ids")
    labels = sample.get("labels", sample.get("label"))
    if not isinstance(ids, list) or not ids:
        raise ValueError("continue-train sample missing compact input_ids")
    if not isinstance(labels, list) or len(labels) != len(ids):
        raise ValueError("continue-train sample need label/labels list aligned with input_ids")
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    label_t = torch.tensor([labels], dtype=torch.long, device=device)
    attn = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "labels": label_t, "attention_mask": attn}


def _continue_max_seq_len(cfg: ContinueTrainConfig) -> int | None:
    raw = (os.environ.get("EIF_CONTINUE_MAX_SEQ_LEN") or "").strip()
    if raw:
        try:
            v = int(raw)
        except ValueError:
            v = int(cfg.max_seq_len)
    else:
        v = int(cfg.max_seq_len)
    return None if v <= 0 else v


def _sample_batch(
    sample: dict,
    tokenizer,
    device: torch.device,
    *,
    max_seq_len: int | None = None,
) -> tuple[dict[str, torch.Tensor], list | None]:
    del tokenizer  # compact path does not re-encode
    edges = sample.get("attention_edges") or sample.get("edges")
    edges_list = edges if isinstance(edges, list) else None
    batch = _compact_batch(sample, device)
    if max_seq_len is not None and int(batch["input_ids"].size(1)) > int(max_seq_len):
        batch, edges_list, dropped = _left_truncate_train_row_for_bank(
            batch, edges_list, int(max_seq_len),
        )
        if dropped:
            print(
                f"[continue-train] left-truncated {dropped} prompt tokens → "
                f"seq={int(batch['input_ids'].size(1))} "
                f"(edges kept={0 if not edges_list else len(edges_list)})",
                flush=True,
            )
    return batch, edges_list


def _trainable_lora_params(model):
    params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
    if not params:
        params = [p for p in model.parameters() if p.requires_grad]
    return params


def run_continue_training(
    model,
    tokenizer,
    train_samples: list[dict],
    *,
    cfg: ContinueTrainConfig,
    bank_cfg: BankLossConfig,
    progress_cb=None,
) -> dict[str, Any]:
    device = _device_of(model)
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.eval()
    # Gradient checkpointing only runs in train mode; keep it on.
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except TypeError:
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass

    params = _trainable_lora_params(model)
    if not params:
        raise RuntimeError("No trainable LoRA parameters found for continue-train.")
    opt = torch.optim.AdamW(params, lr=float(cfg.learning_rate))

    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    n_train = len(train_samples)
    if n_train == 0:
        raise ValueError("No train samples to continue-train on.")

    max_seq = _continue_max_seq_len(cfg)
    if max_seq is not None:
        print(
            f"[continue-train] max_seq_len={max_seq} "
            f"(left-truncate; set EIF_CONTINUE_MAX_SEQ_LEN=0 to disable)",
            flush=True,
        )

    losses: list[float] = []
    t0 = time.time()
    step = 0
    while step < int(cfg.max_steps):
        sample = train_samples[step % n_train]
        batch, edges = _sample_batch(
            sample, tokenizer, device, max_seq_len=max_seq,
        )
        seq_len = int(batch["input_ids"].size(1))
        opt.zero_grad(set_to_none=True)
        if step == 0 and torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        loss, mode_used = compute_bank_loss(
            model,
            batch,
            device=device,
            cfg=bank_cfg,
            edges=edges,
            special_ids=special_ids,
        )
        if loss is None:
            step += 1
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        loss_v = float(loss.detach().float().cpu())
        losses.append(loss_v)
        step += 1
        if progress_cb is not None:
            progress_cb(step, int(cfg.max_steps), loss_v, mode_used)
        if step == 1 or step % 10 == 0 or step == int(cfg.max_steps):
            mem_note = ""
            if torch.cuda.is_available():
                try:
                    peak = torch.cuda.max_memory_allocated() / 1e9
                    free_b, total_b = torch.cuda.mem_get_info()
                    mem_note = f" peak={peak:.1f}G free={free_b/1e9:.1f}/{total_b/1e9:.1f}G"
                except Exception:
                    pass
            print(
                f"[continue-train] step {step}/{cfg.max_steps} "
                f"loss={loss_v:.4f} mode={mode_used} seq={seq_len} n_subset={n_train}"
                f"{mem_note}",
                flush=True,
            )
        del batch, loss
        if step % 20 == 0:
            _release_cuda()

    model.eval()
    return {
        "steps": step,
        "meanLoss": float(sum(losses) / max(1, len(losses))),
        "lastLoss": float(losses[-1]) if losses else None,
        "elapsedSec": round(time.time() - t0, 2),
        "lossModeUsed": bank_cfg.loss_mode,
        "nTrainSamples": n_train,
        "maxSeqLen": max_seq,
    }

# ── Eval dataset + generation (AI4Go-aligned) ─────────────────────────────────

def load_eval_samples(jsonl_path: str) -> list[dict]:
    """Load eval rows: prefer prompt+label/response; else chat input/output."""
    src_name = Path(jsonl_path).name
    rows: list[dict] = []
    with open(jsonl_path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            prompt = obj.get("prompt")
            label = obj.get("response", obj.get("label"))
            if isinstance(prompt, str) and isinstance(label, str):
                rows.append({
                    "task_id": obj.get("task_id", f"row_{line_no}"),
                    "prompt": prompt,
                    "label": label,
                    "source_file": obj.get("source_file") or src_name,
                    "source_line": int(obj.get("source_line") or line_no),
                })
                continue
            # Chat fallback used by EIF load_samples-style files.
            user = obj.get("input") or ""
            if isinstance(obj.get("messages"), list):
                for m in obj["messages"]:
                    if isinstance(m, dict) and m.get("role") == "user":
                        user = m.get("content") or user
                    if isinstance(m, dict) and m.get("role") == "assistant":
                        label = m.get("content") or label
            gold = label if isinstance(label, str) else (obj.get("output") or "")
            if isinstance(user, str) and user and isinstance(gold, str) and gold:
                rows.append({
                    "task_id": obj.get("task_id", f"row_{line_no}"),
                    "prompt": user,
                    "label": gold,
                    "source_file": obj.get("source_file") or src_name,
                    "source_line": int(obj.get("source_line") or line_no),
                })
    if not rows:
        raise ValueError(f"No eval samples in {jsonl_path}")
    return rows


def load_eval_before_cache(
    cache_path: str,
    eval_samples: list[dict],
    *,
    also_truncate_score: bool = True,
) -> dict[str, Any]:
    """Build the same summary shape as ``evaluate_line_hit`` from a precomputed JSONL.

    Each cache line should include ``line_hit_pre`` / ``line_hit_rec`` (and optionally
    trunc variants). Rows are matched to ``eval_samples`` by ``task_id`` first, then
    by positional index.
    """
    path = Path(cache_path)
    if not path.is_file():
        raise FileNotFoundError(f"eval_before cache not found: {cache_path}")

    cache_rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                continue
            obj.setdefault("_cache_line", line_no)
            cache_rows.append(obj)
    if not cache_rows:
        raise ValueError(f"eval_before cache is empty: {cache_path}")

    by_tid: dict[str, dict[str, Any]] = {}
    for row in cache_rows:
        tid = row.get("task_id")
        if tid is None:
            continue
        key = str(tid).strip()
        if key and key not in by_tid:
            by_tid[key] = row

    pres: list[float] = []
    recs: list[float] = []
    trunc_pres: list[float] = []
    trunc_recs: list[float] = []
    per_sample: list[dict[str, Any]] = []
    missing: list[str] = []
    matched_by = {"task_id": 0, "index": 0}

    for i, sample in enumerate(eval_samples):
        tid = str(sample.get("task_id") or "").strip()
        row = by_tid.get(tid) if tid else None
        how = "task_id"
        if row is None and i < len(cache_rows):
            row = cache_rows[i]
            how = "index"
        if row is None:
            missing.append(tid or f"index:{i}")
            continue
        matched_by[how] = matched_by.get(how, 0) + 1

        if "line_hit_pre" not in row or "line_hit_rec" not in row:
            raise ValueError(
                f"Cache row for task_id={tid or i} missing line_hit_pre/line_hit_rec "
                f"(file={cache_path})"
            )
        pre = float(row["line_hit_pre"])
        rec = float(row["line_hit_rec"])
        pres.append(pre)
        recs.append(rec)
        entry: dict[str, Any] = {
            "index": i,
            "task_id": sample.get("task_id") or row.get("task_id"),
            "prompt": sample.get("prompt") or row.get("prompt"),
            "label": sample.get("label") or row.get("label") or row.get("response"),
            "predict": row.get("predict"),
            "source_file": sample.get("source_file") or row.get("source_file"),
            "source_line": sample.get("source_line") or row.get("source_line"),
            "finish_reason": row.get("finish_reason"),
            "prompt_tokens": row.get("prompt_tokens"),
            "generated_tokens": row.get("generated_tokens"),
            "thinking_enabled": bool(row.get("thinking_enabled", False)),
            "score_mode": row.get("score_mode") or "full",
            "line_hit_pre": round(pre, 4),
            "line_hit_rec": round(rec, 4),
            "cache_match": how,
        }
        if also_truncate_score:
            if "line_hit_pre_trunc" in row and "line_hit_rec_trunc" in row:
                t_pre = float(row["line_hit_pre_trunc"])
                t_rec = float(row["line_hit_rec_trunc"])
            else:
                # Fall back to full scores when cache has no trunc fields.
                t_pre, t_rec = pre, rec
            trunc_pres.append(t_pre)
            trunc_recs.append(t_rec)
            entry["line_hit_pre_trunc"] = round(t_pre, 4)
            entry["line_hit_rec_trunc"] = round(t_rec, 4)
        per_sample.append(entry)

    if missing:
        raise ValueError(
            f"eval_before cache missing {len(missing)}/{len(eval_samples)} eval rows "
            f"(examples={missing[:5]}). Cache={cache_path}"
        )
    if len(per_sample) != len(eval_samples):
        raise ValueError(
            f"eval_before cache matched {len(per_sample)} rows but eval has "
            f"{len(eval_samples)} (cache={cache_path})"
        )

    summary: dict[str, Any] = {
        "n": len(per_sample),
        "line_hit_pre": round(sum(pres) / max(1, len(pres)), 4),
        "line_hit_rec": round(sum(recs) / max(1, len(recs)), 4),
        "perSample": per_sample,
        "source": "cache",
        "cachePath": str(path.resolve()),
        "matchedBy": matched_by,
    }
    if also_truncate_score and trunc_pres:
        summary["line_hit_pre_trunc"] = round(sum(trunc_pres) / len(trunc_pres), 4)
        summary["line_hit_rec_trunc"] = round(sum(trunc_recs) / len(trunc_recs), 4)
    print(
        f"[continue-eval] loaded eval_before cache n={summary['n']} "
        f"pre={summary['line_hit_pre']} rec={summary['line_hit_rec']} "
        f"match={matched_by} path={path}",
        flush=True,
    )
    return summary


@torch.no_grad()
def generate_one(tokenizer, model, prompt: str, max_new_tokens: int) -> dict[str, Any]:
    text = _render_eval_prompt(tokenizer, prompt)
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    device = _device_of(model)
    encoded = {k: v.to(device) for k, v in encoded.items()}
    prompt_tokens = int(encoded["input_ids"].shape[1])

    eos_ids = [tokenizer.eos_token_id]
    im_end_id = tokenizer.convert_tokens_to_ids(CHAT_STOP)
    if isinstance(im_end_id, int) and im_end_id >= 0 and im_end_id not in eos_ids:
        eos_ids.append(im_end_id)

    out = model.generate(
        **encoded,
        max_new_tokens=int(max_new_tokens),
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=eos_ids,
    )
    gen = out[0, prompt_tokens:]
    generated_list = gen.tolist()
    text_out = tokenizer.decode(gen, skip_special_tokens=True)
    predict, removed = remove_leading_thinking(text_out)
    finish_reason = "stop" if generated_list and generated_list[-1] in eos_ids else "length"
    row = {
        "predict": predict,
        "finish_reason": finish_reason,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": len(generated_list),
    }
    if removed:
        row["predict_raw"] = text_out
        row["thinking_removed"] = True
    return row


def evaluate_line_hit(
    model,
    tokenizer,
    samples: list[dict],
    *,
    max_new_tokens: int = 1024,
    also_truncate_score: bool = True,
    progress_cb=None,
    source_file: str | None = None,
) -> dict[str, Any]:
    """Return mean line_hit_pre / line_hit_rec (and optional trunc variants).

    ``perSample`` rows follow the AI4Go score-dump shape (prompt/label/predict +
    line_hit_*), so they can be written as a JSONL artifact after continue-train.
    """
    model.eval()
    pres: list[float] = []
    recs: list[float] = []
    trunc_pres: list[float] = []
    trunc_recs: list[float] = []
    per_sample: list[dict[str, Any]] = []
    truncated_cases = 0
    default_source = source_file or ""

    for i, sample in enumerate(samples):
        gen = generate_one(tokenizer, model, sample["prompt"], max_new_tokens)
        pre = line_hit(sample["label"], gen["predict"], "precision")
        rec = line_hit(sample["label"], gen["predict"], "recall")
        pres.append(pre)
        recs.append(rec)
        row: dict[str, Any] = {
            "task_id": sample.get("task_id"),
            "prompt": sample["prompt"],
            "label": sample["label"],
            "predict": gen["predict"],
            "source_file": sample.get("source_file") or default_source or None,
            "source_line": sample.get("source_line", i + 1),
            "finish_reason": gen.get("finish_reason"),
            "prompt_tokens": gen.get("prompt_tokens"),
            "generated_tokens": gen.get("generated_tokens"),
            "thinking_enabled": False,
            "score_mode": "full",
            "line_hit_pre": round(pre, 4),
            "line_hit_rec": round(rec, 4),
            "index": i,
        }
        if gen.get("thinking_removed"):
            row["thinking_removed"] = True
            if gen.get("predict_raw") is not None:
                row["predict_raw"] = gen["predict_raw"]
        if also_truncate_score:
            predict_trunc = truncate_predict_to_label_lines(sample["label"], gen["predict"])
            if predict_trunc != gen["predict"]:
                truncated_cases += 1
                row["predict_trunc"] = predict_trunc
            t_pre = line_hit(sample["label"], predict_trunc, "precision")
            t_rec = line_hit(sample["label"], predict_trunc, "recall")
            trunc_pres.append(t_pre)
            trunc_recs.append(t_rec)
            row["line_hit_pre_trunc"] = round(t_pre, 4)
            row["line_hit_rec_trunc"] = round(t_rec, 4)
        per_sample.append(row)
        if progress_cb is not None:
            progress_cb(i + 1, len(samples), row)
        if (i + 1) % 10 == 0 or (i + 1) == len(samples):
            msg = (
                f"[continue-eval] {i + 1}/{len(samples)} "
                f"pre={sum(pres)/len(pres):.2f} rec={sum(recs)/len(recs):.2f}"
            )
            if trunc_pres:
                msg += (
                    f" trunc_pre={sum(trunc_pres)/len(trunc_pres):.2f} "
                    f"trunc_rec={sum(trunc_recs)/len(trunc_recs):.2f}"
                )
            print(msg, flush=True)

    summary: dict[str, Any] = {
        "n": len(samples),
        "line_hit_pre": round(sum(pres) / max(1, len(pres)), 4),
        "line_hit_rec": round(sum(recs) / max(1, len(recs)), 4),
        "perSample": per_sample,
    }
    if also_truncate_score and trunc_pres:
        summary["line_hit_pre_trunc"] = round(sum(trunc_pres) / len(trunc_pres), 4)
        summary["line_hit_rec_trunc"] = round(sum(trunc_recs) / len(trunc_recs), 4)
        summary["truncated_cases"] = truncated_cases
    return summary


def write_eval_results_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> str:
    """Write one AI4Go-style eval record per line (prompt/label/predict + scores)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            if not isinstance(row, dict):
                continue
            # Stable key order for readability; drop None-only noise keys later.
            payload = {k: v for k, v in row.items() if v is not None or k in (
                "task_id", "prompt", "label", "predict", "line_hit_pre", "line_hit_rec",
            )}
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    print(f"[continue-eval] wrote {len(rows)} rows → {out.resolve()}", flush=True)
    return str(out.resolve())


def _slim_eval_summary(summary: dict[str, Any] | None) -> dict[str, Any] | None:
    """Drop bulky perSample from meta/detail summaries (full rows live in JSONL)."""
    if summary is None:
        return None
    out = {k: v for k, v in summary.items() if k != "perSample"}
    per = summary.get("perSample") or []
    out["nPerSample"] = len(per) if isinstance(per, list) else 0
    return out


def save_adapter(model, tokenizer, output_dir: str, meta: dict[str, Any]) -> str:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(str(out))
    else:
        raise RuntimeError("Model has no save_pretrained; expected PeftModel.")
    try:
        tokenizer.save_pretrained(str(out))
    except Exception:
        pass
    (out / "continue_train_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return str(out.resolve())


def _metric_delta(before: dict | None, after: dict | None) -> dict[str, float | None]:
    if before is None or after is None:
        return {
            "line_hit_pre": None,
            "line_hit_rec": None,
            "line_hit_pre_trunc": None,
            "line_hit_rec_trunc": None,
        }
    def d(key: str):
        a, b = after.get(key), before.get(key)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return float(a) - float(b)
        return None
    return {
        "line_hit_pre": d("line_hit_pre"),
        "line_hit_rec": d("line_hit_rec"),
        "line_hit_pre_trunc": d("line_hit_pre_trunc"),
        "line_hit_rec_trunc": d("line_hit_rec_trunc"),
    }


def summarize_per_sample_line_hit_deltas(
    before: dict[str, Any] | None,
    after: dict[str, Any],
    *,
    eps: float = 1e-9,
) -> dict[str, Any]:
    """Compare per-sample line_hit on EIF_TEST_DATA (eval set), not the continue subset.

    Returns index lists that rose / fell / stayed flat for pre and rec.
    """
    empty = {
        "nCompared": 0,
        "line_hit_pre": {"increased": [], "decreased": [], "unchanged": []},
        "line_hit_rec": {"increased": [], "decreased": [], "unchanged": []},
        "rows": [],
    }
    if before is None:
        return empty
    b_rows = {int(r["index"]): r for r in (before.get("perSample") or []) if isinstance(r, dict)}
    a_rows = {int(r["index"]): r for r in (after.get("perSample") or []) if isinstance(r, dict)}
    common = sorted(set(b_rows) & set(a_rows))
    if not common:
        return empty

    out_pre = {"increased": [], "decreased": [], "unchanged": []}
    out_rec = {"increased": [], "decreased": [], "unchanged": []}
    rows: list[dict[str, Any]] = []

    def _bucket(metric: str, buckets: dict[str, list], idx: int, b_v: float, a_v: float):
        dv = float(a_v) - float(b_v)
        entry = {
            "index": idx,
            "task_id": a_rows[idx].get("task_id") or b_rows[idx].get("task_id"),
            "before": round(float(b_v), 4),
            "after": round(float(a_v), 4),
            "delta": round(dv, 4),
        }
        if dv > eps:
            buckets["increased"].append(entry)
        elif dv < -eps:
            buckets["decreased"].append(entry)
        else:
            buckets["unchanged"].append(entry)
        return entry

    for idx in common:
        b, a = b_rows[idx], a_rows[idx]
        pre_e = _bucket(
            "line_hit_pre", out_pre, idx,
            float(b.get("line_hit_pre") or 0.0),
            float(a.get("line_hit_pre") or 0.0),
        )
        rec_e = _bucket(
            "line_hit_rec", out_rec, idx,
            float(b.get("line_hit_rec") or 0.0),
            float(a.get("line_hit_rec") or 0.0),
        )
        rows.append({
            "index": idx,
            "task_id": a.get("task_id") or b.get("task_id"),
            "line_hit_pre": pre_e,
            "line_hit_rec": rec_e,
        })

    return {
        "nCompared": len(common),
        "line_hit_pre": out_pre,
        "line_hit_rec": out_rec,
        "rows": rows,
    }


def _print_per_sample_delta_report(report: dict[str, Any]) -> None:
    n = int(report.get("nCompared") or 0)
    print(f"[continue-eval] per-sample line_hit deltas on EIF_TEST_DATA (n={n}):", flush=True)
    for metric in ("line_hit_pre", "line_hit_rec"):
        block = report.get(metric) or {}
        up = block.get("increased") or []
        down = block.get("decreased") or []
        flat = block.get("unchanged") or []
        up_idx = [e["index"] for e in up]
        down_idx = [e["index"] for e in down]
        print(
            f"  {metric}: ↑{len(up)} {up_idx[:40]}{'…' if len(up_idx) > 40 else ''} "
            f"↓{len(down)} {down_idx[:40]}{'…' if len(down_idx) > 40 else ''} "
            f"→{len(flat)} unchanged",
            flush=True,
        )
        for e in up[:10]:
            print(
                f"    ↑ idx={e['index']} {e.get('task_id') or ''} "
                f"{e['before']}→{e['after']} (Δ{e['delta']:+})",
                flush=True,
            )
        for e in down[:10]:
            print(
                f"    ↓ idx={e['index']} {e.get('task_id') or ''} "
                f"{e['before']}→{e['after']} (Δ{e['delta']:+})",
                flush=True,
            )


def _pick_continue_attn_implementation() -> str:
    """Continue-train does NOT need output_attentions (saliency recomputes one layer).

    Using eager here OOMs at seq≈2k on 80–96GB GPUs because every decoder layer
    saves HxTxT for backward. Prefer flash_attention_2, else sdpa.
    """
    raw = (os.environ.get("EIF_CONTINUE_ATTN_IMPL") or "").strip().lower()
    if raw in ("eager", "sdpa", "flash_attention_2"):
        return raw
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except Exception:
        return "sdpa"


def _resolve_current_test_sample(
    cfg: ContinueTrainConfig,
    eval_samples: list[dict],
) -> dict[str, Any] | None:
    """Pick the report's current test row for post-train greedy decode."""
    tid = (cfg.current_test_task_id or "").strip()
    if tid:
        for row in eval_samples:
            if str(row.get("task_id") or "") == tid:
                return row
    prompt = (cfg.current_test_prompt or "").strip()
    label = (cfg.current_test_label or "").strip()
    if prompt and label:
        return {
            "task_id": tid or None,
            "prompt": prompt,
            "label": label,
        }
    if eval_samples:
        return eval_samples[0]
    return None


def run_continue_line_hit_compare(cfg: ContinueTrainConfig, progress_cb=None) -> dict[str, Any]:
    """Eval-only: baseline (cache) vs continued adapter on full EIF_TEST_DATA."""
    from src.eif_adapter_env import get_active_adapter_override

    continued = (get_active_adapter_override() or "").strip()
    if not continued or not Path(continued).is_dir():
        raise ValueError(
            "No continued adapter active. Run 续训 first, or set continuedAdapterPath."
        )
    if not cfg.test_data or not Path(cfg.test_data).is_file():
        raise FileNotFoundError(f"test_data not found: {cfg.test_data!r}")

    def _prog(stage: str, message: str, **extra):
        if progress_cb is not None:
            progress_cb(stage, message, extra)

    eval_samples = load_eval_samples(cfg.test_data)
    cache_path = (cfg.eval_before_cache or "").strip()
    cache_ok = bool(cache_path and Path(cache_path).is_file())

    _evict_cached_models()
    before = None
    eval_before_source = None
    if cache_ok:
        _prog("eval_before", "Loading baseline line_hit from cache…")
        before = load_eval_before_cache(
            cache_path,
            eval_samples,
            also_truncate_score=cfg.also_truncate_score,
        )
        eval_before_source = "cache"
    elif cfg.eval_before:
        _prog("eval_before", "Evaluating baseline adapter (line_hit)…")
        base_cfg = ContinueTrainConfig(
            adapter_path=cfg.adapter_path,
            base_model_path=cfg.base_model_path,
            train_data=cfg.train_data,
            test_data=cfg.test_data,
            output_dir=cfg.output_dir,
            max_new_tokens=cfg.max_new_tokens,
            also_truncate_score=cfg.also_truncate_score,
        )
        attn_impl = _pick_continue_attn_implementation()
        model, tokenizer = load_model_and_tokenizer(
            model_path=base_cfg.adapter_path,
            base_model_path=base_cfg.base_model_path,
            attn_implementation=attn_impl,
        )
        before = evaluate_line_hit(
            model, tokenizer, eval_samples,
            max_new_tokens=cfg.max_new_tokens,
            also_truncate_score=cfg.also_truncate_score,
            source_file=Path(cfg.test_data).name,
        )
        eval_before_source = "live"
        del model
        _evict_cached_models()
    else:
        raise ValueError(
            "line_hit compare needs EIF_CONTINUE_EVAL_BEFORE_CACHE "
            "or evalBefore=true for live baseline."
        )

    _prog("eval_after", "Evaluating continued adapter (line_hit)…")
    attn_impl = _pick_continue_attn_implementation()
    model, tokenizer = load_model_and_tokenizer(
        model_path=continued,
        base_model_path=cfg.base_model_path,
        attn_implementation=attn_impl,
    )
    after = evaluate_line_hit(
        model, tokenizer, eval_samples,
        max_new_tokens=cfg.max_new_tokens,
        also_truncate_score=cfg.also_truncate_score,
        source_file=Path(cfg.test_data).name,
        progress_cb=lambda i, n, _r: _prog(
            "eval_after", f"Eval after {i}/{n}", done=i, total=n,
        ),
    )
    delta = _metric_delta(before, after)
    per_sample_deltas = summarize_per_sample_line_hit_deltas(before, after)
    _print_per_sample_delta_report(per_sample_deltas)

    out_root = Path(cfg.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    after_jsonl = write_eval_results_jsonl(
        out_root / "continue_eval_after_compare.jsonl",
        list(after.get("perSample") or []),
    )

    meta = {
        "mode": "line_hit_compare",
        "continuedAdapterPath": continued,
        "evalBeforeSource": eval_before_source,
        "before": _slim_eval_summary(before),
        "after": _slim_eval_summary(after),
        "delta": delta,
        "perSampleDeltas": {
            "nCompared": per_sample_deltas["nCompared"],
            "line_hit_pre": {
                "increased": [e["index"] for e in per_sample_deltas["line_hit_pre"]["increased"]],
                "decreased": [e["index"] for e in per_sample_deltas["line_hit_pre"]["decreased"]],
                "nIncreased": len(per_sample_deltas["line_hit_pre"]["increased"]),
                "nDecreased": len(per_sample_deltas["line_hit_pre"]["decreased"]),
            },
            "line_hit_rec": {
                "increased": [e["index"] for e in per_sample_deltas["line_hit_rec"]["increased"]],
                "decreased": [e["index"] for e in per_sample_deltas["line_hit_rec"]["decreased"]],
                "nIncreased": len(per_sample_deltas["line_hit_rec"]["increased"]),
                "nDecreased": len(per_sample_deltas["line_hit_rec"]["decreased"]),
            },
        },
        "evalAfterJsonl": after_jsonl,
    }
    _prog("completed", "line_hit compare finished.", result=meta)
    del model
    _evict_cached_models()
    return {"status": "success", **meta}


def run_continue_train_and_eval(cfg: ContinueTrainConfig, progress_cb=None) -> dict[str, Any]:
    if not cfg.adapter_path or not Path(cfg.adapter_path).is_dir():
        raise FileNotFoundError(f"adapter_path not found: {cfg.adapter_path!r}")
    if not cfg.test_data or not Path(cfg.test_data).is_file():
        raise FileNotFoundError(f"test_data not found: {cfg.test_data!r}")

    train_jsonl = resolve_continue_train_jsonl(cfg)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    _evict_cached_models()
    torch.manual_seed(int(cfg.seed))

    attn_impl = _pick_continue_attn_implementation()
    print(f"[continue-train] loading adapter={cfg.adapter_path}", flush=True)
    print(f"[continue-train] subset={train_jsonl}", flush=True)
    print(f"[continue-eval] test={cfg.test_data}", flush=True)
    print(
        f"[continue-train] attn_implementation={attn_impl} "
        f"(saliency uses single-layer recompute; avoid eager)",
        flush=True,
    )

    try:
        model, tokenizer = load_model_and_tokenizer(
            model_path=cfg.adapter_path,
            base_model_path=cfg.base_model_path,
            attn_implementation=attn_impl,
        )
    except Exception as exc:
        if attn_impl != "sdpa":
            print(
                f"[continue-train][WARN] load with {attn_impl!r} failed ({exc}); "
                "falling back to sdpa",
                flush=True,
            )
            attn_impl = "sdpa"
            model, tokenizer = load_model_and_tokenizer(
                model_path=cfg.adapter_path,
                base_model_path=cfg.base_model_path,
                attn_implementation=attn_impl,
            )
        else:
            raise
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.requires_grad_(True)
    # Re-assert checkpointing after PEFT wrap (critical with long ChatML).
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except TypeError:
        model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        model.config.use_cache = False

    train_samples = load_train_samples(train_jsonl)
    eval_samples = load_eval_samples(cfg.test_data)
    if not train_samples:
        raise ValueError(f"No train samples in {train_jsonl}")

    n_edges = sum(1 for s in train_samples if s.get("attention_edges") or s.get("edges"))
    print(
        f"[continue-train] subset_size={len(train_samples)} "
        f"(with_edges={n_edges}/{len(train_samples)}) "
        f"eval_n={len(eval_samples)} steps={cfg.max_steps} lr={cfg.learning_rate}",
        flush=True,
    )
    if n_edges == 0 and cfg.loss_mode == "ce_saliency":
        print(
            "[continue-train][WARN] no attention_edges in subset; "
            "ce_saliency will fall back to CE-only per sample.",
            flush=True,
        )

    bank_cfg = load_bank_loss_config(
        cfg.adapter_path,
        Path(cfg.adapter_path).name,
        loss_mode_override=cfg.loss_mode,
    )
    bank_cfg.loss_mode = cfg.loss_mode if cfg.loss_mode in ("ce_only", "ce_saliency") else bank_cfg.loss_mode
    # Continue-train often wants a smaller λ than the original SFT (default 1.5
    # tends to hurt generation on tiny subsets). Prefer EIF_CONTINUE_SALIENCY_LAMBDA,
    # then EIF_BANK_SALIENCY_LAMBDA; leave unset to keep adapter/bank defaults.
    cont_lam = (os.environ.get("EIF_CONTINUE_SALIENCY_LAMBDA") or "").strip()
    bank_lam = (os.environ.get("EIF_BANK_SALIENCY_LAMBDA") or "").strip()
    lam_override = cont_lam or bank_lam
    if lam_override:
        try:
            bank_cfg.saliency_lambda = float(lam_override)
        except ValueError:
            print(
                f"[continue-train][WARN] invalid saliency λ override {lam_override!r}; "
                f"keeping λ={bank_cfg.saliency_lambda}",
                flush=True,
            )
    print(
        f"[continue-train] bank loss_mode={bank_cfg.loss_mode} "
        f"type={bank_cfg.saliency_loss_type} λ={bank_cfg.saliency_lambda}",
        flush=True,
    )

    def _prog(stage: str, message: str, **extra):
        if progress_cb is not None:
            progress_cb(stage, message, extra)

    before = None
    eval_before_source = None
    cache_path = (cfg.eval_before_cache or "").strip()
    cache_ok = bool(cache_path and Path(cache_path).is_file())

    if cache_ok:
        _prog("eval_before", f"Loading baseline line_hit from cache (skip GPU)…")
        before = load_eval_before_cache(
            cache_path,
            eval_samples,
            also_truncate_score=cfg.also_truncate_score,
        )
        eval_before_source = "cache"
        print(
            f"[continue-eval] BEFORE (cache) line_hit_pre={before['line_hit_pre']} "
            f"line_hit_rec={before['line_hit_rec']}",
            flush=True,
        )
        # Model was just loaded for training; still clear allocator before saliency.
        model.zero_grad(set_to_none=True)
        _release_cuda()
    elif cfg.eval_before:
        _prog("eval_before", "Evaluating baseline adapter (line_hit)…")
        before = evaluate_line_hit(
            model, tokenizer, eval_samples,
            max_new_tokens=cfg.max_new_tokens,
            also_truncate_score=cfg.also_truncate_score,
            source_file=Path(cfg.test_data).name,
            progress_cb=lambda i, n, _r: _prog(
                "eval_before", f"Eval before {i}/{n}", done=i, total=n,
            ),
        )
        eval_before_source = "live"
        print(
            f"[continue-eval] BEFORE line_hit_pre={before['line_hit_pre']} "
            f"line_hit_rec={before['line_hit_rec']}",
            flush=True,
        )
        # Free generate KV / allocator fragmentation before saliency train step.
        model.zero_grad(set_to_none=True)
        _release_cuda()
        print("[continue-train] empty_cache after eval_before", flush=True)
    elif cache_path:
        print(
            f"[continue-train][WARN] eval_before cache set but file missing: {cache_path}; "
            "skipping baseline eval",
            flush=True,
        )

    _prog("training", f"Continue-training {cfg.max_steps} steps on {len(train_samples)} samples…")
    if train_samples:
        try:
            seq_lens = [len(s.get("input_ids") or []) for s in train_samples]
            print(
                f"[continue-train] subset seq_len min/max/mean="
                f"{min(seq_lens)}/{max(seq_lens)}/{sum(seq_lens)/len(seq_lens):.0f}",
                flush=True,
            )
        except Exception:
            pass
    if torch.cuda.is_available():
        try:
            free_b, total_b = torch.cuda.mem_get_info()
            print(
                f"[continue-train] before train step: "
                f"free={free_b / 1e9:.2f}G / total={total_b / 1e9:.2f}G "
                f"(ce_saliency uses single-layer attn recompute, not all-layer output_attentions)",
                flush=True,
            )
        except Exception:
            pass
    train_stats = run_continue_training(
        model, tokenizer, train_samples, cfg=cfg, bank_cfg=bank_cfg,
        progress_cb=lambda s, m, loss, mode: _prog(
            "training", f"step {s}/{m} loss={loss:.4f}", step=s, total=m, loss=loss,
        ),
    )

    current_test_out: dict[str, Any] | None = None
    if (
        cfg.current_test_prompt
        or cfg.current_test_label
        or cfg.current_test_task_id
    ):
        _prog("predict_current", "Generating on current test sample…")
        cur = _resolve_current_test_sample(cfg, eval_samples)
        if cur is not None:
            gen = generate_one(tokenizer, model, cur["prompt"], cfg.max_new_tokens)
            pre = line_hit(cur["label"], gen["predict"], "precision")
            rec = line_hit(cur["label"], gen["predict"], "recall")
            current_test_out = {
                "task_id": cur.get("task_id"),
                "prompt": cur["prompt"],
                "label": cur["label"],
                "predict": gen["predict"],
                "line_hit_pre": round(pre, 4),
                "line_hit_rec": round(rec, 4),
                "finish_reason": gen.get("finish_reason"),
            }
            print(
                f"[continue-eval] current test task_id={cur.get('task_id')!r} "
                f"pre={pre:.4f} rec={rec:.4f}",
                flush=True,
            )
            print(
                f"[continue-eval] predict:\n{gen['predict'][:2000]}",
                flush=True,
            )

    after = None
    if cfg.eval_after_full:
        _prog("eval_after", "Evaluating continued adapter (line_hit)…")
        after = evaluate_line_hit(
            model, tokenizer, eval_samples,
            max_new_tokens=cfg.max_new_tokens,
            also_truncate_score=cfg.also_truncate_score,
            source_file=Path(cfg.test_data).name,
            progress_cb=lambda i, n, _r: _prog(
                "eval_after", f"Eval after {i}/{n}", done=i, total=n,
            ),
        )
        print(
            f"[continue-eval] AFTER line_hit_pre={after['line_hit_pre']} "
            f"line_hit_rec={after['line_hit_rec']}",
            flush=True,
        )

    delta = _metric_delta(before, after)
    per_sample_deltas = (
        summarize_per_sample_line_hit_deltas(before, after)
        if before is not None and after is not None
        else None
    )
    if per_sample_deltas:
        _print_per_sample_delta_report(per_sample_deltas)

    out_root = Path(cfg.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    after_jsonl = None
    if after is not None and isinstance(after.get("perSample"), list):
        after_jsonl = write_eval_results_jsonl(
            out_root / "continue_eval_after.jsonl",
            list(after.get("perSample") or []),
        )
    before_jsonl = None
    if before is not None and isinstance(before.get("perSample"), list):
        before_jsonl = write_eval_results_jsonl(
            out_root / "continue_eval_before.jsonl",
            list(before.get("perSample") or []),
        )

    meta = {
        "config": asdict(cfg),
        "continueTrainJsonl": train_jsonl,
        "trainStats": train_stats,
        "evalBeforeSource": eval_before_source,
        "evalBeforeCache": cache_path if eval_before_source == "cache" else None,
        "before": _slim_eval_summary(before),
        "after": _slim_eval_summary(after),
        "delta": delta,
        "currentTest": current_test_out,
        "evalAfterJsonl": after_jsonl,
        "evalBeforeJsonl": before_jsonl,
        "nTrainWithEdges": n_edges,
    }
    if per_sample_deltas:
        meta["perSampleDeltas"] = {
            "nCompared": per_sample_deltas["nCompared"],
            "line_hit_pre": {
                "increased": [e["index"] for e in per_sample_deltas["line_hit_pre"]["increased"]],
                "decreased": [e["index"] for e in per_sample_deltas["line_hit_pre"]["decreased"]],
                "nIncreased": len(per_sample_deltas["line_hit_pre"]["increased"]),
                "nDecreased": len(per_sample_deltas["line_hit_pre"]["decreased"]),
                "nUnchanged": len(per_sample_deltas["line_hit_pre"]["unchanged"]),
                "increasedDetail": per_sample_deltas["line_hit_pre"]["increased"][:50],
                "decreasedDetail": per_sample_deltas["line_hit_pre"]["decreased"][:50],
            },
            "line_hit_rec": {
                "increased": [e["index"] for e in per_sample_deltas["line_hit_rec"]["increased"]],
                "decreased": [e["index"] for e in per_sample_deltas["line_hit_rec"]["decreased"]],
                "nIncreased": len(per_sample_deltas["line_hit_rec"]["increased"]),
                "nDecreased": len(per_sample_deltas["line_hit_rec"]["decreased"]),
                "nUnchanged": len(per_sample_deltas["line_hit_rec"]["unchanged"]),
                "increasedDetail": per_sample_deltas["line_hit_rec"]["increased"][:50],
                "decreasedDetail": per_sample_deltas["line_hit_rec"]["decreased"][:50],
            },
        }
    # Compact detail: scores + deltas; full prompt/predict live in the JSONL files.
    detail = {
        "before": _slim_eval_summary(before),
        "after": _slim_eval_summary(after),
        "perSampleDeltas": per_sample_deltas,
        "currentTest": current_test_out,
        "evalAfterJsonl": after_jsonl,
        "evalBeforeJsonl": before_jsonl,
    }
    out_dir = save_adapter(model, tokenizer, cfg.output_dir, meta)
    detail_path = Path(out_dir) / "continue_eval_detail.json"
    detail_path.write_text(json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")
    meta["outputDir"] = out_dir
    meta["detailPath"] = str(detail_path)
    print(
        f"[continue-eval] saved detail={detail_path} "
        f"after_jsonl={after_jsonl}"
        + (f" before_jsonl={before_jsonl}" if before_jsonl else ""),
        flush=True,
    )
    # Make live probes (token probs / learn) use the continued adapter until recover.
    from src.eif_adapter_env import set_active_adapter_override

    adapter_status = set_active_adapter_override(out_dir, source="continue")
    meta["activeAdapter"] = adapter_status
    print(
        f"[continue-train] activated live adapter override → {out_dir} "
        f"(recover restores env adapter)",
        flush=True,
    )

    _prog(
        "completed",
        "Continue-train finished."
        + (" current-test predict done." if current_test_out else "")
        + (" line_hit eval done." if after is not None else ""),
        result=meta,
    )
    del model
    _evict_cached_models()
    return {"status": "success", **meta}


def build_config_from_request(req: dict[str, Any] | None = None) -> ContinueTrainConfig:
    req = req or {}
    defaults = default_paths_from_env()
    adapter = _resolve_path(req.get("adapterPath") or defaults["adapter_path"])
    base = _resolve_path(req.get("baseModelPath") or defaults["base_model_path"])
    # Prefer explicit small subset path; do not default to full EIF_TRAIN_DATA.
    train = _resolve_path(req.get("trainData") or defaults["continue_train_data"])
    source_train = _resolve_path(req.get("sourceTrainData") or defaults["source_train_data"])
    test = _resolve_path(req.get("testData") or defaults["test_data"])
    out = _resolve_path(req.get("outputDir") or defaults["output_dir"])
    eval_before_cache = _resolve_path(
        req.get("evalBeforeCache") or defaults.get("eval_before_cache")
    )

    raw_ids = req.get("trainSampleIds") or []
    train_ids: list[int] = []
    if isinstance(raw_ids, list):
        for x in raw_ids:
            try:
                train_ids.append(int(x))
            except (TypeError, ValueError):
                pass

    if not adapter:
        raise ValueError("adapterPath / EIF_ADAPTER_PATH_SALIENCY is required")
    if not test:
        raise ValueError("testData / EIF_TEST_DATA is required")
    if not train and not train_ids:
        raise ValueError(
            "Need a small continue-train set: set ANNOTATION_CONTINUE_TRAIN_DATA "
            "or pass trainSampleIds (will slice from EIF_TRAIN_DATA)."
        )

    def _default_max_steps() -> int:
        raw = (os.environ.get("EIF_CONTINUE_MAX_STEPS") or "").strip()
        if raw:
            try:
                return max(1, int(raw))
            except ValueError:
                pass
        return 20

    ct = req.get("currentTest") if isinstance(req.get("currentTest"), dict) else {}
    ct_task = str(ct.get("taskId") or ct.get("task_id") or "").strip() or None
    ct_prompt = str(ct.get("prompt") or "").strip() or None
    ct_label = str(ct.get("label") or ct.get("gold") or "").strip() or None

    return ContinueTrainConfig(
        adapter_path=adapter,
        base_model_path=base,
        train_data=train or "",
        test_data=test,
        output_dir=out or str(REPO_ROOT / "outputs" / "continue_trial"),
        max_steps=max(1, int(req.get("maxSteps", _default_max_steps()))),
        learning_rate=float(req.get("learningRate", 2e-5)),
        loss_mode=str(req.get("lossMode", "ce_saliency") or "ce_saliency").strip().lower(),
        eval_before=bool(req.get("evalBefore", True)),
        eval_after_full=bool(req.get("evalAfterFull", True)),
        current_test_task_id=ct_task,
        current_test_prompt=ct_prompt,
        current_test_label=ct_label,
        eval_before_cache=eval_before_cache,
        max_new_tokens=max(16, int(req.get("maxNewTokens", 1024))),
        seed=int(req.get("seed", 42)),
        train_sample_ids=train_ids,
        source_train_data=source_train,
        also_truncate_score=bool(req.get("alsoTruncateScore", True)),
        max_seq_len=int(
            req.get("maxSeqLen")
            if req.get("maxSeqLen") is not None
            else _DEFAULT_CONTINUE_MAX_SEQ_LEN
        ),
    )


def main():
    defaults = default_paths_from_env()
    p = argparse.ArgumentParser(description="Continue LoRA on small annotated set + line_hit eval")
    p.add_argument("--adapter-path", default=defaults["adapter_path"])
    p.add_argument("--base-model-path", default=defaults["base_model_path"])
    p.add_argument("--continue-train-data", default=defaults["continue_train_data"],
                   help="Small annotated JSONL (NOT the full original train set)")
    p.add_argument("--source-train-data", default=defaults["source_train_data"],
                   help="Full train JSONL; only used with --train-sample-ids")
    p.add_argument("--train-sample-ids", default="",
                   help="Comma-separated 0-based indices to slice from source train")
    p.add_argument("--test-data", default=defaults["test_data"])
    p.add_argument(
        "--eval-before-cache",
        default=defaults.get("eval_before_cache") or "",
        help="Precomputed baseline line_hit JSONL (EIF_CONTINUE_EVAL_BEFORE_CACHE); skips GPU eval_before",
    )
    p.add_argument("--output-dir", default=defaults["output_dir"])
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--loss-mode", default="ce_saliency", choices=["ce_only", "ce_saliency"])
    p.add_argument("--no-eval-before", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument(
        "--max-seq-len",
        type=int,
        default=_DEFAULT_CONTINUE_MAX_SEQ_LEN,
        help="Left-truncate ChatML to this length for continue-train (0=disable). "
             "Also EIF_CONTINUE_MAX_SEQ_LEN.",
    )
    args = p.parse_args()

    ids: list[int] = []
    if args.train_sample_ids.strip():
        ids = [int(x) for x in args.train_sample_ids.split(",") if x.strip() != ""]

    cfg = ContinueTrainConfig(
        adapter_path=str(args.adapter_path or ""),
        base_model_path=args.base_model_path,
        train_data=str(args.continue_train_data or ""),
        test_data=str(args.test_data or ""),
        output_dir=str(args.output_dir or ""),
        max_steps=args.max_steps,
        learning_rate=args.lr,
        loss_mode=args.loss_mode,
        eval_before=not args.no_eval_before,
        eval_before_cache=str(args.eval_before_cache or "").strip() or None,
        max_new_tokens=args.max_new_tokens,
        train_sample_ids=ids,
        source_train_data=args.source_train_data,
        max_seq_len=int(args.max_seq_len),
    )
    result = run_continue_train_and_eval(cfg)
    before = result.get("before") or {}
    after = result.get("after") or {}
    delta = result.get("delta") or {}
    print(json.dumps({
        "outputDir": result.get("outputDir"),
        "continueTrainJsonl": result.get("continueTrainJsonl"),
        "before": {"line_hit_pre": before.get("line_hit_pre"), "line_hit_rec": before.get("line_hit_rec")},
        "after": {"line_hit_pre": after.get("line_hit_pre"), "line_hit_rec": after.get("line_hit_rec")},
        "delta": {"line_hit_pre": delta.get("line_hit_pre"), "line_hit_rec": delta.get("line_hit_rec")},
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
