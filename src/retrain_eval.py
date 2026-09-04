"""Retrain from base LoRA on EIF_TRAIN_DATA + ANNOTATION_CONTINUE_TRAIN_DATA.

Unlike continue-train (few AdamW steps on the existing family adapter + small
subset only), retrain:

  1) concatenates the compact continue subset AFTER the full train bank
  2) starts from EIF_BASE_MODEL_PATH with a fresh LoRA (r=16, α=32, qkvo)
  3) runs CE+contrastive saliency with the SFT hypers (grad accum 4, cosine,
     warmup 0.03, max_len 2000, λ=1.5)
  4) stops at frontend max_steps or num_epochs, then greedy-decodes the
     currently open test sample
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from src.bank_loss import BankLossConfig, compute_bank_loss
from src.continue_train_eval import (
    _evict_cached_models,
    _pick_continue_attn_implementation,
    _release_cuda,
    _resolve_current_test_sample,
    _resolve_path,
    _sample_batch,
    _trainable_lora_params,
    generate_one,
    line_hit,
    load_eval_samples,
    save_adapter,
)
from src.eif_adapter_env import base_model_path_from_env, set_active_adapter_override
from src.intervention_experiment import (
    _cast_lora_params_to_dtype,
    _dispatch_peft_model,
    _patch_model_with_attn_hook,
    load_train_samples,
)


REPO_ROOT = Path(__file__).resolve().parent.parent

RETRAIN_LORA_R = 16
RETRAIN_LORA_ALPHA = 32
RETRAIN_LORA_DROPOUT = 0.05
RETRAIN_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
RETRAIN_BATCH_SIZE = 1
RETRAIN_GRAD_ACCUM = 4
RETRAIN_WARMUP_RATIO = 0.03
RETRAIN_MAX_GRAD_NORM = 1.0
RETRAIN_MAX_LEN = 2000
RETRAIN_LR = 2e-5


@dataclass
class RetrainConfig:
    base_model_path: str
    source_train_data: str
    continue_train_data: str | None
    test_data: str
    output_dir: str
    max_steps: int | None = None
    num_epochs: int | None = None
    learning_rate: float = RETRAIN_LR
    loss_mode: str = "ce_saliency"
    current_test_task_id: str | None = None
    current_test_prompt: str | None = None
    current_test_label: str | None = None
    max_new_tokens: int = 1024
    seed: int = 42
    max_seq_len: int = RETRAIN_MAX_LEN


def default_retrain_paths() -> dict[str, Any]:
    source = (
        (os.environ.get("EIF_TRAIN_DATA") or "").strip()
        or (os.environ.get("ANNOTATION_TRAIN_DATA") or "").strip()
    )
    extra = (os.environ.get("ANNOTATION_CONTINUE_TRAIN_DATA") or "").strip()
    test = (os.environ.get("EIF_TEST_DATA") or "").strip()
    base = base_model_path_from_env() or (os.environ.get("EIF_BASE_MODEL_PATH") or "").strip()
    out = (os.environ.get("EIF_RETRAIN_OUTPUT_DIR") or "").strip() or str(
        REPO_ROOT / "outputs" / "retrain_trial"
    )
    return {
        "base_model_path": _resolve_path(base),
        "source_train_data": _resolve_path(source),
        "continue_train_data": _resolve_path(extra),
        "test_data": _resolve_path(test),
        "output_dir": _resolve_path(out) or str(REPO_ROOT / "outputs" / "retrain_trial"),
        "learning_rate": RETRAIN_LR,
        "num_epochs": 1,
        "grad_accum": RETRAIN_GRAD_ACCUM,
        "max_seq_len": RETRAIN_MAX_LEN,
        "lora_r": RETRAIN_LORA_R,
        "lora_alpha": RETRAIN_LORA_ALPHA,
        "loss_mode": "ce_saliency",
    }


def concat_train_jsonl(source: str, extra: str | None, dest: Path) -> dict[str, Any]:
    src = Path(source)
    if not src.is_file():
        raise FileNotFoundError(f"EIF_TRAIN_DATA not found: {source!r}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    n_src = 0
    n_extra = 0
    with dest.open("w", encoding="utf-8") as out:
        with src.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                out.write(line if line.endswith("\n") else line + "\n")
                n_src += 1
        extra_path = Path(extra) if extra else None
        if extra_path is not None and extra_path.is_file():
            with extra_path.open(encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    out.write(line if line.endswith("\n") else line + "\n")
                    n_extra += 1
        elif extra:
            print(f"[retrain][WARN] continue subset missing, train-only: {extra}", flush=True)
    if n_src == 0:
        raise ValueError(f"No compact rows in {source}")
    print(
        f"[retrain] concatenated train={n_src} + continue={n_extra} → {dest} ({n_src + n_extra})",
        flush=True,
    )
    return {
        "nSource": n_src,
        "nContinue": n_extra,
        "nTotal": n_src + n_extra,
        "path": str(dest.resolve()),
        "continuePath": str(extra_path.resolve()) if extra_path and extra_path.is_file() else None,
    }


def _compute_device(model) -> torch.device:
    for p in model.parameters():
        if getattr(p, "device", None) is not None and p.device.type != "meta":
            return p.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_base_with_fresh_lora(base_path: str, *, attn_implementation: str):
    if not base_path or not Path(base_path).is_dir():
        raise FileNotFoundError(f"EIF_BASE_MODEL_PATH not found: {base_path!r}")
    print(f"[retrain] loading base={base_path} attn={attn_implementation}", flush=True)
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
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    peft_cfg = LoraConfig(
        r=RETRAIN_LORA_R,
        lora_alpha=RETRAIN_LORA_ALPHA,
        lora_dropout=RETRAIN_LORA_DROPOUT,
        target_modules=list(RETRAIN_LORA_TARGETS),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base, peft_cfg)
    try:
        model = _dispatch_peft_model(model, max_memory=None)
    except Exception as exc:
        print(f"[retrain][WARN] dispatch_model failed ({exc}); trying .to(cuda)", flush=True)
        if torch.cuda.is_available():
            model = model.to("cuda")
    n_cast = _cast_lora_params_to_dtype(model, torch.bfloat16)
    if n_cast:
        print(f"[retrain] aligned {n_cast} LoRA tensors to bfloat16", flush=True)
    for n, p in model.named_parameters():
        p.requires_grad = ("lora_" in n)
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
    model = _patch_model_with_attn_hook(model)
    model._eif_grad_space = "lora"
    model._eif_base_model_path = os.path.abspath(base_path)
    model.train()
    return model, tokenizer


def _resolve_max_steps(cfg: RetrainConfig, n_train: int, accum: int) -> int:
    if cfg.max_steps is not None and int(cfg.max_steps) > 0:
        return max(1, int(cfg.max_steps))
    epochs = int(cfg.num_epochs or 1)
    steps_per_epoch = max(1, math.ceil(n_train / max(1, accum)))
    return max(1, epochs * steps_per_epoch)


def run_retrain_loop(
    model,
    tokenizer,
    train_samples: list[dict],
    *,
    cfg: RetrainConfig,
    bank_cfg: BankLossConfig,
    n_steps: int,
    progress_cb=None,
) -> dict[str, Any]:
    device = _compute_device(model)
    params = _trainable_lora_params(model)
    if not params:
        raise RuntimeError("No trainable LoRA parameters for retrain.")
    opt = torch.optim.AdamW(params, lr=float(cfg.learning_rate))
    warmup = max(0, int(RETRAIN_WARMUP_RATIO * n_steps)) if n_steps > 1 else 0
    scheduler = get_cosine_schedule_with_warmup(
        opt, num_warmup_steps=warmup, num_training_steps=n_steps,
    )
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])

    n_train = len(train_samples)
    accum = RETRAIN_GRAD_ACCUM
    max_seq = int(cfg.max_seq_len) if int(cfg.max_seq_len) > 0 else None
    losses: list[float] = []
    started = time.time()
    print(
        f"[retrain] {n_steps} opt step(s) · n={n_train} accum={accum} "
        f"warmup={warmup} lr={cfg.learning_rate} max_len={max_seq}",
        flush=True,
    )
    cursor = 0
    completed = 0
    for step in range(1, n_steps + 1):
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        mode_used = bank_cfg.loss_mode
        micros = 0
        for _micro in range(accum):
            sample = train_samples[cursor % n_train]
            cursor += 1
            batch, edges = _sample_batch(
                sample, None, device, max_seq_len=max_seq,
            )
            loss, mode_used = compute_bank_loss(
                model,
                batch,
                device=device,
                cfg=bank_cfg,
                edges=edges,
                special_ids=special_ids,
            )
            if loss is None:
                del batch
                continue
            (loss / accum).backward()
            step_loss += float(loss.detach().float().cpu())
            micros += 1
            del batch, loss
        if micros == 0:
            scheduler.step()
            continue
        torch.nn.utils.clip_grad_norm_(params, RETRAIN_MAX_GRAD_NORM)
        opt.step()
        scheduler.step()
        avg = step_loss / max(1, micros)
        losses.append(avg)
        completed = step
        lr_now = float(scheduler.get_last_lr()[0]) if scheduler.get_last_lr() else cfg.learning_rate
        if progress_cb is not None:
            progress_cb(step, n_steps, avg, mode_used)
        print(
            f"[retrain] step {step}/{n_steps} loss={avg:.4f} mode={mode_used} "
            f"lr={lr_now:.2e} micros={micros}",
            flush=True,
        )
        if step % 20 == 0:
            _release_cuda()
    model.eval()
    return {
        "steps": completed,
        "plannedSteps": n_steps,
        "meanLoss": float(sum(losses) / max(1, len(losses))),
        "lastLoss": float(losses[-1]) if losses else None,
        "elapsedSec": round(time.time() - started, 2),
        "lossModeUsed": bank_cfg.loss_mode,
        "nTrainSamples": n_train,
        "gradAccum": accum,
        "warmupSteps": warmup,
        "maxSeqLen": max_seq,
        "learningRate": cfg.learning_rate,
    }


def run_retrain_and_eval(cfg: RetrainConfig, progress_cb=None) -> dict[str, Any]:
    if not cfg.base_model_path or not Path(cfg.base_model_path).is_dir():
        raise FileNotFoundError(f"base_model_path not found: {cfg.base_model_path!r}")
    if not cfg.source_train_data or not Path(cfg.source_train_data).is_file():
        raise FileNotFoundError(f"EIF_TRAIN_DATA not found: {cfg.source_train_data!r}")
    if not cfg.test_data or not Path(cfg.test_data).is_file():
        raise FileNotFoundError(f"test_data not found: {cfg.test_data!r}")

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    combined = concat_train_jsonl(
        cfg.source_train_data,
        cfg.continue_train_data,
        out / "retrain_combined.jsonl",
    )

    def _prog(stage: str, message: str, **extra):
        if progress_cb is not None:
            progress_cb(stage, message, extra)

    _evict_cached_models()
    torch.manual_seed(int(cfg.seed))
    attn_impl = _pick_continue_attn_implementation()
    if attn_impl == "flash_attention_2":
        # User config: --use_flash_attention False
        attn_impl = "sdpa"
    _prog("loading", f"Loading base + fresh LoRA ({attn_impl})…")
    model, tokenizer = load_base_with_fresh_lora(
        cfg.base_model_path, attn_implementation=attn_impl,
    )

    train_samples = load_train_samples(combined["path"])
    if not train_samples:
        raise ValueError(f"No compact train samples in {combined['path']}")
    eval_samples = load_eval_samples(cfg.test_data)
    n_steps = _resolve_max_steps(cfg, len(train_samples), RETRAIN_GRAD_ACCUM)
    n_edges = sum(1 for s in train_samples if s.get("attention_edges") or s.get("edges"))
    print(
        f"[retrain] n={len(train_samples)} with_edges={n_edges} steps={n_steps} "
        f"epochs={cfg.num_epochs} max_steps={cfg.max_steps}",
        flush=True,
    )

    bank_cfg = BankLossConfig(
        loss_mode="ce_saliency" if cfg.loss_mode != "ce_only" else "ce_only",
        saliency_loss_type="contrastive",
        saliency_lambda=1.5,
        alpha=1.0,
        margin_plus=2.0,
        neg_sample_k=64,
        saliency_layer=-1,
    )
    _prog("training", f"Retrain {n_steps} opt step(s) on {len(train_samples)} samples…")
    train_stats = run_retrain_loop(
        model, tokenizer, train_samples, cfg=cfg, bank_cfg=bank_cfg, n_steps=n_steps,
        progress_cb=lambda s, m, loss, mode: _prog(
            "training", f"opt {s}/{m} loss={loss:.4f}", step=s, total=m, loss=loss,
        ),
    )
    train_stats["nTrainWithEdges"] = n_edges
    train_stats["concat"] = combined

    try:
        model.gradient_checkpointing_disable()
    except Exception:
        pass
    if hasattr(model, "config"):
        model.config.use_cache = True

    current_test_out: dict[str, Any] | None = None
    if cfg.current_test_prompt or cfg.current_test_label or cfg.current_test_task_id:
        _prog("predict_current", "Generating on current test sample…")
        cur = _resolve_current_test_sample(cfg, eval_samples)
        if cur is not None:
            gen = generate_one(tokenizer, model, cur["prompt"], cfg.max_new_tokens)
            pre = line_hit(cur.get("label") or "", gen["predict"], "precision")
            rec = line_hit(cur.get("label") or "", gen["predict"], "recall")
            current_test_out = {
                "task_id": cur.get("task_id"),
                "prompt": cur["prompt"],
                "label": cur.get("label"),
                "predict": gen["predict"],
                "predict_token_ids": gen.get("predict_token_ids") or [],
                "predict_tokens": gen.get("predict_tokens") or [],
                "line_hit_pre": round(pre, 4),
                "line_hit_rec": round(rec, 4),
                "finish_reason": gen.get("finish_reason"),
            }
            print(
                f"[retrain] current test task_id={cur.get('task_id')!r} "
                f"pre={pre:.4f} rec={rec:.4f}",
                flush=True,
            )
            print(f"[retrain] predict:\n{gen['predict'][:2000]}", flush=True)

    meta = {
        "config": asdict(cfg),
        "concat": combined,
        "trainStats": train_stats,
        "currentTest": current_test_out,
        "plannedSteps": n_steps,
        "attnImplementation": attn_impl,
        "lora": {
            "r": RETRAIN_LORA_R,
            "alpha": RETRAIN_LORA_ALPHA,
            "dropout": RETRAIN_LORA_DROPOUT,
            "target_modules": RETRAIN_LORA_TARGETS,
        },
        "bank": {
            "loss_mode": bank_cfg.loss_mode,
            "saliency_loss_type": bank_cfg.saliency_loss_type,
            "saliency_lambda": bank_cfg.saliency_lambda,
            "alpha": bank_cfg.alpha,
            "margin_plus": bank_cfg.margin_plus,
            "neg_sample_k": bank_cfg.neg_sample_k,
            "saliency_layer": bank_cfg.saliency_layer,
        },
    }
    out_dir = save_adapter(model, tokenizer, cfg.output_dir, meta)
    (Path(out_dir) / "retrain_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    adapter_status = set_active_adapter_override(out_dir, source="retrain")
    meta["outputDir"] = out_dir
    meta["activeAdapter"] = adapter_status
    print(f"[retrain] saved adapter → {out_dir}", flush=True)
    _prog(
        "completed",
        "Retrain finished." + (" current-test predict done." if current_test_out else ""),
        result=meta,
    )
    del model
    _evict_cached_models()
    return {"status": "success", **meta}


def build_retrain_config_from_request(req: dict[str, Any] | None = None) -> RetrainConfig:
    req = req or {}
    defaults = default_retrain_paths()
    base = _resolve_path(req.get("baseModelPath") or defaults["base_model_path"])
    source = _resolve_path(req.get("sourceTrainData") or req.get("trainData") or defaults["source_train_data"])
    extra = _resolve_path(req.get("continueTrainData") or defaults["continue_train_data"])
    test = _resolve_path(req.get("testData") or defaults["test_data"])
    out = _resolve_path(req.get("outputDir") or defaults["output_dir"])
    if not base:
        raise ValueError("baseModelPath / EIF_BASE_MODEL_PATH is required for retrain")
    if not source:
        raise ValueError("sourceTrainData / EIF_TRAIN_DATA is required for retrain")
    if not test:
        raise ValueError("testData / EIF_TEST_DATA is required")

    raw_steps = req.get("maxSteps", req.get("max_steps"))
    raw_epochs = req.get("numEpochs", req.get("epochs", req.get("maxEpochs")))
    max_steps = None
    num_epochs = None
    if raw_steps is not None and str(raw_steps).strip() != "":
        try:
            max_steps = max(1, int(raw_steps))
        except (TypeError, ValueError):
            max_steps = None
    if raw_epochs is not None and str(raw_epochs).strip() != "":
        try:
            num_epochs = max(1, int(raw_epochs))
        except (TypeError, ValueError):
            num_epochs = None
    if max_steps is None and num_epochs is None:
        num_epochs = 1

    ct = req.get("currentTest") if isinstance(req.get("currentTest"), dict) else {}
    lr = req.get("learningRate", req.get("lr", RETRAIN_LR))
    return RetrainConfig(
        base_model_path=base,
        source_train_data=source,
        continue_train_data=extra,
        test_data=test,
        output_dir=out or str(REPO_ROOT / "outputs" / "retrain_trial"),
        max_steps=max_steps,
        num_epochs=num_epochs,
        learning_rate=float(lr),
        loss_mode=str(req.get("lossMode", "ce_saliency") or "ce_saliency").strip().lower(),
        current_test_task_id=str(ct.get("taskId") or ct.get("task_id") or "").strip() or None,
        current_test_prompt=str(ct.get("prompt") or "").strip() or None,
        current_test_label=str(ct.get("label") or ct.get("gold") or "").strip() or None,
        max_new_tokens=max(16, int(req.get("maxNewTokens", 1024))),
        seed=int(req.get("seed", 42)),
        max_seq_len=int(req.get("maxSeqLen") or RETRAIN_MAX_LEN),
    )
