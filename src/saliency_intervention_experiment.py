"""
Causal Intervention Experiment
================================
验证假设：在 correlation_matching_results.json 中找到的 train-test correlation 配对，
         是否真的共享同一组模型参数回路（knowledge circuit）。

方法：
  - 对找到的相关 train sample 的 correlation [A→B] 做 saliency_loss 梯度更新
  - 观察 test sample 上全部 4 个 top-correlation 的 saliency 变化（delta_S）
  - 对照组：选取与 test 无关的 train sample，做同样操作，delta_S 应接近 0

Phase 1: 在 cos_sim 最高的 train sample 上扫描 (lr × steps) 超参组合
Phase 2: 固定超参，对全部实验组 + 对照组样本完整跑
"""

import copy
import json
import os
import re

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from functools import partial
from heapq import nlargest
from tqdm import tqdm
from transformers import DataCollatorForSeq2Seq, set_seed

from src.NIF import (
    CustomCollator,
    DatasetWrapper,
    NewInferenceFunction,
    _find_subseq_start,
    build_single_sample_dataset,
    build_train_dataset,
    load_model_and_tokenizer,
    load_samples_from_formal_jsonl,
    round_floats,
)
from src.loss import (
    compute_full_saliency_vector,
    compute_saliency_score_only,
    do_saliency_loss_step,
)
from src.process_data import process_func_chatml

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

SEED = 42
SELECTED_TEST_SAMPLE_INDEX = 58
TOKEN_INDEX_TO_RETRIEVE = 703   # 我们研究的"第一个预测错误的 token"
TOP_K_PROMPT_TOKENS = 4         # 监测的 test correlation 数量（全部）
SEQUENCE_LENGTH_LIMIT = 3000

# ── Phase 控制 ────────────────────────────────────────────────────────────────
# "phase1" : 超参扫描（在 cos_sim 最高的 train sample 上跑 LR × STEPS 组合）
# "phase2" : 全量实验（固定超参，跑所有实验组 + 对照组）
PHASE: str = "phase1"

# Phase 1 sweep 参数
LR_SWEEP    = [1e-5, 5e-5, 1e-4]
STEPS_SWEEP = [1, 3, 5]

# Phase 2 固定参数（从 Phase 1 结果中选）
PHASE2_LR    = 1e-4
PHASE2_STEPS = 3

# ── 对照组 ────────────────────────────────────────────────────────────────────
# 手动指定 or 自动选取不在实验组中的前 N 个样本
# 若为 None，则自动选取 NUM_CONTROL_SAMPLES 个（跳过实验组 ID）
NUM_CONTROL_SAMPLES: int = 5
MANUAL_CONTROL_IDS: list[int] | None = None   # e.g. [2, 7, 15, 23, 41]


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def qk_last_quarter_filter(name: str, param, num_layers: int = 28) -> bool:
    """只选最后 1/4 层的 Q/K 参数，与 intervention_experiment.py 保持一致。"""
    match = re.search(r"layers\.(\d+)\.", name)
    if match:
        layer_idx = int(match.group(1))
        if layer_idx >= num_layers * 3 // 4:
            if "q_proj" in name or "k_proj" in name:
                return True
    return False


def find_first_valid_token_index(tokenizer, input_ids_tensor, start_idx: int) -> int:
    """
    跳过格式字符（\\n, \\t, 空格, {, } 等），找到第一个有语义意义的 token。
    与 intervention_experiment.py 保持一致。
    """
    idx = start_idx
    seq_len = input_ids_tensor.size(1)
    while idx < seq_len:
        tok_str = tokenizer.decode([input_ids_tensor[0, idx].item()])
        if tok_str.strip() not in ("", "{", "}"):
            break
        idx += 1
    return idx


def save_param_snapshot(model, param_filter_fn) -> dict[str, torch.Tensor]:
    """将 filtered params 的当前值克隆到 CPU 快照。"""
    return {
        name: param.data.clone().cpu()
        for name, param in model.named_parameters()
        if param_filter_fn(name, param)
    }


def restore_param_snapshot(
    model, snapshot: dict[str, torch.Tensor], param_filter_fn
) -> None:
    """从快照恢复 filtered params，并确保它们的 requires_grad = False。"""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in snapshot:
                param.data.copy_(snapshot[name].to(param.device))
                param.requires_grad_(False)
    torch.cuda.empty_cache()


def measure_all_test_saliencies(
    model,
    test_batch: dict,
    test_correlations: list[dict],
    target_token_idx: int,
) -> list[float]:
    """
    一次性计算 test_batch 中所有 4 个 top-correlation 的 saliency score。
    利用 compute_full_saliency_vector 只做一次 forward+backward。
    """
    full_vec = compute_full_saliency_vector(model, test_batch, target_token_idx)
    return [full_vec[tc["source_token_index"]] for tc in test_correlations]


def run_single_intervention(
    model,
    train_batch: dict,
    test_batch: dict,
    train_source_idx: int,
    train_target_idx: int,
    test_correlations: list[dict],
    param_filter_fn,
    lr: float,
    num_steps: int,
    device,
) -> tuple[list[float], list[float], list[float]]:
    """
    完整的一次干预周期：
      1. 测量 S_before（所有 4 个 test correlations）
      2. 保存权重快照
      3. num_steps 步 saliency_loss 梯度更新（在 train correlation 上）
      4. 测量 S_after
      5. 恢复权重快照
    
    Returns:
        s_before  list[float] 长度 4
        s_after   list[float] 长度 4
        step_losses  list[float] 每步的 saliency_loss 值
    """
    # ── Step 1: S_before ──────────────────────────────────────────────────────
    s_before = measure_all_test_saliencies(
        model, test_batch, test_correlations, TOKEN_INDEX_TO_RETRIEVE
    )

    # ── Step 2: 快照 ──────────────────────────────────────────────────────────
    snapshot = save_param_snapshot(model, param_filter_fn)

    # ── Step 3: 梯度更新 ──────────────────────────────────────────────────────
    # 绑定 optimizer（SGD，与理论推导一致，无动量/decay 干扰）
    target_params = [
        p for n, p in model.named_parameters() if param_filter_fn(n, p)
    ]
    # 先全部 freeze，do_saliency_loss_step 内部会按需 unfreeze
    for p in target_params:
        p.requires_grad_(False)

    optimizer = torch.optim.SGD(target_params, lr=lr)

    step_losses = []
    for step in range(num_steps):
        loss_val = do_saliency_loss_step(
            model,
            train_batch,
            train_target_idx,
            train_source_idx,
            param_filter_fn,
            optimizer,
        )
        step_losses.append(loss_val)
        print(
            f"    [{step + 1}/{num_steps}] saliency_loss={loss_val:.6f}  "
            f"(train src={train_source_idx} → tgt={train_target_idx})"
        )

    # ── Step 4: S_after ───────────────────────────────────────────────────────
    s_after = measure_all_test_saliencies(
        model, test_batch, test_correlations, TOKEN_INDEX_TO_RETRIEVE
    )

    # ── Step 5: 恢复 ──────────────────────────────────────────────────────────
    restore_param_snapshot(model, snapshot, param_filter_fn)
    del snapshot

    return s_before, s_after, step_losses


def select_control_ids(
    all_train_ids: set[int],
    treated_ids: set[int],
    n: int,
    manual: list[int] | None,
) -> list[int]:
    """
    选取对照组 train sample IDs。
    优先使用 MANUAL_CONTROL_IDS；否则自动选取不在实验组中的前 n 个。
    """
    if manual is not None:
        bad = set(manual) & treated_ids
        if bad:
            raise ValueError(f"MANUAL_CONTROL_IDS 中有 ID 与实验组重叠：{bad}")
        return manual[:n]

    candidates = sorted(all_train_ids - treated_ids)
    return candidates[:n]


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ══════════════════════════════════════════════════════════════════════════════

def run_saliency_intervention_experiment():
    accelerator = Accelerator()
    set_seed(SEED)

    # ── 加载模型 & 数据 ────────────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer()
    num_layers = len(model.model.layers)
    param_filter = partial(qk_last_quarter_filter, num_layers=num_layers)

    convert_to_chatml = partial(process_func_chatml, tokenizer=tokenizer)
    train_samples = load_samples_from_formal_jsonl("sft_train.jsonl")
    test_samples  = load_samples_from_formal_jsonl("sft_test.jsonl")

    base_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )

    # ── 加载 correlation_matching_results.json ────────────────────────────────
    base_dir     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_path = os.path.join(base_dir, "correlation_matching_results.json")
    if not os.path.exists(results_path):
        raise FileNotFoundError(
            f"找不到 {results_path}，请先运行 intervention_experiment.py"
        )
    with open(results_path, "r", encoding="utf-8") as f:
        corr_results = json.load(f)

    # ── 重建 test_batch（与 intervention_experiment.py 完全相同的逻辑）────────
    print("\n[Setup] 重建 test_batch for sample 58（需要重新 generate）...")
    infer_fw = NewInferenceFunction(
        model=model,
        tokenizer=tokenizer,
        accelerator=accelerator,
        param_filter_fn=param_filter,
        top_k=20,
    )

    test_ds        = build_single_sample_dataset(test_samples[SELECTED_TEST_SAMPLE_INDEX], convert_to_chatml)
    raw_test_batch = base_collator([test_ds[0]])
    raw_test_batch = {k: v.to(accelerator.device) for k, v in raw_test_batch.items()}

    model.eval()
    gen_result = infer_fw.infer(raw_test_batch)
    prompt_len = int(gen_result["target_idx"][0])
    prompt_ids = raw_test_batch["input_ids"][0, :prompt_len]
    pred_ids   = torch.tensor(
        gen_result["pred_ids"][0],
        device=prompt_ids.device,
        dtype=prompt_ids.dtype,
    )
    new_input_ids       = torch.cat([prompt_ids, pred_ids], dim=0).unsqueeze(0)
    new_attention_mask  = torch.ones_like(new_input_ids)
    new_labels          = new_input_ids.clone()
    new_labels[:, :prompt_len] = -100

    test_batch = {
        "input_ids":      new_input_ids,
        "attention_mask": new_attention_mask,
        "labels":         new_labels,
    }

    # ── test correlations（全部 4 个，来自 JSON）──────────────────────────────
    test_correlations = corr_results["test_sample_baseline"]["top_correlations"]
    print(f"[Setup] 监测的 test correlations（{len(test_correlations)} 个）：")
    for i, tc in enumerate(test_correlations):
        print(
            f"  [{i}] source='{tc['source_token']}'(idx={tc['source_token_index']}) "
            f"→ target='{tc['target_token']}'(idx={tc['target_token_index']})  "
            f"saliency={tc['saliency_score']:.5f}"
        )

    interventions = corr_results["interventions"]
    treated_ids   = {item["train_sample_id"] for item in interventions}

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 1: 超参扫描
    # ══════════════════════════════════════════════════════════════════════════
    if PHASE == "phase1":
        print("\n" + "═" * 70)
        print("PHASE 1: 超参扫描")
        print("═" * 70)

        # 选取实验组中 cos_sim 最高的 train sample 作为扫描对象
        best_item = max(
            interventions,
            key=lambda item: max(m["cos_sim"] for m in item["correlation_matches"]),
        )
        best_match = max(best_item["correlation_matches"], key=lambda m: m["cos_sim"])

        train_idx        = best_item["train_sample_id"]
        train_source_idx = best_match["train_correlation"]["source_token_index"]
        train_target_idx = best_match["train_correlation"]["target_token_index"]
        best_cos_sim     = best_match["cos_sim"]

        print(
            f"  选定 train sample ID={train_idx}，cos_sim={best_cos_sim:.5f}\n"
            f"  train correlation:  src='{best_match['train_correlation']['source_token']}' "
            f"(idx={train_source_idx}) → tgt='{best_match['train_correlation']['target_token']}' "
            f"(idx={train_target_idx})"
        )

        tr_ds    = build_single_sample_dataset(train_samples[train_idx], convert_to_chatml)
        tr_batch = base_collator([tr_ds[0]])
        tr_batch = {k: v.to(accelerator.device) for k, v in tr_batch.items()}

        sweep_results = []
        for lr in LR_SWEEP:
            for steps in STEPS_SWEEP:
                print(f"\n  [Sweep] lr={lr:.0e}  steps={steps}")
                s_before, s_after, step_losses = run_single_intervention(
                    model,
                    tr_batch,
                    test_batch,
                    train_source_idx,
                    train_target_idx,
                    test_correlations,
                    param_filter,
                    lr=lr,
                    num_steps=steps,
                    device=accelerator.device,
                )
                delta_S = [a - b for a, b in zip(s_after, s_before)]

                # 理论预测的方向：lr × cos_sim（仅用于验证线性近似）
                # 真实 predicted_delta = lr * cos_sim * ‖test_feat‖ * ‖train_feat‖
                # 这里只记 cos_sim 供后续分析
                per_tc = [
                    {
                        "source_token":       tc["source_token"],
                        "source_token_index": tc["source_token_index"],
                        "S_before":           sb,
                        "S_after":            sa,
                        "delta_S":            ds,
                        "direction_correct":  ds > 0,  # 预期 delta_S > 0 when cos_sim > 0
                    }
                    for tc, sb, sa, ds in zip(test_correlations, s_before, s_after, delta_S)
                ]

                print(f"    delta_S = {[f'{d:.6f}' for d in delta_S]}")
                print(f"    step_losses = {[f'{l:.6f}' for l in step_losses]}")

                sweep_results.append({
                    "lr":             lr,
                    "num_steps":      steps,
                    "train_sample_id": train_idx,
                    "cos_sim":         best_cos_sim,
                    "step_losses":     step_losses,
                    "per_test_correlation": per_tc,
                })

        output = {
            "phase": "phase1",
            "config": {
                "tested_train_sample_id": train_idx,
                "best_cos_sim":           best_cos_sim,
                "train_correlation": {
                    "source_token":       best_match["train_correlation"]["source_token"],
                    "source_token_index": train_source_idx,
                    "target_token":       best_match["train_correlation"]["target_token"],
                    "target_token_index": train_target_idx,
                },
                "param_filter": "qk_last_quarter",
                "test_sample_index":      SELECTED_TEST_SAMPLE_INDEX,
                "test_target_token_index": TOKEN_INDEX_TO_RETRIEVE,
            },
            "sweep_results": sweep_results,
        }

        out_path = os.path.join(base_dir, "causal_intervention_results_phase1.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(round_floats(output, 7), f, indent=2, ensure_ascii=False)
        print(f"\n[Phase 1] 结果已写入 {out_path}")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2: 全量实验（实验组 + 对照组）
    # ══════════════════════════════════════════════════════════════════════════
    elif PHASE == "phase2":
        print("\n" + "═" * 70)
        print(f"PHASE 2: 全量实验  lr={PHASE2_LR:.0e}  steps={PHASE2_STEPS}")
        print("═" * 70)

        marker_ids = tuple(tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False))

        # ── 构建实验组样本列表 ───────────────────────────────────────────────
        treated_group = []
        for item in interventions:
            best_match = max(item["correlation_matches"], key=lambda m: m["cos_sim"])
            treated_group.append({
                "group":           "treated",
                "train_sample_id": item["train_sample_id"],
                "rank":            interventions.index(item) + 1,
                "coarse_cos_sim":  item["coarse_cos_sim"],
                "cos_sim":         best_match["cos_sim"],
                "train_correlation": best_match["train_correlation"],
                "test_correlation":  best_match["test_correlation"],
            })

        # ── 构建对照组样本列表 ───────────────────────────────────────────────
        control_ids = select_control_ids(
            all_train_ids=set(range(len(train_samples))),
            treated_ids=treated_ids,
            n=NUM_CONTROL_SAMPLES,
            manual=MANUAL_CONTROL_IDS,
        )
        print(f"[Phase 2] 对照组 IDs: {control_ids}")

        # 为对照组样本找各自的 top-1 saliency correlation（train's own primary attention）
        control_group = []
        for tid in control_ids:
            tr_ds    = build_single_sample_dataset(train_samples[tid], convert_to_chatml)
            tr_batch = base_collator([tr_ds[0]])
            tr_batch = {k: v.to(accelerator.device) for k, v in tr_batch.items()}

            if tr_batch["input_ids"].size(1) > SEQUENCE_LENGTH_LIMIT:
                print(f"  Control sample {tid} 太长，跳过。")
                continue

            try:
                start_sys = _find_subseq_start(tr_batch["input_ids"][0], marker_ids) + 3
            except ValueError:
                print(f"  Control sample {tid} 找不到 marker，跳过。")
                continue

            valid_tok_idx = find_first_valid_token_index(
                tokenizer, tr_batch["input_ids"], start_sys
            )

            # 用一次 forward 得到完整 saliency 向量，选 top-1 source
            sal_vec      = compute_full_saliency_vector(model, tr_batch, valid_tok_idx)
            top1_src_idx = int(max(range(len(sal_vec)), key=lambda i: sal_vec[i]))

            control_group.append({
                "group":           "control",
                "train_sample_id": tid,
                "cos_sim":         0.0,   # 对照组不计算 cos_sim
                "train_correlation": {
                    "source_token":       tokenizer.decode(
                        [tr_batch["input_ids"][0, top1_src_idx].item()]
                    ),
                    "source_token_index": top1_src_idx,
                    "target_token":       tokenizer.decode(
                        [tr_batch["input_ids"][0, valid_tok_idx].item()]
                    ),
                    "target_token_index": valid_tok_idx,
                },
                "test_correlation": None,
                "_tr_batch":         tr_batch,   # 临时缓存，最后删掉
            })

        # ── 全量运行 ─────────────────────────────────────────────────────────
        all_groups = treated_group + control_group
        results    = []

        for entry in tqdm(all_groups, desc="Running interventions"):
            train_idx  = entry["train_sample_id"]
            group      = entry["group"]
            cos_sim    = entry["cos_sim"]
            train_corr = entry["train_correlation"]

            print(f"\n[{group.upper()} | rank={entry.get('rank','—')}]  "
                  f"train_id={train_idx}  cos_sim={cos_sim:.5f}")

            # 构建 train_batch（对照组已缓存，实验组现在构建）
            if group == "treated":
                tr_ds    = build_single_sample_dataset(train_samples[train_idx], convert_to_chatml)
                tr_batch = base_collator([tr_ds[0]])
                tr_batch = {k: v.to(accelerator.device) for k, v in tr_batch.items()}

                if tr_batch["input_ids"].size(1) > SEQUENCE_LENGTH_LIMIT:
                    print(f"  样本 {train_idx} 太长，跳过。")
                    continue
            else:
                tr_batch = entry.pop("_tr_batch")   # 用完即弃

            train_source_idx = train_corr["source_token_index"]
            train_target_idx = train_corr["target_token_index"]

            print(
                f"  干预 train correlation: "
                f"'{train_corr.get('source_token','')}' (idx={train_source_idx}) "
                f"→ '{train_corr.get('target_token','')}' (idx={train_target_idx})"
            )

            s_before, s_after, step_losses = run_single_intervention(
                model,
                tr_batch,
                test_batch,
                train_source_idx,
                train_target_idx,
                test_correlations,
                param_filter,
                lr=PHASE2_LR,
                num_steps=PHASE2_STEPS,
                device=accelerator.device,
            )

            delta_S = [a - b for a, b in zip(s_after, s_before)]

            per_tc = [
                {
                    "source_token":          tc["source_token"],
                    "source_token_index":    tc["source_token_index"],
                    "matched_cos_sim":       (
                        # 若实验组，找对应 correlation_match 里该 test corr 的 cos_sim
                        next(
                            (
                                m["cos_sim"]
                                for item in interventions
                                if item["train_sample_id"] == train_idx
                                for m in item["correlation_matches"]
                                if m["test_correlation"]["source_token_index"]
                                == tc["source_token_index"]
                            ),
                            None,
                        )
                        if group == "treated"
                        else None
                    ),
                    "S_before":              sb,
                    "S_after":               sa,
                    "delta_S":               ds,
                    "relative_delta":        ds / (sb + 1e-12),
                    "direction_correct":     ds > 0,
                }
                for tc, sb, sa, ds in zip(test_correlations, s_before, s_after, delta_S)
            ]

            print(
                f"  delta_S = {[f'{d:.6f}' for d in delta_S]}  "
                f"step_losses={[f'{l:.6f}' for l in step_losses]}"
            )

            results.append({
                "train_sample_id":   train_idx,
                "group":             group,
                "rank":              entry.get("rank"),
                "coarse_cos_sim":    entry.get("coarse_cos_sim"),
                "cos_sim":           cos_sim,
                "train_correlation": train_corr,
                "step_losses":       step_losses,
                "per_test_correlation": per_tc,
            })

        # ── 写出结果 ─────────────────────────────────────────────────────────
        output = {
            "phase": "phase2",
            "config": {
                "lr":                     PHASE2_LR,
                "num_steps":              PHASE2_STEPS,
                "param_filter":           "qk_last_quarter",
                "test_sample_index":      SELECTED_TEST_SAMPLE_INDEX,
                "test_target_token_index": TOKEN_INDEX_TO_RETRIEVE,
                "num_treated":            len(treated_group),
                "num_control":            len(control_group),
            },
            "test_correlations": test_correlations,
            "results": results,
        }

        out_path = os.path.join(base_dir, "causal_intervention_results_phase2.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(round_floats(output, 7), f, indent=2, ensure_ascii=False)
        print(f"\n[Phase 2] 结果已写入 {out_path}")

    else:
        raise ValueError(f"PHASE 必须是 'phase1' 或 'phase2'，得到：{PHASE!r}")


if __name__ == "__main__":
    run_saliency_intervention_experiment()
