"""
Causal Intervention Experiment (1-to-1 Design)
===============================================
验证假设：在 correlation_matching_results.json 中找到的 train-test correlation 配对，
         是否真的共享同一组模型参数回路（knowledge circuit）。

实验单元：每个 test correlation 独立
  - 为每个 test correlation 找到全局 cos_sim 最高的 (train_sample, train_correlation)
  - 对该 train_correlation 做 saliency_loss 干预
  - 观察这个 test correlation 自己的 delta_S（primary metric）
  - 同时记录其他 3 个 test correlation 的 delta_S（cross-effect，作为参考）
  - 对照组：用 cos_sim 接近 0 的 train_correlation 做同样操作，delta_S 应接近 0

Phase 1: 在全局 cos_sim 最高的那对 (test_corr, train_corr) 上扫描 (lr × steps)
Phase 2: 固定超参，对全部 4 个 test correlation 各自独立跑实验组 + 对照组
"""

import json
import os
import random
import re

import torch
from accelerate import Accelerator
from functools import partial
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
TOKEN_INDEX_TO_RETRIEVE = 703   # 我们研究的第一个预测错误的 token
TOP_K_PROMPT_TOKENS = 4         # 监测的 test correlation 数量
SEQUENCE_LENGTH_LIMIT = 3000

# ── Phase 控制 ────────────────────────────────────────────────────────────────
# "phase1" : 超参扫描（在全局 cos_sim 最高的那对上跑 LR × STEPS 组合）
# "phase2" : 全量实验（固定超参，对每个 test correlation 独立跑 1-to-1 实验）
PHASE: str = "phase1"

# Phase 1 sweep 参数
LR_SWEEP    = [1e-5, 5e-5, 1e-4]
STEPS_SWEEP = [1, 3, 5]

# Phase 2 固定参数（从 Phase 1 结果中选）
PHASE2_LR    = 1e-4
PHASE2_STEPS = 1

# 对照组 A（低 cos_sim）：从 interventions 里取 cos_sim 最低的 N 个
NUM_CONTROL_PER_TC: int = 2
# 对照组 B（真随机）：从完全不在 interventions 里的训练样本中随机选 N 个
NUM_RANDOM_CONTROL_PER_TC: int = 3


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
    """跳过格式字符，找到第一个有语义意义的 token。"""
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
    """从快照恢复 filtered params。"""
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
    一次 forward+backward 得到所有 test correlation 的 saliency score。
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
      1. 测量 S_before（所有 test correlations）
      2. 保存权重快照
      3. num_steps 步 saliency_loss 梯度更新（在 train correlation 上）
      4. 测量 S_after
      5. 恢复权重快照

    Returns:
        s_before     list[float]  长度 = len(test_correlations)
        s_after      list[float]  长度 = len(test_correlations)
        step_losses  list[float]  每步的 saliency_loss 值
    """
    # Step 1: S_before
    s_before = measure_all_test_saliencies(
        model, test_batch, test_correlations, TOKEN_INDEX_TO_RETRIEVE
    )

    # Step 2: 快照
    snapshot = save_param_snapshot(model, param_filter_fn)

    # Step 3: 梯度更新
    target_params = [
        p for n, p in model.named_parameters() if param_filter_fn(n, p)
    ]
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

    # Step 4: S_after
    s_after = measure_all_test_saliencies(
        model, test_batch, test_correlations, TOKEN_INDEX_TO_RETRIEVE
    )

    # Step 5: 恢复
    restore_param_snapshot(model, snapshot, param_filter_fn)
    del snapshot

    return s_before, s_after, step_losses


def find_best_match_for_test_corr(
    interventions: list[dict],
    test_corr_source_idx: int,
) -> dict | None:
    """
    在 correlation_matching_results.json 的 interventions 列表中，
    为指定的 test correlation（by source_token_index）找到全局 cos_sim 最高的
    (train_sample_id, train_correlation, cos_sim) 三元组。

    Returns:
        dict with keys: train_sample_id, train_correlation, cos_sim
        or None if not found
    """
    best = None
    best_cos = -1.0
    for item in interventions:
        for m in item["correlation_matches"]:
            if m["test_correlation"]["source_token_index"] == test_corr_source_idx:
                if m["cos_sim"] > best_cos:
                    best_cos = m["cos_sim"]
                    best = {
                        "train_sample_id":   item["train_sample_id"],
                        "train_correlation": m["train_correlation"],
                        "cos_sim":           m["cos_sim"],
                    }
    return best


def find_control_matches_for_test_corr(
    interventions: list[dict],
    test_corr_source_idx: int,
    treated_train_id: int,
    n: int,
) -> list[dict]:
    """
    为指定 test correlation 找 n 个对照 train sample：
    - 不是 treat 样本（排除 treated_train_id）
    - 按 cos_sim 从低到高（取 cos_sim 接近 0 的，代表无相关）
    """
    candidates = []
    for item in interventions:
        if item["train_sample_id"] == treated_train_id:
            continue
        for m in item["correlation_matches"]:
            if m["test_correlation"]["source_token_index"] == test_corr_source_idx:
                candidates.append({
                    "train_sample_id":   item["train_sample_id"],
                    "train_correlation": m["train_correlation"],
                    "cos_sim":           m["cos_sim"],
                })
    # 取 cos_sim 最低的 n 个（接近 0 的为对照）
    candidates.sort(key=lambda x: x["cos_sim"])
    return candidates[:n]


def build_delta_result(
    test_correlations: list[dict],
    primary_tc_idx: int,
    s_before: list[float],
    s_after: list[float],
) -> dict:
    """
    构建标准化的 delta 结果，区分 primary（当前目标 test corr）和 cross-effect（其他）。
    primary_tc_idx: test_correlations 列表中当前被干预对应的那个的下标
    """
    entries = []
    for i, tc in enumerate(test_correlations):
        ds = s_after[i] - s_before[i]
        entries.append({
            "source_token":       tc["source_token"],
            "source_token_index": tc["source_token_index"],
            "role":               "primary" if i == primary_tc_idx else "cross_effect",
            "S_before":           s_before[i],
            "S_after":            s_after[i],
            "delta_S":            ds,
            "relative_delta":     ds / (s_before[i] + 1e-12),
            "direction_correct":  ds > 0,
        })
    return entries


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

    interventions = corr_results["interventions"]

    # ── 重建 test_batch ────────────────────────────────────────────────────────
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
    new_input_ids      = torch.cat([prompt_ids, pred_ids], dim=0).unsqueeze(0)
    new_attention_mask = torch.ones_like(new_input_ids)
    new_labels         = new_input_ids.clone()
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

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 1: 超参扫描（在全局 cos_sim 最高的那对上）
    # ══════════════════════════════════════════════════════════════════════════
    if PHASE == "phase1":
        print("\n" + "═" * 70)
        print("PHASE 1: 超参扫描（1-to-1）")
        print("═" * 70)

        # 找全局 cos_sim 最高的那对 (test_corr, train_corr)
        best_global_cos = -1.0
        best_tc_idx     = -1
        best_match      = None

        for tc_i, tc in enumerate(test_correlations):
            m = find_best_match_for_test_corr(interventions, tc["source_token_index"])
            if m and m["cos_sim"] > best_global_cos:
                best_global_cos = m["cos_sim"]
                best_tc_idx     = tc_i
                best_match      = m

        if best_match is None:
            raise RuntimeError("找不到任何有效的 correlation match，请检查 correlation_matching_results.json")

        target_tc         = test_correlations[best_tc_idx]
        train_idx         = best_match["train_sample_id"]
        train_source_idx  = best_match["train_correlation"]["source_token_index"]
        train_target_idx  = best_match["train_correlation"]["target_token_index"]

        print(
            f"  全局最优 pair：\n"
            f"    test_corr : '{target_tc['source_token']}'(idx={target_tc['source_token_index']}) "
            f"→ '{target_tc['target_token']}'(idx={target_tc['target_token_index']})\n"
            f"    train_corr: '{best_match['train_correlation']['source_token']}'(idx={train_source_idx}) "
            f"→ '{best_match['train_correlation']['target_token']}'(idx={train_target_idx})\n"
            f"    cos_sim   : {best_global_cos:.5f}  (train_sample_id={train_idx})"
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

                # Primary：目标 test correlation 的 delta_S
                primary_delta_S = s_after[best_tc_idx] - s_before[best_tc_idx]
                all_delta_S     = [a - b for a, b in zip(s_after, s_before)]

                print(f"    primary delta_S          = {primary_delta_S:.6f}  (target: '{target_tc['source_token']}')")
                print(f"    all delta_S              = {[f'{d:.6f}' for d in all_delta_S]}")
                print(f"    step_losses              = {[f'{l:.6f}' for l in step_losses]}")

                sweep_results.append({
                    "lr":               lr,
                    "num_steps":        steps,
                    "train_sample_id":  train_idx,
                    "cos_sim":          best_global_cos,
                    "step_losses":      step_losses,
                    "primary_delta_S":  primary_delta_S,
                    "direction_correct": primary_delta_S > 0,
                    "all_delta_S": [
                        {
                            "source_token":       tc["source_token"],
                            "source_token_index": tc["source_token_index"],
                            "role":               "primary" if i == best_tc_idx else "cross_effect",
                            "S_before":           s_before[i],
                            "S_after":            s_after[i],
                            "delta_S":            all_delta_S[i],
                        }
                        for i, tc in enumerate(test_correlations)
                    ],
                })

        output = {
            "phase": "phase1",
            "config": {
                "primary_test_correlation": {
                    "source_token":       target_tc["source_token"],
                    "source_token_index": target_tc["source_token_index"],
                    "target_token":       target_tc["target_token"],
                    "target_token_index": target_tc["target_token_index"],
                    "tc_list_index":      best_tc_idx,
                },
                "best_train_correlation": {
                    "train_sample_id":   train_idx,
                    "source_token":      best_match["train_correlation"]["source_token"],
                    "source_token_index": train_source_idx,
                    "target_token":      best_match["train_correlation"]["target_token"],
                    "target_token_index": train_target_idx,
                    "cos_sim":           best_global_cos,
                },
                "param_filter":            "qk_last_quarter",
                "test_sample_index":       SELECTED_TEST_SAMPLE_INDEX,
                "test_target_token_index": TOKEN_INDEX_TO_RETRIEVE,
            },
            "sweep_results": sweep_results,
        }

        out_path = os.path.join(base_dir, "causal_intervention_results_phase1.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(round_floats(output, 7), f, indent=2, ensure_ascii=False)
        print(f"\n[Phase 1] 结果已写入 {out_path}")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2: 全量实验（每个 test correlation 独立 1-to-1）
    # ══════════════════════════════════════════════════════════════════════════
    elif PHASE == "phase2":
        print("\n" + "═" * 70)
        print(f"PHASE 2: 全量实验（1-to-1）  lr={PHASE2_LR:.0e}  steps={PHASE2_STEPS}")
        print("═" * 70)

        per_tc_results = []

        for tc_i, tc in enumerate(test_correlations):
            tc_src_idx = tc["source_token_index"]
            print(f"\n{'─' * 60}")
            print(f"[TC {tc_i}] test correlation: '{tc['source_token']}'(idx={tc_src_idx}) "
                  f"→ '{tc['target_token']}'(idx={tc['target_token_index']})")
            print(f"{'─' * 60}")

            # ── 实验组：找该 test corr 全局最优的 train_corr ─────────────────
            treated = find_best_match_for_test_corr(interventions, tc_src_idx)
            if treated is None:
                print(f"  ⚠ 找不到 match，跳过 TC {tc_i}")
                continue

            treated_train_id = treated["train_sample_id"]
            print(
                f"  [TREATED] train_id={treated_train_id}  cos_sim={treated['cos_sim']:.5f}\n"
                f"    train_corr: '{treated['train_correlation']['source_token']}'(idx={treated['train_correlation']['source_token_index']}) "
                f"→ '{treated['train_correlation']['target_token']}'(idx={treated['train_correlation']['target_token_index']})"
            )

            tr_ds    = build_single_sample_dataset(train_samples[treated_train_id], convert_to_chatml)
            tr_batch = base_collator([tr_ds[0]])
            tr_batch = {k: v.to(accelerator.device) for k, v in tr_batch.items()}

            s_before, s_after, step_losses = run_single_intervention(
                model,
                tr_batch,
                test_batch,
                treated["train_correlation"]["source_token_index"],
                treated["train_correlation"]["target_token_index"],
                test_correlations,
                param_filter,
                lr=PHASE2_LR,
                num_steps=PHASE2_STEPS,
                device=accelerator.device,
            )
            treated_entries = build_delta_result(test_correlations, tc_i, s_before, s_after)
            primary_delta   = treated_entries[tc_i]["delta_S"]
            print(f"  TREATED  primary delta_S = {primary_delta:+.6f}  direction_correct={primary_delta > 0}")

            treated_result = {
                "train_sample_id":   treated_train_id,
                "train_correlation": treated["train_correlation"],
                "cos_sim":           treated["cos_sim"],
                "step_losses":       step_losses,
                "per_correlation":   treated_entries,
            }

            # ── 对照组 A：interventions 内 cos_sim 最低的 N 个 ──────────────
            controls_raw = find_control_matches_for_test_corr(
                interventions, tc_src_idx, treated_train_id, NUM_CONTROL_PER_TC
            )
            control_results = []
            for ctrl in controls_raw:
                ctrl_train_id = ctrl["train_sample_id"]
                print(
                    f"\n  [CTRL-A | low cos_sim] train_id={ctrl_train_id}  cos_sim={ctrl['cos_sim']:.5f}\n"
                    f"    train_corr: '{ctrl['train_correlation']['source_token']}'(idx={ctrl['train_correlation']['source_token_index']}) "
                    f"-> '{ctrl['train_correlation']['target_token']}'(idx={ctrl['train_correlation']['target_token_index']})"
                )
                c_ds    = build_single_sample_dataset(train_samples[ctrl_train_id], convert_to_chatml)
                c_batch = base_collator([c_ds[0]])
                c_batch = {k: v.to(accelerator.device) for k, v in c_batch.items()}

                cs_before, cs_after, c_losses = run_single_intervention(
                    model,
                    c_batch,
                    test_batch,
                    ctrl["train_correlation"]["source_token_index"],
                    ctrl["train_correlation"]["target_token_index"],
                    test_correlations,
                    param_filter,
                    lr=PHASE2_LR,
                    num_steps=PHASE2_STEPS,
                    device=accelerator.device,
                )
                c_entries = build_delta_result(test_correlations, tc_i, cs_before, cs_after)
                c_primary = c_entries[tc_i]["delta_S"]
                print(f"  CTRL-A   primary delta_S = {c_primary:+.6f}  direction_correct={c_primary > 0}")

                control_results.append({
                    "train_sample_id":   ctrl_train_id,
                    "train_correlation": ctrl["train_correlation"],
                    "cos_sim":           ctrl["cos_sim"],
                    "step_losses":       c_losses,
                    "per_correlation":   c_entries,
                })

            # ── 对照组 B：真随机样本（完全不在 interventions 里）─────────────
            treated_id_set = {item["train_sample_id"] for item in interventions}
            rand_candidates = [i for i in range(len(train_samples)) if i not in treated_id_set]
            rng = random.Random(SEED + tc_i)
            rng.shuffle(rand_candidates)

            marker_ids_b = tuple(tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False))
            random_control_results = []
            for rand_id in rand_candidates:
                if len(random_control_results) >= NUM_RANDOM_CONTROL_PER_TC:
                    break

                rc_ds    = build_single_sample_dataset(train_samples[rand_id], convert_to_chatml)
                rc_batch = base_collator([rc_ds[0]])
                rc_batch = {k: v.to(accelerator.device) for k, v in rc_batch.items()}

                if rc_batch["input_ids"].size(1) > SEQUENCE_LENGTH_LIMIT:
                    continue
                try:
                    start_sys = _find_subseq_start(rc_batch["input_ids"][0], marker_ids_b) + len(marker_ids_b)
                except ValueError:
                    continue

                valid_tok_rc = find_first_valid_token_index(tokenizer, rc_batch["input_ids"], start_sys)
                sal_vec_rc   = compute_full_saliency_vector(model, rc_batch, valid_tok_rc)
                top1_src_rc  = int(max(range(len(sal_vec_rc)), key=lambda ii: sal_vec_rc[ii]))
                src_tok_rc   = tokenizer.decode([rc_batch["input_ids"][0, top1_src_rc].item()])
                tgt_tok_rc   = tokenizer.decode([rc_batch["input_ids"][0, valid_tok_rc].item()])

                print(
                    f"\n  [CTRL-B | random] train_id={rand_id}  cos_sim~0\n"
                    f"    train_corr (own top-1 saliency): '{src_tok_rc}'(idx={top1_src_rc}) "
                    f"-> '{tgt_tok_rc}'(idx={valid_tok_rc})"
                )

                rc_before, rc_after, rc_losses = run_single_intervention(
                    model, rc_batch, test_batch,
                    top1_src_rc, valid_tok_rc,
                    test_correlations, param_filter,
                    lr=PHASE2_LR, num_steps=PHASE2_STEPS, device=accelerator.device,
                )
                rc_entries = build_delta_result(test_correlations, tc_i, rc_before, rc_after)
                rc_primary = rc_entries[tc_i]["delta_S"]
                print(f"  CTRL-B   primary delta_S = {rc_primary:+.6f}  direction_correct={rc_primary > 0}")

                random_control_results.append({
                    "train_sample_id":   rand_id,
                    "train_correlation": {
                        "source_token":       src_tok_rc,
                        "source_token_index": top1_src_rc,
                        "target_token":       tgt_tok_rc,
                        "target_token_index": valid_tok_rc,
                    },
                    "cos_sim":         None,
                    "step_losses":     rc_losses,
                    "per_correlation": rc_entries,
                })

            per_tc_results.append({
                "tc_index":          tc_i,
                "test_correlation":  tc,
                "treated":           treated_result,
                "controls_low_cos":  control_results,
                "controls_random":   random_control_results,
            })

        # ── 写出结果 ─────────────────────────────────────────────────────────
        output = {
            "phase": "phase2",
            "config": {
                "lr":                      PHASE2_LR,
                "num_steps":               PHASE2_STEPS,
                "param_filter":            "qk_last_quarter",
                "test_sample_index":       SELECTED_TEST_SAMPLE_INDEX,
                "test_target_token_index": TOKEN_INDEX_TO_RETRIEVE,
                "num_control_low_cos_per_tc":  NUM_CONTROL_PER_TC,
                "num_control_random_per_tc":   NUM_RANDOM_CONTROL_PER_TC,
            },
            "test_correlations":   test_correlations,
            "per_tc_results":      per_tc_results,
        }

        out_path = os.path.join(base_dir, "causal_intervention_results_phase2.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(round_floats(output, 7), f, indent=2, ensure_ascii=False)
        print(f"\n[Phase 2] 结果已写入 {out_path}")

        # ── 控制台摘要 ───────────────────────────────────────────────────────
        print("\n" + "═" * 88)
        print("SUMMARY  (treated vs CTRL-A low-cos vs CTRL-B random)")
        print("═" * 88)
        print(f"{'TC':>4}  {'token':>10}  {'cos_sim':>8}  "
              f"{'treated':>9}  {'A-0':>7}  {'A-1':>7}  {'B-0':>7}  {'B-1':>7}  {'B-2':>7}  ok")
        print("─" * 88)
        for r in per_tc_results:
            tc_r    = r["test_correlation"]
            treated = r["treated"]
            ca      = r["controls_low_cos"]
            cb      = r["controls_random"]
            idx     = r["tc_index"]
            t_d  = treated["per_correlation"][idx]["delta_S"]
            ca0  = ca[0]["per_correlation"][idx]["delta_S"] if len(ca) > 0 else float("nan")
            ca1  = ca[1]["per_correlation"][idx]["delta_S"] if len(ca) > 1 else float("nan")
            cb0  = cb[0]["per_correlation"][idx]["delta_S"] if len(cb) > 0 else float("nan")
            cb1  = cb[1]["per_correlation"][idx]["delta_S"] if len(cb) > 1 else float("nan")
            cb2  = cb[2]["per_correlation"][idx]["delta_S"] if len(cb) > 2 else float("nan")
            print(
                f"[{idx}]  {tc_r['source_token']:>10}  "
                f"{treated['cos_sim']:>8.5f}  "
                f"{t_d:>+9.4f}  {ca0:>+7.4f}  {ca1:>+7.4f}  "
                f"{cb0:>+7.4f}  {cb1:>+7.4f}  {cb2:>+7.4f}  "
                f"{'✅' if t_d > 0 else '❌'}"
            )

    else:
        raise ValueError(f"PHASE 必须是 'phase1' 或 'phase2'，得到：{PHASE!r}")


if __name__ == "__main__":
    run_saliency_intervention_experiment()
