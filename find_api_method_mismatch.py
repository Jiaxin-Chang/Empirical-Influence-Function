"""
从 predictions jsonl 中筛选：
1. predict 与 label 不相等（预测错误）
2. 按行对齐后，第一个不同的行双方都是 API 调用行（如 arts.Size(...)）
3. 该行 receiver 相同，但方法名不同（如 arts.Size vs arts.Len）
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# 匹配形如 receiver.Method( 的 API 调用（Go 风格）
API_CALL_RE = re.compile(r"\b([A-Za-z_]\w*)\.([A-Za-z_]\w*)\s*\(")


def extract_api_call(line: str) -> tuple[str, str] | None:
    """从一行中提取第一个 API 调用的 (receiver, method)。"""
    m = API_CALL_RE.search(line)
    if not m:
        return None
    return m.group(1), m.group(2)


def first_diff_line(
    predict: str, label: str
) -> tuple[int, str | None, str | None] | None:
    """
    按行比较 predict / label，返回第一个不同行：
    (0-based index, predict_line, label_line)
    若某侧行数不够，对应侧为 None。
    """
    pred_lines = predict.splitlines()
    label_lines = label.splitlines()
    n = max(len(pred_lines), len(label_lines))
    for i in range(n):
        pl = pred_lines[i] if i < len(pred_lines) else None
        ll = label_lines[i] if i < len(label_lines) else None
        # 用 strip 后比较，忽略纯缩进差异；内容不同才算错误行
        ps = pl.strip() if pl is not None else None
        ls = ll.strip() if ll is not None else None
        if ps != ls:
            return i, pl, ll
    return None


def is_api_method_mismatch(predict: str, label: str) -> tuple[bool, dict]:
    """判断是否满足：预测错误 + 首个差异行为同 receiver 不同 method 的 API 调用。"""
    info: dict = {}
    if predict.strip() == label.strip():
        return False, info

    diff = first_diff_line(predict, label)
    if diff is None:
        return False, info

    idx, pred_line, label_line = diff
    info["diff_line_index"] = idx + 1  # 1-based within predict/label
    info["predict_line"] = (pred_line or "").strip()
    info["label_line"] = (label_line or "").strip()

    if pred_line is None or label_line is None:
        return False, info

    pred_api = extract_api_call(pred_line)
    label_api = extract_api_call(label_line)
    info["predict_api"] = pred_api
    info["label_api"] = label_api

    if pred_api is None or label_api is None:
        return False, info

    pred_recv, pred_method = pred_api
    label_recv, label_method = label_api
    if pred_recv == label_recv and pred_method != label_method:
        return True, info
    return False, info


def main() -> None:
    parser = argparse.ArgumentParser(
        description="找出首个错误行为同 receiver 不同 method 的 API 调用样本"
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="processed_part1.predictions.jsonl",
        help="predictions jsonl 路径",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="同时打印差异行详情",
    )
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"文件不存在: {path}", file=sys.stderr)
        sys.exit(1)

    matched_lines: list[int] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            predict = obj.get("predict", "")
            label = obj.get("label", "")
            ok, info = is_api_method_mismatch(predict, label)
            if not ok:
                continue
            matched_lines.append(lineno)
            if args.verbose:
                pred_api = info["predict_api"]
                label_api = info["label_api"]
                print(f"--- jsonl 行号 {lineno} (差异在代码第 {info['diff_line_index']} 行) ---")
                print(f"  predict: {info['predict_line']}")
                print(f"  label:   {info['label_line']}")
                print(
                    f"  API: {pred_api[0]}.{pred_api[1]}(...) "
                    f"vs {label_api[0]}.{label_api[1]}(...)"
                )
                print()

    if args.verbose:
        print("=" * 40)
    print(f"共 {len(matched_lines)} 行满足条件")
    print("行号:", ", ".join(map(str, matched_lines)) if matched_lines else "(无)")


if __name__ == "__main__":
    main()
