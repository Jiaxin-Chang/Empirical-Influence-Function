#!/usr/bin/env python3
"""End-to-end Go FIM eval: generate with Qwen3-8B (+ optional LoRA), then line_hit.

Input JSONL (CSN-Go / AI4Go style), each line needs:
  {"prompt": "...", "response"|"middle"|"label": "...", "task_id": "..."}

Generation path matches the project official setting:
  ChatML via render_model_prompt(thinking=False),
  eos = [eos_token_id, <|im_end|>],
  remove_leading_thinking,
  line_hit bag-of-lines with del_spaces (same as eval_cli raw).
python /mnt/md124/jiaxin/AI4GoPlayGround-personal-h00613474-v3/eval_go_fim_line_hit.py \
  --input /mnt/md124/jiaxin/training_code/data/csn_go_test_fim_logic1000.jsonl \
  --base-model /mnt/md124/jiaxin/models/Qwen3-8B \
  --adapter /mnt/md124/jiaxin/training_code/Empirical-Influence-Function/outputs/go-sal-7/checkpoint-5104 \
  --output-dir /mnt/md124/jiaxin/training_code/SAL_N_outputs

python /mnt/md124/jiaxin/AI4GoPlayGround-personal-h00613474-v3/eval_go_fim_line_hit.py \
  --input /mnt/md124/jiaxin/training_code/data/csn_py_valid_fim_logic_like_test.jsonl \
  --base-model /mnt/md124/jiaxin/models/Qwen3-8B \
  --adapter /mnt/md124/jiaxin/training_code/Empirical-Influence-Function/outputs/python-ce/checkpoint-5000 \
  --output-dir /mnt/md124/jiaxin/training_code/python_ce_valid

  python /mnt/md124/jiaxin/AI4GoPlayGround-personal-h00613474-v3/eval_go_fim_line_hit.py \
  --input /mnt/md124/jiaxin/Empirical-Influence-Function/csn_go_test_fim_ce6724_imperfect_human_inferable_strict_no_freestr_no_newapi.jsonl \
  --base-model /mnt/md124/jiaxin/models/Qwen3-8B \
  --adapter /mnt/md124/jiaxin/training_code/Empirical-Influence-Function/outputs/qwen3-8b-csn-go-ce-plus/checkpoint-5100 \
  --output-dir ce_outputs

    python /mnt/md124/jiaxin/AI4GoPlayGround-personal-h00613474-v3/eval_go_fim_line_hit.py \
  --input /mnt/md124/jiaxin/Empirical-Influence-Function/csn_go_test_fim_ce6724_imperfect_human_inferable_strict_no_freestr_no_newapi.jsonl \
  --base-model /mnt/md124/jiaxin/models/Qwen3-8B \
  --adapter /mnt/md124/jiaxin/training_code/Empirical-Influence-Function/outputs/qwen3-8b-csn-go-sal-plus/checkpoint-5100 \
  --output-dir sal_outputs

Example (NVIDIA CUDA):
  export CUDA_VISIBLE_DEVICES=0
  python hw_test_data/eval_go_fim_pipeline.py \\
    --input /mnt/md124/jiaxin/path/to/csn_go_fim_test.jsonl \\
    --base-model /mnt/md124/jiaxin/models/Qwen3-8B \\
    --adapter - \\
    --output-dir hw_test_data/go_fim_eval_outputs

  # with LoRA
  python hw_test_data/eval_go_fim_pipeline.py \\
    --input /path/to/go_test.jsonl \\
    --adapter /path/to/lora/checkpoint-xxxx

  # score an existing predictions file only (no GPU)
  python hw_test_data/eval_go_fim_pipeline.py \\
    --score-only hw_test_data/go_fim_eval_outputs/qwen3-8b-go.predictions.jsonl
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generate_local import CHAT_STOP, remove_leading_thinking, render_model_prompt  # noqa: E402

DEFAULT_BASE = "/mnt/md124/jiaxin/models/Qwen3-8B"
DEFAULT_ADAPTER = "-"  # base only by default
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "go_fim_eval_outputs"

# --- normalize (optional, for --score-mode normalized/both) ---
_FENCED_BLOCK_RE = re.compile(
    r"```(?P<language>[^\r\n`]*)[ \t]*\r?\n"
    r"(?P<code>.*?)"
    r"(?:\r?\n)?```",
    flags=re.DOTALL,
)
_LEADING_FENCE_RE = re.compile(r"^[ \t\r\n]*```[^\r\n`]*[ \t]*\r?\n")
_MID_BLOCK_RE = re.compile(
    r"<MID>[ \t]*\r?\n?(?P<code>.*?)\r?\n?[ \t]*</MID>",
    flags=re.DOTALL | re.IGNORECASE,
)
_MID_MARKER_RE = re.compile(
    r"(?m)^[ \t]*(?:</?MID>|MID)[ \t]*(?:\r?\n|$)",
    flags=re.IGNORECASE,
)


def normalize_prediction(prediction: str) -> tuple[str, str]:
    text = prediction
    strategies: list[str] = []
    fenced_blocks = list(_FENCED_BLOCK_RE.finditer(text))
    if fenced_blocks:
        go_blocks = [
            m
            for m in fenced_blocks
            if m.group("language").strip().lower() in {"go", "golang"}
        ]
        selected = (go_blocks or fenced_blocks)[-1]
        text = selected.group("code")
        strategies.append("fenced_code")
    else:
        leading = _LEADING_FENCE_RE.match(text)
        if leading:
            text = text[leading.end() :]
            strategies.append("unclosed_fence")
    mid = _MID_BLOCK_RE.search(text)
    if mid:
        text = mid.group("code")
        strategies.append("mid_block")
    else:
        text, n = _MID_MARKER_RE.subn("", text)
        if n:
            strategies.append("mid_marker")
    if not strategies:
        return prediction, "unchanged"
    return text, "+".join(strategies)


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


def gold_from_obj(obj: dict) -> str | None:
    for key in ("response", "label", "middle"):
        val = obj.get(key)
        if isinstance(val, str):
            return val
    return None


def load_dataset(path: Path, limit: int | None = None) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj.get("prompt"), str):
                raise ValueError(f"{path}:{line_no}: missing prompt")
            label = gold_from_obj(obj)
            if label is None:
                raise ValueError(f"{path}:{line_no}: missing response/label/middle")
            task_id = obj.get("task_id", f"row_{line_no}")
            if not isinstance(task_id, str) or not task_id:
                task_id = f"row_{line_no}"
            rows.append(
                {
                    "task_id": task_id,
                    "prompt": obj["prompt"],
                    "label": label,
                    "source_line": line_no,
                    "language": obj.get("language", "go"),
                }
            )
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"No samples in {path}")
    return rows


class _S:
    def __init__(self, prompt: str):
        self.prompt = prompt


def load_model(base_model: str, adapter: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; this script expects an NVIDIA GPU")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    if adapter and adapter != "-":
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    model.eval()
    return torch, tokenizer, model


def generate_one(torch_mod, tokenizer, model, prompt: str, max_new_tokens: int) -> dict:
    text = render_model_prompt(tokenizer, _S(prompt), enable_thinking=False)
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    device = model.get_input_embeddings().weight.device
    encoded = {k: v.to(device) for k, v in encoded.items()}
    prompt_tokens = int(encoded["input_ids"].shape[1])

    eos_ids = [tokenizer.eos_token_id]
    im_end_id = tokenizer.convert_tokens_to_ids(CHAT_STOP)
    if isinstance(im_end_id, int) and im_end_id >= 0 and im_end_id not in eos_ids:
        eos_ids.append(im_end_id)

    with torch_mod.inference_mode():
        out = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
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
        "thinking_enabled": False,
    }
    if removed:
        row["predict_raw"] = text_out
        row["thinking_removed"] = True
    return row


def score_rows(rows: list[dict], *, normalize: bool) -> tuple[dict, list[dict]]:
    scored: list[dict] = []
    for row in rows:
        predict = row["predict"]
        strategy = "unchanged"
        if normalize:
            predict, strategy = normalize_prediction(predict)
        pre = line_hit(row["label"], predict, "precision")
        rec = line_hit(row["label"], predict, "recall")
        item = dict(row)
        item.update(
            {
                "predict_used": predict,
                "normalize_strategy": strategy,
                "line_hit_pre": round(pre, 4),
                "line_hit_rec": round(rec, 4),
                "perfect": pre == 100.0 and rec == 100.0,
                "hit@0.5": pre > 50.0,
            }
        )
        scored.append(item)
    n = len(scored)
    summary = {
        "score_mode": "normalized" if normalize else "raw",
        "cases": n,
        "line_hit_pre": round(sum(x["line_hit_pre"] for x in scored) / n, 4),
        "line_hit_rec": round(sum(x["line_hit_rec"] for x in scored) / n, 4),
        "perfect": sum(1 for x in scored if x["perfect"]),
        "perfect_rate": round(sum(1 for x in scored if x["perfect"]) / n, 4),
        "hit@0.5": round(sum(1 for x in scored if x["hit@0.5"]) / n, 4),
        "normalized_changed": sum(
            1 for x in scored if x["normalize_strategy"] != "unchanged"
        ),
    }
    return summary, scored


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "model",
        "adapter",
        "score_mode",
        "cases",
        "line_hit_pre",
        "line_hit_rec",
        "perfect",
        "perfect_rate",
        "hit@0.5",
        "normalized_changed",
        "predictions",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_generate_and_eval(args: argparse.Namespace) -> None:
    import torch

    samples = load_dataset(args.input, limit=args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    adapter = None if args.adapter in ("", "-") else args.adapter

    print(f"loaded {len(samples)} samples from {args.input}")
    print(f"base={args.base_model}")
    print(f"adapter={adapter or '-'}")
    print(f"output_dir={args.output_dir}")

    torch_mod, tokenizer, model = load_model(args.base_model, adapter)
    pred_path = args.output_dir / f"{args.name}.predictions.jsonl"
    pred_rows: list[dict] = []

    with pred_path.open("w", encoding="utf-8") as handle:
        for idx, sample in enumerate(samples, start=1):
            gen = generate_one(
                torch_mod, tokenizer, model, sample["prompt"], args.max_new_tokens
            )
            pre = line_hit(sample["label"], gen["predict"], "precision")
            rec = line_hit(sample["label"], gen["predict"], "recall")
            row = {
                "task_id": sample["task_id"],
                "prompt": sample["prompt"],
                "label": sample["label"],
                "predict": gen["predict"],
                "source_file": args.input.name,
                "source_line": sample["source_line"],
                "language": sample.get("language", "go"),
                "finish_reason": gen["finish_reason"],
                "prompt_tokens": gen["prompt_tokens"],
                "generated_tokens": gen["generated_tokens"],
                "thinking_enabled": False,
                "line_hit_pre": round(pre, 4),
                "line_hit_rec": round(rec, 4),
            }
            if gen.get("thinking_removed"):
                row["predict_raw"] = gen["predict_raw"]
                row["thinking_removed"] = True
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            pred_rows.append(row)

            if idx % 10 == 0 or idx == len(samples):
                cur_pre = sum(r["line_hit_pre"] for r in pred_rows) / len(pred_rows)
                cur_rec = sum(r["line_hit_rec"] for r in pred_rows) / len(pred_rows)
                print(
                    f"  [{args.name}] {idx}/{len(samples)} "
                    f"pre={cur_pre:.2f} rec={cur_rec:.2f}",
                    flush=True,
                )

    del model
    gc.collect()
    torch.cuda.empty_cache()

    modes = ("raw", "normalized") if args.score_mode == "both" else (args.score_mode,)
    summaries = []
    for mode in modes:
        summary, _ = score_rows(pred_rows, normalize=(mode == "normalized"))
        summary.update(
            {
                "model": args.name,
                "adapter": adapter or "-",
                "predictions": str(pred_path),
            }
        )
        summaries.append(summary)
        print(
            f"[{summary['score_mode']}] cases={summary['cases']} "
            f"line_hit_pre={summary['line_hit_pre']} "
            f"line_hit_rec={summary['line_hit_rec']} "
            f"perfect={summary['perfect']}/{summary['cases']} "
            f"hit@0.5={summary['hit@0.5']}",
            flush=True,
        )

    summary_path = args.output_dir / "summary_line_hit.csv"
    write_summary_csv(summary_path, summaries)
    print(f"wrote {pred_path}")
    print(f"wrote {summary_path}")


def run_score_only(path: Path, score_mode: str, output: Path | None) -> None:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            label = gold_from_obj(obj)
            predict = obj.get("predict")
            if not isinstance(label, str) or not isinstance(predict, str):
                continue
            rows.append({"label": label, "predict": predict, **obj})
    if not rows:
        raise ValueError(f"no scorable rows in {path}")

    modes = ("raw", "normalized") if score_mode == "both" else (score_mode,)
    summaries = []
    for mode in modes:
        summary, _ = score_rows(rows, normalize=(mode == "normalized"))
        summary.update(
            {"model": path.stem, "adapter": "-", "predictions": str(path)}
        )
        summaries.append(summary)
        print(
            f"[{summary['score_mode']}] cases={summary['cases']} "
            f"line_hit_pre={summary['line_hit_pre']} "
            f"line_hit_rec={summary['line_hit_rec']} "
            f"perfect={summary['perfect']}/{summary['cases']} "
            f"hit@0.5={summary['hit@0.5']}"
        )
    if output:
        write_summary_csv(output, summaries)
        print(f"wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Go FIM with Qwen3-8B then score line_hit pre/rec"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="original Go FIM test jsonl (prompt + response/middle + task_id)",
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE)
    parser.add_argument(
        "--adapter",
        default=DEFAULT_ADAPTER,
        help='LoRA path; "-" or empty = base only',
    )
    parser.add_argument("--name", default="qwen3-8b-go", help="output tag")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None, help="smoke-test limit")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--score-mode",
        choices=("raw", "normalized", "both"),
        default="both",
        help="score after generation (default both)",
    )
    parser.add_argument(
        "--score-only",
        type=Path,
        default=None,
        help="skip generation; only score an existing predictions jsonl",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="summary csv for --score-only (optional)",
    )
    args = parser.parse_args()

    if args.score_only is not None:
        run_score_only(args.score_only, args.score_mode, args.summary_out)
        return

    if args.input is None:
        parser.error("--input is required unless --score-only is set")
    if not args.input.is_file():
        print(f"input not found: {args.input}", file=sys.stderr)
        sys.exit(2)

    run_generate_and_eval(args)


if __name__ == "__main__":
    main()
