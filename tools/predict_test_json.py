#!/usr/bin/env python3
"""Greedy FIM prediction for a single test.json (or JSONL) with Qwen3 + LoRA.

Aligned with continue-train / AI4Go eval: ChatML, thinking off, greedy decode.

Example (this repo's Wrap/New case lives in ./test.json):

  python tools/predict_test_json.py \\
      --base-model /mnt/md124/jiaxin/models/Qwen3-8B \\
      --ce-saliency-adapter /mnt/md124/jiaxin/training_code/Empirical-Influence-Function/outputs/ce_saliency \\
      --input test.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_samples(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"empty input: {path}")
    if path.suffix.lower() == ".jsonl" or "\n" in raw and raw.lstrip()[:1] != "{":
        rows = []
        for i, line in enumerate(raw.splitlines()):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{i} is not an object")
            rows.append(obj)
        return rows
    obj = json.loads(raw)
    if isinstance(obj, list):
        return list(obj)
    if isinstance(obj, dict):
        return [obj]
    raise ValueError(f"{path}: expected JSON object / list / JSONL")


def _user_prompt(sample: dict[str, Any]) -> str:
    for key in ("prompt", "input", "query"):
        val = sample.get(key)
        if isinstance(val, str) and val.strip():
            return val
    raise ValueError("sample missing prompt/input")


def _gold(sample: dict[str, Any]) -> str:
    for key in ("response", "label", "output", "gold"):
        val = sample.get(key)
        if isinstance(val, str):
            return val
    return ""


def _render_chatml(tokenizer, user_text: str, *, enable_thinking: bool = False) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_text},
    ]
    apply = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply):
        try:
            return apply(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except (TypeError, ValueError):
            if not enable_thinking:
                messages[-1] = {
                    "role": "user",
                    "content": f"{user_text}\n/no_think",
                }
            try:
                return apply(messages, tokenize=False, add_generation_prompt=True)
            except (TypeError, ValueError):
                pass
    suffix = "" if enable_thinking else "\n/no_think"
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{user_text}{suffix}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _strip_thinking(text: str) -> str:
    if not text:
        return ""
    pattern = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
    new, n = pattern.subn("", text, count=1)
    if n:
        return new
    if re.match(r"^\s*<think>", text, re.IGNORECASE):
        return ""
    return text


def _del_spaces(txt: str) -> str:
    return re.sub(r"\s", "", txt or "")


def _to_lines(text: str) -> list[str]:
    rows: list[str] = []
    for raw in text.split("\n"):
        item = _del_spaces(raw)
        if item and item not in "{}":
            rows.append(item)
    return rows


def line_hit(expect: str, actual: str, method: str) -> float:
    expects = _to_lines(expect)
    actuals = _to_lines(actual)
    denom = len(expects) if method == "recall" else len(actuals)
    if denom == 0:
        return 0.0
    pool = list(actuals)
    cnt = 0
    for item in expects:
        if item in pool:
            cnt += 1
            pool.remove(item)
    return min(1.0, cnt / denom) * 100.0


def load_ce_saliency(base_model: str, adapter: str):
    print(f"[predict] base={base_model}", flush=True)
    print(f"[predict] adapter={adapter}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(model, adapter, local_files_only=True)
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = True
    return tokenizer, model


@torch.inference_mode()
def greedy_predict(
    tokenizer,
    model,
    user_text: str,
    *,
    max_new_tokens: int,
) -> dict[str, Any]:
    chat = _render_chatml(tokenizer, user_text, enable_thinking=False)
    encoded = tokenizer(chat, return_tensors="pt", add_special_tokens=False)
    device = next(model.parameters()).device
    encoded = {k: v.to(device) for k, v in encoded.items()}
    prompt_tokens = int(encoded["input_ids"].shape[1])

    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos_ids = sorted({
        i for i in [tokenizer.eos_token_id, im_end] if isinstance(i, int) and i >= 0
    })

    out = model.generate(
        **encoded,
        max_new_tokens=max(1, int(max_new_tokens)),
        do_sample=False,
        num_beams=1,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=eos_ids or tokenizer.eos_token_id,
    )
    gen = out[0, prompt_tokens:]
    raw = tokenizer.decode(gen, skip_special_tokens=False)
    predict = _strip_thinking(tokenizer.decode(gen, skip_special_tokens=True))
    finish = "stop" if gen.numel() and int(gen[-1]) in eos_ids else "length"
    return {
        "predict": predict,
        "predict_raw": raw,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": int(gen.numel()),
        "finish_reason": finish,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predict FIM completion for test.json")
    parser.add_argument(
        "--base-model",
        default="/mnt/md124/jiaxin/models/Qwen3-8B",
        help="Qwen3-8B (or other) base checkpoint",
    )
    parser.add_argument(
        "--ce-saliency-adapter",
        default="/mnt/md124/jiaxin/training_code/Empirical-Influence-Function/outputs/ce_saliency",
        help="ce_saliency LoRA adapter directory",
    )
    parser.add_argument(
        "--input",
        default=str(REPO_ROOT / "test.json"),
        help="JSON object or JSONL with prompt + response",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Write predictions JSON here (default: <input>.predict.json)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args(argv)

    in_path = Path(args.input).expanduser().resolve()
    if not in_path.is_file():
        print(f"[predict] input not found: {in_path}", file=sys.stderr)
        return 2
    out_path = Path(args.output).expanduser().resolve() if args.output else in_path.with_suffix(
        in_path.suffix + ".predict.json" if in_path.suffix else ".predict.json"
    )
    if not args.output:
        out_path = in_path.with_name(in_path.stem + ".predict.json")

    samples = _load_samples(in_path)
    tokenizer, model = load_ce_saliency(args.base_model, args.ce_saliency_adapter)

    results: list[dict[str, Any]] = []
    for i, sample in enumerate(samples):
        prompt = _user_prompt(sample)
        gold = _gold(sample)
        print(
            f"[predict] sample {i}/{len(samples)} task_id={sample.get('task_id')!r} "
            f"prompt_chars={len(prompt)}",
            flush=True,
        )
        gen = greedy_predict(
            tokenizer, model, prompt, max_new_tokens=args.max_new_tokens,
        )
        pred = gen["predict"]
        exact = pred.strip() == gold.strip() if gold else None
        row = {
            "task_id": sample.get("task_id"),
            "predict": pred,
            "label": gold,
            "exact_match": exact,
            "line_hit_pre": line_hit(gold, pred, "precision") if gold else None,
            "line_hit_rec": line_hit(gold, pred, "recall") if gold else None,
            "prompt_tokens": gen["prompt_tokens"],
            "generated_tokens": gen["generated_tokens"],
            "finish_reason": gen["finish_reason"],
        }
        results.append(row)

        print("=" * 72, flush=True)
        print(f"task_id: {row['task_id']}", flush=True)
        print(f"tokens: prompt={row['prompt_tokens']} gen={row['generated_tokens']} "
              f"finish={row['finish_reason']}", flush=True)
        if gold:
            print("--- gold ---", flush=True)
            print(gold, end="" if gold.endswith("\n") else "\n", flush=True)
        print("--- predict ---", flush=True)
        print(pred, end="" if pred.endswith("\n") else "\n", flush=True)
        if gold:
            print(
                f"exact_match={exact}  line_hit_pre={row['line_hit_pre']:.1f}  "
                f"line_hit_rec={row['line_hit_rec']:.1f}",
                flush=True,
            )
        print("=" * 72, flush=True)

    payload = {
        "base_model": args.base_model,
        "ce_saliency_adapter": args.ce_saliency_adapter,
        "input": str(in_path),
        "n": len(results),
        "results": results,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[predict] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
