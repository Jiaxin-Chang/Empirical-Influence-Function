"""
Automated GPT annotation for training samples.

Replaces the manual copy-paste workflow:
  Old: write samples -> paste to GPT manually -> copy response -> parse
  New: call OpenAI API directly -> parse response -> align to tokenizer

Usage:
    from src.auto_annotate import annotate_samples
    results = annotate_samples(sample_texts, tokenizer)
"""

from __future__ import annotations

import asyncio
import os
import re
from typing import TYPE_CHECKING

from openai import AsyncOpenAI

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

from .NIF import convert_sample_to_full_text, tokenize_with_marked_tokens


# ---------------------------------------------------------------------------
# Prompt (matches NIF.py L1384-1386)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = "You are an expert code analyst."

USER_PROMPT_TEMPLATE = """Below is a full training sample of go code completion:

```
{sample_text}
```

Please:
1. The `<|im_start|>user` part is the question part, and the `<|im_start|>assistant` part is the answer part as the correct answer. The `<MID>` is a completion placeholder which you should not modify. Mark 3-10 words in the question part that you think that are most contributive to the correct answer with `<ATTN></ATTN>`, but don't mark the `<MID>`. These words must natively appear in the context, and not in the natural language text, but code text.
2. Then output only one code block containing the marked full training sample again, not just the marked part."""


# ---------------------------------------------------------------------------
# Code-block extraction (matches NIF.py L1396-1400)
# ---------------------------------------------------------------------------

_CODE_BLOCK_RE = re.compile(r"```[^\n]*\n(.*?)\n```", re.DOTALL)


def extract_fenced_code_blocks(text: str) -> list[str]:
    """Extract content from fenced code blocks in markdown text."""
    return _CODE_BLOCK_RE.findall(text)


# ---------------------------------------------------------------------------
# Single-sample annotation
# ---------------------------------------------------------------------------

async def _annotate_one(
    client: AsyncOpenAI,
    sample_text: str,
    model: str,
    semaphore: asyncio.Semaphore,
) -> str:
    """Call the API for one sample and return the raw response text."""
    async with semaphore:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(sample_text=sample_text)},
            ],
            temperature=0.2,
            max_tokens=8192,
        )
        return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def annotate_sample(
    sample_text: str,
    tokenizer: PreTrainedTokenizer,
    *,
    model: str = "gpt-5.2",
    api_key: str | None = None,
    base_url: str | None = None,
) -> dict:
    """
    Annotate a single training sample using GPT.

    Returns:
        {
            "marked_token_indices": [int, ...],
            "marked_text": str,
            "token_result": dict,
            "rationale": str,  # raw GPT response for debugging
        }
    """
    return asyncio.run(
        _annotate_samples_async(
            [sample_text], tokenizer,
            model=model, api_key=api_key, base_url=base_url,
            concurrency=1,
        )
    )[0]


def annotate_samples(
    sample_texts: list[str],
    tokenizer: PreTrainedTokenizer,
    *,
    model: str = "gpt-5.2",
    api_key: str | None = None,
    base_url: str | None = None,
    concurrency: int = 5,
) -> list[dict]:
    """
    Annotate multiple training samples concurrently.

    Args:
        sample_texts: List of full sample texts (from convert_sample_to_full_text).
        tokenizer: The model's tokenizer (for character-level alignment).
        model: OpenAI model name.
        api_key: Override OPENAI_API_KEY env var.
        base_url: Override OPENAI_BASE_URL for compatible endpoints.
        concurrency: Max parallel API calls.

    Returns:
        List of dicts, each:
        {
            "marked_token_indices": [int, ...],
            "marked_text": str,
            "token_result": dict,
            "rationale": str,
        }
    """
    return asyncio.run(
        _annotate_samples_async(
            sample_texts, tokenizer,
            model=model, api_key=api_key, base_url=base_url,
            concurrency=concurrency,
        )
    )


def annotate_raw_samples(
    samples: list[dict],
    tokenizer: PreTrainedTokenizer,
    **kwargs,
) -> list[dict]:
    """
    Convenience: takes raw sample dicts (with 'system', 'input', 'output' keys)
    and converts them to full text before annotating.
    """
    texts = [convert_sample_to_full_text(s) for s in samples]
    return annotate_samples(texts, tokenizer, **kwargs)


# ---------------------------------------------------------------------------
# Async internals
# ---------------------------------------------------------------------------

async def _annotate_samples_async(
    sample_texts: list[str],
    tokenizer: PreTrainedTokenizer,
    *,
    model: str,
    api_key: str | None,
    base_url: str | None,
    concurrency: int,
) -> list[dict]:
    # Default to xi-api format if no base_url is provided
    final_base_url = base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.xi-ai.cn/v1"
    
    client = AsyncOpenAI(
        api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        base_url=final_base_url,
    )
    semaphore = asyncio.Semaphore(concurrency)

    tasks = [
        _annotate_one(client, text, model, semaphore)
        for text in sample_texts
    ]
    raw_responses = await asyncio.gather(*tasks, return_exceptions=True)

    results = []
    for i, resp in enumerate(raw_responses):
        if isinstance(resp, Exception):
            print(f"[auto_annotate] Sample {i} failed: {resp}")
            results.append({
                "marked_token_indices": [], 
                "marked_text": "",
                "token_result": {},
                "rationale": f"ERROR: {resp}"
            })
            continue

        # Parse code blocks from GPT response
        code_blocks = extract_fenced_code_blocks(resp)
        if not code_blocks:
            print(f"[auto_annotate] Sample {i}: no code block found in response")
            results.append({
                "marked_token_indices": [], 
                "marked_text": "",
                "token_result": {},
                "rationale": resp
            })
            continue

        # Use the first (and ideally only) code block
        marked_text = code_blocks[0]

        # Align to tokenizer using existing infrastructure
        token_result = {}
        try:
            token_result = tokenize_with_marked_tokens(marked_text, tokenizer)
            marked_indices = token_result["marked_indices"]
        except Exception as e:
            print(f"[auto_annotate] Sample {i}: alignment failed: {e}")
            marked_indices = []

        results.append({
            "marked_token_indices": marked_indices,
            "marked_text": marked_text,
            "token_result": token_result,
            "rationale": resp,
        })

    return results

if __name__ == "__main__":
    import os
    import json
    from transformers import AutoTokenizer

    print("Running auto_annotate.py test...")
    if not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable is not set. The API call may fail.")

    sample_text = """<|im_start|>system
You are an expert go assistant.<|im_end|>
<|im_start|>user
```go
package main

import "fmt"

func main() {
    fmt.Println("Hello, World!")
    <MID>
}
```<|im_end|>
<|im_start|>assistant
    fmt.Println("End")<|im_end|>"""

    abs_model_path = os.path.join(os.path.dirname(__file__), "sft/scripts/checkpoint-full")
    if os.path.exists(abs_model_path):
        print(f"Loading tokenizer from {abs_model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(abs_model_path, local_files_only=True)
    else:
        # Fallback to a common tokenizer from huggingface hub if local doesn't exist just for testing
        print(f"Model path {abs_model_path} doesn't exist, falling back to Qwen/Qwen2.5-Coder-7B-Instruct tokenizer from hub...")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
    
    print("Sending request to GPT...")
    try:
        results = annotate_sample(
            sample_text=sample_text,
            tokenizer=tokenizer
        )
        print("\n--- Test Successful ---")
        print(f"Marked Indices (Count: {len(results.get('marked_token_indices', []))}):", results.get("marked_token_indices"))
        print("\n--- Marked Text returned from API ---")
        print(results.get("marked_text"))
    except Exception as e:
        print(f"Test failed: {e}")
