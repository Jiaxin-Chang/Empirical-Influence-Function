#!/usr/bin/env python3
"""Fix garbled Chinese in full_tokens / correct_full_tokens fields.

Root cause: old code used tokenizer.convert_ids_to_tokens() which encodes
every byte as a Latin-1 Unicode codepoint (GPT-2 byte map).  Chinese UTF-8
characters (3 bytes each) are sometimes split across *multiple* BPE tokens,
so fixing one token at a time fails for the split-byte cases.

Fix strategy:
  1. Find consecutive runs of GPT-2-encoded tokens.
  2. Concatenate their raw bytes into one stream.
  3. Decode the whole stream as UTF-8.
  4. Re-split the decoded Unicode back into the original number of slots
     (a char is assigned to the slot whose byte range contains the char's
     first byte).  Split-byte leftovers become empty strings.

Already-decoded tokens (proper Chinese characters, etc.) are detected by
having at least one character outside the GPT-2 byte-map range and are
left untouched.
"""
import bisect
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# GPT-2 byte map (reversed): Unicode codepoint → raw byte value
# ---------------------------------------------------------------------------

def _build_cp2byte() -> dict[int, int]:
    bs = (
        list(range(ord("!"), ord("~") + 1))      # 0x21-0x7e  printable ASCII
        + list(range(ord("¡"), ord("¬") + 1))    # 0xa1-0xac
        + list(range(ord("®"), ord("ÿ") + 1))    # 0xae-0xff
    )
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(cs, bs))   # codepoint → byte


_CP2BYTE: dict[int, int] = _build_cp2byte()


def _is_bpe_encoded(token: str) -> bool:
    """True iff every character in *token* is in the GPT-2 byte map.

    After the first (incomplete) fix some tokens may already be proper Unicode
    (e.g. '适用于').  Those have codepoints outside the byte-map range and
    will return False here — leaving them untouched.
    """
    return bool(token) and all(ord(c) in _CP2BYTE for c in token)


# ---------------------------------------------------------------------------
# Core decoder
# ---------------------------------------------------------------------------

def _decode_run(tokens: list[str]) -> list[str]:
    """Decode a consecutive BPE-encoded run as one UTF-8 byte stream,
    then redistribute the decoded characters back to the original slots."""
    per_bytes = [bytes(_CP2BYTE[ord(c)] for c in t) for t in tokens]
    all_bytes = b"".join(per_bytes)

    full_text = all_bytes.decode("utf-8", errors="replace")

    # Build char → byte-start offset table (needed for binary search below)
    char_byte_starts: list[int] = []
    pos = 0
    for ch in full_text:
        char_byte_starts.append(pos)
        pos += len(ch.encode("utf-8"))

    # Assign each decoded char to the token slot whose byte range contains
    # that char's first byte.  Use binary search for O(n log n) overall.
    result: list[str] = [""] * len(tokens)
    cum = 0
    for ti, tb in enumerate(per_bytes):
        start_b = cum
        end_b   = cum + len(tb)
        lo = bisect.bisect_left(char_byte_starts, start_b)
        hi = bisect.bisect_left(char_byte_starts, end_b)
        result[ti] = full_text[lo:hi]
        cum = end_b

    return result


def fix_token_sequence(tokens: list[str]) -> tuple[list[str], int]:
    """Fix an entire token list in place, handling partial UTF-8 splits."""
    result = list(tokens)
    changed = 0

    i = 0
    while i < len(result):
        if _is_bpe_encoded(result[i]):
            # Collect the contiguous BPE-encoded run
            j = i
            while j < len(result) and _is_bpe_encoded(result[j]):
                j += 1
            # Decode the whole run at once
            fixed = _decode_run(result[i:j])
            for k, (orig, new) in enumerate(zip(result[i:j], fixed)):
                if orig != new:
                    result[i + k] = new
                    changed += 1
            i = j
        else:
            i += 1

    return result, changed


# ---------------------------------------------------------------------------
# File-level entry point
# ---------------------------------------------------------------------------

def fix_file(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    total_changed = 0

    # test_sample_baseline: full_tokens (model output) + correct_full_tokens (ground truth)
    baseline = data.get("test_sample_baseline", {})
    for field in ("full_tokens", "correct_full_tokens"):
        if field in baseline:
            baseline[field], n = fix_token_sequence(baseline[field])
            total_changed += n

    # train_sample_details: full_tokens for every train sample
    for detail in data.get("train_sample_details", {}).values():
        if "full_tokens" in detail:
            detail["full_tokens"], n = fix_token_sequence(detail["full_tokens"])
            total_changed += n

    if total_changed:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[fixed]     {path.name}  ({total_changed} tokens corrected)")
    else:
        print(f"[no change] {path.name}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 fix_garbled_tokens.py <file1.json> [file2.json ...]")
        sys.exit(1)
    for arg in sys.argv[1:]:
        fix_file(Path(arg))
