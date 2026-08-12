import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from src.export_ttav_bundle import (
    build_short_token_label,
    compute_projection,
    decode_token,
    infer_sample_id,
    normalize_token_for_display,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = REPO_ROOT / "src" / "sft" / "scripts" / "nif-checkpoints" / "checkpoint-full" / "checkpoint-2785"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "ttav_bundles_real"
_MODEL_AND_TOKENIZER_CACHE: dict[tuple[str, str], tuple[object, object]] = {}
ProgressCallback = Callable[[str, str], None]


def load_report(report_json_path: str) -> dict:
    with open(report_json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fffd_decode_index(tokenizer) -> dict[str, list[int]]:
    """Map decode(id) → [ids] for vocab entries whose decode contains U+FFFD."""
    cached = getattr(tokenizer, "_eif_fffd_decode_index", None)
    if cached is not None:
        return cached
    index: dict[str, list[int]] = {}
    for tid in range(len(tokenizer)):
        dec = tokenizer.decode([tid])
        if "\ufffd" in dec:
            index.setdefault(dec, []).append(tid)
    setattr(tokenizer, "_eif_fffd_decode_index", index)
    return index


def _lookup_report_token_id(tokenizer, tok: str, *, vocab: dict, unk_id, unk_token) -> int | None:
    """Map one report token surface back to a vocab id.

    Reports from ``intervention_experiment`` store ``tokenizer.decode([id])``
    strings (e.g. ``'\\n'``, Chinese chars), not SentencePiece piece names
    (``Ċ``, …). ``convert_tokens_to_ids`` only handles the latter, so we fall
    back to single-piece ``encode``. Surfaces containing U+FFFD must NOT use
    encode (it collapses many byte-tokens to one fake id); use a unique reverse
    decode hit instead.
    """
    token_id = tokenizer.convert_tokens_to_ids(tok)
    is_missing = token_id is None
    if not is_missing and unk_id is not None and token_id == unk_id and tok != unk_token:
        is_missing = tok not in vocab
    if not is_missing:
        return int(token_id)

    vocab_id = vocab.get(tok)
    if vocab_id is not None:
        return int(vocab_id)

    if "\ufffd" in tok:
        hits = _fffd_decode_index(tokenizer).get(tok) or []
        if len(hits) == 1:
            return int(hits[0])
        return None

    # Display / decode surfaces → single model token.
    encoded = tokenizer.encode(tok, add_special_tokens=False)
    if len(encoded) == 1:
        return int(encoded[0])

    # Older reports sometimes keep GPT-2/Qwen marker forms.
    alts: list[str] = []
    if " " in tok or "\n" in tok or "\t" in tok:
        alts.append(tok.replace(" ", "Ġ").replace("\n", "Ċ").replace("\t", "ĉ"))
    if tok and not tok.startswith(("Ġ", "Ċ", "ĉ")):
        alts.extend([f"Ġ{tok}", f"Ċ{tok}", f"ĉ{tok}"])
    for alt in alts:
        if alt in vocab:
            return int(vocab[alt])
        enc = tokenizer.encode(alt, add_special_tokens=False)
        if len(enc) == 1:
            return int(enc[0])

    return None


def convert_report_tokens_to_ids(
    tokenizer,
    report_tokens: list[str],
    *,
    hint_ids: list[int] | None = None,
    hint_until: int | None = None,
) -> list[int]:
    """Map report token surfaces back to vocab ids.

    ``hint_ids`` / ``hint_until``: when a surface cannot be mapped (common for
    U+FFFD byte-fallback pieces), reuse ``hint_ids[i]`` for ``i < hint_until``
    (typically the shared prompt from ``full_token_ids``).
    """
    vocab = tokenizer.get_vocab()
    unk_id = getattr(tokenizer, "unk_token_id", None)
    unk_token = getattr(tokenizer, "unk_token", None)

    missing: list[tuple[int, str]] = []
    recovered_ids: list[int] = []
    hint_cap = len(hint_ids) if hint_ids is not None else 0
    if hint_until is not None:
        hint_cap = min(hint_cap, int(hint_until))

    for idx, tok in enumerate(report_tokens):
        resolved = _lookup_report_token_id(
            tokenizer, tok, vocab=vocab, unk_id=unk_id, unk_token=unk_token
        )
        if resolved is None and hint_ids is not None and idx < hint_cap:
            recovered_ids.append(int(hint_ids[idx]))
            continue
        if resolved is None:
            missing.append((idx, tok))
            recovered_ids.append(-1)
        else:
            recovered_ids.append(resolved)

    if missing:
        # Last resort: re-tokenize the joined surfaces when length is preserved.
        joined = "".join(report_tokens)
        all_ids = tokenizer.encode(joined, add_special_tokens=False)
        if len(all_ids) == len(report_tokens):
            return [int(x) for x in all_ids]
        preview = ", ".join(f"{idx}:{repr(tok)}" for idx, tok in missing[:10])
        raise ValueError(
            "Failed to map some report tokens back to tokenizer ids. "
            f"Examples: {preview}. "
            "Usually U+FFFD byte-fallback surfaces without stored token ids — "
            "re-run intervention to write correct_full_token_ids, or ensure "
            "full_token_ids covers the prompt prefix."
        )

    return recovered_ids


def validate_roundtrip_tokens(tokenizer, token_ids: list[int], report_tokens: list[str]):
    """Accept either vocab-piece equality or decode-surface equality."""
    mismatches: list[tuple[int, str, str, int]] = []
    for idx, (src, tid) in enumerate(zip(report_tokens, token_ids)):
        vocab_tok = tokenizer.convert_ids_to_tokens(int(tid))
        decoded = tokenizer.decode([int(tid)])
        if src == vocab_tok or src == decoded:
            continue
        # Corrupted replacement chars in old JSON cannot round-trip; skip soft.
        if "\ufffd" in src:
            continue
        mismatches.append((idx, src, decoded, int(tid)))
    if mismatches:
        preview = ", ".join(
            f"{idx}:{repr(src)}->{repr(back)}(id={tok_id})"
            for idx, src, back, tok_id in mismatches[:10]
        )
        raise ValueError(
            "Tokenizer round-trip mismatch for some report tokens. "
            f"Examples: {preview}"
        )


def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def load_model_and_tokenizer(
    model_path: str,
    dtype_name: str,
    progress_callback: ProgressCallback | None = None,
):
    cache_key = (str(Path(model_path).resolve()), dtype_name)
    cached = _MODEL_AND_TOKENIZER_CACHE.get(cache_key)
    if cached is not None:
        if progress_callback:
            progress_callback("model_ready", "Reusing cached model and tokenizer")
        return cached

    if progress_callback:
        progress_callback("loading_model", "Loading model and tokenizer from local checkpoint")

    config = AutoConfig.from_pretrained(
        model_path,
        attn_implementation="eager",
        output_attentions=False,
        use_cache=False,
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    dtype = resolve_torch_dtype(dtype_name)
    model_kwargs = {
        "config": config,
        "torch_dtype": dtype,
        "trust_remote_code": True,
        "local_files_only": True,
        # Stream weights shard-by-shard instead of materializing a full extra
        # copy during load. Cuts peak host RAM roughly in half — critical on
        # this 7GB-RAM box where loading otherwise OOM-kills the API.
        "low_cpu_mem_usage": True,
    }
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs).eval()
    _MODEL_AND_TOKENIZER_CACHE[cache_key] = (model, tokenizer)
    if progress_callback:
        progress_callback("model_ready", "Model and tokenizer loaded")
    return model, tokenizer


def compute_input_embeddings(model, input_ids: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        embed_layer = model.get_input_embeddings()
        embeddings = embed_layer(input_ids)[0].detach().float().cpu().numpy()
    return embeddings.astype(np.float32)


def compute_contextual_embeddings(model, input_ids: torch.Tensor, hidden_layer: int) -> np.ndarray:
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Model did not return hidden_states.")
        embeddings = hidden_states[hidden_layer][0].detach().float().cpu().numpy()
    return embeddings.astype(np.float32)


def build_real_bundle_payload(
    report_json_path: str,
    model_path: str,
    sample_id: str | None = None,
    embedding_type: str = "contextual",
    hidden_layer: int = -1,
    vis_method: str = "TimeVis",
    vis_id: str = "1",
    dtype_name: str = "bfloat16",
    progress_callback: ProgressCallback | None = None,
) -> dict:
    if progress_callback:
        progress_callback("loading_report", "Loading EIF report JSON")
    report = load_report(report_json_path)
    sample_id = sample_id or infer_sample_id(report_json_path)

    report_tokens = report["test_sample_baseline"]["full_tokens"]
    prompt_len = int(report["test_sample_baseline"]["prompt_len"])
    test_index = int(report["experiment_meta"]["test_sample_index"])

    model, tokenizer = load_model_and_tokenizer(
        model_path,
        dtype_name=dtype_name,
        progress_callback=progress_callback,
    )
    if progress_callback:
        progress_callback("mapping_tokens", "Mapping report tokens to tokenizer ids")
    token_ids = convert_report_tokens_to_ids(tokenizer, report_tokens)
    validate_roundtrip_tokens(tokenizer, token_ids, report_tokens)

    if getattr(model, "device", None) is not None:
        device = model.device
    else:
        device = next(model.parameters()).device
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    if progress_callback:
        progress_callback("computing_embeddings", "Computing real token embeddings")
    if embedding_type == "input":
        embeddings = compute_input_embeddings(model, input_ids)
        model_note = "input_embedding_lookup"
    elif embedding_type == "contextual":
        embeddings = compute_contextual_embeddings(model, input_ids, hidden_layer)
        model_note = f"contextual_hidden_state_layer_{hidden_layer}"
    else:
        raise ValueError(f"Unsupported embedding_type: {embedding_type}")

    labels = [0 if i < prompt_len else 1 for i in range(len(report_tokens))]
    text_list = []
    token_list = []
    for idx, token in enumerate(report_tokens):
        role = "prompt" if idx < prompt_len else "output"
        text_list.append(build_short_token_label(token, idx, role))
        token_list.append(normalize_token_for_display(token))

    if progress_callback:
        progress_callback("projecting_embeddings", "Projecting embeddings to 2D via UMAP")
    projection = compute_projection(embeddings, use_umap=True)

    if progress_callback:
        progress_callback("packaging_bundle", "Packaging TTAV bundle payload")

    return {
        "sample_id": sample_id,
        "vis_method": vis_method,
        "vis_id": vis_id,
        "overwrite": True,
        "bundle": {
            "model": Path(model_path).name,
            "checkpoint_path": model_path,
            "embedding_type": embedding_type,
            "embedding_note": model_note,
            "embedding_dtype": dtype_name,
            "classes": ["prompt", "output"],
            "sample_index": test_index,
            "prompt_len": prompt_len,
            "labels": labels,
            "text_list": text_list,
            "text_data": text_list,
            "token_list": token_list,
            "raw_tokens": report_tokens,
            "reconstructed_text": "".join(decode_token(tok) for tok in report_tokens),
            "token_ids": token_ids,
            "index": {"train": list(range(len(report_tokens))), "test": []},
            "embeddings": embeddings.tolist(),
            "projection": projection.tolist(),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Export a TTAV bundle with real model-derived token embeddings."
    )
    parser.add_argument(
        "--report-json",
        required=True,
        help="Path to correlation_matching_results_*_all_tokens*.json",
    )
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to the local Hugging Face checkpoint directory.",
    )
    parser.add_argument(
        "--sample-id",
        default=None,
        help="Override the inferred sample id used for bundle naming.",
    )
    parser.add_argument(
        "--embedding-type",
        choices=["input", "contextual"],
        default="contextual",
        help="Use model input embeddings or contextual hidden states.",
    )
    parser.add_argument(
        "--hidden-layer",
        type=int,
        default=-1,
        help="Layer index for contextual embeddings. -1 means the last hidden layer.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
        help="Torch dtype used to load the checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write the TTAV bundle to. Defaults to ttav_bundles_real/{sampleId}.",
    )
    parser.add_argument("--vis-method", default="TimeVis")
    parser.add_argument("--vis-id", default="1")
    args = parser.parse_args()

    payload = build_real_bundle_payload(
        report_json_path=args.report_json,
        model_path=args.model_path,
        sample_id=args.sample_id,
        embedding_type=args.embedding_type,
        hidden_layer=args.hidden_layer,
        vis_method=args.vis_method,
        vis_id=args.vis_id,
        dtype_name=args.dtype,
    )

    from src.ttav_bundle_api import write_local_bundle_cache

    sample_id = payload["sample_id"]
    output_dir = Path(args.output_dir) if args.output_dir else (DEFAULT_OUTPUT_ROOT / sample_id)
    write_local_bundle_cache(sample_id, payload, explicit_path=str(output_dir))

    summary = {
        "status": "ok",
        "sample_id": sample_id,
        "output_dir": str(output_dir),
        "embedding_type": args.embedding_type,
        "hidden_layer": args.hidden_layer if args.embedding_type == "contextual" else None,
        "dtype": args.dtype,
        "shape": list(np.asarray(payload["bundle"]["embeddings"]).shape),
        "model_path": args.model_path,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
