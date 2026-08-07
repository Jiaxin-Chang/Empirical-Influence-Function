"""Turn two precomputed token-embedding dumps into a TTAV bundle.

This is the "any two pieces of code" path that HANDOFF_2026-08-03_new_task_pending.md
left unstarted: unlike export_ttav_bundle.py and export_train_probe_bundle.py, it does
not read an attribution report and does not need a model loaded — the embeddings
already exist on disk. Point-and-click validation of "does this pipeline run at all"
is the only goal of the first version; there is no attribution relationship between
the two inputs, so no pair_signature, no cos_sim, no drawn edges.

Input format (one file per sample), produced by whatever computed the embeddings:
    {
        "hidden": FloatTensor [n_tokens, hidden_dim],   # last-layer embedding, one per token
        "input_ids": LongTensor [n_tokens],
        "labels": LongTensor [n_tokens],                 # -100 = prompt, anything else = answer
        "token_surfaces": list[str],                     # already-decoded token strings
        "text": str,
        "layer": int,
        "model_name_or_path": str,
        "adapter_path": str,
    }

Both files must come from the same model and layer — comparing token vectors from
different embedding spaces is nonsense, so that's a hard error, not a warning.
"""

import argparse
import hashlib
import io
import json
import webbrowser
from pathlib import Path
from urllib import parse as urllib_parse
from urllib import request as urllib_request

import numpy as np
import torch

from .export_ttav_bundle import compute_projection, upload_bundle

# One entry per (side, role): the bundle's class list, in the order labels index into.
# Order matches export_train_probe_bundle.py's PROBE_CLASSES convention (side first,
# then role) so a reader already used to that scheme reads this the same way.
PAIR_CLASSES = ["a_prompt", "a_answer", "b_prompt", "b_answer"]
PAIR_CLASS_TO_ID = {name: idx for idx, name in enumerate(PAIR_CLASSES)}


def load_pair_sample(pt_path: str) -> dict:
    payload = torch.load(pt_path, map_location="cpu", weights_only=False)
    required = {"hidden", "input_ids", "labels", "token_surfaces", "model_name_or_path", "layer"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"{pt_path}: missing required field(s) {sorted(missing)}")
    return payload


def _short_label(side: str, role: str, token_idx: int, token: str) -> str:
    # Mirrors build_short_token_label's P{idx}/O{idx} convention (export_ttav_bundle.py)
    # extended with the side prefix, so formatPointLabel's regex on the TTAV/report side
    # — [PO]\d+|[TX][CST]\d+ — has to be widened for this bundle kind to read its labels
    # instead of falling into the generic "233. …" branch. See ONBOARDING notes.
    tag = f"{side.upper()}{'A' if role == 'answer' else 'P'}"
    normalized = token.replace("\n", "↵").replace("\t", "⇥")
    shortened = normalized[:15] + "…" if len(normalized) > 18 else normalized
    return f"{tag}{token_idx}: {shortened or '·'}"


def _side_points(
    side: str,
    sample: dict,
    point_records: list[dict],
    label_ids: list[int],
    text_list: list[str],
    token_list: list[str],
    embedding_rows: list[np.ndarray],
) -> None:
    tokens = sample["token_surfaces"]
    labels = sample["labels"]
    input_ids = sample["input_ids"]
    hidden = sample["hidden"].numpy()

    n = len(tokens)
    if hidden.shape[0] != n or labels.shape[0] != n or input_ids.shape[0] != n:
        raise ValueError(
            f"{side}: token_surfaces ({n}) does not match hidden/labels/input_ids "
            f"({hidden.shape[0]}/{labels.shape[0]}/{input_ids.shape[0]})"
        )

    is_answer = (labels != -100).tolist()
    for token_idx in range(n):
        role = "answer" if is_answer[token_idx] else "prompt"
        class_name = f"{side}_{role}"
        token = tokens[token_idx]

        label_ids.append(PAIR_CLASS_TO_ID[class_name])
        text_list.append(_short_label(side, role, token_idx, token))
        token_list.append(token)
        embedding_rows.append(hidden[token_idx])
        point_records.append({
            "point_index": len(point_records),
            "side": side,
            "role": role,
            "token_index": token_idx,
            "token_id": int(input_ids[token_idx]),
            "token": token,
            "token_display": token,
        })


def upload_array(ttav_url: str, sample_id: str, kind: str, array: np.ndarray) -> dict:
    """Send one array to TTAV as raw .npy bytes.

    Inlining these as JSON numbers costs ~12 bytes per float, which turned a
    3839x4096 bundle into ~182 MB of text — over nginx's body limit, and enough
    to get the backend worker OOM-killed while parsing it. As .npy the same array
    is 63 MB in float32 and 31 MB in float16, and the server streams it to disk
    without holding it in memory.

    float16 is a transport choice only: the server stores float32 either way.
    """
    buf = io.BytesIO()
    np.save(buf, array, allow_pickle=False)
    body = buf.getvalue()

    url = f"{ttav_url.rstrip('/')}/registerEIFBundleArray?sample_id={sample_id}&kind={kind}"
    req = urllib_request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/octet-stream"},
        method="POST",
    )
    with urllib_request.urlopen(req) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    print(f"[export_pair_bundle] uploaded {kind}: {array.shape} {array.dtype}, {len(body) / 1e6:.1f} MB")
    return result


def infer_pair_sample_id(a_path: str, b_path: str) -> str:
    a_stem = Path(a_path).stem
    b_stem = Path(b_path).stem
    # A short content hash rides along so two different pairs that happen to share
    # filenames (e.g. re-running against a new pair both named "000000"/"000001")
    # don't collide on the same ttav_bundles_real/ directory.
    digest = hashlib.sha1(f"{a_path}|{b_path}".encode("utf-8")).hexdigest()[:8]
    return f"pair_{a_stem}_{b_stem}_{digest}"


def build_pair_bundle_payload(
    a_path: str,
    b_path: str,
    sample_id: str | None = None,
    use_umap: bool = False,
) -> dict:
    a = load_pair_sample(a_path)
    b = load_pair_sample(b_path)

    # Comparing vectors across embedding spaces (different base model, different
    # layer) produces numbers that look like cosines but mean nothing — this has
    # to fail loudly rather than silently plot garbage.
    if a["model_name_or_path"] != b["model_name_or_path"]:
        raise ValueError(
            f"model mismatch: {a_path} used {a['model_name_or_path']!r}, "
            f"{b_path} used {b['model_name_or_path']!r}"
        )
    if a["layer"] != b["layer"]:
        raise ValueError(f"layer mismatch: {a_path} used layer {a['layer']}, {b_path} used layer {b['layer']}")
    if a["hidden"].shape[1] != b["hidden"].shape[1]:
        raise ValueError(
            f"embedding dim mismatch: {a_path} is {a['hidden'].shape[1]}-d, "
            f"{b_path} is {b['hidden'].shape[1]}-d"
        )

    sample_id = sample_id or infer_pair_sample_id(a_path, b_path)

    point_records: list[dict] = []
    label_ids: list[int] = []
    text_list: list[str] = []
    token_list: list[str] = []
    embedding_rows: list[np.ndarray] = []

    _side_points("a", a, point_records, label_ids, text_list, token_list, embedding_rows)
    a_count = len(point_records)
    _side_points("b", b, point_records, label_ids, text_list, token_list, embedding_rows)

    embeddings = np.stack(embedding_rows).astype(np.float32)
    # Projected together, not per-side: fitting PCA separately would give each side
    # its own basis, and "A's points are near B's points" would say nothing —
    # export_train_probe_bundle.py hit exactly this failure mode with an earlier,
    # asymmetric version of the probe bundle (see ONBOARDING_visualization.md §4).
    projection = compute_projection(embeddings, use_umap=use_umap)

    payload = {
        "sample_id": sample_id,
        "vis_method": "TimeVis",
        "vis_id": "1",
        "overwrite": True,
        "bundle": {
            "model": a["model_name_or_path"],
            "checkpoint_path": a.get("adapter_path", ""),
            "embedding_type": "contextual",
            "embedding_note": f"contextual_hidden_state_layer_{a['layer']}",
            "classes": PAIR_CLASSES,
            "sample_index": 0,
            # Reused as the A/B split point, the same way probe bundles overload
            # it as the train/test split rather than an actual prompt length
            # (see ONBOARDING_visualization.md §4's comparison table).
            "prompt_len": a_count,
            "labels": label_ids,
            "text_list": text_list,
            "text_data": text_list,
            "token_list": token_list,
            "index": {
                "train": list(range(a_count)),
                "test": list(range(a_count, len(point_records))),
            },
            "probe_metadata": {
                "kind": "arbitrary_pair",
                "a_path": a_path,
                "b_path": b_path,
                "a_token_count": a_count,
                "b_token_count": len(point_records) - a_count,
                "point_records": point_records,
                # No attribution relationship between these two inputs — nothing
                # to draw an edge for. See probeOverlay.ts / the ONBOARDING note
                # this feature is recorded under for why that's a deliberate
                # empty list rather than a missing field.
                "pair_signature": [],
            },
        },
    }
    # Returned alongside rather than inside the payload: the caller decides
    # whether they travel as .npy uploads or get inlined into the JSON.
    return payload, embeddings, projection


def _inline_arrays(payload: dict, embeddings: np.ndarray, projection: np.ndarray) -> dict:
    """Fold the arrays into the JSON — the original protocol, kept as a fallback."""
    payload["bundle"]["embeddings"] = embeddings.astype(np.float32).tolist()
    payload["bundle"]["projection"] = projection.tolist()
    return payload


def main():
    parser = argparse.ArgumentParser(
        description="Build a TTAV bundle from two precomputed token-embedding .pt files "
                     "and upload it, with no attribution report involved."
    )
    parser.add_argument("--a", required=True, help="Path to the first sample's .pt file.")
    parser.add_argument("--b", required=True, help="Path to the second sample's .pt file.")
    parser.add_argument("--sample-id", default=None, help="Override the inferred sample id.")
    parser.add_argument("--projection", choices=["pca", "umap"], default="pca")
    parser.add_argument("--ttav-url", default="http://1.94.115.154/", help="TTAV base URL.")
    parser.add_argument(
        "--out-dir", default=None,
        help="If set, also write bundle_payload.json under <out-dir>/<sample_id>/ "
             "(same layout as ttav_bundles_real/), independent of the upload."
    )
    parser.add_argument("--no-upload", action="store_true", help="Build and optionally save, but skip the TTAV POST.")
    parser.add_argument(
        "--open", action="store_true",
        help="Open the result in a browser. Only works where one exists — on a "
             "headless server it prints the URL instead."
    )
    parser.add_argument(
        "--array-transport", choices=["binary", "inline"], default="binary",
        help="binary (default) streams embeddings/projection as .npy to "
             "/registerEIFBundleArray; inline embeds them as JSON numbers, which "
             "is ~6x larger and only works for small bundles."
    )
    parser.add_argument(
        "--array-dtype", choices=["float16", "float32"], default="float16",
        help="Wire dtype for binary transport. The server stores float32 either "
             "way; float16 just halves what crosses the network."
    )
    args = parser.parse_args()

    payload, embeddings, projection = build_pair_bundle_payload(
        a_path=args.a,
        b_path=args.b,
        sample_id=args.sample_id,
        use_umap=(args.projection == "umap"),
    )
    sample_id = payload["sample_id"]
    n_points = len(payload["bundle"]["labels"])
    print(f"[export_pair_bundle] sample_id={sample_id} points={n_points} dim={embeddings.shape[1]}")

    if args.out_dir:
        # The saved copy is self-contained, so it always inlines the arrays —
        # it's a cache/artifact, not something that has to cross a network.
        out_path = Path(args.out_dir) / sample_id / "bundle_payload.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(_inline_arrays(dict(payload, bundle=dict(payload["bundle"])), embeddings, projection)),
            encoding="utf-8",
        )
        print(f"[export_pair_bundle] wrote {out_path}")

    if args.no_upload:
        return

    if args.array_transport == "binary":
        wire_dtype = np.float16 if args.array_dtype == "float16" else np.float32
        upload_array(args.ttav_url, sample_id, "embeddings", embeddings.astype(wire_dtype, copy=False))
        # Projection stays float32: it's only 2 columns, so there is nothing to
        # save, and these are the coordinates every point is drawn at.
        upload_array(args.ttav_url, sample_id, "projection", projection.astype(np.float32, copy=False))
        payload["bundle"]["arrays_uploaded"] = True
    else:
        _inline_arrays(payload, embeddings, projection)

    upload_url = args.ttav_url.rstrip("/") + "/registerEIFBundle"
    result = upload_bundle(upload_url, payload)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # Only what plotView actually reads and can't default: it falls back to
    # visId '1', dataType 'Text' and taskType 'Alignment' on its own, and never
    # looks at promptLen. Dropping those keeps the URL short enough to paste.
    jump_payload = {
        "source": "eif",
        "sampleId": sample_id,
        "contentPath": result.get("content_path", ""),
        "visMethod": payload["vis_method"],
    }
    # Percent-encoded: the raw JSON contains {}, quotes and spaces, which
    # terminals wrap and some of which browsers reject when pasted bare.
    jump_url = (
        f"{args.ttav_url.rstrip('/')}/?eif_jump="
        f"{urllib_parse.quote(json.dumps(jump_payload, separators=(',', ':')))}"
    )
    print("\nOpen in browser:")
    print(jump_url)

    if args.open:
        # Only meaningful where a browser exists — on a headless server this
        # quietly does nothing, so it stays opt-in rather than the default.
        if webbrowser.open(jump_url):
            print("(opened in your default browser)")
        else:
            print("(no browser available here — copy the URL above)")


if __name__ == "__main__":
    main()
