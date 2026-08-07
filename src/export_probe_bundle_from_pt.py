"""Build a TTAV probe bundle from an attribution report plus precomputed embeddings.

Same output as export_train_probe_bundle.py, but without loading a model. That
script's only GPU dependency is _compute_sequence_embeddings(); when the token
vectors already exist on disk, that step becomes a lookup and the whole probe can
be built on a machine that cannot hold a 7B model — which is the situation on the
visualizer host (EIF_CACHE_ONLY=1).

The report supplies what embeddings cannot: which train sample pairs with which
test token, the source→target edges, and cos_sim. cos_sim lives in gradient space
and is not recoverable from representations, so a probe without a report is not
possible — see ONBOARDING_visualization.md §4 on the three similarities.

    python -m src.export_probe_bundle_from_pt \
        --report  reports/report_ce_only.json \
        --test-pt test/test_ce_only.pt \
        --train-pt 0=train/000000.pt \
        --test-target 2950 --train-id 0

## Index alignment

A report's full_tokens is frequently a *window* of what the .pt holds: reports get
truncated to a fixed length while the .pt keeps the whole sequence. So report index
i maps to .pt index i + offset, and the offset is discovered by locating the
report's token sequence inside the .pt's, then verified token by token. Getting
this wrong silently indexes the wrong vector — the plot still renders, it is just
describing different tokens than the report is talking about.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .export_pair_bundle import upload_array
from .export_ttav_bundle import (
    compute_projection,
    infer_sample_id,
    normalize_token_for_display,
    upload_bundle,
)

# Same six classes, in the same order, as export_train_probe_bundle.py — the
# frontend colours by label id, so this ordering is part of the wire format.
PROBE_CLASSES = [
    "train_context",
    "train_source",
    "train_target",
    "test_context",
    "test_source",
    "test_target",
]
PROBE_CLASS_TO_ID = {name: idx for idx, name in enumerate(PROBE_CLASSES)}

_LABEL_PREFIX = {
    ("train", "context"): "TC",
    ("train", "source"): "TS",
    ("train", "target"): "TT",
    ("test", "context"): "XC",
    ("test", "source"): "XS",
    ("test", "target"): "XT",
}


def load_pt(path: str) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    missing = {"hidden", "token_surfaces", "model_name_or_path", "layer"} - payload.keys()
    if missing:
        raise ValueError(f"{path}: missing required field(s) {sorted(missing)}")
    return payload


def align_report_to_pt(
    report_tokens: list[str],
    pt_tokens: list[str],
    what: str,
    allow_truncate: bool = False,
) -> tuple[int, int]:
    """Find (offset, usable) with pt_tokens[i + offset] == report_tokens[i] for i < usable.

    A report's tokens are normally a window of the .pt's, so `usable` is the whole
    report. But the two can also record *different generations of the same prompt*:
    the prompt matches exactly and the outputs diverge partway. There is then no
    offset that maps the whole report, and mapping it anyway would silently pair
    report tokens with unrelated vectors.

    With allow_truncate, the aligned prefix is kept and the rest dropped — losing
    points is recoverable, mislabelled points are not.
    """
    n, m = len(report_tokens), len(pt_tokens)
    if n > m:
        raise ValueError(
            f"{what}: report has {n} tokens but the .pt only has {m}; "
            f"they are not the same sample."
        )

    if "\x00".join(report_tokens) in "\x00".join(pt_tokens):
        for off in range(m - n + 1):
            if pt_tokens[off:off + n] == report_tokens:
                return off, n

    # No full match: locate the offset that agrees longest, which for a
    # same-prompt/different-generation pair is the one aligning the prompts.
    best_off, best_agree = 0, -1
    for off in range(m - n + 1):
        agree = 0
        while agree < n and pt_tokens[off + agree] == report_tokens[agree]:
            agree += 1
        if agree > best_agree:
            best_off, best_agree = off, agree

    detail = (
        f"{what}: the report's token sequence does not occur in the .pt. "
        f"Offset {best_off} agrees for {best_agree}/{n} tokens, then the report has "
        f"{report_tokens[best_agree]!r} where the .pt has {pt_tokens[best_off + best_agree]!r}. "
        f"The two most likely record different generations of the same prompt."
    )
    if not allow_truncate:
        raise ValueError(detail + "\nPass --truncate-to-aligned to keep the matching prefix only.")

    print(f"[warn] {detail}")
    print(f"[warn] --truncate-to-aligned given: keeping {best_agree}/{n} {what} tokens")
    return best_off, best_agree


def write_frontend_cache(
    out_dir: Path,
    payload: dict,
    projection: np.ndarray,
    embeddings: np.ndarray | None = None,
) -> Path:
    """Write the cache the report page reads, as projection.json (+ optional full bundle).

    Splitting the two matters more than it looks. The in-page canvas needs the 2-D
    coordinates and the token metadata; `embeddings` is ~99% of the bytes and is
    only consumed by TTAV's neighbour lines and refine. Keeping them in one file
    meant the dev server re-parsed a 250 MB document on every request just to drop
    a field — measured at 250 MB versus 1.3 MB for the same picture.

    Pass `embeddings` only when the full TTAV path is also wanted; without it the
    probe still renders in the page identically.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    slim = json.loads(json.dumps(payload))
    slim["bundle"]["projection"] = projection.tolist()
    slim_path = out_dir / "projection.json"
    slim_path.write_text(json.dumps(slim), encoding="utf-8")
    print(f"[probe_from_pt] wrote {slim_path} ({slim_path.stat().st_size / 1e6:.1f} MB)")

    if embeddings is not None:
        full = json.loads(json.dumps(payload))
        full["bundle"]["projection"] = projection.tolist()
        full["bundle"]["embeddings"] = embeddings.tolist()
        full_path = out_dir / "bundle_payload.json"
        full_path.write_text(json.dumps(full), encoding="utf-8")
        print(f"[probe_from_pt] wrote {full_path} ({full_path.stat().st_size / 1e6:.0f} MB)")

    return slim_path


def collect_pairs(report: dict, test_target: int, train_id: int, test_source: int | None) -> list[dict]:
    """The attribution edges anchoring this probe, in report index space."""
    out = []
    for entry in report.get("per_token_results", []):
        if entry.get("target_token_index") != test_target:
            continue
        for pair in entry.get("correlation_pairs", []):
            if int(pair.get("train_sample_id", -1)) != train_id:
                continue
            tc, rc = pair.get("test_correlation", {}), pair.get("train_correlation", {})
            if test_source is not None and tc.get("source_token_index") != test_source:
                continue
            out.append({
                "id": pair.get("id"),
                "cos_sim": pair.get("cos_sim"),
                "trainSourceIndex": rc.get("source_token_index"),
                "trainTargetIndex": rc.get("target_token_index"),
                "testSourceIndex": tc.get("source_token_index"),
                "testTargetIndex": tc.get("target_token_index"),
            })
    return out


def _append_side(
    side: str,
    report_tokens: list[str],
    hidden: np.ndarray,
    offset: int,
    source_indices: set[int],
    target_indices: set[int],
    pair_ids_by_index: dict[int, list[str]],
    acc: dict,
) -> None:
    """Add every token of one sample as a point, in report index space."""
    for token_index, token in enumerate(report_tokens):
        role = "target" if token_index in target_indices else \
               "source" if token_index in source_indices else "context"

        display = normalize_token_for_display(token)
        shortened = display[:15] + "…" if len(display) > 18 else display
        acc["labels"].append(PROBE_CLASS_TO_ID[f"{side}_{role}"])
        acc["text_list"].append(f"{_LABEL_PREFIX[(side, role)]}{token_index}: {shortened}")
        acc["token_list"].append(display)
        # offset is where the report's window sits inside the full .pt sequence.
        acc["rows"].append(hidden[token_index + offset])
        acc["point_records"].append({
            "point_index": len(acc["point_records"]),
            "side": side,
            "role": role,
            "token_index": token_index,
            "token": token,
            "token_display": display,
            "pair_ids": pair_ids_by_index.get(token_index, []),
            "is_focus": role != "context",
        })
        if role != "context":
            acc["selected"].append(len(acc["point_records"]) - 1)


def build_probe_payload(
    report_path: str,
    test_pt_path: str,
    train_pt_path: str,
    train_id: int,
    test_target: int,
    test_source: int | None = None,
    sample_id: str | None = None,
    use_umap: bool = False,
    allow_adapter_mismatch: bool = False,
    truncate_to_aligned: bool = False,
    focus_pair_ids: list[str] | None = None,
) -> dict:
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    test_pt = load_pt(test_pt_path)
    train_pt = load_pt(train_pt_path)

    if test_pt["model_name_or_path"] != train_pt["model_name_or_path"]:
        raise ValueError(
            f"base model mismatch: test .pt is {test_pt['model_name_or_path']!r}, "
            f"train .pt is {train_pt['model_name_or_path']!r}"
        )
    if test_pt["layer"] != train_pt["layer"]:
        raise ValueError(f"layer mismatch: {test_pt['layer']} vs {train_pt['layer']}")

    # A differing LoRA adapter keeps the dimensions compatible but moves the
    # representations: measured on this data, identical tokens at identical
    # positions sit at cosine ~0.89 across adapters versus ~0.9998 within one. A
    # joint projection over two such spaces can separate points for reasons that
    # have nothing to do with the tokens, so this needs a deliberate override.
    if test_pt.get("adapter_path") != train_pt.get("adapter_path"):
        message = (
            f"adapter mismatch:\n"
            f"  test  .pt: {test_pt.get('adapter_path')}\n"
            f"  train .pt: {train_pt.get('adapter_path')}\n"
            f"Their embeddings come from different fine-tunes, so distances between "
            f"a test point and a train point are not purely semantic."
        )
        if not allow_adapter_mismatch:
            raise ValueError(message + "\nPass --allow-adapter-mismatch to build it anyway.")
        print(f"[warn] {message}\n[warn] proceeding because --allow-adapter-mismatch was given")

    pairs = collect_pairs(report, test_target, train_id, test_source)
    if not pairs:
        raise ValueError(
            f"no attribution pairs for test target {test_target}"
            + (f" / source {test_source}" if test_source is not None else "")
            + f" / train {train_id} in {report_path}"
        )

    test_tokens = report["test_sample_baseline"]["full_tokens"]
    train_detail = report["train_sample_details"].get(str(train_id))
    if train_detail is None:
        raise ValueError(f"train sample {train_id} is not in the report's train_sample_details")
    train_tokens = train_detail["full_tokens"]

    test_offset, test_usable = align_report_to_pt(
        test_tokens, test_pt["token_surfaces"], "test", truncate_to_aligned)
    train_offset, train_usable = align_report_to_pt(
        train_tokens, train_pt["token_surfaces"], "train", truncate_to_aligned)

    # Truncation may only drop context. An edge endpoint outside the aligned region
    # has no trustworthy vector, and the probe exists to show exactly those edges.
    for pair in pairs:
        for key, limit, side in (("testSourceIndex", test_usable, "test"),
                                 ("testTargetIndex", test_usable, "test"),
                                 ("trainSourceIndex", train_usable, "train"),
                                 ("trainTargetIndex", train_usable, "train")):
            if pair[key] >= limit:
                raise ValueError(
                    f"pair {pair['id']}: {key}={pair[key]} lies beyond the aligned "
                    f"{side} region (first {limit} tokens). This edge cannot be placed "
                    f"without guessing which vector it refers to."
                )

    test_tokens = test_tokens[:test_usable]
    train_tokens = train_tokens[:train_usable]

    test_hidden = test_pt["hidden"].numpy()
    train_hidden = train_pt["hidden"].numpy()

    tr_pid, te_pid = defaultdict(list), defaultdict(list)
    for pair in pairs:
        tr_pid[pair["trainSourceIndex"]].append(pair["id"])
        tr_pid[pair["trainTargetIndex"]].append(pair["id"])
        te_pid[pair["testSourceIndex"]].append(pair["id"])
        te_pid[pair["testTargetIndex"]].append(pair["id"])

    acc = {"labels": [], "text_list": [], "token_list": [], "rows": [],
           "point_records": [], "selected": []}

    # Roles — and therefore which points are coloured as source/target and ringed
    # as selected — come from the *focus* pairs only, while pair_ids and
    # pair_signature still cover the whole group. A group here runs to 100-200
    # pairs; marking every endpoint would ring most of the plot and bury the pair
    # the user actually ticked. Matches how tools/ttav-pipeline's build_probe()
    # scopes it ("角色(高亮)只看 focus pair").
    focus_set = set(focus_pair_ids) if focus_pair_ids else {p["id"] for p in pairs}
    unknown = focus_set - {p["id"] for p in pairs}
    if unknown:
        raise ValueError(f"focus pair id(s) not in this group: {sorted(unknown)}")
    fpairs = [p for p in pairs if p["id"] in focus_set]

    _append_side("train", train_tokens, train_hidden, train_offset,
                 {p["trainSourceIndex"] for p in fpairs},
                 {p["trainTargetIndex"] for p in fpairs}, tr_pid, acc)
    train_count = len(acc["point_records"])
    _append_side("test", test_tokens, test_hidden, test_offset,
                 {p["testSourceIndex"] for p in fpairs},
                 {p["testTargetIndex"] for p in fpairs}, te_pid, acc)

    embeddings = np.stack(acc["rows"]).astype(np.float32)
    # Both sides projected together: fitting per side would give each its own
    # basis and make "the test points land near the train points" an artefact of
    # the projection rather than an observation about the tokens.
    projection = compute_projection(embeddings, use_umap=use_umap)

    if sample_id is None:
        base = Path(report_path).stem
        spart = f"_s{test_source}" if test_source is not None else ""
        sample_id = f"{base}_t{test_target}{spart}_train{train_id}_probe"

    return {
        "sample_id": sample_id,
        "vis_method": "TimeVis",
        "vis_id": "1",
        "overwrite": True,
        "selected_indices": acc["selected"],
        "target_index": None,
        "bundle": {
            "model": report.get("experiment_meta", {}).get("model_name", "unknown"),
            "checkpoint_path": test_pt.get("adapter_path", ""),
            "embedding_type": "contextual",
            "embedding_note": f"contextual_hidden_state_layer_{test_pt['layer']} (from .pt)",
            "classes": PROBE_CLASSES,
            "sample_index": report.get("experiment_meta", {}).get("test_sample_index", 0),
            # Train/test split point, not a prompt length — probe bundles overload
            # this field the same way (ONBOARDING_visualization.md §4).
            "prompt_len": train_count,
            "labels": acc["labels"],
            "text_list": acc["text_list"],
            "text_data": acc["text_list"],
            "token_list": acc["token_list"],
            "index": {
                "train": list(range(train_count)),
                "test": list(range(train_count, len(acc["point_records"]))),
            },
            "probe_metadata": {
                "kind": "report_plus_pt",
                "report_path": report_path,
                "test_pt": test_pt_path,
                "train_pt": train_pt_path,
                "train_sample_id": train_id,
                "test_target_index": test_target,
                "test_source_index": test_source,
                "focus_pair_ids": sorted(focus_set),
                "test_report_offset": test_offset,
                "train_report_offset": train_offset,
                "point_records": acc["point_records"],
                "pair_signature": [
                    {"id": p["id"], "cos_sim": p["cos_sim"],
                     "trainSourceIndex": p["trainSourceIndex"],
                     "trainTargetIndex": p["trainTargetIndex"],
                     "testSourceIndex": p["testSourceIndex"],
                     "testTargetIndex": p["testTargetIndex"]}
                    for p in pairs
                ],
            },
        },
    }, embeddings, projection


def main():
    parser = argparse.ArgumentParser(
        description="Build a TTAV probe bundle from a report plus precomputed .pt embeddings (no model needed)."
    )
    parser.add_argument("--report", required=True)
    parser.add_argument("--test-pt", required=True)
    parser.add_argument("--train-pt", required=True)
    parser.add_argument("--train-id", type=int, required=True)
    parser.add_argument("--test-target", type=int, required=True)
    parser.add_argument("--test-source", type=int, default=None,
                        help="Pin one source→target edge, matching what the report UI launches.")
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--projection", choices=["pca", "umap"], default="pca")
    parser.add_argument("--allow-adapter-mismatch", action="store_true")
    parser.add_argument("--truncate-to-aligned", action="store_true",
                        help="If the report and .pt diverge (different generations of the "
                             "same prompt), keep only the aligned prefix instead of failing.")
    parser.add_argument("--ttav-url", default="http://1.94.115.154/")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument(
        "--for-frontend", action="store_true",
        help="Write to ttav_bundles_real/ under the name the EIF API looks for, so "
             "the report page's 'Open Full Probe' finds it as a precomputed cache. "
             "Implies --no-upload: the API uploads it itself when the button is used."
    )
    parser.add_argument(
        "--with-embeddings", action="store_true",
        help="With --for-frontend: also write the full bundle_payload.json. Only "
             "needed to keep the TTAV new-window path working; the in-page canvas "
             "does not read embeddings and the file is ~190x larger."
    )
    parser.add_argument(
        "--base-sample-id", default=None,
        help="With --for-frontend: the id the report filename resolves to. Defaults "
             "to infer_sample_id() of --report, which is what the frontend computes."
    )
    parser.add_argument("--list-targets", action="store_true",
                        help="List the (target, source, train) combinations the report offers, then exit.")
    args = parser.parse_args()

    if args.list_targets:
        report = json.loads(Path(args.report).read_text(encoding="utf-8"))
        for entry in report.get("per_token_results", []):
            target = entry.get("target_token_index")
            combos = defaultdict(set)
            for pair in entry.get("correlation_pairs", []):
                combos[int(pair["train_sample_id"])].add(pair["test_correlation"]["source_token_index"])
            for train_id, sources in sorted(combos.items()):
                print(f"  --test-target {target} --train-id {train_id} "
                      f"--test-source {sorted(sources)}")
        return

    sample_id = args.sample_id
    if args.for_frontend and sample_id is None:
        # Must reproduce _stable_probe_sample_id() in ttav_bundle_api.py exactly —
        # that is the name the API derives from the button press and looks for
        # under ttav_bundles_real/. A different name here means a guaranteed 404
        # with nothing to indicate why. The report filename drives the base id, the
        # same way the frontend's inferSampleIdFromMeta() computes it.
        base = args.base_sample_id or infer_sample_id(args.report)
        if args.test_source is None:
            raise ValueError("--for-frontend needs --test-source: the button lives inside "
                             "one source→target edge, and the cache name records it.")
        sample_id = f"{base}_t{args.test_target}_s{args.test_source}_train{args.train_id}_probe"

    payload, embeddings, projection = build_probe_payload(
        report_path=args.report,
        test_pt_path=args.test_pt,
        train_pt_path=args.train_pt,
        train_id=args.train_id,
        test_target=args.test_target,
        test_source=args.test_source,
        sample_id=sample_id,
        use_umap=(args.projection == "umap"),
        allow_adapter_mismatch=args.allow_adapter_mismatch,
        truncate_to_aligned=args.truncate_to_aligned,
    )

    sample_id = payload["sample_id"]
    meta = payload["bundle"]["probe_metadata"]
    print(f"[probe_from_pt] sample_id={sample_id}")
    print(f"[probe_from_pt] points={len(payload['bundle']['labels'])} "
          f"(train {payload['bundle']['prompt_len']} + test "
          f"{len(payload['bundle']['labels']) - payload['bundle']['prompt_len']})")
    print(f"[probe_from_pt] pairs={len(meta['pair_signature'])} "
          f"offsets: test+{meta['test_report_offset']} train+{meta['train_report_offset']}")

    if args.for_frontend:
        root = Path(__file__).resolve().parent.parent / "ttav_bundles_real"
        write_frontend_cache(
            root / sample_id, payload, projection,
            embeddings=embeddings if args.with_embeddings else None,
        )
        print(f"[probe_from_pt] the report page's 'Open Full Probe' will now find this "
              f"for target {args.test_target} / source {args.test_source} / TRAIN #{args.train_id}")
        return

    if args.out_dir:
        out = Path(args.out_dir) / sample_id / "bundle_payload.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        full = json.loads(json.dumps(payload))
        full["bundle"]["embeddings"] = embeddings.tolist()
        full["bundle"]["projection"] = projection.tolist()
        out.write_text(json.dumps(full), encoding="utf-8")
        print(f"[probe_from_pt] wrote {out}")

    if args.no_upload:
        return

    # Binary transport: these bundles are far too large to inline as JSON numbers.
    upload_array(args.ttav_url, sample_id, "embeddings", embeddings.astype(np.float16))
    upload_array(args.ttav_url, sample_id, "projection", projection.astype(np.float32))
    payload["bundle"]["arrays_uploaded"] = True

    result = upload_bundle(args.ttav_url.rstrip("/") + "/registerEIFBundle", payload)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    from urllib import parse as urllib_parse
    jump = {"source": "eif", "sampleId": sample_id,
            "contentPath": result.get("content_path", ""), "visMethod": payload["vis_method"]}
    print("\nOpen in browser:")
    print(f"{args.ttav_url.rstrip('/')}/?eif_jump={urllib_parse.quote(json.dumps(jump, separators=(',', ':')))}")


if __name__ == "__main__":
    main()
