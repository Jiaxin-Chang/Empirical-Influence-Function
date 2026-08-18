import argparse
import hashlib
import json
import os
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from time import time
from urllib.parse import parse_qs, urlparse

import numpy as np

from src.export_real_ttav_bundle import (
    DEFAULT_MODEL_PATH,
    build_real_bundle_payload,
    token_surfaces_for_display,
)
from src.export_probe_bundle_from_pt import build_probe_payload as build_probe_from_pt
from src.export_train_probe_bundle import build_train_probe_bundle_payload
from src.export_ttav_bundle import build_bundle_payload, infer_sample_id, upload_bundle
from src.unlearn_pair_probe import (
    _get_model,
    _resolve_paths,
    compute_next_token_probs,
    recover_pair_intervention,
    run_unlearn_pair_probe,
)
from src.gold_live_attribution import (
    _gold_tokens_and_ids,
    _hydrate_eif_env,
    gold_retrieve_and_stage3,
    gold_saliency_top_k,
)
from src.continue_train_eval import (
    build_config_from_request,
    default_paths_from_env,
    run_continue_train_and_eval,
)
from src.eif_adapter_env import (
    get_active_adapter_status,
    set_active_adapter_override,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
# Load eif_api.env early so CACHE_ONLY_MODE / model paths see it.
_hydrate_eif_env()
CORR_RESULTS_DIR = REPO_ROOT / "correlation_matching_results"
EIF_BUNDLE_CACHE_ROOT = REPO_ROOT / "ttav_bundles"
PREGENERATED_REAL_BUNDLE_ROOT = REPO_ROOT / "ttav_bundles_real"


def _resolve_corr_report_path(report_file_name: str):
    from src.eif_adapter_env import resolve_report_json_path
    from pathlib import Path

    p = resolve_report_json_path(CORR_RESULTS_DIR, report_file_name)
    if p is not None:
        return p
    cand = CORR_RESULTS_DIR / report_file_name
    if cand.exists():
        return cand
    alt = REPO_ROOT / report_file_name
    if alt.exists():
        return alt
    return None
PREPARE_STATUS_LOCK = Lock()
PREPARE_STATUS: dict[str, dict] = {}
# One lock for gold-live + unlearn/probs: they share (or overwrite) the same
# in-process PEFT model. Separate locks allowed parallel loads and mid-flight
# session swaps, which intermittently triggered bf16/float matmul errors.
SHARED_MODEL_LOCK = Lock()
UNLEARN_PROBE_LOCK = SHARED_MODEL_LOCK
GOLD_LIVE_LOCK = SHARED_MODEL_LOCK
CONTINUE_TRAIN_LOCK = Lock()
CONTINUE_TRAIN_JOBS: dict[str, dict] = {}
CONTINUE_TRAIN_JOBS_LOCK = Lock()

def _set_continue_job(job_id: str, **fields):
    with CONTINUE_TRAIN_JOBS_LOCK:
        cur = dict(CONTINUE_TRAIN_JOBS.get(job_id) or {"jobId": job_id})
        cur.update(fields)
        cur["updatedAt"] = int(time() * 1000)
        CONTINUE_TRAIN_JOBS[job_id] = cur
    print(
        f"[continue-train][{job_id}] {fields.get('stage', '?')}: {fields.get('message', '')}",
        flush=True,
    )


def _get_continue_job(job_id: str) -> dict:
    with CONTINUE_TRAIN_JOBS_LOCK:
        payload = CONTINUE_TRAIN_JOBS.get(job_id)
    if payload is None:
        return {
            "jobId": job_id,
            "stage": "unknown",
            "message": "Unknown jobId",
            "active": False,
            "error": True,
            "updatedAt": int(time() * 1000),
        }
    return dict(payload)


def _run_continue_train_job(job_id: str, req: dict):
    try:
        cfg = build_config_from_request(req)

        def progress(stage: str, message: str, extra: dict | None = None):
            payload = {
                "stage": stage,
                "message": message,
                "active": stage not in {"completed", "error"},
                "error": False,
            }
            if isinstance(extra, dict):
                # Don't put huge result blobs into status repeatedly.
                slim = {k: v for k, v in extra.items() if k != "result"}
                payload["progress"] = slim
                if "result" in extra:
                    payload["result"] = extra["result"]
            _set_continue_job(job_id, **payload)

        with CONTINUE_TRAIN_LOCK:
            result = run_continue_train_and_eval(cfg, progress_cb=progress)
        _set_continue_job(
            job_id,
            stage="completed",
            message="Continue-train + eval finished.",
            active=False,
            error=False,
            result=result,
        )
    except Exception as exc:
        _set_continue_job(
            job_id,
            stage="error",
            message=str(exc),
            active=False,
            error=True,
        )


# Server-side kill switch for live model loading. Independent of (and stronger
# than) the per-request `requireCached` flag: that one is client-supplied and
# not trustworthy — any request that omits/flips it would still fall through
# to loading the ~7GB model on this 7.1GB-RAM box. Set EIF_CACHE_ONLY=1 (in
# the systemd unit's Environment=, or the shell that launches the API) to make
# every request behave as cache-only server-wide, regardless of what the
# client sends: a cache miss returns an error instead of ever loading a model.
CACHE_ONLY_MODE = os.environ.get("EIF_CACHE_ONLY", "").strip().lower() in ("1", "true", "yes")


def _set_prepare_status(sample_id: str, stage: str, message: str, *, active: bool, error: bool = False):
    payload = {
        "sampleId": sample_id,
        "stage": stage,
        "message": message,
        "active": active,
        "error": error,
        "updatedAt": int(time() * 1000),
    }
    with PREPARE_STATUS_LOCK:
        PREPARE_STATUS[sample_id] = payload
    print(f"[{sample_id}] {stage}: {message}", flush=True)


def _get_prepare_status(sample_id: str) -> dict:
    with PREPARE_STATUS_LOCK:
        payload = PREPARE_STATUS.get(sample_id)
    if payload is None:
        return {
            "sampleId": sample_id,
            "stage": "idle",
            "message": "No active prepare job.",
            "active": False,
            "error": False,
            "updatedAt": int(time() * 1000),
        }
    return dict(payload)


def _cache_dir(sample_id: str, explicit_path: str | None = None) -> Path:
    if explicit_path:
        return Path(explicit_path)
    return EIF_BUNDLE_CACHE_ROOT / sample_id


def _cache_payload_path(sample_id: str, explicit_path: str | None = None) -> Path:
    return _cache_dir(sample_id, explicit_path) / "bundle_payload.json"


def _resolve_cache_payload_path(sample_id: str, explicit_path: str | None = None) -> Path:
    payload_path = _cache_payload_path(sample_id, explicit_path)
    if payload_path.exists():
        return payload_path

    # A caller-supplied explicit_path (from the frontend's "EIF Bundle Cache
    # Path" field / localStorage) can go stale or point at a path that only
    # ever existed on a different machine. Rather than treat that as a hard
    # cache miss and fall through to a live model load -- which requires
    # loading a ~7GB checkpoint on this 7.1GB-RAM box and reliably OOMs, see
    # HANDOFF_2026-07-13_EIF_TTAV_OOM.md -- fall back to the server's own
    # default cache locations first.
    if explicit_path:
        default_path = _cache_payload_path(sample_id, None)
        if default_path.exists():
            print(f"[prepare] sampleId={sample_id} cache=default_fallback (explicit path missing: {explicit_path})", flush=True)
            return default_path

    pregenerated_path = PREGENERATED_REAL_BUNDLE_ROOT / sample_id / "bundle_payload.json"
    if pregenerated_path.exists():
        print(f"[prepare] sampleId={sample_id} cache=pregenerated_real_bundle", flush=True)
        return pregenerated_path

    return payload_path


def write_local_bundle_cache(sample_id: str, payload: dict, explicit_path: str | None = None):
    cache_dir = _cache_dir(sample_id, explicit_path)
    bundle = payload["bundle"]
    vis_method = str(payload.get("vis_method", "TimeVis"))
    vis_id = str(payload.get("vis_id", "1"))
    num_points = len(bundle["labels"])

    dataset_dir = cache_dir / "dataset"
    epoch_dir = cache_dir / "epochs" / "epoch_1"
    vis_info_dir = cache_dir / "visualize" / f"{vis_method}_{vis_id}"
    vis_dir = vis_info_dir / "epochs" / "epoch_1"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    epoch_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    (_cache_payload_path(sample_id, explicit_path)).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    dataset_info = {
        "model": bundle.get("model", "EIFStaticTokenBundle"),
        "classes": bundle.get("classes", ["prompt", "output"]),
        "eif_bundle": True,
        "sample_id": sample_id,
        "prompt_len": bundle.get("prompt_len"),
        "cache_source": "EIF",
    }
    (dataset_dir / "info.json").write_text(json.dumps(dataset_info, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(dataset_dir / "labels.npy", np.asarray(bundle["labels"], dtype=np.int64))
    (dataset_dir / "index.json").write_text(
        json.dumps(bundle.get("index", {"train": list(range(num_points)), "test": []}), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (dataset_dir / "text.txt").write_text("\n".join(str(x) for x in bundle["text_list"]), encoding="utf-8")
    (dataset_dir / "token_list.json").write_text(json.dumps(bundle.get("token_list", []), ensure_ascii=False), encoding="utf-8")
    (dataset_dir / "text_data.json").write_text(json.dumps(bundle.get("text_data", []), ensure_ascii=False), encoding="utf-8")

    np.save(epoch_dir / "embeddings.npy", np.asarray(bundle["embeddings"], dtype=np.float32))
    np.save(vis_dir / "projection.npy", np.asarray(bundle["projection"], dtype=np.float32))

    vis_info = {
        "content_path": str(cache_dir),
        "vis_method": vis_method,
        "vis_id": vis_id,
        "data_type": "Text",
        "task_type": "Alignment",
        "vis_config": {"gpu_id": -1},
        "sample_id": sample_id,
        "eif_bundle": True,
        "cache_source": "EIF",
    }
    (vis_info_dir / "info.json").write_text(json.dumps(vis_info, ensure_ascii=False, indent=2), encoding="utf-8")


def load_local_bundle_cache(sample_id: str, explicit_path: str | None = None) -> dict:
    payload_path = _resolve_cache_payload_path(sample_id, explicit_path)
    return json.loads(payload_path.read_text(encoding="utf-8"))


def _find_token_embeddings(base_sample_id: str, train_sample_id: int) -> tuple[str, str] | None:
    """Locate the test and train .pt for a sample, or None if either is absent.

    Layout, one directory per sample id (i.e. per model variant):

        token_embeddings/{base_sample_id}/test.pt
        token_embeddings/{base_sample_id}/train_{train_sample_id}.pt

    Keeping a variant's test and train vectors together is what makes adapter
    consistency structural rather than a thing to remember: embeddings from two
    different fine-tunes sit at cosine ~0.89 for identical tokens (versus ~0.9998
    within one), so a probe mixing them measures the checkpoints as much as the
    tokens. The directory name reuses the sample id the report filename already
    resolves to — no new naming rule to drift out of sync.

    Override the root with EIF_TOKEN_EMBEDDING_ROOT when the files live off-repo.
    """
    root = Path(os.environ.get("EIF_TOKEN_EMBEDDING_ROOT", "").strip() or (REPO_ROOT / "token_embeddings"))
    test_pt = root / base_sample_id / "test.pt"
    train_pt = root / base_sample_id / f"train_{train_sample_id}.pt"
    if test_pt.exists() and train_pt.exists():
        return str(test_pt), str(train_pt)
    return None


def _write_probe_projection_cache(probe_sample_id: str, payload: dict, projection) -> Path:
    """Persist the slim view the report page reads, under the probe's cache name.

    Only projection.json: the report page's canvas never reads `embeddings`, and
    they are ~99% of the bytes (1.3 MB versus 250 MB for one probe here). The full
    payload still goes to TTAV in-memory when the new-window mode asks for it, so
    nothing is lost by leaving it off disk.
    """
    out_dir = PREGENERATED_REAL_BUNDLE_ROOT / probe_sample_id
    out_dir.mkdir(parents=True, exist_ok=True)
    slim = json.loads(json.dumps(payload))
    slim["bundle"]["projection"] = projection.tolist()
    slim["bundle"].pop("embeddings", None)
    path = out_dir / "projection.json"
    path.write_text(json.dumps(slim), encoding="utf-8")
    return path


def _make_probe_sample_id(
    base_sample_id: str,
    train_sample_id: int,
    probe_pairs: list[dict],
    context_radius: int,
    include_full_train: bool,
) -> str:
    # focus_train_indices is deliberately NOT part of this signature: it only
    # controls which points get highlighted/selected in the UI, not which points
    # are in the bundle or what their embeddings are. Including it meant every
    # checkbox toggle produced a different cache key, so a cached (or
    # precomputed) probe could never be reused.
    signature = json.dumps(
        {
            "trainSampleId": train_sample_id,
            "contextRadius": context_radius,
            "includeFullTrain": include_full_train,
            "pairs": [
                {
                    "id": pair.get("id"),
                    "trainSourceIndex": pair.get("trainSourceIndex"),
                    "trainTargetIndex": pair.get("trainTargetIndex"),
                    "testSourceIndex": pair.get("testSourceIndex"),
                    "testTargetIndex": pair.get("testTargetIndex"),
                }
                for pair in probe_pairs
            ],
        },
        sort_keys=True,
    )
    digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:8]
    return f"{base_sample_id}_train{train_sample_id}_probe_{digest}"


def _probe_anchor_indices(probe_pairs: list[dict]) -> tuple[int | None, int | None]:
    """The (test target, test source) edge a probe request is anchored on.

    "Open Full Probe" sits inside one source→target edge of one test token, so
    every pair it sends shares both indices. Both are part of the precomputed
    bundle's directory name, so both are needed to find the file. A mixed set
    can't be named, so that component comes back None and the lookup falls back
    to a shorter, older name.
    """
    def sole(key: str) -> int | None:
        values = {
            int(pair[key])
            for pair in probe_pairs
            if isinstance(pair, dict) and pair.get(key) is not None
        }
        return values.pop() if len(values) == 1 else None

    return sole("testTargetIndex"), sole("testSourceIndex")


def _stable_probe_sample_id(
    base_sample_id: str,
    train_sample_id: int,
    test_target_index: int | None = None,
    test_source_index: int | None = None,
) -> str:
    """Parameter-independent probe id, used to look up precomputed bundles.

    Precomputed probes are produced on another machine that can't reproduce this
    server's request-dependent hash, so they're stored under a name derived only
    from the anchoring edge — the same id the generator puts in the payload, and
    the same encoding the report uses for pair ids (`t136_s60_...`).

    The naming has widened twice. `_t{target}` alone still lumped together every
    source edge feeding one target token, which doesn't match what the button
    actually launches; `_s{source}` splits those apart. Omitting either segment
    reproduces an older form, kept only so existing bundles stay reachable.
    """
    if test_target_index is None:
        return f"{base_sample_id}_train{train_sample_id}_probe"
    if test_source_index is None:
        return f"{base_sample_id}_t{test_target_index}_train{train_sample_id}_probe"
    return (
        f"{base_sample_id}_t{test_target_index}_s{test_source_index}"
        f"_train{train_sample_id}_probe"
    )


def _default_probe_cache_path(base_sample_id: str, train_sample_id: int, probe_sample_id: str) -> str:
    return str(EIF_BUNDLE_CACHE_ROOT / "probes" / base_sample_id / f"train_{train_sample_id}" / probe_sample_id)


def _find_cached_probe_payload(
    base_sample_id: str,
    train_sample_id: int,
    explicit_cache_path: str,
    test_target_index: int | None = None,
    test_source_index: int | None = None,
) -> tuple[dict, str] | None:
    """Look for an already-built probe bundle, most-specific-first: this server's
    own cache, then a bundle precomputed elsewhere and dropped into
    ttav_bundles_real/. Returns (payload, source) or None."""
    candidates = [
        (Path(explicit_cache_path) / "bundle_payload.json", "local_cache"),
    ]
    if test_target_index is not None and test_source_index is not None:
        candidates.append((
            PREGENERATED_REAL_BUNDLE_ROOT
            / _stable_probe_sample_id(base_sample_id, train_sample_id, test_target_index, test_source_index)
            / "bundle_payload.json",
            "pregenerated",
        ))
    # Older layouts, each a degraded match rather than an equal option: the
    # _t-only form mixed every source edge under one target, and the form with
    # neither segment also cut the test side down to the pairs' tokens ±1. Both
    # are consulted only when nothing in the current format exists.
    if test_target_index is not None:
        candidates.append((
            PREGENERATED_REAL_BUNDLE_ROOT
            / _stable_probe_sample_id(base_sample_id, train_sample_id, test_target_index)
            / "bundle_payload.json",
            "pregenerated_legacy_target_only",
        ))
    candidates.append((
        PREGENERATED_REAL_BUNDLE_ROOT
        / _stable_probe_sample_id(base_sample_id, train_sample_id)
        / "bundle_payload.json",
        "pregenerated_legacy",
    ))
    for path, source in candidates:
        if not path.exists():
            continue
        try:
            return json.loads(path.read_text(encoding="utf-8")), source
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[probe] ignoring unreadable cache {path}: {exc}", flush=True)
    return None


def _payload_matches_request(payload: dict, bundle_mode: str, embedding_type: str, model_path: str | None) -> bool:
    bundle = payload.get("bundle") if isinstance(payload, dict) else None
    if not isinstance(bundle, dict):
        return False

    payload_embedding_type = str(bundle.get("embedding_type", "")).strip().lower()
    payload_model_path = str(bundle.get("checkpoint_path", "")).strip()

    if bundle_mode == "real":
        if payload_embedding_type != embedding_type:
            return False
        if model_path and payload_model_path and payload_model_path != model_path:
            return False
        return True

    return payload_embedding_type in {"", "static"}


def _build_requested_payload(
    report_json_path: Path,
    test_data: str | None,
    model_path: str | None,
    sample_id: str | None,
    bundle_mode: str,
    embedding_type: str,
    hidden_layer: int,
    vis_method: str,
    vis_id: str,
    progress_callback=None,
) -> dict:
    if bundle_mode == "real":
        resolved_model_path = model_path or str(DEFAULT_MODEL_PATH)
        return build_real_bundle_payload(
            report_json_path=str(report_json_path),
            model_path=resolved_model_path,
            sample_id=sample_id,
            embedding_type=embedding_type,
            hidden_layer=hidden_layer,
            vis_method=vis_method,
            vis_id=vis_id,
            progress_callback=progress_callback,
        )

    return build_bundle_payload(
        report_json_path=str(report_json_path),
        test_data_path=test_data,
        model_path=model_path,
        sample_id=sample_id,
    )


class TTAVBundleRequestHandler(BaseHTTPRequestHandler):
    server_version = "EIFTTAVBundleAPI/0.1"

    def log_message(self, format: str, *args):
        print(f"[http] {self.address_string()} - {format % args}", flush=True)

    def _send_json(self, status: int, payload: dict):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send_json(200, {"status": "ok"})

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/continue-train-eval-status":
            job_id = parse_qs(parsed.query).get("jobId", [""])[0].strip()
            if not job_id:
                self._send_json(400, {"status": "error", "message": "jobId is required"})
                return
            self._send_json(200, {"status": "success", **_get_continue_job(job_id)})
            return
        if parsed.path == "/api/continue-adapter-status":
            self._send_json(200, {"status": "success", **get_active_adapter_status()})
            return
        if parsed.path == "/api/continue-train-eval-defaults":
            self._send_json(200, {
                "status": "success",
                "defaults": default_paths_from_env(),
            })
            return
        if parsed.path != "/api/prepare-ttav-bundle-status":
            self._send_json(404, {"status": "error", "message": "Not found"})
            return

        sample_id = parse_qs(parsed.query).get("sampleId", [""])[0].strip()
        if not sample_id:
            self._send_json(400, {"status": "error", "message": "sampleId is required"})
            return

        print(f"[status] sampleId={sample_id}", flush=True)
        self._send_json(200, {"status": "success", **_get_prepare_status(sample_id)})

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/continue-train-eval":
            self._handle_continue_train_eval()
            return
        if parsed.path == "/api/continue-adapter-recover":
            self._handle_continue_adapter_recover()
            return
        if parsed.path == "/api/structural-pair-retrieve":
            self._handle_structural_pair_retrieve()
            return
        if parsed.path == "/api/prepare-ttav-train-probe":
            self._handle_prepare_train_probe()
            return
        if parsed.path == "/api/unlearn-pair-probe":
            self._handle_unlearn_pair_probe()
            return
        if parsed.path == "/api/pair-intervene-recover":
            self._handle_pair_intervene_recover()
            return
        if parsed.path == "/api/next-token-probs":
            self._handle_next_token_probs()
            return
        if parsed.path == "/api/degradation-retrieve":
            self._handle_degradation_retrieve()
            return
        if parsed.path == "/api/token-display-surfaces":
            self._handle_token_display_surfaces()
            return
        if parsed.path == "/api/gold-saliency":
            self._handle_gold_saliency()
            return
        if parsed.path == "/api/gold-retrieve-stage3":
            self._handle_gold_retrieve_stage3()
            return
        if parsed.path != "/api/prepare-ttav-bundle":
            self._send_json(404, {"status": "error", "message": "Not found"})
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)
        try:
            req = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json(400, {"status": "error", "message": "Invalid JSON body"})
            return

        report_file_name = str(req.get("reportFileName", "")).strip()
        if not report_file_name:
            self._send_json(400, {"status": "error", "message": "reportFileName is required"})
            return

        report_json_path = _resolve_corr_report_path(report_file_name)
        if report_json_path is None or not report_json_path.exists():
            self._send_json(404, {"status": "error", "message": f"Report JSON not found: {report_file_name}"})
            return

        test_data = str(req.get("testData", "sft_test.jsonl")).strip() or "sft_test.jsonl"
        raw_model_path = req.get("modelPath")
        model_path = str(raw_model_path).strip() if raw_model_path else None
        sample_id = req.get("sampleId")
        ttav_upload_url = str(req.get("ttavUploadUrl", "")).strip()
        ttav_url = str(req.get("ttavUrl", "")).strip()
        vis_method = str(req.get("visMethod", "TimeVis")).strip() or "TimeVis"
        vis_id = str(req.get("visId", "1")).strip() or "1"
        selected_indices = req.get("selectedIndices", [])
        target_index = req.get("targetIndex")
        client_require_cached = bool(req.get("requireCached", False))
        require_cached = client_require_cached or CACHE_ONLY_MODE
        explicit_cache_path = str(req.get("eifBundleCachePath", "")).strip() or None
        bundle_mode = str(req.get("bundleMode", "real")).strip().lower() or "real"
        embedding_type = str(req.get("embeddingType", "contextual")).strip().lower() or "contextual"
        hidden_layer = int(req.get("hiddenLayer", -1))
        overwrite_req = req.get("overwrite")
        overwrite_remote = bool(overwrite_req) if overwrite_req is not None else (bundle_mode == "real")
        resolved_sample_id = str(sample_id).strip() if sample_id else infer_sample_id(str(report_json_path))
        started_at = time()
        print(
            f"[prepare] sampleId={resolved_sample_id} requireCached={require_cached} bundleMode={bundle_mode} "
            f"embeddingType={embedding_type} vis={vis_method}/{vis_id}",
            flush=True,
        )

        try:
            _set_prepare_status(resolved_sample_id, "checking_cache", "Checking EIF local bundle cache", active=True)
            payload_path = _resolve_cache_payload_path(resolved_sample_id, explicit_cache_path)
            cache_hit = False

            if payload_path.exists():
                cached_payload = json.loads(payload_path.read_text(encoding="utf-8"))
                if _payload_matches_request(cached_payload, bundle_mode, embedding_type, model_path):
                    payload = cached_payload
                    cache_hit = True
                    _set_prepare_status(resolved_sample_id, "cache_hit", "Matching EIF local bundle cache found", active=True)
                    print(f"[prepare] sampleId={resolved_sample_id} cache=hit", flush=True)
                else:
                    payload = cached_payload
                    _set_prepare_status(resolved_sample_id, "cache_miss", "Existing cache does not match the requested real bundle", active=True)
                    print(f"[prepare] sampleId={resolved_sample_id} cache=mismatch", flush=True)
            if not cache_hit:
                if require_cached:
                    if CACHE_ONLY_MODE and not client_require_cached:
                        message = (
                            f"EIF local bundle cache not found for sample: {resolved_sample_id}. "
                            "This server has live model loading disabled (EIF_CACHE_ONLY=1); "
                            "only precomputed bundles can be served. Precompute this sample "
                            "elsewhere and drop it into ttav_bundles_real/ first."
                        )
                    else:
                        message = f"EIF local bundle cache not found for sample: {resolved_sample_id}. Please prepare the sample first."
                    _set_prepare_status(resolved_sample_id, "error", message, active=False, error=True)
                    self._send_json(404, {"status": "error", "message": message})
                    return
                _set_prepare_status(resolved_sample_id, "building_bundle", "Building TTAV bundle from EIF report", active=True)
                payload = _build_requested_payload(
                    report_json_path=report_json_path,
                    test_data=test_data,
                    model_path=model_path,
                    sample_id=resolved_sample_id,
                    bundle_mode=bundle_mode,
                    embedding_type=embedding_type,
                    hidden_layer=hidden_layer,
                    vis_method=vis_method,
                    vis_id=vis_id,
                    progress_callback=lambda stage, message: _set_prepare_status(resolved_sample_id, stage, message, active=True),
                )
                num_points = len(payload.get("bundle", {}).get("labels", []))
                print(f"[prepare] sampleId={resolved_sample_id} bundle_points={num_points}", flush=True)

            payload["vis_method"] = vis_method
            payload["vis_id"] = vis_id
            payload["overwrite"] = overwrite_remote
            payload["build_trainable_session"] = True
            payload["wait_until_ready"] = False
            payload["data_type"] = "Text"
            payload["task_type"] = "Alignment"
            payload["vis_config"] = {
                "gpu_id": -1,
                "n_neighbors": 10,
                "max_epochs": 10,
                "patient": 3,
                "s_n_epochs": 500,
                "b_n_epochs": 0,
                "t_n_epochs": 5,
                "lambda": 1.0,
                "refine_hd_k": 15,
            }
            _set_prepare_status(resolved_sample_id, "writing_local_cache", "Writing EIF local bundle cache", active=True)
            write_local_bundle_cache(resolved_sample_id, payload, explicit_path=explicit_cache_path)
            real_bundle_dir = REPO_ROOT / "ttav_bundles_real" / resolved_sample_id
            write_local_bundle_cache(resolved_sample_id, payload, explicit_path=str(real_bundle_dir))
            _set_prepare_status(resolved_sample_id, "uploading_to_ttav", "Uploading bundle to TTAV", active=True)
            upload_result = upload_bundle(ttav_upload_url, payload)
            elapsed = time() - started_at
            ttav_cached = upload_result.get("cached") is True
            print(f"[prepare] sampleId={resolved_sample_id} ttav_cached={ttav_cached} elapsed={elapsed:.2f}s", flush=True)
            _set_prepare_status(resolved_sample_id, "completed", f"Prepare sample completed in {elapsed:.1f}s", active=False)
        except Exception as exc:
            elapsed = time() - started_at
            print(f"[prepare] sampleId={resolved_sample_id} error after {elapsed:.2f}s: {exc}", flush=True)
            _set_prepare_status(resolved_sample_id, "error", str(exc), active=False, error=True)
            self._send_json(500, {"status": "error", "message": str(exc)})
            return

        content_path = upload_result.get("content_path")
        response = {
            "status": "success",
            "sampleId": upload_result.get("sample_id") or payload.get("sample_id"),
            "contentPath": content_path,
            "visMethod": upload_result.get("vis_method", vis_method),
            "visId": upload_result.get("vis_id", vis_id),
            "ttavUrl": ttav_url,
            "selectedIndices": selected_indices,
            "targetIndex": target_index,
            "eifBundleCachePath": str(_cache_dir(resolved_sample_id, explicit_cache_path)),
            "eifCacheHit": cache_hit,
            "bundleMode": bundle_mode,
            "embeddingType": embedding_type,
            "overwrite": overwrite_remote,
            "uploadResult": upload_result,
            "refineReady": upload_result.get("refineReady", False),
            "trainableSessionStatus": upload_result.get("trainableSessionStatus", "registered"),
            "statusMessage": upload_result.get("statusMessage"),
        }
        self._send_json(200, response)

    def _handle_continue_train_eval(self):
        """Start async continue-train on small annotated subset + line_hit eval."""
        if CACHE_ONLY_MODE:
            self._send_json(503, {
                "status": "error",
                "message": "EIF_CACHE_ONLY=1 — continue-train disabled on this server.",
            })
            return
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)
        try:
            req = json.loads(raw_body.decode("utf-8")) if raw_body else {}
        except json.JSONDecodeError:
            self._send_json(400, {"status": "error", "message": "Invalid JSON body"})
            return
        if not isinstance(req, dict):
            self._send_json(400, {"status": "error", "message": "JSON body must be an object"})
            return

        try:
            cfg = build_config_from_request(req)
        except Exception as exc:
            self._send_json(400, {"status": "error", "message": str(exc)})
            return

        # Reject if another continue-train is already active.
        with CONTINUE_TRAIN_JOBS_LOCK:
            active = [
                j for j in CONTINUE_TRAIN_JOBS.values()
                if j.get("active")
            ]
        if active:
            self._send_json(409, {
                "status": "error",
                "message": f"Another continue-train job is active: {active[0].get('jobId')}",
                "jobId": active[0].get("jobId"),
            })
            return

        job_id = uuid.uuid4().hex[:12]
        _set_continue_job(
            job_id,
            stage="queued",
            message="Continue-train job queued…",
            active=True,
            error=False,
            config={
                "adapterPath": cfg.adapter_path,
                "trainData": cfg.train_data,
                "trainSampleIds": cfg.train_sample_ids,
                "sourceTrainData": cfg.source_train_data,
                "testData": cfg.test_data,
                "outputDir": cfg.output_dir,
                "maxSteps": cfg.max_steps,
                "learningRate": cfg.learning_rate,
                "lossMode": cfg.loss_mode,
                "evalBefore": cfg.eval_before,
                "metrics": ["line_hit_pre", "line_hit_rec"],
            },
        )
        thread = threading.Thread(
            target=_run_continue_train_job,
            args=(job_id, req),
            name=f"continue-train-{job_id}",
            daemon=True,
        )
        thread.start()
        self._send_json(200, {
            "status": "success",
            "jobId": job_id,
            "message": "Continue-train job started (small subset + line_hit)",
            "config": {
                "adapterPath": cfg.adapter_path,
                "trainData": cfg.train_data,
                "trainSampleIds": cfg.train_sample_ids,
                "testData": cfg.test_data,
                "outputDir": cfg.output_dir,
                "maxSteps": cfg.max_steps,
                "learningRate": cfg.learning_rate,
                "lossMode": cfg.loss_mode,
                "metrics": ["line_hit_pre", "line_hit_rec"],
            },
        })

    def _handle_continue_adapter_recover(self):
        """Clear continue-train live adapter override; reload from eif_api.env paths."""
        if CACHE_ONLY_MODE:
            self._send_json(503, {
                "status": "error",
                "message": "EIF_CACHE_ONLY=1 — adapter recover disabled.",
            })
            return
        before = get_active_adapter_status()
        # Also undo any in-memory Learn/Unlearn stack so we truly match env weights.
        try:
            recover_pair_intervention()
        except Exception as exc:
            print(f"[continue-recover] pair intervene recover skipped: {exc}", flush=True)
        status = set_active_adapter_override(None)
        try:
            from src.continue_train_eval import _evict_cached_models
            _evict_cached_models()
        except Exception as exc:
            print(f"[continue-recover] cache eviction failed: {exc}", flush=True)
        env_path = status.get("envAdapterPath")
        print(
            f"[continue-recover] cleared override "
            f"(was {before.get('overridePath')}); env adapter={env_path}",
            flush=True,
        )
        self._send_json(200, {
            "status": "success",
            "recovered": bool(before.get("overrideActive")),
            "before": before,
            "activeAdapter": status,
            "message": (
                "Restored live probes to EIF_ADAPTER_PATH_* from env."
                if before.get("overrideActive")
                else "No continue-adapter override was active."
            ),
        })

    def _handle_structural_pair_retrieve(self):
        """Rank train pairs by structure+text similarity (flat pair list)."""
        _hydrate_eif_env()
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)
        try:
            req = json.loads(raw_body.decode("utf-8")) if raw_body else {}
        except json.JSONDecodeError:
            self._send_json(400, {"status": "error", "message": "Invalid JSON body"})
            return
        if not isinstance(req, dict):
            self._send_json(400, {"status": "error", "message": "JSON body must be an object"})
            return

        src_tok = str(req.get("sourceToken") or "").strip()
        dst_tok = str(req.get("targetToken") or "").strip()
        if not src_tok or not dst_tok:
            self._send_json(400, {
                "status": "error",
                "message": "sourceToken and targetToken are required",
            })
            return

        subtype = req.get("subtype")
        subtype_s = str(subtype).strip() if subtype is not None else None
        try:
            top_k = int(req.get("topK", 40) or 40)
        except (TypeError, ValueError):
            top_k = 40
        try:
            sw = float(req.get("structWeight", 0.8))
            tw = float(req.get("textWeight", 0.2))
        except (TypeError, ValueError):
            sw, tw = 0.8, 0.2

        src_idx = req.get("sourceIndex")
        dst_idx = req.get("targetIndex")
        try:
            src_i = int(src_idx) if src_idx is not None else None
            dst_i = int(dst_idx) if dst_idx is not None else None
        except (TypeError, ValueError):
            src_i, dst_i = None, None
        if src_i is None or dst_i is None:
            self._send_json(400, {
                "status": "error",
                "message": "sourceIndex and targetIndex are required for AST pair scoring",
            })
            return

        raw_tokens = req.get("tokens")
        query_tokens: list[str] | None = None
        if isinstance(raw_tokens, list) and raw_tokens:
            query_tokens = [str(t) for t in raw_tokens]
        try:
            prompt_len = int(req["promptLen"]) if req.get("promptLen") is not None else None
        except (TypeError, ValueError):
            prompt_len = None
        try:
            max_train = int(req["maxTrainSamples"]) if req.get("maxTrainSamples") is not None else None
        except (TypeError, ValueError):
            max_train = None

        print(
            f"[structural-ast] retrieve {src_tok!r}@{src_i} → {dst_tok!r}@{dst_i} "
            f"tokens={len(query_tokens) if query_tokens else 0} topK={top_k}",
            flush=True,
        )
        try:
            from src.structural_pair_retrieval import retrieve_structural_pairs

            result = retrieve_structural_pairs(
                query_src_token=src_tok,
                query_dst_token=dst_tok,
                query_src_index=src_i,
                query_dst_index=dst_i,
                query_tokens=query_tokens,
                query_prompt_len=prompt_len,
                top_k=top_k,
                struct_weight=sw,
                text_weight=tw,
                max_train_samples=max_train,
            )
        except Exception as exc:
            print(f"[structural-ast] failed: {exc}", flush=True)
            self._send_json(500, {"status": "error", "message": str(exc)})
            return
        self._send_json(200, result)

    def _load_report_from_req(self, req: dict) -> tuple[dict | None, str | None]:
        from src.eif_adapter_env import resolve_report_json_path, stamp_report_family

        report_file_name = str(req.get("reportFileName", "")).strip()
        if not report_file_name:
            return None, "reportFileName is required"
        report_json_path = resolve_report_json_path(CORR_RESULTS_DIR, report_file_name)
        if report_json_path is None:
            # Legacy / absolute fallback
            cand = CORR_RESULTS_DIR / report_file_name
            if cand.exists():
                report_json_path = cand
            else:
                alt = REPO_ROOT / report_file_name
                if alt.exists():
                    report_json_path = alt
        if report_json_path is None or not report_json_path.exists():
            return None, f"Report JSON not found: {report_file_name}"
        try:
            report = json.loads(report_json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None, "Report JSON is invalid"
        family = stamp_report_family(report, report_file_name)
        print(
            f"[report] loaded {report_file_name} family={family} "
            f"path={report_json_path}",
            flush=True,
        )
        return report, None

    def _handle_gold_saliency(self):
        """Stage1: teacher-force gold → top-k saliency sources for one gold target."""
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)
        try:
            req = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json(400, {"status": "error", "message": "Invalid JSON body"})
            return
        if CACHE_ONLY_MODE:
            self._send_json(503, {
                "status": "error",
                "message": "Gold live attribution needs a live model (EIF_CACHE_ONLY=1).",
            })
            return
        report, err = self._load_report_from_req(req)
        if err:
            self._send_json(400 if "required" in err else 404, {"status": "error", "message": err})
            return
        try:
            target_index = int(req["targetIndex"])
        except (KeyError, TypeError, ValueError):
            self._send_json(400, {"status": "error", "message": "targetIndex is required"})
            return
        top_k = req.get("topK")
        mode = str(req.get("mode", "gold") or "gold").strip().lower()
        source_index = req.get("sourceIndex")
        print(
            f"[live-saliency] mode={mode} targetIndex={target_index}"
            + (f" sourceIndex={source_index}" if source_index is not None else ""),
            flush=True,
        )
        try:
            with GOLD_LIVE_LOCK:
                result = gold_saliency_top_k(
                    report,
                    target_index=target_index,
                    top_k=int(top_k) if top_k is not None else None,
                    mode=mode,
                    source_index=int(source_index) if source_index is not None else None,
                )
        except Exception as exc:
            print(f"[live-saliency] failed: {exc}", flush=True)
            self._send_json(500, {"status": "error", "message": str(exc)})
            return
        self._send_json(200, result)

    def _handle_gold_retrieve_stage3(self):
        """Stage2+3: bank Top-K trains + Stage3 pairs for one gold saliency edge."""
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)
        try:
            req = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json(400, {"status": "error", "message": "Invalid JSON body"})
            return
        if CACHE_ONLY_MODE:
            self._send_json(503, {
                "status": "error",
                "message": "Gold live attribution needs a live model (EIF_CACHE_ONLY=1).",
            })
            return
        report, err = self._load_report_from_req(req)
        if err:
            self._send_json(400 if "required" in err else 404, {"status": "error", "message": err})
            return
        try:
            source_index = int(req["sourceIndex"])
            target_index = int(req["targetIndex"])
        except (KeyError, TypeError, ValueError):
            self._send_json(400, {
                "status": "error",
                "message": "sourceIndex and targetIndex are required integers",
            })
            return
        top_trains = req.get("topTrains")
        print(
            f"[gold] retrieve+stage3 source={source_index} target={target_index}",
            flush=True,
        )
        try:
            with GOLD_LIVE_LOCK:
                result = gold_retrieve_and_stage3(
                    report,
                    source_index=source_index,
                    target_index=target_index,
                    top_trains=int(top_trains) if top_trains is not None else None,
                )
        except Exception as exc:
            print(f"[gold] retrieve/stage3 failed: {exc}", flush=True)
            self._send_json(500, {"status": "error", "message": str(exc)})
            return
        print(
            f"[gold] done trains={len(result.get('relatedTrains') or [])} "
            f"pairs={len(result.get('correlationPairs') or [])}",
            flush=True,
        )
        self._send_json(200, result)

    def _handle_unlearn_pair_probe(self):
        """One-step LoRA learn/unlearn on CE + single-edge contrastive."""
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)
        try:
            req = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json(400, {"status": "error", "message": "Invalid JSON body"})
            return

        if CACHE_ONLY_MODE:
            self._send_json(503, {
                "status": "error",
                "message": (
                    "Unlearn probe needs a live model, but this server has "
                    "EIF_CACHE_ONLY=1. Restart without that flag (and with GPU) to run it."
                ),
            })
            return

        report, err = self._load_report_from_req(req)
        if err:
            self._send_json(400 if "required" in err else 404, {"status": "error", "message": err})
            return
        # Keep local name used below for report_json_path / unlearn call.
        report_file_name = str(req.get("reportFileName", "")).strip()
        report_json_path = _resolve_corr_report_path(report_file_name)
        if report_json_path is None:
            report_json_path = CORR_RESULTS_DIR / report_file_name

        try:
            train_sample_id = int(req["trainSampleId"])
            test_source_index = int(req["testSourceIndex"])
            test_target_index = int(req["testTargetIndex"])
            train_source_index = int(req["trainSourceIndex"])
            train_target_index = int(req["trainTargetIndex"])
        except (KeyError, TypeError, ValueError):
            self._send_json(400, {
                "status": "error",
                "message": (
                    "trainSampleId, testSourceIndex, testTargetIndex, "
                    "trainSourceIndex, trainTargetIndex are required integers"
                ),
            })
            return

        pair_id = str(req.get("pairId", "")).strip() or None
        raw_model_path = req.get("modelPath")
        model_path = str(raw_model_path).strip() if raw_model_path else None
        raw_base_path = req.get("baseModelPath")
        base_model_path = str(raw_base_path).strip() if raw_base_path else None
        unlearn_lr = float(req["unlearnLr"]) if req.get("unlearnLr") is not None else None
        normalize_grad = not bool(req.get("noNormalizeGrad", False))
        recompute_saliency = bool(req.get("recomputeSaliency", True))
        direction = str(req.get("direction", "unlearn") or "unlearn").strip().lower()
        # UI learn/unlearn keeps the step applied so the prob chart can refresh.
        persist = bool(req.get("persist", True))
        completion_mode = str(req.get("completionMode", "predict") or "predict").strip().lower()
        train_detail = req.get("trainSampleDetail")
        if train_detail is not None and not isinstance(train_detail, dict):
            train_detail = None

        print(
            f"[{direction}] pair={pair_id or '?'} train={train_sample_id} "
            f"test={test_source_index}->{test_target_index} "
            f"lr={unlearn_lr if unlearn_lr is not None else '(default)'} "
            f"persist={persist} mode={completion_mode} "
            f"adapter={model_path or '(env/report)'} base={base_model_path or '(auto/env)'}",
            flush=True,
        )

        try:
            with UNLEARN_PROBE_LOCK:
                result = run_unlearn_pair_probe(
                    report,
                    train_sample_id=train_sample_id,
                    test_source_index=test_source_index,
                    test_target_index=test_target_index,
                    train_source_index=train_source_index,
                    train_target_index=train_target_index,
                    pair_id=pair_id,
                    model_path=model_path,
                    base_model_path=base_model_path,
                    unlearn_lr=unlearn_lr,
                    normalize_grad=normalize_grad,
                    recompute_saliency=recompute_saliency,
                    sample_id=str(req.get("sampleId", "")).strip() or None,
                    report_json_path=str(report_json_path),
                    direction=direction,
                    persist=persist,
                    completion_mode=completion_mode,
                    train_sample_detail=train_detail,
                )
        except Exception as exc:
            print(f"[{direction}] failed: {exc}", flush=True)
            try:
                from src.unlearn_pair_probe import _release_cuda_memory
                _release_cuda_memory(reason="unlearn_api_error")
            except Exception:
                pass
            self._send_json(500, {"status": "error", "message": str(exc)})
            return

        print(
            f"[{direction}] done verdict={result.get('verdict')} "
            f"dLogP={result.get('delta', {}).get('logprob')} "
            f"dSal={result.get('delta', {}).get('saliency')} "
            f"restored={result.get('restored')}",
            flush=True,
        )
        self._send_json(200, result)

    def _handle_pair_intervene_recover(self):
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length:
            self.rfile.read(content_length)
        if CACHE_ONLY_MODE:
            self._send_json(503, {
                "status": "error",
                "message": "Recover needs a live model (EIF_CACHE_ONLY=1).",
            })
            return
        try:
            with UNLEARN_PROBE_LOCK:
                result = recover_pair_intervention()
        except Exception as exc:
            self._send_json(500, {"status": "error", "message": str(exc)})
            return
        print(
            f"[recover] recovered={result.get('recovered')} "
            f"direction={result.get('direction')} pair={result.get('pairId')}",
            flush=True,
        )
        self._send_json(200, result)

    def _handle_token_display_surfaces(self):
        """Rebuild Model/Gold display strings from token ids (fix U+FFFD chips)."""
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)
        try:
            req = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json(400, {"status": "error", "message": "Invalid JSON body"})
            return

        if CACHE_ONLY_MODE:
            self._send_json(503, {
                "status": "error",
                "message": "token-display-surfaces needs a live tokenizer (EIF_CACHE_ONLY=1).",
            })
            return

        report, err = self._load_report_from_req(req)
        if err:
            self._send_json(400 if "required" in err else 404, {"status": "error", "message": err})
            return

        baseline = report.get("test_sample_baseline") or {}
        # Already written by newer intervention runs.
        full_disp = baseline.get("full_tokens_display")
        gold_disp = baseline.get("correct_full_tokens_display")
        if (
            isinstance(full_disp, list) and full_disp
            and isinstance(gold_disp, list)
        ):
            self._send_json(200, {
                "status": "success",
                "fullTokensDisplay": full_disp,
                "correctFullTokensDisplay": gold_disp,
                "cached": True,
            })
            return

        try:
            with UNLEARN_PROBE_LOCK:
                _hydrate_eif_env()
                resolved_model, resolved_base = _resolve_paths(report, None, None)
                _model, tokenizer = _get_model(resolved_model, resolved_base)

                full_ids = baseline.get("full_token_ids")
                if not isinstance(full_ids, list) or not full_ids:
                    self._send_json(400, {
                        "status": "error",
                        "message": "Report missing full_token_ids; cannot rebuild display surfaces.",
                    })
                    return
                full_out = token_surfaces_for_display(tokenizer, [int(x) for x in full_ids])

                gold_out: list[str] = []
                gold_ids = baseline.get("correct_full_token_ids")
                if isinstance(gold_ids, list) and gold_ids:
                    gold_out = token_surfaces_for_display(tokenizer, [int(x) for x in gold_ids])
                else:
                    try:
                        _toks, g_ids, _pl = _gold_tokens_and_ids(report, tokenizer)
                        gold_out = token_surfaces_for_display(tokenizer, g_ids)
                    except Exception as exc:
                        print(f"[display] gold surfaces fallback failed: {exc}", flush=True)
                        gold_out = list(baseline.get("correct_full_tokens") or [])
        except Exception as exc:
            print(f"[display] failed: {exc}", flush=True)
            self._send_json(500, {"status": "error", "message": str(exc)})
            return

        print(
            f"[display] rebuilt full={len(full_out)} gold={len(gold_out)}",
            flush=True,
        )
        self._send_json(200, {
            "status": "success",
            "fullTokensDisplay": full_out,
            "correctFullTokensDisplay": gold_out,
            "cached": False,
        })

    def _handle_next_token_probs(self):
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)
        try:
            req = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json(400, {"status": "error", "message": "Invalid JSON body"})
            return

        if CACHE_ONLY_MODE:
            self._send_json(503, {
                "status": "error",
                "message": "Next-token probs need a live model (EIF_CACHE_ONLY=1).",
            })
            return

        report, err = self._load_report_from_req(req)
        if err:
            self._send_json(400 if "required" in err else 404, {"status": "error", "message": err})
            return

        try:
            target_index = int(req["targetIndex"])
        except (KeyError, TypeError, ValueError):
            self._send_json(400, {
                "status": "error",
                "message": "targetIndex is required (int > 0)",
            })
            return

        mode = str(req.get("mode", "predict") or "predict").strip().lower()
        top_k = int(req.get("topK", 10) or 10)
        raw_model_path = req.get("modelPath")
        model_path = str(raw_model_path).strip() if raw_model_path else None
        raw_base_path = req.get("baseModelPath")
        base_model_path = str(raw_base_path).strip() if raw_base_path else None
        view_family = str(req.get("viewFamily") or req.get("view_family") or "live").strip()
        gained_token_id = req.get("gainedTokenId")
        lost_token_id = req.get("lostTokenId")

        print(
            f"[probs] mode={mode} targetIndex={target_index} topK={top_k} view={view_family}",
            flush=True,
        )
        try:
            with UNLEARN_PROBE_LOCK:
                result = compute_next_token_probs(
                    report,
                    mode=mode,
                    target_index=target_index,
                    top_k=top_k,
                    model_path=model_path,
                    base_model_path=base_model_path,
                    view_family=view_family,
                    gained_token_id=int(gained_token_id) if gained_token_id is not None else None,
                    lost_token_id=int(lost_token_id) if lost_token_id is not None else None,
                )
        except Exception as exc:
            print(f"[probs] failed: {exc}", flush=True)
            self._send_json(500, {"status": "error", "message": str(exc)})
            return
        self._send_json(200, result)

    def _handle_degradation_retrieve(self):
        """Bank retrieve for ∇(logit_gained − logit_lost) vs a compare adapter."""
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)
        try:
            req = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json(400, {"status": "error", "message": "Invalid JSON body"})
            return

        if CACHE_ONLY_MODE:
            self._send_json(503, {
                "status": "error",
                "message": "Degradation retrieve needs a live model (EIF_CACHE_ONLY=1).",
            })
            return

        report, err = self._load_report_from_req(req)
        if err:
            self._send_json(400 if "required" in err else 404, {"status": "error", "message": err})
            return

        try:
            target_index = int(req["targetIndex"])
        except (KeyError, TypeError, ValueError):
            self._send_json(400, {
                "status": "error",
                "message": "targetIndex is required (int > 0)",
            })
            return

        mode = str(req.get("mode", "predict") or "predict").strip().lower()
        compare_family = str(req.get("compareFamily") or req.get("viewFamily") or "ce").strip()
        top_trains = req.get("topTrains")
        gained_token_id = req.get("gainedTokenId")
        lost_token_id = req.get("lostTokenId")
        print(
            f"[degrade] retrieve mode={mode} targetIndex={target_index} compare={compare_family}",
            flush=True,
        )
        try:
            from src.degradation_attribution import retrieve_degradation
            with GOLD_LIVE_LOCK:
                result = retrieve_degradation(
                    report,
                    mode=mode,
                    target_index=target_index,
                    compare_family=compare_family,
                    gained_token_id=int(gained_token_id) if gained_token_id is not None else None,
                    lost_token_id=int(lost_token_id) if lost_token_id is not None else None,
                    top_trains=int(top_trains) if top_trains is not None else None,
                )
        except Exception as exc:
            print(f"[degrade] retrieve failed: {exc}", flush=True)
            self._send_json(500, {"status": "error", "message": str(exc)})
            return
        self._send_json(200, result)

    def _handle_prepare_train_probe(self):
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)
        try:
            req = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json(400, {"status": "error", "message": "Invalid JSON body"})
            return

        report_file_name = str(req.get("reportFileName", "")).strip()
        if not report_file_name:
            self._send_json(400, {"status": "error", "message": "reportFileName is required"})
            return

        report_json_path = _resolve_corr_report_path(report_file_name)
        if report_json_path is None or not report_json_path.exists():
            self._send_json(404, {"status": "error", "message": f"Report JSON not found: {report_file_name}"})
            return

        train_sample_id = req.get("trainSampleId")
        probe_pairs = req.get("probePairs")
        if not isinstance(train_sample_id, int):
            self._send_json(400, {"status": "error", "message": "trainSampleId is required"})
            return
        if not isinstance(probe_pairs, list) or not probe_pairs:
            self._send_json(400, {"status": "error", "message": "probePairs is required"})
            return

        base_sample_id = str(req.get("sampleId", "")).strip() or infer_sample_id(str(report_json_path))
        context_radius = int(req.get("contextRadius", 1))
        include_full_train = bool(req.get("includeFullTrain", False))
        raw_focus_train_indices = req.get("focusTrainIndices", [])
        focus_train_indices = raw_focus_train_indices if isinstance(raw_focus_train_indices, list) else []
        # Which surface will draw this probe. Defaults to "window" so any caller
        # that predates this field keeps the original upload-and-navigate flow.
        render_target = "inline" if str(req.get("renderTarget", "")).strip() == "inline" else "window"
        raw_focus_pair_ids = req.get("focusPairIds", [])
        focus_pair_ids = [str(x) for x in raw_focus_pair_ids] if isinstance(raw_focus_pair_ids, list) else []
        probe_built_from_pt = False
        pt_arrays = None
        model_path = str(req.get("modelPath", "")).strip() or str(DEFAULT_MODEL_PATH)
        ttav_upload_url = str(req.get("ttavUploadUrl", "")).strip()
        ttav_url = str(req.get("ttavUrl", "")).strip()
        vis_method = str(req.get("visMethod", "TimeVis")).strip() or "TimeVis"
        vis_id = str(req.get("visId", "1")).strip() or "1"
        test_target_index, test_source_index = _probe_anchor_indices(probe_pairs)
        probe_sample_id = _make_probe_sample_id(
            base_sample_id,
            train_sample_id,
            probe_pairs,
            context_radius,
            include_full_train,
        )
        explicit_cache_path = str(req.get("probeCachePath", "")).strip() or _default_probe_cache_path(base_sample_id, train_sample_id, probe_sample_id)
        status_key = probe_sample_id
        started_at = time()

        print(
            f"[probe] sampleId={base_sample_id} trainSampleId={train_sample_id} pairs={len(probe_pairs)} vis={vis_method}/{vis_id}",
            flush=True,
        )

        try:
            _set_prepare_status(status_key, "checking_cache", "Checking train probe bundle cache", active=True)
            cached = _find_cached_probe_payload(
                base_sample_id,
                train_sample_id,
                explicit_cache_path,
                test_target_index,
                test_source_index,
            )
            probe_cache_hit = cached is not None

            if probe_cache_hit:
                payload, cache_source = cached
                print(f"[probe] sampleId={base_sample_id} trainSampleId={train_sample_id} cache={cache_source}", flush=True)
                _set_prepare_status(status_key, "cache_hit", "Matching train probe bundle cache found", active=True)
            else:
                # Precomputed token vectors turn the expensive step into a lookup:
                # building a probe normally needs the model in memory, which is what
                # CACHE_ONLY_MODE exists to prevent, but with .pt files on disk the
                # whole thing takes ~3s on CPU. Tried before the cache-only refusal
                # so that path only fires when there are genuinely no embeddings.
                pt_paths = _find_token_embeddings(base_sample_id, train_sample_id)
                if pt_paths is not None:
                    test_pt, train_pt = pt_paths
                    print(f"[probe] sampleId={base_sample_id} trainSampleId={train_sample_id} "
                          f"building from .pt (no model load)", flush=True)
                    _set_prepare_status(status_key, "building_probe",
                                        "Building probe from precomputed embeddings", active=True)
                    try:
                        payload, _embeddings, _projection = build_probe_from_pt(
                            report_path=str(report_json_path),
                            test_pt_path=test_pt,
                            train_pt_path=train_pt,
                            train_id=train_sample_id,
                            test_target=test_target_index,
                            test_source=test_source_index,
                            sample_id=probe_sample_id,
                            # The report's tokens and the .pt's can diverge inside the
                            # generated output when they record different runs; keep
                            # the aligned prefix rather than refusing outright, and
                            # let the endpoint-range check reject edges it can't place.
                            truncate_to_aligned=True,
                            # Only the ticked pairs get source/target roles, so the
                            # plot rings what the user chose instead of every
                            # endpoint in a 100+ pair group.
                            focus_pair_ids=focus_pair_ids or None,
                        )
                    except Exception as build_exc:
                        message = f"Building the probe from precomputed embeddings failed: {build_exc}"
                        print(f"[probe] sampleId={base_sample_id} build_from_pt failed: {build_exc}", flush=True)
                        _set_prepare_status(status_key, "error", message, active=False, error=True)
                        self._send_json(400, {"status": "error", "message": message})
                        return

                    # Written as projection.json so the report page can read it
                    # without parsing a payload that would be ~190x larger.
                    #
                    # Under the *stable* name, not probe_sample_id: that one hashes
                    # the request parameters, while the report page fetches the id
                    # returned as pregeneratedSampleId, which is derived from the
                    # anchoring edge. Writing under the hash puts the file somewhere
                    # nothing ever looks — a silent 404 with a success response.
                    _write_probe_projection_cache(
                        _stable_probe_sample_id(
                            base_sample_id, train_sample_id, test_target_index, test_source_index
                        ),
                        payload,
                        _projection,
                    )
                    probe_built_from_pt = True
                    # build_probe_from_pt hands the arrays back separately so the
                    # inline path never has to materialise them. The TTAV upload
                    # below does need them in the payload, so keep them to hand
                    # and fold them in only on that branch.
                    pt_arrays = (_embeddings, _projection)

                elif CACHE_ONLY_MODE:
                    # Building a probe always requires loading the model, and this
                    # server is configured never to do that.
                    message = (
                        f"No precomputed train probe for {base_sample_id} / TRAIN #{train_sample_id}. "
                        "This server has live model loading disabled (EIF_CACHE_ONLY=1); "
                        "precompute the probe elsewhere and drop it into "
                        f"ttav_bundles_real/{_stable_probe_sample_id(base_sample_id, train_sample_id, test_target_index, test_source_index)}/ first."
                    )
                    print(f"[probe] sampleId={base_sample_id} trainSampleId={train_sample_id} cache=miss (cache-only mode)", flush=True)
                    _set_prepare_status(status_key, "error", message, active=False, error=True)
                    self._send_json(404, {"status": "error", "message": message})
                    return

                else:
                    _set_prepare_status(status_key, "building_probe", "Building train-sample embedding probe", active=True)
                    payload = build_train_probe_bundle_payload(
                        report_json_path=str(report_json_path),
                        model_path=model_path,
                        train_sample_id=train_sample_id,
                        probe_pairs=probe_pairs,
                        sample_id=base_sample_id,
                        embedding_type="contextual",
                        hidden_layer=-1,
                        vis_method=vis_method,
                        vis_id=vis_id,
                        context_radius=context_radius,
                        include_full_train=include_full_train,
                        focus_train_indices=focus_train_indices,
                        progress_callback=lambda stage, message: _set_prepare_status(status_key, stage, message, active=True),
                    )

            payload["sample_id"] = probe_sample_id
            payload["vis_method"] = vis_method
            payload["vis_id"] = vis_id
            payload["overwrite"] = True
            # A probe built from .pt has already been written as projection.json,
            # which is the only file the in-page canvas reads. Re-writing the full
            # cache would materialise the embeddings this path exists to avoid.
            if not probe_cache_hit and not probe_built_from_pt:
                _set_prepare_status(status_key, "writing_local_cache", "Writing train probe bundle cache", active=True)
                write_local_bundle_cache(probe_sample_id, payload, explicit_path=explicit_cache_path)

            upload_result = None
            upload_error = None
            browser_upload_required = False

            # The report page draws the plot itself in "本页显示" mode, reading the
            # projection straight off disk — TTAV is not in that picture at all, so
            # uploading to it is pure cost (tens of MB per click). "新窗口" mode is
            # unchanged: it navigates to the TTAV app, which can only show a bundle
            # that was uploaded.
            if render_target == "inline":
                _set_prepare_status(status_key, "completed",
                                    "Probe ready for in-page rendering (TTAV upload skipped)",
                                    active=False)
                print(f"[probe] sampleId={base_sample_id} trainSampleId={train_sample_id} "
                      f"renderTarget=inline — skipping TTAV upload", flush=True)
                self._send_json(200, {
                    "status": "success",
                    "sampleId": probe_sample_id,
                    "pregeneratedSampleId": _stable_probe_sample_id(
                        base_sample_id, train_sample_id, test_target_index, test_source_index
                    ),
                    "renderTarget": "inline",
                    "builtFromPrecomputedEmbeddings": probe_built_from_pt,
                    "probeCacheHit": probe_cache_hit,
                    "contentPath": None,
                    "visMethod": vis_method,
                    "visId": vis_id,
                    "trainSampleId": train_sample_id,
                    "selectedIndices": payload.get("selected_indices", []),
                    "targetIndex": payload.get("target_index"),
                    "promptLen": payload.get("bundle", {}).get("prompt_len", 0),
                    "comparisonSummary": payload.get("comparison_summary"),
                    "browserUploadRequired": False,
                    "bundlePayload": None,
                })
                return

            # registerEIFBundle validates that bundle.embeddings and .projection are
            # present and the right length, so a .pt-built payload has to be
            # completed before it can be uploaded — without this it fails with a
            # bare HTTP 400 and falls back to asking the browser to upload the same
            # incomplete payload.
            if probe_built_from_pt and pt_arrays is not None:
                payload["bundle"]["embeddings"] = pt_arrays[0].tolist()
                payload["bundle"]["projection"] = pt_arrays[1].tolist()

            _set_prepare_status(status_key, "uploading_to_ttav", "Uploading train probe to TTAV", active=True)
            try:
                upload_result = upload_bundle(ttav_upload_url, payload)
            except Exception as upload_exc:
                upload_error = str(upload_exc)
                browser_upload_required = True
                print(
                    f"[probe] sampleId={base_sample_id} trainSampleId={train_sample_id} server_upload_failed: {upload_error}",
                    flush=True,
                )
                _set_prepare_status(
                    status_key,
                    "browser_upload_required",
                    "Server-side TTAV upload failed; browser upload fallback required",
                    active=False,
                )

            elapsed = time() - started_at
            print(f"[probe] sampleId={base_sample_id} trainSampleId={train_sample_id} elapsed={elapsed:.2f}s", flush=True)
            if not browser_upload_required:
                _set_prepare_status(status_key, "completed", f"Train probe completed in {elapsed:.1f}s", active=False)
        except Exception as exc:
            elapsed = time() - started_at
            print(f"[probe] sampleId={base_sample_id} trainSampleId={train_sample_id} error after {elapsed:.2f}s: {exc}", flush=True)
            _set_prepare_status(status_key, "error", str(exc), active=False, error=True)
            self._send_json(500, {"status": "error", "message": str(exc)})
            return

        self._send_json(200, {
            "status": "success",
            "sampleId": (upload_result or {}).get("sample_id") or probe_sample_id,
            # The directory a precomputed probe actually lives in under
            # ttav_bundles_real/. "sampleId" above is a request hash, so it can't
            # be used to find the file — the report's in-page plot reads the
            # bundle straight off disk and needs this name instead. Sent from
            # here rather than rebuilt in the frontend so the naming rule stays
            # in the two places that already own it.
            "pregeneratedSampleId": _stable_probe_sample_id(
                base_sample_id, train_sample_id, test_target_index, test_source_index
            ),
            "contentPath": (upload_result or {}).get("content_path"),
            "visMethod": (upload_result or {}).get("vis_method", vis_method),
            "visId": (upload_result or {}).get("vis_id", vis_id),
            "ttavUrl": ttav_url,
            "trainSampleId": train_sample_id,
            "selectedIndices": payload.get("selected_indices", []),
            "targetIndex": payload.get("target_index"),
            "promptLen": payload.get("bundle", {}).get("prompt_len", 0),
            "probeCachePath": explicit_cache_path,
            "probeCacheHit": probe_cache_hit,
            "comparisonSummary": payload.get("comparison_summary"),
            "uploadResult": upload_result,
            "browserUploadRequired": browser_upload_required,
            "uploadError": upload_error,
            "bundlePayload": payload if browser_upload_required else None,
        })


def main():
    parser = argparse.ArgumentParser(description="EIF API for preparing and uploading TTAV bundles.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    httpd = ThreadingHTTPServer((args.host, args.port), TTAVBundleRequestHandler)
    print(f"EIF TTAV bundle API listening on http://{args.host}:{args.port}", flush=True)
    print(f"CACHE_ONLY_MODE={'ON — live model loading disabled server-wide' if CACHE_ONLY_MODE else 'off'}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
