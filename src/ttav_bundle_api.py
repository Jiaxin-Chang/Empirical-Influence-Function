import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from time import time
from urllib.parse import parse_qs, urlparse

import numpy as np

from src.export_real_ttav_bundle import DEFAULT_MODEL_PATH, build_real_bundle_payload
from src.export_ttav_bundle import build_bundle_payload, infer_sample_id, upload_bundle


REPO_ROOT = Path(__file__).resolve().parent.parent
CORR_RESULTS_DIR = REPO_ROOT / "correlation_matching_results"
EIF_BUNDLE_CACHE_ROOT = REPO_ROOT / "ttav_bundles"
PREPARE_STATUS_LOCK = Lock()
PREPARE_STATUS: dict[str, dict] = {}


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
    payload_path = _cache_payload_path(sample_id, explicit_path)
    return json.loads(payload_path.read_text(encoding="utf-8"))


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

        report_json_path = CORR_RESULTS_DIR / report_file_name
        if not report_json_path.exists():
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
        require_cached = bool(req.get("requireCached", False))
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
            payload_path = _cache_payload_path(resolved_sample_id, explicit_cache_path)
            cache_hit = False

            if payload_path.exists():
                cached_payload = load_local_bundle_cache(resolved_sample_id, explicit_cache_path)
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
                    _set_prepare_status(resolved_sample_id, "error", "EIF local bundle cache not found. Prepare sample first.", active=False, error=True)
                    self._send_json(404, {
                        "status": "error",
                        "message": f"EIF local bundle cache not found for sample: {resolved_sample_id}. Please prepare the sample first.",
                    })
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
            _set_prepare_status(resolved_sample_id, "writing_local_cache", "Writing EIF local bundle cache", active=True)
            write_local_bundle_cache(resolved_sample_id, payload, explicit_path=explicit_cache_path)
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
        }
        self._send_json(200, response)


def main():
    parser = argparse.ArgumentParser(description="EIF API for preparing and uploading TTAV bundles.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    httpd = ThreadingHTTPServer((args.host, args.port), TTAVBundleRequestHandler)
    print(f"EIF TTAV bundle API listening on http://{args.host}:{args.port}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
