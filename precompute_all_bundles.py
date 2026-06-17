"""
Batch pre-compute real TTAV bundles for all EIF correlation result files.

Loads the LLM model once, then processes every sample.
Output goes to ttav_bundles_real/{sampleId}/ so the EIF frontend fallback
(loadPrecomputedRealBundle) can serve them without the EIF API running.

Usage:
    conda run -n eif python precompute_all_bundles.py
    conda run -n eif python precompute_all_bundles.py --force  # overwrite existing
"""

import argparse
import json
from pathlib import Path

from src.export_real_ttav_bundle import DEFAULT_MODEL_PATH, build_real_bundle_payload
from src.export_ttav_bundle import infer_sample_id
from src.ttav_bundle_api import write_local_bundle_cache

REPO_ROOT = Path(__file__).resolve().parent
CORR_DIR = REPO_ROOT / "correlation_matching_results"
OUTPUT_ROOT = REPO_ROOT / "ttav_bundles_real"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--force", action="store_true", help="Overwrite existing bundles")
    parser.add_argument("--vis-method", default="TimeVis")
    parser.add_argument("--vis-id", default="1")
    args = parser.parse_args()

    report_files = sorted(CORR_DIR.glob("*.json"))
    if not report_files:
        print(f"No JSON files found in {CORR_DIR}")
        return

    print(f"Found {len(report_files)} report files. Model: {args.model_path}\n")

    results = []
    for report_path in report_files:
        sample_id = infer_sample_id(str(report_path))
        output_dir = OUTPUT_ROOT / sample_id
        bundle_file = output_dir / "bundle_payload.json"

        if bundle_file.exists() and not args.force:
            print(f"[SKIP] {sample_id} — bundle already exists")
            results.append((sample_id, "skipped"))
            continue

        print(f"[COMPUTING] {sample_id} ...")
        try:
            def progress(stage, msg):
                print(f"  [{stage}] {msg}")

            payload = build_real_bundle_payload(
                report_json_path=str(report_path),
                model_path=args.model_path,
                sample_id=sample_id,
                vis_method=args.vis_method,
                vis_id=args.vis_id,
                dtype_name=args.dtype,
                progress_callback=progress,
            )
            write_local_bundle_cache(sample_id, payload, explicit_path=str(output_dir))
            n = len(payload["bundle"]["labels"])
            print(f"  → saved {n} tokens to {output_dir}\n")
            results.append((sample_id, "ok"))
        except Exception as e:
            print(f"  [ERROR] {e}\n")
            results.append((sample_id, f"error: {e}"))

    print("\n=== Summary ===")
    for sample_id, status in results:
        print(f"  {sample_id}: {status}")


if __name__ == "__main__":
    main()
