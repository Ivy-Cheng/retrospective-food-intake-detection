#!/usr/bin/env python3
"""Fit/apply an unrounded dataset-wide Q75 from historical candidate exports.

Fit from full-fusion candidate exports. Reuse the resulting threshold file when applying
ablations or sensitivity variants. This entry point does not require clinical
reference events, raw data access or UKF recalculation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pipelines.prominence_threshold import (
    detect_dataset_records, fit_dataset_prominence_threshold,
    load_score_manifest, load_candidate_manifest, read_dataset_threshold,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="Score manifest for detection; can also name historical candidate files.")
    parser.add_argument("--candidate-manifest", type=Path, help="Historical full-fusion candidate exports for calibration.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--threshold-file", type=Path, help="Reuse a full-fusion calibration for this dataset.")
    parser.add_argument("--fit-only", action="store_true")
    args = parser.parse_args()
    if args.manifest is None and not args.fit_only:
        parser.error("--manifest is required when applying the threshold.")
    if args.manifest is None and args.candidate_manifest is None and args.threshold_file is None:
        parser.error("Provide historical candidate exports or an existing calibration.")
    records, contexts, entries = load_score_manifest(args.manifest) if args.manifest else ({}, {}, {})
    if args.threshold_file is None:
        candidate_path = args.candidate_manifest or args.manifest
        exports = load_candidate_manifest(candidate_path)
        metadata = fit_dataset_prominence_threshold(records or None, candidate_exports=exports)
        metadata["candidate_manifest_sha256"] = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    else:
        metadata = read_dataset_threshold(args.threshold_file)
    metadata = dict(metadata)
    metadata["application_record_ids"] = list(records)
    metadata["application_manifest_sha256"] = hashlib.sha256(args.manifest.read_bytes()).hexdigest() if args.manifest else None
    if args.threshold_file:
        metadata["reused_calibration"] = True
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "dataset_threshold.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Q75 threshold (unrounded): {metadata['threshold']:.17g}; records={len(records)}")
    if not args.fit_only:
        for record_id, events in detect_dataset_records(records, metadata, contexts).items():
            # Keep record identifiers out of paths; retain their mapping.
            name = hashlib.sha256(record_id.encode()).hexdigest()[:16] + "_events.csv"
            events.to_csv(args.out_dir / name, index=False)
            print(f"{record_id}: {len(events)} events -> {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
