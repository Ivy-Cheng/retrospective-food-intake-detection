#!/usr/bin/env python3
"""Create a local, date-shifted standard input bundle without modifying source data."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from parse_nhs_data import parse_nhs_meal_file

ROOT = Path(__file__).resolve().parent

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, date_format="%Y-%m-%dT%H:%M:%S", float_format="%.17g")

def prepare(raw_root, reference_root, candidate_root, out_dir, provenance_path):
    cohort = json.loads((ROOT / "configs/nhs_cohort.json").read_text())["included_sessions"]
    out_dir = out_dir.resolve()
    provenance_path = provenance_path.resolve()
    for source_root in [raw_root.resolve(), reference_root.resolve(), candidate_root.resolve()]:
        if out_dir == source_root or source_root in out_dir.parents or provenance_path == source_root or source_root in provenance_path.parents:
            raise ValueError("Output and provenance must be outside every read-only source root.")
    if provenance_path.exists():
        raise FileExistsError(f"Refusing to overwrite private provenance: {provenance_path}")
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing bundle: {out_dir}")
    inputs = []
    hashes = {}
    for item in cohort:
        pid, variant = item["participant_id"], item["variant"]
        paths = {
            "raw": raw_root / f"xt_apman{pid}_{variant}.txt",
            "reference": reference_root / f"ground_truth_{pid}_{variant}.csv",
            "candidate": candidate_root / f"nhs_{pid}/plot_nhs_{pid}_{variant}_local_maxima.csv",
        }
        for p in paths.values():
            hashes[str(p)] = digest(p)
        raw = parse_nhs_meal_file(paths["raw"])
        glucose = raw["glucose_concentration"][["Time", "Concentration"]].rename(columns={"Time": "ts", "Concentration": "value"})
        bolus = raw["insulin_bolus"][["Time", "Bolus"]].rename(columns={"Time": "ts", "Bolus": "dose"})
        refs = pd.read_csv(paths["reference"], float_precision="round_trip")[["ts"]]
        candidates = pd.read_csv(paths["candidate"], float_precision="round_trip")[["delta_S"]]
        for frame in [glucose, bolus, refs]:
            frame["ts"] = pd.to_datetime(frame["ts"], format="mixed", errors="raise")
            if frame.ts.isna().any():
                raise ValueError(f"Invalid time in {pid}/{variant}")
            frame.sort_values("ts", kind="stable", inplace=True)
        if not np.isfinite(glucose.value.to_numpy(dtype=float)).all() or not np.isfinite(bolus.dose.to_numpy(dtype=float)).all():
            raise ValueError(f"Invalid measurements in {pid}/{variant}")
        inputs.append((item, paths, glucose, bolus, refs, candidates))
    # One integer-day offset per participant preserves times of day and spacing between both sessions.
    anchors = {}
    for item, _, glucose, bolus, refs, _ in inputs:
        pid = item["participant_id"]
        lo = min(f.ts.min() for f in [glucose, bolus, refs] if not f.empty).normalize()
        anchors[pid] = min(anchors.get(pid, lo), lo)
    offsets = {pid: pd.Timestamp("2000-01-01") - lo for pid, lo in anchors.items()}
    records, calibration_records, private_records = [], [], []
    for item, paths, glucose, bolus, refs, candidates in inputs:
        pid, variant = item["participant_id"], item["variant"]
        rid = f"nhs_{pid}_{variant}"
        entry = {"record_id": rid, "participant_id": pid, "session_id": variant, "variant": variant, "monitoring_days": 1.0, "sha256": {}}
        for name, frame, key in [
            ("glucose", glucose, "glucose_file"), ("bolus", bolus, "bolus_file"), ("references", refs, "reference_file")
        ]:
            shifted = frame.copy()
            shifted["ts"] = shifted.ts + offsets[pid]
            # Validate lossless timing shift before writing.
            assert shifted.ts.equals(frame.ts + offsets[pid])
            relative = f"records/{rid}/{name}.csv"
            target = out_dir / relative
            write_csv(shifted, target)
            entry[key] = relative
            entry["sha256"][key] = digest(target)
        relative = f"calibration/{rid}_candidates.csv"
        write_csv(candidates, out_dir / relative)
        calibration_records.append({"record_id": rid, "candidate_file": Path(relative).name, "evidence": "full_fusion", "sha256": digest(out_dir / relative)})
        records.append(entry)
        private_records.append({"record_id": rid, "day_shift": int(offsets[pid].days), "sources": {k: str(v) for k, v in paths.items()}})
    calibration_manifest = {"records": calibration_records, "description": "Historical full-fusion export rows, delta_S only. No deduplication or acceptance filtering."}
    (out_dir / "calibration/manifest.json").write_text(json.dumps(calibration_manifest, indent=2) + "\n")
    manifest = {
        "schema_version": 1, "dataset_id": "nhs_v1", "time_basis": "local_naive",
        "units": {"glucose": "mmol/L", "bolus": "U"},
        "date_handling": "Constant integer-day shift per participant to a virtual 2000-01-01 anchor; no source dates or shift keys distributed.",
        "release_status": "LOCAL_ONLY_PENDING_DATA_SHARING_PERMISSION",
        "calibration": {"mode": "historical_candidate_exports", "manifest_file": "calibration/manifest.json"},
        "records": records,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for source, before in hashes.items():
        if digest(source) != before:
            raise RuntimeError(f"Source changed: {source}")
    summary = {
        "sessions": len(records), "participants": len(anchors),
        "glucose_rows": sum(len(x[2]) for x in inputs), "bolus_rows": sum(len(x[3]) for x in inputs),
        "reference_events": sum(len(x[4]) for x in inputs), "candidate_rows": sum(len(x[5]) for x in inputs),
        "source_hashes_unchanged": True,
    }
    (out_dir / "bundle_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / "README.md").write_text(
        "# NHS standard input bundle\n\nLocal staging only; sharing permission is pending. Date shifting is a preparation step, not a determination of disclosure approval.\n\n"
        "Raw CGM and bolus measurements are in records/. Duplicate CGM rows are preserved for the original preprocessing. "
        "References are the already prepared evaluation labels, not detector inputs. calibration/ preserves historical prominence rows, without source timestamps or record headers. "
        "No original raw file headers, email identifiers or date-shift keys are included. See ../../docs/data_format.md for the reusable input contract.\n"
    )
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(json.dumps({"source_sha256": hashes, "private_date_mapping": private_records, "summary": summary}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return manifest

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--reference-root", type=Path, required=True)
    p.add_argument("--candidate-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=ROOT / "datasets/nhs_v1")
    p.add_argument("--private-provenance", type=Path, default=ROOT / "audit/nhs_preparation_provenance.json")
    a = p.parse_args()
    prepare(a.raw_root.resolve(), a.reference_root.resolve(), a.candidate_root.resolve(), a.out_dir, a.private_provenance)

if __name__ == "__main__":
    main()
