#!/usr/bin/env python3
"""Create an entirely synthetic standard input bundle for a connectivity run.

The generated manifest works with reproduce.py. The generator itself does not
run the detector or reproduce a paper result.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timedelta
from pathlib import Path

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "generated"


def example_rows():
    start = datetime(2000, 1, 1, 0, 0)
    glucose = []
    for minute in range(24 * 60):
        baseline = 6.0 + 0.15 * math.sin(2 * math.pi * minute / 360)
        excursions = sum(
            2.5 * math.exp(-((minute - onset - 45) / 40) ** 2)
            for onset in (420, 720, 1140) if minute >= onset
        )
        if not (300 <= minute < 350):  # Retained long gap exercises masking.
            glucose.append(((start + timedelta(minutes=minute)).isoformat(), round(baseline + excursions, 4)))
    bolus = [((start + timedelta(minutes=m)).isoformat(), 2.0) for m in (405, 705, 1125)]
    references = [((start + timedelta(minutes=m)).isoformat(),) for m in (420, 720, 1140)]
    return glucose, bolus, references


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true", help="Print row counts without writing files.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    glucose, bolus, references = example_rows()
    if args.preview:
        print(f"Synthetic rows: glucose={len(glucose)}, bolus={len(bolus)}, references={len(references)}")
        print("Input-schema illustration only; no detection analysis was run.")
        return 0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for filename, header, rows in (
        ("glucose.csv", ("ts", "value"), glucose),
        ("bolus.csv", ("ts", "dose"), bolus),
        ("reference_events.csv", ("ts",), references),
    ):
        target = args.out_dir / filename
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite {target}")
        with target.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)
        print(target)
    manifest = {
        "schema_version": 1, "dataset_id": "synthetic_demo", "time_basis": "local_naive",
        "units": {"glucose": "mmol/L", "bolus": "U"},
        "records": [{"record_id": "demo_01", "participant_id": "demo_01", "session_id": "synthetic",
                     "monitoring_days": 1.0, "glucose_file": "glucose.csv", "bolus_file": "bolus.csv",
                     "reference_file": "reference_events.csv"}],
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Manifest created. Run: python reproduce.py --manifest", args.out_dir / "manifest.json", "--smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
