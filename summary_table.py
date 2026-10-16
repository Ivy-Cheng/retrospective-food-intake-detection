#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a participant × method summary table from evaluation_new.py outputs.

Reads:
  results/evaluation/primary/evaluation_by_participant.csv
  results/evaluation/primary/onset_errors_{patient}_{variant}.csv

Writes:
  results/evaluation/primary/summary_table_by_participant_method.csv

Run from Retrospective_meal_detection after evaluation_new.py:

  python summary_table.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PRIMARY_DIR = SCRIPT_DIR / "results" / "evaluation" / "primary"

SYSTEM_ORDER = [
    "baseline",
    "fusion",
    "signal_based_only",
    "physio_based_only",
]

METHOD_LABELS = {
    "baseline": "Baseline",
    "fusion": "Fusion",
    "physio_based_only": "Physio-only",
    "signal_based_only": "Signal-only",
}

SUMMARY_COLUMNS = [
    "Participant ID",
    "Method",
    "Days",
    "Reference events",
    "Detected events",
    "TP",
    "FP",
    "FN",
    "Sensitivity",
    "FP/day",
    "Median |Abs onset diff|",
    "Median signed onset diff",
]


def _ordered_methods(methods: list[str]) -> list[str]:
    order = [METHOD_LABELS[s] for s in SYSTEM_ORDER if METHOD_LABELS.get(s) in methods]
    extra = sorted(m for m in methods if m not in order)
    return order + extra


def load_participant_metrics(primary_dir: Path) -> pd.DataFrame:
    path = primary_dir / "evaluation_by_participant.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run evaluation_new.py first."
        )
    df = pd.read_csv(path, dtype={"patient_id": str})
    df["patient_id"] = df["patient_id"].astype(str).str.zfill(2)
    return df


def load_median_signed_onset(primary_dir: Path) -> pd.DataFrame:
    paths = sorted(primary_dir.glob("onset_errors_*.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No onset_errors_*.csv under {primary_dir}. Run evaluation_new.py first."
        )
    onset = pd.concat(
        [pd.read_csv(p, dtype={"patient_id": str}) for p in paths],
        ignore_index=True,
    )
    onset["patient_id"] = onset["patient_id"].astype(str).str.zfill(2)
    if "signed_onset_min" not in onset.columns:
        raise ValueError("onset_errors CSVs missing column signed_onset_min")
    return (
        onset.groupby(["patient_id", "system"], as_index=False)["signed_onset_min"]
        .median()
        .rename(columns={"signed_onset_min": "median_signed_onset_min"})
    )


def build_summary_table(primary_dir: Path) -> pd.DataFrame:
    part = load_participant_metrics(primary_dir)
    signed_med = load_median_signed_onset(primary_dir)

    required = {"patient_id", "system", "TP", "FN", "FP", "n_days", "sensitivity", "fp_per_day", "median_abs_onset_min"}
    missing = required - set(part.columns)
    if missing:
        raise ValueError(f"evaluation_by_participant.csv missing columns: {sorted(missing)}")

    df = part.merge(signed_med, on=["patient_id", "system"], how="left")
    df["Method"] = df["system"].map(METHOD_LABELS).fillna(df["system"])
    df["Reference events"] = df["TP"] + df["FN"]
    df["Detected events"] = df["TP"] + df["FP"]

    out = pd.DataFrame(
        {
            "Participant ID": df["patient_id"],
            "Method": df["Method"],
            "Days": df["n_days"],
            "Reference events": df["Reference events"],
            "Detected events": df["Detected events"],
            "TP": df["TP"],
            "FP": df["FP"],
            "FN": df["FN"],
            "Sensitivity": pd.to_numeric(df["sensitivity"], errors="coerce").round(3),
            "FP/day": df["fp_per_day"],
            "Median |Abs onset diff|": df["median_abs_onset_min"],
            "Median signed onset diff": df["median_signed_onset_min"],
        }
    )

    method_order = {m: i for i, m in enumerate(_ordered_methods(out["Method"].unique().tolist()))}
    out["_method_order"] = out["Method"].map(method_order)
    out = out.sort_values(["Participant ID", "_method_order"]).drop(columns="_method_order")
    return out[SUMMARY_COLUMNS]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build participant × method summary table from primary evaluation outputs."
    )
    parser.add_argument(
        "--primary-dir",
        type=Path,
        default=DEFAULT_PRIMARY_DIR,
        help="Directory with evaluation_by_participant.csv and onset_errors_*.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path (default: <primary-dir>/summary_table_by_participant_method.csv)",
    )
    args = parser.parse_args(argv)

    primary_dir = args.primary_dir.expanduser().resolve()
    out_path = (
        args.out.expanduser().resolve()
        if args.out
        else primary_dir / "summary_table_by_participant_method.csv"
    )

    summary = build_summary_table(primary_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)

    print(f"Wrote {out_path} ({len(summary)} rows)")
    print(summary.head(8).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
