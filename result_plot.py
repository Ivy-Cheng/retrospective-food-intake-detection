#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate onset_errors_{patient_id}_{label}.csv from results/evaluation/primary,
write summary statistics and a box plot under results/evaluation/primary/onset/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Match evaluation_new.py method order when present
SYSTEM_ORDER = [
    "baseline",
    "fusion",
    "signal_based_only",
    "physio_based_only",
]

METHOD_LABELS = {
    "baseline": "Baseline",
    "fusion": "Fusion",
    "signal_based_only": "Signal-only",
    "physio_based_only": "Physio-only",
}

SCRIPT_DIR = Path(__file__).resolve().parent
PRIMARY_DIR = SCRIPT_DIR / "results" / "evaluation" / "primary"
OUT_DIR = PRIMARY_DIR / "onset"


def _ordered_systems(systems: list[str]) -> list[str]:
    known = [s for s in SYSTEM_ORDER if s in systems]
    extra = sorted(s for s in systems if s not in SYSTEM_ORDER)
    return known + extra


def _iqr(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    q1, q3 = np.percentile(values, [25, 75])
    return float(q3 - q1)


def load_onset_errors(primary_dir: Path) -> pd.DataFrame:
    """Read onset_errors_*.csv files directly under primary (not subfolders)."""
    paths = sorted(primary_dir.glob("onset_errors_*.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No onset_errors_*.csv found in {primary_dir}"
        )

    frames: list[pd.DataFrame] = []
    for path in paths:
        df = pd.read_csv(path)
        required = {"system", "abs_onset_min"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path.name}: missing columns {missing}")
        df = df.copy()
        df["source_file"] = path.name
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined["abs_onset_min"] = pd.to_numeric(combined["abs_onset_min"], errors="coerce")
    if "signed_onset_min" in combined.columns:
        combined["signed_onset_min"] = pd.to_numeric(
            combined["signed_onset_min"], errors="coerce"
        )
    combined = combined.dropna(subset=["system", "abs_onset_min"])
    return combined


def build_summary_by_method(df: pd.DataFrame) -> pd.DataFrame:
    """Per-method event counts, median and IQR of absolute onset error (minutes)."""
    rows: list[dict] = []
    for system in _ordered_systems(df["system"].unique().tolist()):
        sub = df.loc[df["system"] == system, "abs_onset_min"].to_numpy(dtype=float)
        n = int(sub.size)
        if n == 0:
            rows.append(
                {
                    "system": system,
                    "n_events": 0,
                    "median_abs_onset_min": np.nan,
                    "q1_abs_onset_min": np.nan,
                    "q3_abs_onset_min": np.nan,
                    "iqr_abs_onset_min": np.nan,
                    "mean_abs_onset_min": np.nan,
                }
            )
            continue
        q1, med, q3 = np.percentile(sub, [25, 50, 75])
        rows.append(
            {
                "system": system,
                "n_events": n,
                "median_abs_onset_min": float(med),
                "q1_abs_onset_min": float(q1),
                "q3_abs_onset_min": float(q3),
                "iqr_abs_onset_min": _iqr(sub),
                "mean_abs_onset_min": float(np.mean(sub)),
            }
        )
    return pd.DataFrame(rows)


def plot_boxplot_by_method(df: pd.DataFrame, out_path: Path, dpi: int = 150) -> None:
    systems = _ordered_systems(df["system"].unique().tolist())
    data = [
        df.loc[df["system"] == s, "abs_onset_min"].dropna().to_numpy(dtype=float)
        for s in systems
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    positions = np.arange(1, len(systems) + 1)
    bp = ax.boxplot(
        data,
        positions=positions,
        showfliers=True,
        patch_artist=True,
        vert=False,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#cfe2ff")
        patch.set_edgecolor("#0b5fff")

    ylabels = [
        f"{METHOD_LABELS.get(s, s)}\n(n={arr.size})" for s, arr in zip(systems, data)
    ]
    ax.set_yticks(positions)
    ax.set_yticklabels(ylabels)
    ax.invert_yaxis()
    ax.set_xlabel("|onset difference| (minutes)")
    ax.set_ylabel("method")
    # ax.set_title("Absolute onset difference by method")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


FUSION_SIGNED_BINS = [-np.inf, -30, -15, -5, 0, 5, 15, 30, np.inf]
FUSION_SIGNED_BIN_LABELS = [
    "<-30",
    "-30:-15",
    "-15:-5",
    "-5:0",
    "0:5",
    "5:15",
    "15:30",
    ">30",
]
# Finite edges for matplotlib hist (data roughly within ±40 min)
FUSION_SIGNED_HIST_BINS = [-40, -30, -15, -5, 0, 5, 15, 30, 40]


def _fusion_signed_array(df: pd.DataFrame) -> np.ndarray:
    if "signed_onset_min" not in df.columns:
        raise ValueError("onset_errors data missing column signed_onset_min")
    sub = df.loc[df["system"] == "fusion", "signed_onset_min"]
    return pd.to_numeric(sub, errors="coerce").dropna().to_numpy(dtype=float)


def build_fusion_signed_metrics(signed: np.ndarray) -> pd.DataFrame:
    """
    Long-format metrics for fusion signed onset (detection - GT, minutes).
    Includes summary stats, sign counts, and histogram bin counts.
    """
    rows: list[dict[str, str | float | int]] = []

    def add(section: str, metric: str, value: float | int) -> None:
        rows.append({"section": section, "metric": metric, "value": value})

    n = int(signed.size)
    add("summary", "n_events", n)
    if n == 0:
        return pd.DataFrame(rows)

    q1, med, q3 = np.percentile(signed, [25, 50, 75])
    add("summary", "mean_signed_onset_min", float(np.mean(signed)))
    add("summary", "std_signed_onset_min", float(np.std(signed, ddof=1)) if n > 1 else 0.0)
    add("summary", "median_signed_onset_min", float(med))
    add("summary", "q1_signed_onset_min", float(q1))
    add("summary", "q3_signed_onset_min", float(q3))
    add("summary", "iqr_signed_onset_min", float(q3 - q1))
    add("summary", "min_signed_onset_min", float(np.min(signed)))
    add("summary", "max_signed_onset_min", float(np.max(signed)))
    for p in (5, 95):
        add("summary", f"p{p}_signed_onset_min", float(np.percentile(signed, p)))

    n_early = int((signed < 0).sum())
    n_on_time = int((signed == 0).sum())
    n_late = int((signed > 0).sum())
    add("sign", "n_early_detect_before_gt", n_early)
    add("sign", "n_on_time", n_on_time)
    add("sign", "n_late_detect_after_gt", n_late)
    add("sign", "pct_early", float(100.0 * n_early / n))
    add("sign", "pct_on_time", float(100.0 * n_on_time / n))
    add("sign", "pct_late", float(100.0 * n_late / n))

    cats = pd.cut(
        signed,
        bins=FUSION_SIGNED_BINS,
        labels=FUSION_SIGNED_BIN_LABELS,
        right=False,
    )
    for label in FUSION_SIGNED_BIN_LABELS:
        count = int((cats == label).sum())
        add("histogram_bin", str(label), count)

    return pd.DataFrame(rows)


def plot_fusion_signed_histogram(signed: np.ndarray, out_path: Path) -> None:
    """Histogram of fusion signed onset; negative = detect before GT."""
    fig, ax = plt.subplots(figsize=(9, 5))
    if signed.size == 0:
        ax.text(0.5, 0.5, "No fusion TP onset events", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.hist(
            signed,
            bins=FUSION_SIGNED_HIST_BINS,
            edgecolor="white",
            color="#0b5fff",
            alpha=0.85,
        )
        ax.axvline(0.0, color="#d62728", linestyle="--", linewidth=1.5, label="0 (aligned with GT)")
        ax.legend(loc="upper right")
    ax.set_xlabel("Signed onset difference (minutes)\ndetection − ground truth")
    ax.set_ylabel("Count")
    ax.set_title("Fusion: signed onset distribution")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate paper onset summaries and Figure 3.")
    parser.add_argument("--primary-dir", type=Path, default=PRIMARY_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--formats", nargs="+", choices=("png", "svg"), default=["png"])
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args(argv)
    primary_dir = args.primary_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve() if args.out_dir else primary_dir / "onset"
    out_dir.mkdir(parents=True, exist_ok=True)

    combined = load_onset_errors(primary_dir)
    summary = build_summary_by_method(combined)

    summary_path = out_dir / "onset_summary_by_method.csv"
    summary.to_csv(summary_path, index=False)

    plot_paths = []
    for fmt in dict.fromkeys(args.formats):
        plot_path = out_dir / f"abs_onset_difference_by_method.{fmt}"
        plot_boxplot_by_method(combined, plot_path, dpi=args.dpi)
        plot_paths.append(plot_path)

    fusion_signed = _fusion_signed_array(combined)
    fusion_metrics = build_fusion_signed_metrics(fusion_signed)
    fusion_metrics_path = out_dir / "fusion_signed_onset_metrics.csv"
    fusion_metrics.to_csv(fusion_metrics_path, index=False)

    fusion_hist_path = out_dir / "fusion_signed_onset_histogram.png"
    plot_fusion_signed_histogram(fusion_signed, fusion_hist_path)

    print(f"Read {combined['source_file'].nunique()} onset_errors files from {primary_dir}")
    print(f"Total TP onset events: {len(combined)}")
    print(f"Wrote {summary_path}")
    for plot_path in plot_paths:
        print(f"Wrote {plot_path}")
    print(f"Wrote {fusion_metrics_path}")
    print(f"Wrote {fusion_hist_path}")
    print("\nSummary by method:")
    print(summary.to_string(index=False))
    print("\nFusion signed onset metrics:")
    print(fusion_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
