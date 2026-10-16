#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a 3-panel shared-x figure for NHS meal-detection results.

Panel (a): raw CGM, smoothed CGM, reference meals, bolus insulin
Panel (b): normalised evidence trajectories and fusion score
Panel (c): baseline vs fusion detections on swimlanes
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from pipelines.event_matching import match_event_times

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_FUSION_DIR = SCRIPT_DIR / "results" / "fusion_outputs" / "test"
DEFAULT_OUT_DIR = SCRIPT_DIR / "results" / "display"
MATCH_WINDOW_MIN = 45.0
BOLUS_STEM_SCALE = 0.25

COLOR_CGM = "#1f77b4"
COLOR_SMOOTH = "#888888"
COLOR_REFERENCE = "#ff7f0e"
COLOR_BOLUS = "#2ca02c"
COLOR_ESTAT = "#9ecae1"
COLOR_ETREND = "#ffd700" # yellow
COLOR_EPHYS = "#c7e9c0"
COLOR_FUSION_SCORE = "#8b0000"
COLOR_BASELINE = "#7b3294"
COLOR_FUSION_DET = "#d62728"


@dataclass
class DetectionClassification:
    tp: list[pd.Timestamp]
    fp: list[pd.Timestamp]
    fn: list[pd.Timestamp]


@dataclass
class PatientContext:
    patient_id: str
    variant_label: str
    merged: pd.DataFrame
    smooth_cgm: pd.DataFrame
    reference_meals: list[pd.Timestamp]
    scores: pd.DataFrame
    baseline_detections: pd.DataFrame
    fusion_detections: pd.DataFrame
    baseline_class: DetectionClassification
    fusion_class: DetectionClassification


def _ensure_cwd() -> None:
    os.chdir(SCRIPT_DIR)


def _tag(patient_id: str, variant_label: str) -> str:
    return f"nhs_{patient_id}_{variant_label}"


def classify_detections(
    ground_truth: list[pd.Timestamp],
    detections: list[pd.Timestamp],
    *,
    window_min: float = MATCH_WINDOW_MIN,
) -> DetectionClassification:
    gt_ts = sorted(pd.to_datetime(ground_truth))
    det_ts = sorted(pd.to_datetime(detections))
    matches = match_event_times(gt_ts, det_ts, window_min=window_min)
    used = {di for _, di in matches}
    matched_gt = {gi for gi, _ in matches}
    tp = [det_ts[i] for i in sorted(used)]
    fp = [det_ts[i] for i in range(len(det_ts)) if i not in used]
    fn = [gt_ts[i] for i in range(len(gt_ts)) if i not in matched_gt]
    return DetectionClassification(tp=tp, fp=fp, fn=fn)


def _require_file(path: Path, hint: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}\n{hint}")


def load_patient_context(
    patient_id: str,
    variant_label: str,
    fusion_dir: Path,
) -> PatientContext:
    from evaluation_new import read_ground_truth
    from fusion_based import process_nhs_data
    from pipelines.simple_heuristic_utils import filter_cgm_data

    tag = _tag(patient_id, variant_label)
    fusion_patient_dir = fusion_dir / f"nhs_{patient_id}"

    scores_path = fusion_patient_dir / f"fusion_scores_{tag}.csv"
    fusion_events_path = fusion_patient_dir / f"fusion_events_{tag}.csv"
    baseline_path = (
        SCRIPT_DIR
        / "results"
        / "nhs"
        / "baseline"
        / "multi-meals"
        / f"simple_heuristic_{patient_id}_{variant_label}.csv"
    )

    run_hint = (
        f"Run: python fusion_main.py --dataset nhs --patient_id {patient_id} "
        f"--variant_label {variant_label}"
    )
    _require_file(scores_path, run_hint)
    _require_file(fusion_events_path, run_hint)
    _require_file(baseline_path, f"Baseline results missing. Expected: {baseline_path}")

    merged = process_nhs_data(patient_id, variant_label=variant_label)
    if not isinstance(merged.index, pd.DatetimeIndex):
        merged = merged.set_index(pd.to_datetime(merged["ts"]))

    reference_meals, _ = read_ground_truth(patient_id, variant_label)
    reference_meals = [pd.Timestamp(t) for t in reference_meals]

    glu_df = merged.reset_index().rename(columns={"index": "ts"})
    if "ts" not in glu_df.columns:
        glu_df["ts"] = merged.index
    day_list = pd.to_datetime(glu_df["ts"]).dt.date.unique().tolist()
    smooth_cgm = filter_cgm_data(
        glu_data=glu_df,
        day_list=day_list,
        window_length=125,
        polyorder=3,
        plot=False,
    )

    scores = pd.read_csv(scores_path, parse_dates=["start", "end"])
    baseline_detections = pd.read_csv(baseline_path, parse_dates=["tmeal"])
    fusion_detections = pd.read_csv(fusion_events_path, parse_dates=["onset_time"])

    baseline_ts = pd.to_datetime(baseline_detections["tmeal"], errors="coerce").dropna().tolist()
    fusion_ts = pd.to_datetime(fusion_detections["onset_time"], errors="coerce").dropna().tolist()

    baseline_class = classify_detections(reference_meals, baseline_ts)
    fusion_class = classify_detections(reference_meals, fusion_ts)

    return PatientContext(
        patient_id=patient_id,
        variant_label=variant_label,
        merged=merged,
        smooth_cgm=smooth_cgm,
        reference_meals=reference_meals,
        scores=scores,
        baseline_detections=baseline_detections,
        fusion_detections=fusion_detections,
        baseline_class=baseline_class,
        fusion_class=fusion_class,
    )


def _draw_reference_lines(ax: plt.Axes, reference_meals: list[pd.Timestamp], *, zorder: int = 1) -> None:
    for i, t in enumerate(reference_meals):
        ax.axvline(
            t,
            color=COLOR_REFERENCE,
            linestyle="--",
            linewidth=1.2,
            alpha=0.85,
            zorder=zorder,
            label="Reference meal" if i == 0 else None,
        )


def _plot_panel_a(ax: plt.Axes, ctx: PatientContext) -> None:
    merged = ctx.merged.sort_index()
    ax.plot(merged.index, merged["value"], color=COLOR_CGM, linewidth=1.5, label="CGM", zorder=3)
    ax.plot(
        ctx.smooth_cgm["ts"],
        ctx.smooth_cgm["ycgm_F"],
        color=COLOR_SMOOTH,
        linewidth=1.2,
        label="Smoothed CGM",
        zorder=2,
    )

    _draw_reference_lines(ax, ctx.reference_meals, zorder=1)

    bolus = merged[merged["dose"].fillna(0) > 0]
    if not bolus.empty:
        y_min = float(merged["value"].min())
        display_heights = y_min + bolus["dose"] * BOLUS_STEM_SCALE
        markerline, stemlines, _baseline = ax.stem(
            bolus.index,
            display_heights,
            bottom=y_min,
            linefmt="-",
            markerfmt="^",
            basefmt=" ",
        )
        markerline.set_color(COLOR_BOLUS)
        markerline.set_markersize(8)
        stemlines.set_color(COLOR_BOLUS)
        stemlines.set_linewidth(1.0)
        markerline.set_label("Bolus insulin")

        # ax.scatter(
        #     bolus.index,
        #     bolus["value"],          # 画在 CGM 曲线上；也可改成固定 y
        #     marker="^",
        #     s=200,                   # 越大 star 越大（scatter 用面积，不是 markersize）
        #     color=COLOR_BOLUS,
        #     zorder=5,
        #     label="Bolus insulin",
        # )


    ax.set_ylabel("Glucose (mmol/L)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.text(0.01, 0.97, "(a)", transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")


def _plot_panel_b(ax: plt.Axes, ctx: PatientContext) -> None:
    scores = ctx.scores.sort_values("end")
    t = scores["end"]
    ephys = scores["bergman_ra_post_iob_rise_len_min_normalized"].fillna(0.0)

    ax.plot(t, scores["pos_grad_energy_normalized"], color=COLOR_ESTAT, linewidth=1.5, alpha=0.9, label=r"$E_{stat}(t)$")
    ax.plot(t, scores["trend_auc_medtronic_pos_normalized"], color=COLOR_ETREND, linewidth=1.5, alpha=0.9, label=r"$E_{trend}(t)$")
    ax.plot(t, ephys, color=COLOR_EPHYS, linewidth=1.5, alpha=0.9, label=r"$E_{phys}(t)$")
    ax.plot(t, scores["score_6"], color=COLOR_FUSION_SCORE, linewidth=2.5, label=r"$S(t)$")

    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Normalised evidence")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.text(0.01, 0.97, "(b)", transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")


def _plot_lane_detections(
    ax: plt.Axes,
    times: list[pd.Timestamp],
    y: float,
    *,
    marker: str,
    color: str,
    label: str,
) -> None:
    if not times:
        return
    ax.scatter(
        times,
        [y] * len(times),
        marker=marker,
        s=100,
        color=color,
        edgecolors="white",
        linewidths=0.6,
        zorder=4,
        label=label,
    )


def _plot_panel_c(ax: plt.Axes, ctx: PatientContext) -> None:
    y_baseline = 0.0
    y_fusion = 1.0

    _draw_reference_lines(ax, ctx.reference_meals, zorder=1)

    baseline_ts = pd.to_datetime(ctx.baseline_detections["tmeal"], errors="coerce").dropna().tolist()
    fusion_ts = pd.to_datetime(ctx.fusion_detections["onset_time"], errors="coerce").dropna().tolist()

    _plot_lane_detections(
        ax,
        baseline_ts,
        y_baseline,
        marker="o",
        color=COLOR_BASELINE,
        label="Baseline",
    )
    _plot_lane_detections(
        ax,
        fusion_ts,
        y_fusion,
        marker="v",
        color=COLOR_FUSION_DET,
        label="Fusion",
    )

    ax.axhline(y_baseline, color="#dddddd", linewidth=0.8, zorder=0)
    ax.axhline(y_fusion, color="#dddddd", linewidth=0.8, zorder=0)

    ax.set_yticks([y_baseline, y_fusion])
    ax.set_yticklabels(["Baseline", "Fusion"])
    ax.set_ylim(-0.2, 1.2)
    handles, labels = ax.get_legend_handles_labels()
    preferred_order = ["Reference meal", "Fusion", "Baseline"]
    ordered = [
        (handles[labels.index(label)], label)
        for label in preferred_order
        if label in labels
    ]
    ax.legend(
        [handle for handle, _label in ordered],
        [label for _handle, label in ordered],
        loc="upper right",
        fontsize=8,
    )
    ax.grid(True, axis="x", alpha=0.25)
    ax.text(0.01, 0.97, "(c)", transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")


def build_three_panel_figure(ctx: PatientContext) -> plt.Figure:
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1.6, 0.7], "hspace": 0.08},
    )

    _plot_panel_a(axes[0], ctx)
    _plot_panel_b(axes[1], ctx)
    _plot_panel_c(axes[2], ctx)

    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    axes[2].tick_params(axis="x", rotation=30)
    axes[2].set_xlabel("Time")

    # fig.suptitle(f"NHS patient {ctx.patient_id} ({ctx.variant_label})", fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def output_path(out_dir: Path, patient_id: str, variant_label: str, fmt: str) -> Path:
    return out_dir / f"display_nhs_{patient_id}_{variant_label}.{fmt}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 3-panel NHS results display figure.")
    parser.add_argument("--patient_id", type=str, required=True, help="NHS patient ID, e.g. 01")
    parser.add_argument("--dataset", type=str, default="nhs", choices=["nhs"], help="Dataset (NHS only for now)")
    parser.add_argument("--variant_label", type=str, default="aspart", help="NHS insulin variant")
    parser.add_argument("--fusion_dir", type=Path, default=DEFAULT_FUSION_DIR, help="Fusion outputs root")
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory")
    parser.add_argument("--dpi", type=int, default=150, help="Figure DPI")
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf"], help="Output format")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    if args.dataset != "nhs":
        raise ValueError("Only --dataset nhs is supported in this version.")

    _ensure_cwd()
    ctx = load_patient_context(args.patient_id, args.variant_label, args.fusion_dir)

    fig = build_three_panel_figure(ctx)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = output_path(args.out_dir, args.patient_id, args.variant_label, args.format)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out}")
    print(
        f"Baseline  TP={len(ctx.baseline_class.tp)} FP={len(ctx.baseline_class.fp)} FN={len(ctx.baseline_class.fn)}"
    )
    print(
        f"Fusion    TP={len(ctx.fusion_class.tp)} FP={len(ctx.fusion_class.fp)} FN={len(ctx.fusion_class.fn)}"
    )
    return out


if __name__ == "__main__":
    main()
