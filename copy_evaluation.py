# Evaluate supplied prepared NHS reference events.
# Scheduled-meal merging is upstream dataset preparation, outside this repository.
from __future__ import annotations

from pathlib import Path
from typing import Any

from pipelines.event_matching import match_event_times

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd








def _to_ground_truth_times(
    ground_truth: pd.DataFrame | pd.Series | list,
    gt_time_col: str = "ts",
) -> list[pd.Timestamp]:
    if isinstance(ground_truth, pd.DataFrame):
        if gt_time_col in ground_truth.columns:
            s = pd.to_datetime(ground_truth[gt_time_col], errors="coerce")
        else:
            s = pd.to_datetime(ground_truth.iloc[:, 0], errors="coerce")
    elif isinstance(ground_truth, pd.Series):
        s = pd.to_datetime(ground_truth, errors="coerce")
    else:
        s = pd.to_datetime(pd.Series(list(ground_truth)), errors="coerce")
    s = s.dropna().sort_values()
    return [pd.Timestamp(t) for t in s.tolist()]


def _to_detection_times(
    method_results: pd.DataFrame | pd.Series | list,
    det_time_col: str | None = None,
) -> list[pd.Timestamp]:
    if isinstance(method_results, pd.DataFrame):
        if det_time_col and det_time_col in method_results.columns:
            s = pd.to_datetime(method_results[det_time_col], errors="coerce")
        else:
            for c in ("onset_time", "tmeal", "ts", "time"):
                if c in method_results.columns:
                    s = pd.to_datetime(method_results[c], errors="coerce")
                    break
            else:
                s = pd.to_datetime(method_results.iloc[:, 0], errors="coerce")
    elif isinstance(method_results, pd.Series):
        s = pd.to_datetime(method_results, errors="coerce")
    else:
        s = pd.to_datetime(pd.Series(list(method_results)), errors="coerce")
    s = s.dropna().sort_values()
    return [pd.Timestamp(t) for t in s.tolist()]


def _n_calendar_days(gt_ts: list[pd.Timestamp], det_ts: list[pd.Timestamp]) -> float:
    all_t = gt_ts + det_ts
    if not all_t:
        return 1.0
    dates = {pd.Timestamp(t).normalize() for t in all_t}
    return max(1.0, float(len(dates)))


# NHS three-meals trials: ~24 h monitoring per patient-variant (may span 2 calendar dates).
NHS_MONITORING_DAYS_PER_VARIANT = 1.0


def _n_days_for_fp_rate(
    gt_ts: list[pd.Timestamp],
    det_ts: list[pd.Timestamp],
    *,
    dataset: str = "nhs",
) -> float:
    """Days denominator for FP/day. NHS uses one monitoring day per variant, not calendar-day count."""
    if dataset == "nhs":
        return NHS_MONITORING_DAYS_PER_VARIANT
    return _n_calendar_days(gt_ts, det_ts)


def evaluate(
    ground_truth: pd.DataFrame | pd.Series | list,
    method_results: pd.DataFrame | pd.Series | list,
    *,
    window_min: float = 45,
    gt_time_col: str = "ts",
    det_time_col: str | None = None,
) -> dict[str, Any]:
    """Match candidate pairs in ascending absolute difference within ±window_min.

    Each reference and detection participates in at most one accepted pair.
    Signed onset difference is detection minus reference, in minutes.
    """
    gt_ts = _to_ground_truth_times(ground_truth, gt_time_col)
    det_ts = _to_detection_times(method_results, det_time_col)
    matches = match_event_times(gt_ts, det_ts, window_min=window_min)
    onset_difference_min = [
        float((det_ts[di] - gt_ts[gi]).total_seconds() / 60)
        for gi, di in matches
    ]
    TP = len(matches)
    return {
        "TP": TP,
        "FN": len(gt_ts) - TP,
        "FP": len(det_ts) - TP,
        "onset_difference_min": onset_difference_min,
        "abs_onset_difference_min": [abs(x) for x in onset_difference_min],
    }


NHS_LIST = [
    "01",
    "02",
    "03",
    "04",
    "06",
    "07",
    "08",
    "09",
    "10",
    "11",
    "13",
    "14",
    "15",
    "16",
    "17",
    "18",
]
VARIANTS = ["fiasp", "aspart"]
SCENARIOS = ("_weights/0.5phys","_weights/0.3phys", "_weights/0.34phys")
SENSITIVITY_ANALYSIS_ROOT = Path("results/evaluation/primary/sensitivity_analysis")


def fusion_events_path(scenario: str, patient_id: str, variant_label: str) -> Path:
    return Path(
        f"results/fusion_outputs/{scenario}/nhs_{patient_id}/"
        f"fusion_events_nhs_{patient_id}_{variant_label}.csv"
    )


def gt_path(patient_id: str, variant_label: str) -> Path:
    return Path(
        f"results/evaluation/primary/ground_truth/"
        f"ground_truth_{patient_id}_{variant_label}.csv"
    )


def scenario_out_dir(scenario: str) -> Path:
    return SENSITIVITY_ANALYSIS_ROOT / scenario


def _participant_metrics_from_sums(
    TP: int,
    FN: int,
    FP: int,
    n_days: float,
    abs_list: list[float],
) -> dict[str, Any]:
    """Participant-level metrics from pooled variant counts (n_days summed across variants)."""
    denom = TP + FN
    sensitivity = float(TP / denom) if denom > 0 else float("nan")
    fp_per_day = float(FP / n_days) if n_days > 0 else float("nan")
    arr = np.asarray(abs_list, dtype=float)
    mean_abs = float(np.mean(arr)) if arr.size else float("nan")
    median_abs = float(np.median(arr)) if arr.size else float("nan")
    return {
        "TP": int(TP),
        "FN": int(FN),
        "FP": int(FP),
        "n_days": float(n_days),
        "sensitivity": sensitivity,
        "fp_per_day": fp_per_day,
        "mean_abs_onset_min": mean_abs,
        "median_abs_onset_min": median_abs,
    }


def _cohort_summary_by_system(participant_df: pd.DataFrame) -> pd.DataFrame:
    """Macro mean ± std of participant-level sensitivity and fp_per_day per system."""
    rows: list[dict[str, Any]] = []
    for system, grp in participant_df.groupby("system", sort=False):
        sens = grp["sensitivity"].dropna()
        fpd = grp["fp_per_day"].dropna()
        rows.append(
            {
                "system": system,
                "sensitivity_mean": float(sens.mean()) if len(sens) else float("nan"),
                "sensitivity_std": float(sens.std(ddof=1)) if len(sens) > 1 else float("nan"),
                "fp_per_day_mean": float(fpd.mean()) if len(fpd) else float("nan"),
                "fp_per_day_std": float(fpd.std(ddof=1)) if len(fpd) > 1 else float("nan"),
                "n_participants": int(len(grp)),
            }
        )
    return pd.DataFrame(rows)


def run_scenario_evaluation(scenario: str) -> list[dict[str, Any]]:
    """
    Evaluate fusion events under results/fusion_outputs/{scenario}/ against primary GT.
    Writes only under results/evaluation/primary/sensitivity_analysis/{scenario}/.
    """
    out = scenario_out_dir(scenario)
    out.mkdir(parents=True, exist_ok=True)

    plot_rows: list[dict[str, Any]] = []
    participant_acc: dict[str, dict[str, Any]] = {}

    for patient_id in NHS_LIST:
        for variant_label in VARIANTS:
            if patient_id == "17" and variant_label == "aspart":
                continue

            gpath = gt_path(patient_id, variant_label)
            try:
                ground_truth = pd.read_csv(gpath)
            except FileNotFoundError:
                print(f"[{scenario}] Skip (missing GT): {gpath}")
                continue

            events_path = fusion_events_path(scenario, patient_id, variant_label)
            try:
                events_df = pd.read_csv(events_path)
            except FileNotFoundError:
                print(f"[{scenario}] Skip (missing events): {events_path}")
                continue

            ev = evaluate(ground_truth, events_df, window_min=45, gt_time_col="ts")
            gt_ts = _to_ground_truth_times(ground_truth, "ts")
            det_ts = _to_detection_times(events_df, None)
            n_days = _n_days_for_fp_rate(gt_ts, det_ts, dataset="nhs")

            TP, FN, FP = ev["TP"], ev["FN"], ev["FP"]
            sens = float(TP / (TP + FN)) if (TP + FN) > 0 else float("nan")
            fp_per_day = float(FP / n_days) if n_days > 0 else float("nan")
            abs_list = ev["abs_onset_difference_min"]
            arr = np.asarray(abs_list, dtype=float)
            mean_abs = float(np.mean(arr)) if arr.size else float("nan")
            median_abs = float(np.median(arr)) if arr.size else float("nan")

            variant_row = {
                "patient_id": patient_id,
                "variant_label": variant_label,
                "system": scenario,
                "TP": TP,
                "FN": FN,
                "FP": FP,
                "sensitivity": sens,
                "fp_per_day": fp_per_day,
                "n_days": n_days,
                "mean_abs_onset_min": mean_abs,
                "median_abs_onset_min": median_abs,
            }
            pd.DataFrame([variant_row]).to_csv(
                out / f"evaluation_{patient_id}_{variant_label}.csv",
                index=False,
            )

            onset_long_rows: list[dict[str, Any]] = []
            for signed, ab in zip(
                ev["onset_difference_min"],
                ev["abs_onset_difference_min"],
            ):
                onset_long_rows.append(
                    {
                        "patient_id": patient_id,
                        "variant_label": variant_label,
                        "system": scenario,
                        "signed_onset_min": signed,
                        "abs_onset_min": ab,
                    }
                )
                plot_rows.append(
                    {
                        "scenario": scenario,
                        "patient_id": patient_id,
                        "variant_label": variant_label,
                        "system": scenario,
                        "abs_onset_min": ab,
                    }
                )

            long_cols = [
                "patient_id",
                "variant_label",
                "system",
                "signed_onset_min",
                "abs_onset_min",
            ]
            out_long = out / f"onset_errors_{patient_id}_{variant_label}.csv"
            if onset_long_rows:
                pd.DataFrame(onset_long_rows).to_csv(out_long, index=False)
            else:
                pd.DataFrame(columns=long_cols).to_csv(out_long, index=False)

            if patient_id not in participant_acc:
                participant_acc[patient_id] = {
                    "TP": 0,
                    "FN": 0,
                    "FP": 0,
                    "n_days": 0.0,
                    "abs_list": [],
                }
            bucket = participant_acc[patient_id]
            bucket["TP"] += TP
            bucket["FN"] += FN
            bucket["FP"] += FP
            bucket["n_days"] += float(n_days)
            bucket["abs_list"].extend(abs_list)

    participant_rows: list[dict[str, Any]] = []
    for patient_id, bucket in sorted(participant_acc.items()):
        metrics = _participant_metrics_from_sums(
            bucket["TP"],
            bucket["FN"],
            bucket["FP"],
            bucket["n_days"],
            bucket["abs_list"],
        )
        participant_rows.append(
            {
                "patient_id": patient_id,
                "system": scenario,
                **metrics,
            }
        )

    if participant_rows:
        participant_df = pd.DataFrame(participant_rows)
        out_participant = out / "evaluation_by_participant.csv"
        participant_df.to_csv(out_participant, index=False)
        print(f"[{scenario}] Wrote {out_participant} ({len(participant_df)} rows)")

        cohort_df = _cohort_summary_by_system(participant_df)
        out_cohort = out / "evaluation_summary_by_system.csv"
        cohort_df.to_csv(out_cohort, index=False)
        print(f"[{scenario}] Wrote {out_cohort}")
        print(f"[{scenario}] Cohort summary:\n{cohort_df.to_string(index=False)}")
    else:
        print(f"[{scenario}] No participant-level metrics; skip participant/cohort CSVs.")

    return plot_rows


def main() -> None:
    SENSITIVITY_ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    all_plot_rows: list[dict[str, Any]] = []

    for scenario in SCENARIOS:
        all_plot_rows.extend(run_scenario_evaluation(scenario))

    plot_path = SENSITIVITY_ANALYSIS_ROOT / "abs_onset_difference_dexcom_vs_hovorka.png"
    plot_df = pd.DataFrame(all_plot_rows)
    if plot_df.empty:
        print(f"No TP onset errors to plot; skip {plot_path}")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    data = [
        plot_df.loc[plot_df["scenario"] == s, "abs_onset_min"].dropna().values
        for s in SCENARIOS
    ]
    positions = np.arange(1, len(SCENARIOS) + 1)
    ax.boxplot(data, positions=positions, showfliers=True)
    ax.set_xticks(positions)
    ax.set_xticklabels(list(SCENARIOS))
    ax.set_ylabel("|onset error| (minutes)")
    ax.set_xlabel("score configuration")
    ax.set_title("Absolute onset difference: dexcom vs hovorka (all NHS evaluations)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {plot_path}")

if __name__ == "__main__":
    main()
