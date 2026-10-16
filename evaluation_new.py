# Evaluate supplied prepared NHS reference events.
# Scheduled-meal merging is upstream dataset preparation, outside this repository.
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from pipelines.event_matching import match_event_times

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_ground_truth(
    patient_id: str,
    variant_label: str,
    reference_dir: str | Path = "results/evaluation/primary/ground_truth",
) -> tuple[list, pd.DataFrame]:
    """Read supplied prepared reference events without adding scheduled meals.

    Scheduled-meal merging is upstream NHS dataset preparation and is outside
    this repository's reproduction scope.
    """
    path = Path(reference_dir) / f"ground_truth_{patient_id}_{variant_label}.csv"
    frame = pd.read_csv(path)
    if "ts" not in frame.columns:
        raise ValueError(f"Prepared reference file must contain a 'ts' column: {path}")
    return frame["ts"].tolist(), frame









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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate supplied primary reference/detection files.")
    parser.add_argument("--input-dir", type=Path, default=Path("results/evaluation/primary"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/evaluation/primary"))
    parser.add_argument("--strict", action="store_true", help="Fail on missing cohort inputs.")
    args = parser.parse_args(argv)
    input_root = args.input_dir.expanduser().resolve()
    nhs_list = [
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
    variants = ["fiasp", "aspart"]
    primary = args.output_dir.expanduser().resolve()
    primary.mkdir(parents=True, exist_ok=True)

    def method_csv(system: str, patient_id: str, variant_label: str) -> Path:
        if system == "baseline":
            return input_root / "baseline" / f"simple_heuristic_{patient_id}_{variant_label}.csv"
        if system == "fusion":
            return input_root / "fusion" / f"fusion_events_{patient_id}_{variant_label}.csv"
        if system == "signal_based_only":
            return (
                input_root
                / "signal_based_only"
                / f"signal_based_{patient_id}_{variant_label}.csv"
            )
        if system == "physio_based_only":
            return (
                input_root
                / "physio_based_only"
                / f"physio_based_{patient_id}_{variant_label}.csv"
            )
        raise ValueError(system)

    system_order = [
        "baseline",
        "fusion",
        "signal_based_only",
        "physio_based_only",
    ]

    all_plot_rows: list[dict[str, Any]] = []
    participant_acc: dict[tuple[str, str], dict[str, Any]] = {}

    for patient_id in nhs_list:
        for variant_label in variants:
            if patient_id == "17" and variant_label == "aspart":
                continue

            gt_path = input_root / "ground_truth"/ f"ground_truth_{patient_id}_{variant_label}.csv"
            try:
                ground_truth = pd.read_csv(gt_path)
            except FileNotFoundError:
                if args.strict:
                    raise
                print(f"Skip (missing GT): {gt_path}")
                continue

            summary_rows: list[dict[str, Any]] = []
            onset_long_rows: list[dict[str, Any]] = []

            for system in system_order:
                cmp_path = method_csv(system, patient_id, variant_label)
                try:
                    method_df = pd.read_csv(cmp_path)
                except FileNotFoundError:
                    if args.strict:
                        raise
                    print(f"Skip {system}: {cmp_path}")
                    continue

                ev = evaluate(ground_truth, method_df, window_min=45, gt_time_col="ts")
                gt_ts = _to_ground_truth_times(ground_truth, "ts")
                det_ts = _to_detection_times(method_df, None)
                n_days = _n_days_for_fp_rate(gt_ts, det_ts, dataset="nhs")

                TP, FN, FP = ev["TP"], ev["FN"], ev["FP"]
                sens = float(TP / (TP + FN)) if (TP + FN) > 0 else float("nan")
                fp_per_day = float(FP / n_days) if n_days > 0 else float("nan")

                abs_list = ev["abs_onset_difference_min"]
                arr = np.asarray(abs_list, dtype=float)
                mean_abs = float(np.mean(arr)) if arr.size else float("nan")
                median_abs = float(np.median(arr)) if arr.size else float("nan")

                summary_rows.append(
                    {
                        "patient_id": patient_id,
                        "variant_label": variant_label,
                        "system": system,
                        "TP": TP,
                        "FN": FN,
                        "FP": FP,
                        "sensitivity": sens,
                        "fp_per_day": fp_per_day,
                        "n_days": n_days,
                        "mean_abs_onset_min": mean_abs,
                        "median_abs_onset_min": median_abs,
                    }
                )

                for signed, ab in zip(
                    ev["onset_difference_min"],
                    ev["abs_onset_difference_min"],
                ):
                    onset_long_rows.append(
                        {
                            "patient_id": patient_id,
                            "variant_label": variant_label,
                            "system": system,
                            "signed_onset_min": signed,
                            "abs_onset_min": ab,
                        }
                    )
                    all_plot_rows.append(
                        {
                            "patient_id": patient_id,
                            "variant_label": variant_label,
                            "system": system,
                            "abs_onset_min": ab,
                        }
                    )

                key = (patient_id, system)
                if key not in participant_acc:
                    participant_acc[key] = {
                        "TP": 0,
                        "FN": 0,
                        "FP": 0,
                        "n_days": 0.0,
                        "abs_list": [],
                    }
                bucket = participant_acc[key]
                bucket["TP"] += TP
                bucket["FN"] += FN
                bucket["FP"] += FP
                bucket["n_days"] += float(n_days)
                bucket["abs_list"].extend(abs_list)

            if not summary_rows:
                print(f"No method files for {patient_id}_{variant_label}; skip write.")
                continue

            out_sum = primary / f"evaluation_{patient_id}_{variant_label}.csv"
            pd.DataFrame(summary_rows).to_csv(out_sum, index=False)
            print(f"Wrote {out_sum}")

            out_long = primary / f"onset_errors_{patient_id}_{variant_label}.csv"
            long_cols = [
                "patient_id",
                "variant_label",
                "system",
                "signed_onset_min",
                "abs_onset_min",
            ]
            if onset_long_rows:
                pd.DataFrame(onset_long_rows).to_csv(out_long, index=False)
            else:
                pd.DataFrame(columns=long_cols).to_csv(out_long, index=False)
            print(f"Wrote {out_long}")

    # Participant-level metrics (pooled TP/FN/FP and summed n_days across variants)
    participant_rows: list[dict[str, Any]] = []
    for (patient_id, system), bucket in sorted(participant_acc.items()):
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
                "system": system,
                **metrics,
            }
        )

    if participant_rows:
        participant_df = pd.DataFrame(participant_rows)
        out_participant = primary / "evaluation_by_participant.csv"
        participant_df.to_csv(out_participant, index=False)
        print(f"Wrote {out_participant} ({len(participant_df)} rows)")

        cohort_df = _cohort_summary_by_system(participant_df)
        out_cohort = primary / "evaluation_summary_by_system.csv"
        cohort_df.to_csv(out_cohort, index=False)
        print(f"Wrote {out_cohort}")
        print("\nCohort summary by system (participant-level macro):")
        print(cohort_df.to_string(index=False))
    else:
        print("No participant-level metrics accumulated; skip participant/cohort CSVs.")

    # Aggregate plot: abs onset difference by method (all patients / variants)
    plot_df = pd.DataFrame(all_plot_rows)
    plot_path = primary / "abs_onset_difference_by_method.png"
    if plot_df.empty:
        print(f"No TP onset errors to plot; skip {plot_path}")
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        data = [
            plot_df.loc[plot_df["system"] == s, "abs_onset_min"].dropna().values
            for s in system_order
        ]
        positions = np.arange(1, len(system_order) + 1)
        ax.boxplot(data, positions=positions, showfliers=True)
        ax.set_xticks(positions)
        ax.set_xticklabels(system_order)
        ax.set_ylabel("|onset difference| (minutes)")
        ax.set_xlabel("method")
        ax.set_title("Absolute onset difference by method")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Wrote {plot_path}")


    # for patient_id in nhs_list:
    #     for variant_label in variants:
    #         if patient_id == "17" and variant_label == "aspart":
    #             continue
            # ============================= baseline results =============================
            # read the baseline results from /results/nhs/baseline/multi-meals/
            # ===============================================================================================
            # baseline_results = pd.read_csv(
            #     Path(f"results/nhs/baseline/multi-meals/simple_heuristic_{patient_id}_{variant_label}.csv")
            # )
            # baseline_results = baseline_results['tmeal']
            # baseline_results.to_csv(f"results/evaluation/primary/baseline/simple_heuristic_{patient_id}_{variant_label}.csv", index=False)
            
            # ============================= fusion results =============================
            # read the fusion results from /results/fusion_outputs/newest/
            # ===============================================================================================
            # fusion_results = pd.read_csv(
            #     Path(f"results/fusion_outputs/test/nhs_{patient_id}/fusion_events_nhs_{patient_id}_{variant_label}.csv")
            # )
            # fusion_results = fusion_results['onset_time']
            # fusion_results.to_csv(f"results/evaluation/primary/fusion/fusion_events_{patient_id}_{variant_label}.csv", index=False)

            # ============================= signal-based only component results =============================
            # read the signal-based only component results 
            # ===============================================================================================
            # signal_based_only_results = pd.read_csv(
            #     Path(f"results/fusion_outputs/signal_only/nhs_{patient_id}/fusion_events_nhs_{patient_id}_{variant_label}.csv")
            # )
            # signal_based_only_results = signal_based_only_results['onset_time']
            # signal_based_only_results.to_csv(f"results/evaluation/primary/signal_based_only/signal_based_{patient_id}_{variant_label}.csv", index=False)

            # ============================= physio-based only component results =============================
            # read the physio-based only component results 
            # ===============================================================================================
            # physio_based_only_results = pd.read_csv(
            #     Path(f"results/fusion_outputs/physio_only/nhs_{patient_id}/fusion_events_nhs_{patient_id}_{variant_label}.csv")
            # )
            # physio_based_only_results = physio_based_only_results['onset_time']
            # physio_based_only_results.to_csv(f"results/evaluation/primary/physio_based_only/physio_based_{patient_id}_{variant_label}.csv", index=False)

            # ============================= evaluate =============================
            # read the ground truth from /results/evaluation/primary/ground_truth_{patient_id}_{variant_label}.csv
            # read the baseline results from /results/evaluation/primary/baseline/simple_heuristic_{patient_id}_{variant_label}.csv
            # read the fusion results from /results/evaluation/primary/fusion/fusion_events_{patient_id}_{variant_label}.csv
            # evaluate the baseline results and fusion results
            # save the evaluation results to /results/evaluation/primary/evaluation_{patient_id}_{variant_label}.csv

if __name__ == "__main__":
    main()
