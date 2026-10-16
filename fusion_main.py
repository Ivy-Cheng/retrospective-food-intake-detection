#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-click runner for the fusion-based pipeline.

Usage:
  python fusion_main.py --dataset uom --patient_id 2307
  python fusion_main.py --dataset nhs --patient_id 01 --variant_label fiasp

Requirements:
  - Ensure fusion_based.ipynb has been exported to fusion_based.py in the same folder,
    so we can import functions like process_uom_data / process_nhs_data /
    build_ra_features / compute_components_only that you assembled in the notebook.
    You can export via:
      jupyter nbconvert --to=python fusion-based.ipynb
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import pandas as pd
import importlib.util
import io
import json

from pipelines.prominence_threshold import fit_dataset_prominence_threshold, load_candidate_manifest, read_dataset_threshold

uom_list = ['2301', '2302', '2304', '2305', '2306', '2307', '2308', '2309', '2313', '2314', '2320', '2401', '2403', '2404', '2405']
nhs_list = ['01', '02', '03', '04', '06', '07', '08', '09', '10', '11', '13', '14', '15', '16', '17', '18']

# 本脚本依赖在 notebook 中已实现的核心函数
# 需要将 fusion-based.ipynb 导出为 fusion_based.py
from fusion_based import (                    # type: ignore
    process_uom_data,
    process_nhs_data,
    build_ra_features,
    compute_components_only,
    pos_grad_energy,
    trend_auc_medtronic_pos,
    trend_auc_dexcom_pos,
    attach_physio_kf_series
)

from pipelines.fusion_refinement import (
    detect_meals_from_score, 
    detect_meals_start_guard, 
    smooth_score_minimal, 
    detect_meals_mean_shift,
    detect_meals_mean_shift_or_rise,
    detect_meals_simple_peaks_then_refine,
    detect_meals_peakvalley_refine
)
from pipelines.fusion_refinement_new import (
    detect_meals_adaptive_region_refine,
    detect_meals_adaptive_rise_based,
    detect_meals_slope_prominence,
    export_slope_local_maxima_csv,
)
# from glucose_visualization import create_interactive_trend_figure, create_basic_interactive_trend_figure

# from glucose_visualization_v2 import create_wider_figure, create_interactive_trend_figure, create_basic_interactive_trend_figure
from pipelines.fusion_scoring import (
    align_scores_to_plot,
    build_uom_l2_merged_and_scores,
)
from pipelines.uom_study_windows import (
    filter_events_to_days,
    mask_by_days,
    save_pool_days_cache,
)

from visualization_final import create_wider_figure, create_interactive_trend_figure, write_html_with_prev_next

def run_pipeline(
    dataset: str,
    patient_id: str,
    variant_label: str | None = None,
    window_length_min: int = 90,
    step_min: int = 1,
    detector: str = "peakvalley",
    out_dir: str = "results/fusion_outputs/all",
    pool_last_n_days: int = 84,
    skip_l3_filter: bool = False,
    weighting: str = "default",
    uom_block_mode: str = "daily_independent",
    dataset_threshold: float | None = None,
    scores_only: bool = False,
    cached_score_dir: str | Path | None = None,
    plot_results: bool = True,
    dataset_manifest: str | Path | None = None,
    merged_input: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """
    High-level runner:
      1) 读取并构建 merged
      2) 计算生理 RA 特征（bergman/hovorka）
      3) 计算 rolling window 组件（pos_grad_energy / trend AUC / 生理组件）
      4) 基于组件做 robust z+sigmoid（在 compute_components_only 里），并输出时序表
      5) 对 score 做极值法检测餐点（可选 90min 起点截断）
      6) 保存 csv + 绘图（可选）
    """
    dataset = dataset.lower()

    pool_days = None
    sampled_days = None
    gap_ranges = None

    if cached_score_dir is None:
       # 1) 读取 merged
       if dataset == "uom":
           merged = process_uom_data(patient_id)
           merged_ra, fusion_results, pool_days, sampled_days, gap_ranges = build_uom_l2_merged_and_scores(
               patient_id,
               merged,
               pool_last_n_days=pool_last_n_days,
               window_length_min=window_length_min,
               step_min=step_min,
               block_mode=uom_block_mode,
           )
           save_pool_days_cache(patient_id, pool_days)
           print(
               f"[UoM L2/L3] patient={patient_id} pool_days={len(pool_days)} "
               f"sampled_days={len(sampled_days)} gap_blocks={len(gap_ranges or [])}"
           )
           df_plot = align_scores_to_plot(merged_ra, fusion_results)
       elif dataset == "nhs":
           if not variant_label:
               raise ValueError("NHS 数据集需要 --variant_label，例如 --variant_label fiasp")
           merged = merged_input.copy() if merged_input is not None else process_nhs_data(patient_id, variant_label=variant_label, dataset_manifest=dataset_manifest)
       else:
           raise ValueError(f"未知数据集: {dataset}. 仅支持 'uom' 或 'nhs'")

       if dataset == "nhs":
           # 2) 生理模型序列（bergman/hovorka）+ 评分列
           from pipelines.gap_masking import prepared_gap_intervals, split_processing_blocks
           from pipelines.fusion_scoring import attach_physio_kf_per_ranges
           masks = prepared_gap_intervals(merged)
           if masks:
               gap_ranges = split_processing_blocks([(merged.index.min(), merged.index.max())], masks)
               merged_ra = attach_physio_kf_per_ranges(merged, gap_ranges)
           else:
               merged_ra, _ = attach_physio_kf_series(
                   merged, models=("bergman", "hovorka"), reindex_method="nearest"
               )

           # 3) 只计算组件（不在此阶段做融合总分）
           comp_df = compute_components_only(
               merged_ra,
               window_length_min=window_length_min,
               step_min=step_min,
               component_fns={
                   "pos_grad_energy": lambda w: pos_grad_energy(w, "value"),
                   "trend_auc_medtronic_pos": trend_auc_medtronic_pos,
                   "trend_auc_dexcom_pos": trend_auc_dexcom_pos,
               },
               include_prob=True,
               sigmoid_k=1.0,
               contiguous_blocks=gap_ranges,
           )

           # 4) 生成与 notebook 相同的归一化与多种 score（score_1 ~ score_6）
           def _safe_minmax(s: pd.Series) -> pd.Series:
               s = pd.to_numeric(s, errors="coerce")
               if not s.notna().any():
                   return pd.Series(float("nan"), index=s.index)
               vmin, vmax = s.min(), s.max()
               denom = (vmax - vmin)
               if pd.isna(denom) or denom == 0:
                   return pd.Series(0.0, index=s.index).where(s.notna()) if gap_ranges is not None else pd.Series(0.0, index=s.index)
               return (s - vmin) / denom

           fusion_results = comp_df.copy()
           # 保留 end 作为锚点
           if "end" not in fusion_results.columns:
               raise RuntimeError("components dataframe must contain 'end' column")
           # 归一化（与 notebook 保持一致的列名）
           if "pos_grad_energy" in fusion_results:
               fusion_results["pos_grad_energy_normalized"] = _safe_minmax(fusion_results["pos_grad_energy"])
           if "trend_auc_medtronic_pos" in fusion_results:
               fusion_results["trend_auc_medtronic_pos_normalized"] = _safe_minmax(fusion_results["trend_auc_medtronic_pos"])
           if "trend_auc_dexcom_pos" in fusion_results:
               fusion_results["trend_auc_dexcom_pos_normalized"] = _safe_minmax(fusion_results["trend_auc_dexcom_pos"])
           if "bergman_ra_post_iob_rise_len_min" in fusion_results:
               fusion_results["bergman_ra_post_iob_rise_len_min_normalized"] = _safe_minmax(
                   fusion_results["bergman_ra_post_iob_rise_len_min"]
               )
           if "bergman_ra_post_iob_rise_auc" in fusion_results:
               fusion_results["bergman_ra_post_iob_rise_auc_normalized"] = _safe_minmax(
                   fusion_results["bergman_ra_post_iob_rise_auc"]
               )
           if "hovorka_ra_post_iob_rise_len_min" in fusion_results:
               fusion_results["hovorka_ra_post_iob_rise_len_min_normalized"] = _safe_minmax(
                   fusion_results["hovorka_ra_post_iob_rise_len_min"]
               )
           if "hovorka_ra_post_iob_rise_auc" in fusion_results:
               fusion_results["hovorka_ra_post_iob_rise_auc_normalized"] = _safe_minmax(
                   fusion_results["hovorka_ra_post_iob_rise_auc"]
               )

           need_cols = {
               "pos_grad_energy_normalized",
               "trend_auc_medtronic_pos_normalized",
               "trend_auc_dexcom_pos_normalized",
               "bergman_ra_post_iob_rise_len_min_normalized",
               "bergman_ra_post_iob_rise_auc_normalized",
               "hovorka_ra_post_iob_rise_len_min_normalized",
               "hovorka_ra_post_iob_rise_auc_normalized",
           }
           if need_cols.issubset(set(fusion_results.columns)):
               mask = fusion_results["bergman_ra_post_iob_rise_auc_normalized"].notna()
               w1 = mask.map({True: 0.3, False: 0.5}).to_numpy()
               w2 = mask.map({True: 0.3, False: 0.5}).to_numpy()
               w3 = mask.map({True: 0.4, False: 0.0}).to_numpy()
               fusion_results["score_6"] = (
                   fusion_results["pos_grad_energy_normalized"].to_numpy() * w1
                   + fusion_results["trend_auc_medtronic_pos_normalized"].to_numpy() * w2
                   + fusion_results["bergman_ra_post_iob_rise_len_min_normalized"].fillna(0.0).to_numpy() * w3
               )

           df_plot = merged_ra.copy()
           df_plot = df_plot.sort_index()
           score_series = comp_df.set_index("end")["pos_grad_energy_p"]
           df_plot["fusion_score"] = score_series.reindex(df_plot.index)
           fusion_results = fusion_results.copy()
           fusion_results["end"] = pd.to_datetime(fusion_results["end"], errors="coerce")
           for k in range(1, 7):
               col = f"score_{k}"
               if col in fusion_results:
                   s = fusion_results.set_index("end")[col]
                   df_plot[f"fusion_{col}"] = s.reindex(df_plot.index)

       # 5) 先把窗口“结束时刻”作为 score 时间轴锚点，并对齐到 merged 索引（UoM 已在 align_scores_to_plot 完成）
       if dataset == "nhs":
           pass  # df_plot / fusion_results already built above
       else:
           fusion_results = fusion_results.copy()
           fusion_results["end"] = pd.to_datetime(fusion_results["end"], errors="coerce")

    else:
        cached_path = Path(cached_score_dir)
        cached_tag = f"{dataset}_{patient_id}" if dataset != "nhs" else f"nhs_{patient_id}_{variant_label}"
        merged_ra = pd.read_csv(cached_path / f"overall_{cached_tag}.csv", parse_dates=["ts"], float_precision="round_trip").set_index("ts")
        fusion_results = pd.read_csv(cached_path / f"fusion_scores_{cached_tag}.csv", parse_dates=["start", "end"], float_precision="round_trip")
        context = json.loads((cached_path / f"score_context_{cached_tag}.json").read_text())
        pool_days = {pd.Timestamp(d).date() for d in context["pool_days"]} if context["pool_days"] is not None else None
        sampled_days = {pd.Timestamp(d).date() for d in context["sampled_days"]} if context["sampled_days"] is not None else None
        gap_ranges = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in context["gap_ranges"]] if context["gap_ranges"] is not None else None
        from pipelines.gap_masking import prepared_gap_intervals
        if prepared_gap_intervals(merged_ra) and context.get("gap_policy") != "exclude_prepared_gaps_over_30_minutes":
            raise ValueError("Cached scores contain gaps without verified masking; regenerate the cache.")
        df_plot = align_scores_to_plot(merged_ra, fusion_results)

    if weighting not in {"default", "equal"}:
        raise ValueError("weighting must be 'default' or 'equal'")
    if weighting == "equal":
        from pipelines.equal_weighting import equal_weight_score
        fusion_results["score_6"] = equal_weight_score(fusion_results)
        score_series = fusion_results.set_index("end")["score_6"]
        df_plot["fusion_score_6"] = score_series.reindex(df_plot.index)

    if scores_only:
        score_path = Path(out_dir)
        score_path.mkdir(parents=True, exist_ok=True)
        score_tag = f"{dataset}_{patient_id}" if dataset != "nhs" else f"nhs_{patient_id}_{variant_label}"
        merged_csv = score_path / f"overall_{score_tag}.csv"
        scores_csv = score_path / f"fusion_scores_{score_tag}.csv"
        trajectory_csv = score_path / f"score_trajectory_{score_tag}.csv"
        context_csv = score_path / f"score_context_{score_tag}.json"
        merged_ra.to_csv(merged_csv, index=True)
        fusion_results.to_csv(scores_csv, index=False)
        detect_col = "fusion_score_6" if "fusion_score_6" in df_plot else "fusion_score"
        pd.DataFrame({"ts": df_plot.index, "score": df_plot[detect_col].to_numpy()}).to_csv(trajectory_csv, index=False)
        context = {
            "pool_days": sorted(str(d) for d in pool_days) if pool_days is not None else None,
            "sampled_days": sorted(str(d) for d in sampled_days) if sampled_days is not None else None,
            "gap_ranges": [[str(a), str(b)] for a, b in gap_ranges] if gap_ranges is not None else None,
            "gap_policy": "exclude_prepared_gaps_over_30_minutes",
        }
        context_csv.write_text(json.dumps(context, indent=2)+"\n")
        return {"scores_csv": scores_csv, "merged_csv": merged_csv, "trajectory_csv": trajectory_csv, "context_json": context_csv}

    detect_col = "fusion_score_6" if "fusion_score_6" in df_plot else "fusion_score"
    df_evt = pd.DataFrame({"ts": df_plot.index, detect_col: df_plot[detect_col].to_numpy()})

    # 6) 事件检测（默认用 fusion_score_6，如不存在则回退到 fusion_score）
    detect_col = "fusion_score_6" if "fusion_score_6" in df_plot.columns else "fusion_score"
    df_evt = (
        df_plot.reset_index()
        .rename(columns={"index": "ts"})
        .loc[:, ["ts", detect_col]]
    )
    detector = str(detector).lower()
    if detector == "adaptive_region":
        detects = detect_meals_adaptive_region_refine(
            df=df_evt,
            time_col="ts",
            score_col=detect_col,
            window_len_min=window_length_min,
            merge_within_minutes=60,
        )
    elif detector == "adaptive_rise":
        detects = detect_meals_adaptive_rise_based(
            df=df_evt,
            time_col="ts",
            score_col=detect_col,
            merge_within_minutes=60,
        )
    elif detector == "slope_prominence":
        # theta_slope = 0.003 if dataset == "nhs" else 0.0016
        theta_slope = 0
        if dataset_threshold is None:
            raise ValueError("Fit one dataset-wide Q75 before detection; use the CLI two-pass workflow or supply dataset_threshold.")
        theta_prom_fixed = float(dataset_threshold)
        prominence_mode = "fixed"
        tag_q = f"{dataset}_{patient_id}" if dataset != "nhs" else f"nhs_{patient_id}_{variant_label}"
        q_csv = Path(out_dir) / f"plot_{tag_q}_candidate_prominence_slope_quantiles.csv"
        q_csv.parent.mkdir(parents=True, exist_ok=True)
        detects = detect_meals_slope_prominence(
            df=df_evt,
            time_col="ts",
            score_col=detect_col,
            lookback_min=90,
            theta_prom=theta_prom_fixed,
            theta_slope=theta_slope,
            merge_within_minutes=60,
            candidate_quantiles_csv=q_csv,
            prominence_mode=prominence_mode,
            prominence_quantile=0.75,
            quantile_fit_days=pool_days if dataset == "uom" else None,
            gap_blocks=gap_ranges,
            boundary_blocks=gap_ranges,
            smooth_blocks=gap_ranges,
        )
        theta_prom_used = float(detects.attrs.get("theta_prom", theta_prom_fixed))
        if dataset == "uom":
            print(f"[UoM L2/L3] theta_prom={theta_prom_used:.4f} raw_events={len(detects)}")
    elif detector == "peakvalley":
        df_evt[detect_col] = smooth_score_minimal(df_evt[detect_col])
        if dataset == "nhs":
            detects = detect_meals_peakvalley_refine(
                df=df_evt,
                dataset="nhs",
                time_col="ts",
                score_col=detect_col,
                lookback=90,
                peak_thr=0.15,
                dy_thr=0.15,
                slope_thr=0.003,
                seg=20,
                delta_D=0.12,
                backshift=15,
                merge_within_minutes=60,
                plot=False,
            )
        else:
            detects = detect_meals_peakvalley_refine(
                df=df_evt,
                dataset="uom",
                time_col="ts",
                score_col=detect_col,
                lookback=90,
                peak_thr=0.15,
                dy_thr=0.15,
                slope_thr=0.0016,
                seg=20,
                delta_D=0.12,
                backshift=15,
                merge_within_minutes=60,
                plot=False,
            )
    else:
        raise ValueError(
            f"Unknown detector: {detector}. Use 'peakvalley', 'adaptive_region', 'adaptive_rise', or 'slope_prominence'."
        )

    # 5) 保存
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    tag = f"{dataset}_{patient_id}" if dataset != "nhs" else f"nhs_{patient_id}_{variant_label}"

    merged_csv = out_path / f"overall_{tag}.csv"
    evt_csv = out_path / f"fusion_events_{tag}.csv"
    scores_csv = out_path / f"fusion_scores_{tag}.csv"
    merged_ra.to_csv(merged_csv, index=True)

    detects_save = detects
    if dataset == "uom" and sampled_days and not skip_l3_filter:
        detects_l2_full = detects.copy()
        detects_l2_full.to_csv(out_path / f"fusion_events_{tag}_l2_full.csv", index=False)
        detects_save = filter_events_to_days(detects, sampled_days)
        print(f"[UoM L2/L3] events saved L3={len(detects_save)} (from L2={len(detects_l2_full)})")

    detects_save.to_csv(evt_csv, index=False)
    try:
        fusion_results.to_csv(scores_csv, index=False)
    except Exception:
        pass

    if detector == "slope_prominence":
        export_slope_local_maxima_csv(
            df_evt,
            out_path / f"plot_{tag}_local_maxima.csv",
            time_col="ts",
            score_col=detect_col,
            smooth_win=7,
            lookback_min=90,
            theta_prom=theta_prom_used,
            theta_slope=theta_slope,
            final_events=detects_save,
        )

    if not plot_results:
        return {"events_csv": evt_csv, "scores_csv": scores_csv}

    # 7) 可选：画图（右轴叠加 fusion_score / fusion_score_6）
    df_plot_vis = df_plot
    if dataset == "uom" and sampled_days and not skip_l3_filter:
        df_plot_vis = df_plot.loc[mask_by_days(df_plot.index, sampled_days)].copy()

    if dataset == "nhs":
        baseline_results = pd.read_csv(f'results/nhs/baseline/multi-meals/simple_heuristic_{patient_id}_{variant_label}.csv')
    elif dataset == "uom":
        baseline_results = pd.read_csv(f'results/uom/baseline/simple_heuristic_{patient_id}.csv')
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    baseline_simple_ts = baseline_results['tmeal']
    fusion_ts = detects_save['onset_time'] if not detects_save.empty else pd.Series(dtype="datetime64[ns]")
    if dataset == "uom" and sampled_days and not skip_l3_filter:
        baseline_simple_ts = baseline_simple_ts[
            pd.to_datetime(baseline_simple_ts).dt.date.isin(sampled_days)
        ]
    result_ts = {
    'simple_heuristic': pd.to_datetime(baseline_simple_ts),
    'fusion_based': pd.to_datetime(fusion_ts)
    }
    def to_ts_list(v):
            t = pd.to_datetime(v)
            return t.tolist() if hasattr(t, "tolist") else [t]
    rt_clean = {k: to_ts_list(v) for k, v in result_ts.items() if v is not None and len(v) > 0}
    df_plot_vis[detect_col] = smooth_score_minimal(df_plot_vis[detect_col])
    try:
        fig = create_interactive_trend_figure(
            df=df_plot_vis,
            features=["value"],
            event_features=["carbs", "dose"] if all(c in df_plot_vis.columns for c in ["carbs", "dose"]) else [],
            result_timestamps=rt_clean,
            ra_series=df_plot_vis[detect_col],
            ra_name=detect_col,
        )
        fig.update_yaxes(title_text="Probability (0-1)", secondary_y=True, row=1, col=1)
        fig_path = out_path / f"plot_{tag}.html"
        fig.write_html(str(fig_path))

        # draw the figure only including glucose/bolus data
        fig1 = create_wider_figure(
            df=df_plot_vis,
            features=["value"],
            event_features=["dose"] if "dose" in df_plot.columns else [],
        )
        fig1.update_yaxes(title_text="Glucose (mmol/L)", secondary_y=False, row=1, col=1)
        fig1_path = out_path / f"plot_{tag}_glucose_wider.html"
        # 以第一天为默认视图，注入 Prev/Next 控件（纯静态可用）
        if dataset == "uom" and sampled_days and not skip_l3_filter:
            dates_list = sorted(sampled_days)
        else:
            dates_list = sorted(set(pd.to_datetime(df_plot_vis.index).date))
        write_html_with_prev_next(fig1, dates_list, str(fig1_path))
    except Exception:
        # 绘图失败不影响主流程
        pass

    return {"events_csv": evt_csv, "scores_csv": scores_csv}


def main():
    parser = argparse.ArgumentParser(description="Run fusion-based pipeline")
    parser.add_argument("--dataset", type=str, required=True, choices=["uom", "nhs"], help="Dataset name")
    parser.add_argument("--patient_id", type=str, default="all", help="Patient ID or 'all' to run all in dataset list")
    parser.add_argument("--variant_label", type=str, default=None, help="NHS 变体标签（仅 NHS 需要）")
    parser.add_argument("--window", type=int, default=90, help="窗口长度（分钟）")
    parser.add_argument("--step", type=int, default=1, help="滑动步长（分钟）")
    parser.add_argument(
        "--detector",
        type=str,
        default="slope_prominence",
        choices=["peakvalley", "adaptive_region", "adaptive_rise", "slope_prominence"],
        help="事件检测器：peakvalley / adaptive_region / adaptive_rise / slope_prominence（峰+ prominence）",
    )
    parser.add_argument("--out_dir", type=str, default="results/fusion_outputs/annot_", help="输出目录")
    parser.add_argument("--pool_last_n_days", type=int, default=84, help="UoM L2 pool 最近 N 天（仅 uom）")
    parser.add_argument("--skip_l3_filter", action="store_true", help="UoM 不做 L3 过滤（调试用）")
    parser.add_argument("--threshold_file", type=Path, help="Reuse the unrounded Q75 fitted from full-fusion scores.")
    parser.add_argument("--candidate_manifest", type=Path, help="Historical full-fusion candidate exports; required unless --threshold_file is supplied.")
    parser.add_argument("--uom-block-mode", choices=["daily_independent", "contiguous"], default="daily_independent", help="Daily-independent physiology and windows, as in the clinician evaluation.")
    parser.add_argument("--weighting", choices=["default", "equal"], default="default", help="Use default fusion weights or exactly one third per component.")
    parser.add_argument("--dataset-manifest", type=Path, help="Standard CSV dataset manifest; NHS defaults to datasets/nhs_v1/manifest.json.")
    parser.add_argument("--no-plots", action="store_true", help="Write numerical outputs without interactive plots.")
    args = parser.parse_args()

    if args.detector == "slope_prominence":
        if args.threshold_file is None and args.candidate_manifest is None:
            parser.error("Supply --candidate_manifest to preserve historical calibration, or --threshold_file to reuse it.")
        ids = (uom_list if args.dataset == "uom" else nhs_list) if args.patient_id == "all" else [args.patient_id]
        variants = [None] if args.dataset == "uom" else ([args.variant_label] if args.variant_label else ["fiasp", "aspart"])
        jobs = [(pid, variant) for pid in ids for variant in variants if not (args.dataset == "nhs" and pid == "17" and variant == "aspart")]
        if not jobs:
            parser.error("No included records selected.")
        frames, options, cached_jobs = {}, {}, []
        for pid, variant in jobs:
            record_id = f"{args.dataset}_{pid}" if args.dataset != "nhs" else f"nhs_{pid}_{variant}"
            record_dir = Path(args.out_dir) / f"{args.dataset}_{pid}"
            paths = run_pipeline(
                args.dataset, pid, variant_label=variant, window_length_min=args.window,
                step_min=args.step, detector=args.detector, out_dir=str(record_dir),
                pool_last_n_days=args.pool_last_n_days, skip_l3_filter=args.skip_l3_filter,
                scores_only=True, plot_results=False, weighting=args.weighting, uom_block_mode=args.uom_block_mode, dataset_manifest=args.dataset_manifest,
            )
            frames[record_id] = pd.read_csv(paths["trajectory_csv"], parse_dates=["ts"], float_precision="round_trip")
            cached_jobs.append((pid, variant, record_dir))
        metadata = read_dataset_threshold(args.threshold_file) if args.threshold_file else fit_dataset_prominence_threshold(
            frames, candidate_exports=load_candidate_manifest(args.candidate_manifest)
        )
        metadata["weighting"] = args.weighting
        metadata["dataset"] = args.dataset
        metadata["application_record_ids"] = list(frames)
        threshold_path = Path(args.out_dir) / "dataset_threshold.json"
        threshold_path.write_text(json.dumps(metadata, indent=2)+"\n")
        print(f"Dataset-wide unrounded Q75: {metadata['threshold']:.17g}")
        for pid, variant, record_dir in cached_jobs:
            run_pipeline(
                args.dataset, pid, variant_label=variant, window_length_min=args.window,
                step_min=args.step, detector=args.detector, out_dir=str(record_dir),
                pool_last_n_days=args.pool_last_n_days, skip_l3_filter=args.skip_l3_filter,
                dataset_threshold=metadata["threshold"], cached_score_dir=record_dir,
                plot_results=not args.no_plots, weighting=args.weighting, uom_block_mode=args.uom_block_mode, dataset_manifest=args.dataset_manifest,
            )
        print(f"Threshold provenance: {threshold_path}")
        return

    # Decide which patient IDs and (for NHS) variants to run
    if args.dataset == "uom":
        ids = uom_list if args.patient_id == "all" else [args.patient_id]
        variants = [None]
    else:
        ids = nhs_list if args.patient_id == "all" else [args.patient_id]
        if args.variant_label:
            variants = [args.variant_label]
        else:
            # 默认同时跑 fiasp 与 aspart
            variants = ["fiasp", "aspart"]

    all_outs = {}
    for pid in ids:
        if args.dataset == "uom":
            out_paths = run_pipeline(
                weighting=args.weighting, uom_block_mode=args.uom_block_mode, dataset_manifest=args.dataset_manifest,
                dataset=args.dataset,
                patient_id=pid,
                variant_label=None,
                window_length_min=args.window,
                step_min=args.step,
                detector=args.detector,
                out_dir=args.out_dir+f"/{args.dataset}_{pid}",
                pool_last_n_days=args.pool_last_n_days,
                skip_l3_filter=args.skip_l3_filter,
            )
            all_outs[pid] = {k: str(v) for k, v in out_paths.items()}
            print(f"[{args.dataset}:{pid}] -> {all_outs[pid]}")
        else:
            for variant in variants:
                out_paths = run_pipeline(
                    weighting=args.weighting, uom_block_mode=args.uom_block_mode, dataset_manifest=args.dataset_manifest,
                    dataset=args.dataset,
                    patient_id=pid,
                    variant_label=variant,
                    window_length_min=args.window,
                    step_min=args.step,
                    detector=args.detector,
                    out_dir=args.out_dir+f"/{args.dataset}_{pid}",
                )
                key = f"{pid}_{variant}"
                all_outs[key] = {k: str(v) for k, v in out_paths.items()}
                print(f"[{args.dataset}:{pid}:{variant}] -> {all_outs[key]}")

    print("Completed. Outputs:", all_outs)


if __name__ == "__main__":
    main()

