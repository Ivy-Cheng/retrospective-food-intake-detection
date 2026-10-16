"""Build fusion scores with optional L2-fit normalization and gap-aware windows."""
from __future__ import annotations

from datetime import date
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from fusion_based import (
    attach_physio_kf_series,
    compute_components_only,
    pos_grad_energy,
    robust_zscore,
    sigmoid,
    trend_auc_dexcom_pos,
    trend_auc_medtronic_pos,
    full_trend_indicators,
)
from pipelines.uom_study_windows import (
    contiguous_day_blocks,
    day_blocks_to_timestamp_ranges,
    filter_merged_to_pool_days,
    independent_day_blocks,
)


from pipelines.gap_masking import prepared_gap_intervals, split_processing_blocks

def attach_physio_kf_per_ranges(merged, processing_blocks, models=("bergman", "hovorka")):
    """Estimate physiology and trends separately in each unmasked segment."""
    out = merged.copy()
    for model in models:
        out[f"{model}_ra_series"] = np.nan
        out[f"{model}_extra_series"] = np.nan
    for start, end in processing_blocks:
        chunk = merged.loc[start:end].copy()
        if len(chunk) < 2 or not pd.to_numeric(chunk["value"], errors="coerce").notna().any():
            continue
        chunk = full_trend_indicators(chunk)
        fitted, _ = attach_physio_kf_series(chunk, models=models, reindex_method="nearest")
        for col in fitted.columns:
            if col not in out.columns:
                out[col] = pd.Series(np.nan, index=out.index, dtype=float if pd.api.types.is_numeric_dtype(fitted[col]) else object)
            out.loc[fitted.index, col] = fitted[col]
    return out


def safe_minmax_fit(s: pd.Series, fit_mask: pd.Series | None = None) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    ref = s[fit_mask] if fit_mask is not None else s
    ref = ref.dropna()
    if ref.empty:
        return pd.Series(np.nan, index=s.index)
    vmin, vmax = ref.min(), ref.max()
    denom = vmax - vmin
    if pd.isna(denom) or denom == 0:
        return pd.Series(0.0, index=s.index).where(s.notna())
    return (s - vmin) / denom


def robust_zscore_fit(s: pd.Series, fit_mask: pd.Series | None = None, eps: float = 1e-9) -> pd.Series:
    ref = s[fit_mask] if fit_mask is not None else s
    ref = pd.to_numeric(ref, errors="coerce").dropna()
    if ref.empty:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    med = float(np.nanmedian(ref.values))
    mad = float(np.nanmedian(np.abs(ref.values - med)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < eps:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    return (s - med) / scale


def attach_physio_kf_per_blocks(
    merged: pd.DataFrame,
    day_blocks: list[list[date]],
    models: Sequence[str] = ("bergman", "hovorka"),
    reindex_method: str = "nearest",
) -> pd.DataFrame:
    if not day_blocks:
        return merged.copy()
    parts: list[pd.DataFrame] = []
    for block in day_blocks:
        t0 = pd.Timestamp(min(block))
        t1 = pd.Timestamp(max(block)) + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
        chunk = merged.loc[t0:t1]
        if chunk.empty:
            continue
        chunk_ra, _ = attach_physio_kf_series(
            chunk,
            models=models,
            reindex_method=reindex_method,
        )
        parts.append(chunk_ra)
    if not parts:
        return merged.copy()
    return pd.concat(parts).sort_index()


def build_fusion_results(
    merged_ra: pd.DataFrame,
    *,
    pool_days: set[date] | None = None,
    day_blocks: list[list[date]] | None = None,
    window_length_min: int = 90,
    step_min: int = 1,
    component_fns: dict[str, Callable[[pd.DataFrame], Any]] | None = None,
    processing_blocks: list[tuple[pd.Timestamp, pd.Timestamp]] | None = None,
) -> pd.DataFrame:
    if component_fns is None:
        component_fns = {
            "pos_grad_energy": lambda w: pos_grad_energy(w, "value"),
            "trend_auc_medtronic_pos": trend_auc_medtronic_pos,
            "trend_auc_dexcom_pos": trend_auc_dexcom_pos,
        }

    ts_ranges = processing_blocks if processing_blocks is not None else (day_blocks_to_timestamp_ranges(day_blocks) if day_blocks else None)

    comp_df = compute_components_only(
        merged_ra,
        window_length_min=window_length_min,
        step_min=step_min,
        component_fns=component_fns,
        include_prob=False,
        sigmoid_k=1.0,
        contiguous_blocks=ts_ranges,
    )

    fusion_results = comp_df.copy()
    ends = pd.to_datetime(fusion_results["end"], errors="coerce")
    if pool_days is not None:
        fit_mask = pd.Series([e.date() in pool_days if pd.notna(e) else False for e in ends], index=fusion_results.index)
    else:
        fit_mask = None

    # signal z/p with L2-fit when pool_days provided
    for name in component_fns.keys():
        if name in fusion_results.columns:
            z = robust_zscore_fit(fusion_results[name], fit_mask=fit_mask)
            fusion_results[f"{name}_z"] = z
            fusion_results[f"{name}_p"] = sigmoid(z, k=1.0)

    norm_cols = [
        "pos_grad_energy",
        "trend_auc_medtronic_pos",
        "trend_auc_dexcom_pos",
        "bergman_ra_post_iob_rise_len_min",
        "bergman_ra_post_iob_rise_auc",
        "hovorka_ra_post_iob_rise_len_min",
        "hovorka_ra_post_iob_rise_auc",
    ]
    for col in norm_cols:
        if col in fusion_results.columns:
            fusion_results[f"{col}_normalized"] = safe_minmax_fit(fusion_results[col], fit_mask=fit_mask)

    if {"pos_grad_energy_normalized", "trend_auc_medtronic_pos_normalized"}.issubset(fusion_results.columns):
        fusion_results["score_1"] = (
            fusion_results["pos_grad_energy_normalized"] + fusion_results["trend_auc_medtronic_pos_normalized"]
        ) / 2.0
    if {"pos_grad_energy_normalized", "trend_auc_dexcom_pos_normalized"}.issubset(fusion_results.columns):
        fusion_results["score_2"] = (
            fusion_results["pos_grad_energy_normalized"] + fusion_results["trend_auc_dexcom_pos_normalized"]
        ) / 2.0
    if "pos_grad_energy_normalized" in fusion_results.columns:
        fusion_results["score_3"] = fusion_results["pos_grad_energy_normalized"]
    if "bergman_ra_post_iob_rise_len_min_normalized" in fusion_results.columns:
        fusion_results["score_4"] = fusion_results["bergman_ra_post_iob_rise_len_min_normalized"]
    if "bergman_ra_post_iob_rise_auc_normalized" in fusion_results.columns:
        fusion_results["score_5"] = fusion_results["bergman_ra_post_iob_rise_auc_normalized"]

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

    fusion_results["end"] = ends
    return fusion_results


def build_uom_l2_merged_and_scores(
    patient_id: str,
    merged_full: pd.DataFrame,
    *,
    pool_last_n_days: int = 84,
    window_length_min: int = 90,
    step_min: int = 1,
    block_mode: str = "daily_independent",
) -> tuple[pd.DataFrame, pd.DataFrame, set[date], set[date], list[tuple[pd.Timestamp, pd.Timestamp]]]:
    from pipelines.uom_study_windows import load_pool_days, load_sampled_days

    pool_days = load_pool_days(patient_id, last_n_days=pool_last_n_days)
    sampled_days = load_sampled_days(patient_id)
    if block_mode == "contiguous":
        day_blocks = contiguous_day_blocks(pool_days)
    elif block_mode == "daily_independent":
        day_blocks = independent_day_blocks(pool_days)
    else:
        raise ValueError(f"Unknown UoM block_mode: {block_mode}")
    merged_l2 = filter_merged_to_pool_days(merged_full, pool_days)
    original_ranges = day_blocks_to_timestamp_ranges(day_blocks)
    masks = prepared_gap_intervals(merged_full)
    processing_ranges = split_processing_blocks(original_ranges, masks)
    merged_ra = attach_physio_kf_per_ranges(merged_l2, processing_ranges)
    fusion_results = build_fusion_results(
        merged_ra,
        pool_days=pool_days,
        day_blocks=day_blocks,
        processing_blocks=processing_ranges,
        window_length_min=window_length_min,
        step_min=step_min,
    )
    gap_ranges = processing_ranges
    return merged_ra, fusion_results, pool_days, sampled_days, gap_ranges


def align_scores_to_plot(merged_ra: pd.DataFrame, fusion_results: pd.DataFrame) -> pd.DataFrame:
    df_plot = merged_ra.copy().sort_index()
    score_series = fusion_results.set_index("end").get("pos_grad_energy_p")
    if score_series is not None:
        df_plot["fusion_score"] = score_series.reindex(df_plot.index)
    fusion_results = fusion_results.copy()
    fusion_results["end"] = pd.to_datetime(fusion_results["end"], errors="coerce")
    for k in range(1, 7):
        col = f"score_{k}"
        if col in fusion_results.columns:
            s = fusion_results.set_index("end")[col]
            df_plot[f"fusion_{col}"] = s.reindex(df_plot.index)
    return df_plot
