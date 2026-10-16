"""UoM L2/L3 study window helpers (patient_pool vs sampled days)."""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Iterable

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_UOM_DIR = SCRIPT_DIR / "results" / "uom"
SELECTION_DIR = SCRIPT_DIR / "results" / "uom" / "sampling" / "day_cv_selected"
POOL_CACHE_DIR = SCRIPT_DIR / "results" / "uom" / "sampling" / "day_cv_pool"


def _read_patient_csv(data_dir: Path, patient_id: str) -> pd.DataFrame | None:
    csv_path = data_dir / f"overall_data_{patient_id}.csv"
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    ts_col = None
    if "ts" in df.columns:
        ts_col = "ts"
    elif len(df.columns) > 0 and str(df.columns[0]).lower().startswith("unnamed"):
        ts_col = df.columns[0]
    if ts_col is None:
        return None
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col]).rename(columns={ts_col: "ts"})
    return df


def compute_daily_cv_1440(df: pd.DataFrame, last_n_days: int | None = None) -> pd.Series:
    """Same logic as sampling.compute_daily_cv_1440 (inlined to avoid import cycles)."""
    import numpy as np

    if df.empty or "ts" not in df.columns:
        return pd.Series(dtype=float)
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df = df.dropna(subset=["ts"])
    vals = pd.to_numeric(df.get("value"), errors="coerce")
    med = vals.median(skipna=True)
    if pd.notna(med) and med > 30:
        df["value"] = vals / 18.0
    else:
        df["value"] = vals

    df["ts_min"] = df["ts"].dt.floor("min")
    per_min = df.groupby("ts_min", as_index=True)["value"].mean().to_frame("value")
    if per_min.empty:
        return pd.Series(dtype=float)

    per_min["day"] = per_min.index.date
    grp = per_min.groupby("day")
    counts = grp["value"].size()
    means = grp["value"].mean()
    stds = grp["value"].std()

    if last_n_days is not None and last_n_days > 0 and len(counts) > last_n_days:
        day_order = sorted(counts.index)
        keep_set = set(day_order[-last_n_days:])
        counts = counts[counts.index.map(lambda d: d in keep_set)]
        means = means[means.index.map(lambda d: d in keep_set)]
        stds = stds[stds.index.map(lambda d: d in keep_set)]

    cv = stds / means.replace(0, np.nan)
    valid = (counts == 1440) & means.ne(0) & cv.notna()
    return cv[valid].sort_index()


def load_pool_days(patient_id: str, last_n_days: int = 84, input_dir: str | Path | None = None) -> set[dt.date]:
    data_dir = Path(input_dir) if input_dir is not None else DEFAULT_UOM_DIR
    df = _read_patient_csv(data_dir, patient_id)
    if df is None or df.empty:
        return set()
    daily_cv = compute_daily_cv_1440(df, last_n_days=last_n_days)
    return set(daily_cv.index)


def load_sampled_days(patient_id: str, selection_dir: str | Path | None = None) -> set[dt.date]:
    sel_dir = Path(selection_dir) if selection_dir is not None else SELECTION_DIR
    sel_path = sel_dir / f"uom_{patient_id}_selected_dates.csv"
    if not sel_path.exists():
        return set()
    sel = pd.read_csv(sel_path)
    return set(pd.to_datetime(sel["day"]).dt.date.tolist())


def contiguous_day_blocks(days: Iterable[dt.date]) -> list[list[dt.date]]:
    sorted_days = sorted(set(days))
    if not sorted_days:
        return []
    blocks: list[list[dt.date]] = []
    current = [sorted_days[0]]
    for d in sorted_days[1:]:
        if (d - current[-1]).days == 1:
            current.append(d)
        else:
            blocks.append(current)
            current = [d]
    blocks.append(current)
    return blocks


def independent_day_blocks(days: Iterable[dt.date]) -> list[list[dt.date]]:
    """Return one block per day so rolling windows never cross midnight."""
    return [[d] for d in sorted(set(days))]


def day_blocks_to_timestamp_ranges(blocks: list[list[dt.date]]) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for block in blocks:
        t0 = pd.Timestamp(block[0])
        t1 = pd.Timestamp(block[-1]) + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
        ranges.append((t0, t1))
    return ranges


def filter_merged_to_pool_days(df: pd.DataFrame, pool_days: set[dt.date]) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        out = df.copy()
        out.index = pd.to_datetime(out.index)
    else:
        out = df.sort_index()
    mask = pd.Index([ts.date() in pool_days for ts in out.index])
    return out.loc[mask]


def mask_by_days(index: pd.Index, days: set[dt.date]) -> pd.Index:
    idx = pd.to_datetime(index)
    return pd.Index([ts.date() in days for ts in idx])


def filter_events_to_days(
    events_df: pd.DataFrame,
    days: set[dt.date],
    col: str = "onset_time",
) -> pd.DataFrame:
    if events_df is None or events_df.empty or not days:
        return events_df
    out = events_df.copy()
    ts = pd.to_datetime(out[col], errors="coerce")
    keep = ts.dt.date.isin(days)
    return out.loc[keep].reset_index(drop=True)


def save_pool_days_cache(patient_id: str, pool_days: set[dt.date]) -> Path:
    POOL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = POOL_CACHE_DIR / f"uom_{patient_id}_pool_dates.csv"
    rows = sorted(pool_days)
    pd.DataFrame({"day": [d.isoformat() for d in rows]}).to_csv(out, index=False)
    return out
