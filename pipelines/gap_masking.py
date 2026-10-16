"""Shared exclusion of retained missing-CGM intervals.

Short gaps retain existing preparation. A gap is an open interval between
successive finite prepared-CGM observations separated by more than 30 minutes.
Breaks left by removed days are also excluded, regardless of their length.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

def prepared_gap_intervals(frame, *, time_col="ts", value_col="value", max_short_gap_min=30):
    times = frame.index if isinstance(frame.index, pd.DatetimeIndex) else pd.to_datetime(frame[time_col])
    valid = pd.to_numeric(frame[value_col], errors="coerce").to_numpy()
    ts = pd.Series(pd.to_datetime(times)[np.isfinite(valid)]).dropna().drop_duplicates().sort_values().reset_index(drop=True)
    delta = ts.diff().dt.total_seconds() / 60.0
    return [(ts.iloc[i-1], ts.iloc[i]) for i in delta.index[delta > max_short_gap_min]]

def split_processing_blocks(base_blocks, masked_intervals):
    """Return inclusive observed segments, excluding the open masked intervals."""
    result = []
    for lo, hi in base_blocks:
        lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
        cursor = lo
        for a, b in sorted(masked_intervals):
            a, b = pd.Timestamp(a), pd.Timestamp(b)
            if b <= lo or a >= hi:
                continue
            stop = min(hi, a)
            if cursor <= stop:
                result.append((cursor, stop))
            cursor = max(cursor, b)
        if cursor <= hi:
            result.append((cursor, hi))
    return result

def masked_time_flags(times, masked_intervals):
    times = pd.Series(pd.to_datetime(times))
    flags = pd.Series(False, index=times.index)
    for a, b in masked_intervals:
        flags |= (times > a) & (times < b)
    return flags.to_numpy()

def recompute_trends_by_blocks(frame, blocks, trend_fn):
    """Prevent trend lookbacks from using observations across a masked interval."""
    out = frame.copy()
    for a, b in blocks:
        chunk = out.loc[a:b].copy()
        if chunk.empty:
            continue
        chunk = trend_fn(chunk)
        for col in chunk.columns:
            if col not in out.columns:
                out[col] = np.nan
            out.loc[chunk.index, col] = chunk[col]
    return out
