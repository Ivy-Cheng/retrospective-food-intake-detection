from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple, Any
import numpy as np
import pandas as pd




def detect_extrema_1d(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect local maxima/minima indices for 1D array x without any smoothing.
    Local max: x[i-1] < x[i] >= x[i+1]
    Local min: x[i-1] > x[i] <= x[i+1]
    """
    x = np.asarray(x, dtype=float)
    if x.size < 3:
        return np.array([], dtype=int), np.array([], dtype=int)

    maxima = np.where((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]))[0] + 1
    minima = np.where((x[1:-1] < x[:-2]) & (x[1:-1] <= x[2:]))[0] + 1
    return maxima.astype(int), minima.astype(int)

# s 为 [0,1] 区间的 score Series，索引为 DatetimeIndex（或 RangeIndex 均可）
def smooth_score_minimal(s, win=7):
    # 先用中位数压毛刺，再用很轻的均值，均为居中窗口
    s1 = s.rolling(window=win, center=True, min_periods=1).median()
    s2 = s1.rolling(window=win, center=True, min_periods=1).mean()
    return s2.clip(0, 1)


def smooth_score_by_blocks(
    s: pd.Series,
    times: pd.Series | pd.Index,
    blocks: list[tuple[pd.Timestamp, pd.Timestamp]] | None,
    win: int = 7,
    preserve_nan: bool = True,
) -> pd.Series:
    """Apply score smoothing independently inside each time block.

    ``preserve_nan`` keeps intentionally missing score positions missing, which
    is needed for daily-independent runs where each day's first 90 minutes are
    undefined by design.
    """
    raw = pd.to_numeric(s, errors="coerce")
    if blocks is None:
        smoothed = smooth_score_minimal(raw, win=win)
        return smoothed.where(raw.notna()) if preserve_nan else smoothed

    t = pd.to_datetime(pd.Series(times, index=raw.index), errors="coerce")
    out = pd.Series(np.nan, index=raw.index, dtype=float)
    for b0, b1 in blocks:
        mask = (t >= pd.Timestamp(b0)) & (t <= pd.Timestamp(b1))
        if not mask.any():
            continue
        smoothed = smooth_score_minimal(raw.loc[mask], win=win)
        if preserve_nan:
            smoothed = smoothed.where(raw.loc[mask].notna())
        out.loc[mask] = smoothed
    return out.clip(0, 1)

def detect_meals_from_score(
    df: pd.DataFrame,
    time_col: str = "ts",
    score_col: str = "fusion_score",
    peak_thr: float = 0.2,
    base_thr: float = 0.2,
    merge_within_minutes: Optional[int] = None,
) -> pd.DataFrame:
    """
    Implements your rule (no filtering):
      For each local maximum tmax:
        if score(tmax) > peak_thr:
            let tmin be the immediately preceding local minimum
            if score(tmin) < base_thr:
                meal onset = tmin

    Args:
      df: DataFrame with time_col (datetime) and score_col (float in [0,1])
      merge_within_minutes: if provided, merge multiple onsets closer than this (keep earliest)

    Returns:
      DataFrame with detected meal onsets and associated peak info.
    """
  

    d = df[[time_col, score_col]].copy()
    d[time_col] = pd.to_datetime(d[time_col])
    d = d.sort_values(time_col).reset_index(drop=True)

    x = d[score_col].to_numpy(dtype=float)
    # x = pd.Series(x_raw).rolling(window=5, center=True, min_periods=1).median().to_numpy()
    maxima, minima = detect_extrema_1d(x)

    # For fast "previous minimum" lookup: keep minima sorted
    minima = np.sort(minima)

    detections = []
    for p in maxima:
        if x[p] <= peak_thr:
            continue

        # previous local minimum index before p
        prev_mins = minima[minima < p]
        if prev_mins.size == 0:
            continue
        m = int(prev_mins[-1])

        if x[m] < base_thr:
            detections.append(
                {
                    "onset_idx": m,
                    "onset_time": d.loc[m, time_col],
                    "onset_score": float(x[m]),
                    "peak_idx": int(p),
                    "peak_time": d.loc[p, time_col],
                    "peak_score": float(x[p]),
                }
            )

    out = pd.DataFrame(detections).sort_values("onset_time").reset_index(drop=True)

    # Optional merge: if multiple onsets are within X minutes, keep earliest
    if merge_within_minutes is not None and not out.empty:
        keep_rows = []
        last_kept_time = None
        for i, row in out.iterrows():
            t = row["onset_time"]
            if last_kept_time is None:
                keep_rows.append(i)
                last_kept_time = t
            else:
                if (t - last_kept_time).total_seconds() / 60.0 >= merge_within_minutes:
                    keep_rows.append(i)
                    last_kept_time = t
                # else: skip (treated as same meal episode)
        out = out.loc[keep_rows].reset_index(drop=True)

    return out

# include the start guard for NHS dataset
def detect_meals_start_guard(
    df: pd.DataFrame,
    time_col: str = "ts",
    score_col: str = "fusion_score",
    peak_thr: float = 0.2,
    base_thr: float = 0.2,
    window_len_min: int = 90,
    merge_within_minutes: int | None = None,
):
    # 准备数据
    d = df[[time_col, score_col]].copy()
    d[time_col] = pd.to_datetime(d[time_col])
    d = d.sort_values(time_col).reset_index(drop=True)
    x = d[score_col].to_numpy(dtype=float)

    # 简单极值
    def extrema_1d(arr):
        if arr.size < 3:
            return np.array([], int), np.array([], int)
        maxima = np.where((arr[1:-1] > arr[:-2]) & (arr[1:-1] >= arr[2:]))[0] + 1
        minima = np.where((arr[1:-1] < arr[:-2]) & (arr[1:-1] <= arr[2:]))[0] + 1
        return maxima.astype(int), minima.astype(int)

    maxima, minima = extrema_1d(x)
    minima = np.sort(minima)

    t_start = d[time_col].min()
    lb = pd.Timedelta(minutes=window_len_min)

    detections = []
    for k, p in enumerate(maxima):
        if x[p] <= peak_thr:
            continue

        if k == 0:
            # 第一个峰值：优先找“后一个”极小
            next_mins = minima[minima > p]
            if next_mins.size > 0:
                m = int(next_mins[0])
            else:
                # 兜底：没有后继极小就用“前一个”
                prev_mins = minima[minima < p]
                if prev_mins.size == 0:
                    continue
                m = int(prev_mins[-1])
        else:
            # 其余峰值：用“前一个”极小
            prev_mins = minima[minima < p]
            if prev_mins.size == 0:
                continue
            m = int(prev_mins[-1])

        if x[m] >= base_thr:
            continue

        t_peak = d.loc[p, time_col]
        if k == 0:
            # 第一个峰：用 start guard（与序列起点取较大值）
            onset_time = min(t_peak - lb, t_start)
        else:
            # 后续峰：用“前一个局部极小”的时间（与起点取较大值）
            onset_time = max(d.loc[m, time_col], t_start)

        detections.append({
            "onset_idx": None,  # 可选：用 searchsorted 再映射到最近索引
            "onset_time": onset_time,
            "onset_score": float(x[m]),   # 触发基线得分，供参考
            "peak_idx": int(p),
            "peak_time": t_peak,
            "peak_score": float(x[p]),
        })

    out = pd.DataFrame(detections).sort_values("onset_time").reset_index(drop=True)

    # 可选：近邻合并
    if merge_within_minutes is not None and not out.empty:
        keep, last_t = [], None
        for i, row in out.iterrows():
            t = row["onset_time"]
            if last_t is None or (t - last_t).total_seconds() / 60.0 >= merge_within_minutes:
                keep.append(i); last_t = t
        out = out.loc[keep].reset_index(drop=True)

    return out


# mean-shift algorithm to detect the changing points of the probability score 
def detect_meals_mean_shift(
    df: pd.DataFrame,
    time_col: str = "ts",
    score_col: str = "fusion_score",
    window_len_min: int = 90,
    merge_within_minutes: int | None = None,
    # mean-shift settings
    seg_min: int = 30,            # compare last seg_min vs previous seg_min
    delta: float = 0.15,          # D(t) threshold
    rho_10min: float = 0.05,      # D(t)-D(t-10) threshold (rise)
    hold_min: int = 7,            # require D(t)>=delta for last hold_min minutes
    onset_backshift_min: int = 15 # onset = first-cross - backshift
) -> pd.DataFrame:
    """
    Plan B (1-min, score already smoothed):
      D(t)=mean(score[t-seg+1:t]) - mean(score[t-2seg+1:t-seg])
      Trigger if D holds above delta for hold_min mins AND rises over last 10 mins by rho_10min.
      Onset localized as earliest u in [t-window_len_min, t] where D(u) crosses delta, then backshift.
    """
    if df.empty:
        return pd.DataFrame(columns=["onset_idx","onset_time","onset_score","peak_idx","peak_time","peak_score"])

    d = df.copy()
    t = pd.to_datetime(d[time_col])
    s = d[score_col].astype(float).to_numpy()

    seg = int(seg_min)
    win = int(window_len_min)
    hold = int(hold_min)
    rise_n = 10
    back = int(onset_backshift_min)

    def nanmean(a: np.ndarray) -> float:
        v = a[np.isfinite(a)]
        return float(v.mean()) if v.size else float("nan")

    # D(t)
    D = np.full(len(s), np.nan, dtype=float)
    for i in range(2*seg, len(s)):
        D[i] = nanmean(s[i-seg+1:i+1]) - nanmean(s[i-2*seg+1:i-seg+1])

    dets: list[dict[str, Any]] = []

    refractory_min = merge_within_minutes if merge_within_minutes is not None else win
    refractory = int(refractory_min)

    i = 0
    while i < len(s):
        if i < max(win, 2*seg, hold, rise_n):
            i += 1
            continue
        if not (np.isfinite(D[i]) and np.isfinite(D[i-rise_n])):
            i += 1
            continue

        h0 = i - hold + 1
        ok_hold = np.all(np.isfinite(D[h0:i+1]) & (D[h0:i+1] >= delta))
        ok_rise = (D[i] - D[i-rise_n]) >= rho_10min
        if not (ok_hold and ok_rise):
            i += 1
            continue

        # localize onset: first crossing of delta within the last window_len_min minutes
        w0 = max(0, i - win)
        u = next((j for j in range(w0, i+1) if j >= 2*seg and np.isfinite(D[j]) and D[j] >= delta), i)
        onset_idx = max(0, u - back)

        dets.append({
            "onset_idx": int(onset_idx),
            "onset_time": t.iloc[onset_idx],
            "onset_score": float(s[onset_idx]) if np.isfinite(s[onset_idx]) else float("nan"),
            "peak_idx": int(i),
            "peak_time": t.iloc[i],
            "peak_score": float(s[i]) if np.isfinite(s[i]) else float("nan"),
        })

        i = min(len(s), i + refractory)

    out = pd.DataFrame(dets)
    if out.empty:
        return out
    return out.sort_values("onset_time").reset_index(drop=True)


def detect_meals_mean_shift_or_rise(
    df: pd.DataFrame,
    time_col: str = "ts",
    score_col: str = "fusion_score",
    window_len_min: int = 90,
    merge_within_minutes: int | None = None,
    # ----- mean-shift D(t) settings -----
    seg_min: int = 20,           # D(t)=mean(last seg)-mean(prev seg)
    delta_D: float = 0.12,       # threshold on D(t)
    hold_D_min: int = 5,         # require D(t)>=delta_D for last hold_D_min mins
    # ----- rise-path settings -----
    rise_diff_min: int = 10,     # rise(t)=score(t)-score(t-rise_diff_min)
    rise_thr: float = 0.10,      # threshold on rise(t)
    hold_rise_min: int = 3,      # require rise(t)>=rise_thr for last hold_rise_min mins
    # ----- baseline gate (for rise path) -----
    base_early_min: int = 60,    # base(t)=median(score[t-60:t-30])
    base_late_min: int = 30,
    delta_S: float = 0.10,       # require score(t) >= base(t) + delta_S
    # ----- onset localization -----
    onset_backshift_min: int = 15,
) -> pd.DataFrame:
    """
    1-min, score already smoothed.

    Trigger at time t if:
      A) Mean-shift path: D(t) >= delta_D for hold_D_min minutes, where
           D(t) = mean(score[t-seg+1:t]) - mean(score[t-2seg+1:t-seg])
      OR
      B) Rise path: rise(t) >= rise_thr for hold_rise_min minutes, where
           rise(t) = score(t) - score(t-rise_diff_min),
         AND score(t) is above a dynamic baseline:
           score(t) >= median(score[t-base_early:t-base_late]) + delta_S

    Once triggered at peak time t_peak:
      - Find earliest u in [t_peak-window_len_min, t_peak] where either path becomes true.
      - onset = u - onset_backshift_min

    Output columns:
      onset_idx, onset_time, onset_score, peak_idx, peak_time, peak_score
    """
    if df.empty:
        return pd.DataFrame(columns=["onset_idx","onset_time","onset_score","peak_idx","peak_time","peak_score"])

    d = df.copy()
    t = pd.to_datetime(d[time_col])
    s = d[score_col].astype(float).to_numpy()

    n = len(s)
    seg = int(seg_min)
    win = int(window_len_min)
    holdD = int(hold_D_min)
    riseN = int(rise_diff_min)
    holdR = int(hold_rise_min)
    bE = int(base_early_min)
    bL = int(base_late_min)
    back = int(onset_backshift_min)

    def nanmean(a: np.ndarray) -> float:
        v = a[np.isfinite(a)]
        return float(v.mean()) if v.size else float("nan")

    def nanmedian(a: np.ndarray) -> float:
        v = a[np.isfinite(a)]
        return float(np.median(v)) if v.size else float("nan")

    # ----- precompute D(t) and rise(t) -----
    D = np.full(n, np.nan, dtype=float)
    for i in range(2 * seg, n):
        D[i] = nanmean(s[i-seg+1:i+1]) - nanmean(s[i-2*seg+1:i-seg+1])

    R = np.full(n, np.nan, dtype=float)
    for i in range(riseN, n):
        if np.isfinite(s[i]) and np.isfinite(s[i-riseN]):
            R[i] = s[i] - s[i-riseN]

    dets: list[dict[str, Any]] = []

    refractory_min = merge_within_minutes if merge_within_minutes is not None else win
    refractory = int(refractory_min)

    # helpers: path truth at time i
    def path_D(i: int) -> bool:
        if i < max(2*seg, holdD-1):
            return False
        h0 = i - holdD + 1
        v = D[h0:i+1]
        return bool(v.size == holdD and np.all(np.isfinite(v) & (v >= delta_D)))

    def path_rise(i: int) -> bool:
        if i < max(riseN, bE, holdR-1):
            return False
        # baseline gate: median over [i-bE, i-bL]
        j0, j1 = i - bE, i - bL
        if j1 <= j0:
            return False
        base = nanmedian(s[j0:j1])
        if not np.isfinite(base) or not np.isfinite(s[i]):
            return False
        if s[i] < base + delta_S:
            return False
        # hold on rise
        h0 = i - holdR + 1
        v = R[h0:i+1]
        return bool(v.size == holdR and np.all(np.isfinite(v) & (v >= rise_thr)))

    i = 0
    while i < n:
        if i < max(win, 2*seg, riseN, bE, holdD, holdR):
            i += 1
            continue

        trig = path_D(i) or path_rise(i)
        if not trig:
            i += 1
            continue

        # ----- localize onset: earliest u within [i-win, i] where either path becomes true -----
        w0 = max(0, i - win)
        u = None
        for j in range(w0, i + 1):
            if path_D(j) or path_rise(j):
                u = j
                break
        if u is None:
            u = i

        onset_idx = max(0, u - back)

        dets.append({
            "onset_idx": int(onset_idx),
            "onset_time": t.iloc[onset_idx],
            "onset_score": float(s[onset_idx]) if np.isfinite(s[onset_idx]) else float("nan"),
            "peak_idx": int(i),
            "peak_time": t.iloc[i],
            "peak_score": float(s[i]) if np.isfinite(s[i]) else float("nan"),
        })

        i = min(n, i + refractory)

    out = pd.DataFrame(dets)
    if out.empty:
        return out
    return out.sort_values("onset_time").reset_index(drop=True)




def detect_meals_simple_peaks_then_refine(
    df: pd.DataFrame,
    time_col: str = "ts",
    score_col: str = "fusion_score",
    window_len_min: int = 90,
    merge_within_minutes: int | None = 60,
    # ---- simple candidate rules (your idea) ----
    thr: float = 0.20,             # for local min/max and "high" region
    rise_diff_min: int = 10,       # rise(t)=score(t)-score(t-rise_diff_min)
    rise_thr: float = 0.20,        # rise threshold
    valley_lookback_min: int = 30, # search valley in [t-valley_lookback_min, t]
    hold_min: int = 5,             # require score >= thr for last hold_min mins at trigger
    # ---- onset refine (mean-shift D) ----
    seg_min: int = 20,             # D(t)=mean(last seg)-mean(prev seg)
    delta_D: float = 0.12,         # D threshold inside refine window
    onset_backshift_min: int = 15, # onset = first-cross - backshift
) -> pd.DataFrame:
    """
    Stage 1 (candidate, super simple):
      candidate at time t if:
        A) rise(t) = score(t)-score(t-rise_diff_min) >= rise_thr
        OR
        B) min(score[t-valley_lookback:t]) < thr and score(t) >= thr
      AND
        C) score has been high for hold_min minutes: score[t-hold_min+1:t] >= thr

    Stage 2 (refine onset):
      inside [t-window_len_min, t], compute D(u)=mean(score[u-seg+1:u]) - mean(score[u-2seg+1:u-seg])
      onset is earliest u where D(u) >= delta_D, then subtract onset_backshift_min.
      fallback: earliest time score crosses thr in [t-window_len_min, t].

    Output columns:
      onset_idx, onset_time, onset_score, peak_idx, peak_time, peak_score
    """
    if df.empty:
        return pd.DataFrame(columns=["onset_idx","onset_time","onset_score","peak_idx","peak_time","peak_score"])

    d = df.copy()
    t = pd.to_datetime(d[time_col])
    s = d[score_col].astype(float).to_numpy()
    n = len(s)

    win = int(window_len_min)
    seg = int(seg_min)
    back = int(onset_backshift_min)
    riseN = int(rise_diff_min)
    vwin = int(valley_lookback_min)
    hold = int(hold_min)

    def nanmean(a: np.ndarray) -> float:
        v = a[np.isfinite(a)]
        return float(v.mean()) if v.size else float("nan")

    # precompute D(u) for refine (global, cheap)
    D = np.full(n, np.nan, dtype=float)
    for i in range(2 * seg, n):
        D[i] = nanmean(s[i-seg+1:i+1]) - nanmean(s[i-2*seg+1:i-seg+1])

    dets: list[dict[str, Any]] = []
    refractory = int(merge_within_minutes) if merge_within_minutes is not None else win

    i = 0
    while i < n:
        if i < max(win, 2*seg, riseN, vwin, hold):
            i += 1
            continue

        # ---- Stage 1: candidate trigger at i ----
        rise_ok = np.isfinite(s[i]) and np.isfinite(s[i-riseN]) and (s[i] - s[i-riseN] >= rise_thr)

        lo = max(0, i - vwin)
        valley = np.nanmin(s[lo:i+1]) if np.any(np.isfinite(s[lo:i+1])) else np.nan
        valley_ok = np.isfinite(valley) and np.isfinite(s[i]) and (valley < thr) and (s[i] >= thr)

        hold_ok = np.all(np.isfinite(s[i-hold+1:i+1]) & (s[i-hold+1:i+1] >= thr))

        if not ((rise_ok or valley_ok) and hold_ok):
            i += 1
            continue

        # ---- Stage 2: refine onset within [i-win, i] ----
        w0 = max(0, i - win)
        w1 = i

        # 2.1 earliest D-cross
        u = None
        for j in range(w0, w1 + 1):
            if j >= 2*seg and np.isfinite(D[j]) and D[j] >= delta_D:
                u = j
                break

        # 2.2 fallback: earliest score-threshold crossing
        if u is None:
            for j in range(w0, w1 + 1):
                if np.isfinite(s[j]) and s[j] >= thr:
                    u = j
                    break
        if u is None:
            u = i

        onset_idx = max(0, u - back)

        dets.append({
            "onset_idx": int(onset_idx),
            "onset_time": t.iloc[onset_idx],
            "onset_score": float(s[onset_idx]) if np.isfinite(s[onset_idx]) else float("nan"),
            "peak_idx": int(i),
            "peak_time": t.iloc[i],
            "peak_score": float(s[i]) if np.isfinite(s[i]) else float("nan"),
        })

        i = min(n, i + refractory)

    out = pd.DataFrame(dets)
    if out.empty:
        return out
    return out.sort_values("onset_time").reset_index(drop=True)


import numpy as np
import pandas as pd


def detect_meals_peakvalley_refine(
    df,
    dataset: str = "nhs",
    time_col="ts",
    score_col="fusion_score",
    lookback=90,
    peak_thr=0.2,
    dy_thr=0.15,
    slope_thr=0.004,
    seg=20,
    delta_D=0.12,
    backshift=15,
    merge_within_minutes=60,
    plot=True,
):

    t = pd.to_datetime(df[time_col])
    s = df[score_col].to_numpy()
    n = len(s)

    # ---------- find peaks ----------
    peaks = [i for i in range(1,n-1) if s[i-1] < s[i] >= s[i+1] and s[i] >= peak_thr]

    # ---------- candidate peaks ----------
    cands = []

    for k, p in enumerate(peaks):

        lo = max(0,p-lookback)
        m = lo + np.argmin(s[lo:p])

        dy = s[p] - s[m]
        dt = p - m
        slope = dy/dt if dt>0 else 0

        if dy >= dy_thr and slope >= slope_thr:
            cands.append((m,p,dy,dt,slope))
    # ... your loop that fills cands ...
    cands = merge_cands_by_peak_score(cands, s, merge_within_minutes=merge_within_minutes)

    # ---------- compute mean shift ----------
    D = np.full(n,np.nan)

    for i in range(2*seg,n):
        D[i] = np.mean(s[i-seg+1:i+1]) - np.mean(s[i-2*seg+1:i-seg+1])

    # ---------- refine onset ----------
    events=[]

    for m,p,dy,dt,slope in cands:

        u=None

        for j in range(max(0,p-lookback),p):
            if D[j] >= delta_D:
                u=j
                break

        if u is None:
            u=m

        onset=max(0,u-backshift)

        events.append({
            "onset_idx":onset,
            "onset_time":t.iloc[onset],
            "peak_idx":p,
            "peak_time":t.iloc[p],
            "peak_score":s[p],
            "delta_y":dy,
            "delta_t":dt,
            "slope":slope
        })
    
    if dataset == "nhs":
        t_start = df[time_col].min()
        lb = pd.Timedelta(minutes=int(lookback))

        # 计算极值（若你方法里已有 maxima/minima，可直接复用）
        x = df[score_col].to_numpy(dtype=float)
        maxima, minima = detect_extrema_1d(x)
        minima = np.sort(minima)

        if maxima.size > 0:
            p0 = int(maxima[0])
            next_mins = minima[minima > p0]
            if next_mins.size > 0:
                m0 = int(next_mins[0])
                # 使用 peak_thr 作为 0.2 的阈值：>0.2 → 触发，<=0.2 → 回落
                if (x[p0] > peak_thr) and (x[m0] <= peak_thr):
                    t_peak = df.loc[p0, time_col]
                    onset_time = max(t_peak - lb, t_start)
                    events.append({
                        "onset_time":onset_time,
                        "peak_time":t_peak,
                        "peak_score":x[p0],
                    })

    events=pd.DataFrame(events)

    if events.empty:
        return events

    # ---------- merge events ----------
    events=events.sort_values("onset_time").reset_index(drop=True)

    merged=[events.iloc[0].to_dict()]

    for r in events.iloc[1:].to_dict("records"):

        prev=merged[-1]

        if pd.to_datetime(r["onset_time"]) - pd.to_datetime(prev["onset_time"]) <= pd.Timedelta(minutes=merge_within_minutes):

            if r["peak_score"] > prev["peak_score"]:
                merged[-1]=r

        else:
            merged.append(r)

    events=pd.DataFrame(merged)

    # ---------- plots ----------
    if plot:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.scatter(events.delta_t,events.delta_y)
        plt.xlabel("Δt")
        plt.ylabel("Δy")
        plt.title("Δt vs Δy")
        plt.show()

        plt.figure()
        plt.plot(t,s)
        plt.scatter(events.peak_time,events.peak_score)
        plt.scatter(events.onset_time,events.peak_score*0+0.05)
        plt.title("Detected meals")
        plt.show()

    return events

def merge_cands_by_peak_score(cands, s, merge_within_minutes=60):
    """
    cands: list of tuples (m,p,dy,dt,slope)
    keep within merge window the one with larger s[p].
    """
    if not cands:
        return cands

    # sort by peak time (index)
    cands_sorted = sorted(cands, key=lambda x: x[1])
    merged = [cands_sorted[0]]

    for cur in cands_sorted[1:]:
        m,p,dy,dt,sl = cur
        m0,p0,dy0,dt0,sl0 = merged[-1]

        if (p - p0) <= merge_within_minutes:
            # keep the one with higher peak score s[p]
            if s[p] > s[p0]:
                merged[-1] = cur
            # else keep merged[-1]
        else:
            merged.append(cur)

    return merged


def _find_contiguous_regions(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive (start, end) index pairs where mask is continuously True."""
    m = np.asarray(mask, dtype=bool)
    if m.size == 0:
        return []
    starts = np.where(m & ~np.r_[False, m[:-1]])[0]
    ends = np.where(m & ~np.r_[m[1:], False])[0]
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def _calc_adaptive_thresholds(
    valid: pd.Series,
    high_k: float,
    low_k: float,
    min_high_clip: float,
    max_high_clip: float,
    min_low_clip: float,
) -> tuple[float, float]:
    """Adaptive high/low thresholds with robust fallback and guaranteed ordering."""
    v = pd.to_numeric(valid, errors="coerce").dropna()
    if v.empty:
        theta_high = float(min_high_clip)
        theta_low = float(min_low_clip)
        if theta_low >= theta_high:
            theta_low = max(min_low_clip, theta_high - 0.02)
        return theta_high, theta_low

    median_score = float(np.median(v.to_numpy()))
    mad_score = float(np.median(np.abs(v.to_numpy() - median_score)))
    fallback = (not np.isfinite(mad_score)) or (mad_score == 0.0)

    if not fallback:
        theta_high = median_score + float(high_k) * mad_score
        theta_low = median_score + float(low_k) * mad_score
        theta_high = float(np.clip(theta_high, min_high_clip, max_high_clip))
        theta_low = float(np.clip(theta_low, min_low_clip, theta_high - 0.02))
        invalid = (not np.isfinite(theta_high)) or (not np.isfinite(theta_low)) or (theta_low >= theta_high)
        fallback = invalid

    if fallback:
        theta_high = max(float(min_high_clip), float(np.quantile(v.to_numpy(), 0.75)))
        theta_low = max(float(min_low_clip), float(np.quantile(v.to_numpy(), 0.50)))
        theta_low = min(theta_low, theta_high - 0.02)

    # Final guarantee
    if not np.isfinite(theta_high):
        theta_high = float(min_high_clip)
    if not np.isfinite(theta_low):
        theta_low = float(min_low_clip)
    if theta_low >= theta_high:
        theta_high = max(theta_high, min_high_clip)
        theta_low = max(min_low_clip, theta_high - 0.02)
        if theta_low >= theta_high:
            theta_low = theta_high - 0.02
    return float(theta_high), float(theta_low)


def _expand_region_by_low(
    s: np.ndarray,
    t: pd.Series,
    start_idx: int,
    end_idx: int,
    theta_low: float,
    lookback_limit_min: int,
) -> tuple[int, int]:
    """Expand high-threshold region to low-threshold candidate region."""
    n = len(s)
    c0 = int(start_idx)
    c1 = int(end_idx)

    # backward expansion with lookback limit
    while c0 - 1 >= 0:
        prev = c0 - 1
        if not (np.isfinite(s[prev]) and s[prev] >= theta_low):
            break
        if (t.iloc[start_idx] - t.iloc[prev]).total_seconds() / 60.0 > float(lookback_limit_min):
            break
        c0 = prev

    # forward expansion while low-threshold holds
    while c1 + 1 < n:
        nxt = c1 + 1
        if not (np.isfinite(s[nxt]) and s[nxt] >= theta_low):
            break
        c1 = nxt

    return int(c0), int(c1)


def _compute_mean_shift_D(score_smooth: np.ndarray, seg_min: int) -> np.ndarray:
    """D(t)=mean(last L)-mean(previous L) with NaN-safe means."""
    s = np.asarray(score_smooth, dtype=float)
    n = len(s)
    L = int(seg_min)
    D = np.full(n, np.nan, dtype=float)
    if L <= 0:
        return D
    for i in range(2 * L, n):
        recent = s[i - L + 1 : i + 1]
        previous = s[i - 2 * L + 1 : i - L + 1]
        D[i] = np.nanmean(recent) - np.nanmean(previous)
    return D


def _compute_adaptive_D_thr(D: np.ndarray, mean_shift_k: float, min_D_thr: float) -> float:
    """Adaptive D threshold with MAD and fallback."""
    d_valid = D[np.isfinite(D)]
    if d_valid.size == 0:
        return float(min_D_thr)
    d_median = float(np.median(d_valid))
    d_mad = float(np.median(np.abs(d_valid - d_median)))
    if (not np.isfinite(d_mad)) or (d_mad == 0.0):
        return float(min_D_thr)
    d_thr = d_median + float(mean_shift_k) * d_mad
    if not np.isfinite(d_thr):
        return float(min_D_thr)
    return float(max(d_thr, min_D_thr))


def _merge_events_keep_higher_peak(events: list[dict[str, Any]], merge_within_minutes: int) -> list[dict[str, Any]]:
    """Merge close events by onset_time, keep the one with higher peak_score."""
    if not events:
        return []
    items = sorted(events, key=lambda r: pd.to_datetime(r["onset_time"]))
    merged = [items[0]]
    for cur in items[1:]:
        prev = merged[-1]
        dt_min = (pd.to_datetime(cur["onset_time"]) - pd.to_datetime(prev["onset_time"])).total_seconds() / 60.0
        if dt_min <= float(merge_within_minutes):
            prev_peak = float(prev.get("peak_score", float("-inf")))
            cur_peak = float(cur.get("peak_score", float("-inf")))
            if cur_peak > prev_peak:
                merged[-1] = cur
        else:
            merged.append(cur)
    return merged


def _merge_events_keep_higher_prominence_then_peak(
    events: list[dict[str, Any]], merge_within_minutes: int
) -> list[dict[str, Any]]:
    """Merge close events by onset_time; keep larger delta_S, tie-break by peak_score."""
    if not events:
        return []
    items = sorted(events, key=lambda r: pd.to_datetime(r["onset_time"]))
    merged = [items[0]]
    for cur in items[1:]:
        prev = merged[-1]
        dt_min = (pd.to_datetime(cur["onset_time"]) - pd.to_datetime(prev["onset_time"])).total_seconds() / 60.0
        if dt_min <= float(merge_within_minutes):
            prev_ds = float(prev.get("delta_S", float("-inf")))
            cur_ds = float(cur.get("delta_S", float("-inf")))
            prev_pk = float(prev.get("peak_score", float("-inf")))
            cur_pk = float(cur.get("peak_score", float("-inf")))
            if (cur_ds > prev_ds) or (cur_ds == prev_ds and cur_pk > prev_pk):
                merged[-1] = cur
        else:
            merged.append(cur)
    return merged


def detect_meals_adaptive_region_refine(
    df: pd.DataFrame,
    time_col: str = "ts",
    score_col: str = "fusion_score",
    smooth_win: int = 7,
    high_k: float = 2.0,
    low_k: float = 0.8,
    min_high_clip: float = 0.15,
    max_high_clip: float = 0.85,
    min_low_clip: float = 0.05,
    min_high_duration: int = 8,
    lookback_limit_min: int = 120,
    mean_shift_seg_min: int = 20,
    mean_shift_k: float = 1.5,
    min_D_thr: float = 0.03,
    window_len_min: int = 90,
    onset_backshift_min: int = 15,
    merge_within_minutes: int = 60,
) -> pd.DataFrame:
    """
    Adaptive region-based meal detection:
      1) slight smoothing
      2) adaptive high/low thresholds
      3) sustained high-score regions
      4) low-threshold expansion
      5) mean-shift D(t) + adaptive D threshold
      6) onset refinement with start-boundary rule
      7) merge nearby events by onset, keep higher peak score
    """
    cols = [
        "onset_idx", "onset_time", "onset_score",
        "peak_idx", "peak_time", "peak_score",
        "region_start_idx", "region_end_idx",
        "candidate_start_idx", "candidate_end_idx",
        "theta_low", "theta_high", "D_thr",
        "detection_reason", "is_boundary_event",
    ]
    if df.empty:
        return pd.DataFrame(columns=cols)

    d = df[[time_col, score_col]].copy()
    d[time_col] = pd.to_datetime(d[time_col], errors="coerce")
    d = d.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    if d.empty:
        return pd.DataFrame(columns=cols)

    raw_s = pd.to_numeric(d[score_col], errors="coerce")
    score_smooth = smooth_score_minimal(raw_s, win=int(smooth_win))
    s = pd.to_numeric(score_smooth, errors="coerce").to_numpy(dtype=float)
    t = d[time_col]
    n = len(d)

    valid = pd.Series(s).dropna()
    theta_high, theta_low = _calc_adaptive_thresholds(
        valid=valid,
        high_k=high_k,
        low_k=low_k,
        min_high_clip=min_high_clip,
        max_high_clip=max_high_clip,
        min_low_clip=min_low_clip,
    )

    high_mask = np.isfinite(s) & (s >= theta_high)
    high_regions = _find_contiguous_regions(high_mask)

    kept_regions: list[tuple[int, int, int, pd.Timestamp, float]] = []
    for rs, re in high_regions:
        dur_min = (t.iloc[re] - t.iloc[rs]).total_seconds() / 60.0 + 1.0
        if dur_min < float(min_high_duration):
            continue
        seg = s[rs : re + 1]
        if not np.any(np.isfinite(seg)):
            continue
        peak_rel = int(np.nanargmax(seg))
        peak_idx = rs + peak_rel
        kept_regions.append((rs, re, peak_idx, t.iloc[peak_idx], float(s[peak_idx])))

    D = _compute_mean_shift_D(s, seg_min=int(mean_shift_seg_min))
    D_thr = _compute_adaptive_D_thr(D, mean_shift_k=float(mean_shift_k), min_D_thr=float(min_D_thr))

    record_start = t.iloc[0]
    first_valid_score_time = record_start + pd.Timedelta(minutes=int(window_len_min))
    backshift = pd.Timedelta(minutes=int(onset_backshift_min))
    lookback_delta = pd.Timedelta(minutes=int(window_len_min))

    events: list[dict[str, Any]] = []
    for region_start_idx, region_end_idx, peak_idx, peak_time, peak_score in kept_regions:
        cand_start_idx, cand_end_idx = _expand_region_by_low(
            s=s,
            t=t,
            start_idx=region_start_idx,
            end_idx=region_end_idx,
            theta_low=theta_low,
            lookback_limit_min=int(lookback_limit_min),
        )

        lookback_start_time = peak_time - lookback_delta
        is_boundary_event = bool(lookback_start_time < first_valid_score_time)
        onset_time: pd.Timestamp | None = None
        detection_reason = ""

        if not is_boundary_event:
            mask_win = (t >= lookback_start_time) & (t <= peak_time)
            idxs = np.where(mask_win.to_numpy())[0]
            u_idx = None
            for u in idxs:
                if np.isfinite(D[u]) and D[u] >= D_thr and np.isfinite(s[u]) and s[u] >= theta_low:
                    u_idx = int(u)
                    break
            if u_idx is not None:
                onset_time = max(t.iloc[u_idx] - backshift, record_start)
                detection_reason = "mean_shift_onset"
            else:
                if 0 <= cand_start_idx < n:
                    onset_time = pd.to_datetime(t.iloc[cand_start_idx])
                    detection_reason = "low_threshold_fallback"
                else:
                    onset_time = max(peak_time - lookback_delta, record_start)
                    detection_reason = "lookback_fallback"
        else:
            mask_avail = (t >= first_valid_score_time) & (t <= peak_time)
            idxs = np.where(mask_avail.to_numpy())[0]
            u_idx = None
            for u in idxs:
                if np.isfinite(D[u]) and D[u] >= D_thr and np.isfinite(s[u]) and s[u] >= theta_low:
                    u_idx = int(u)
                    break
            if u_idx is not None:
                onset_time = max(t.iloc[u_idx] - backshift, record_start)
                detection_reason = "mean_shift_onset"
            else:
                onset_time = max(peak_time - lookback_delta, record_start)
                detection_reason = "start_boundary_fallback"

        if onset_time is None:
            continue

        onset_idx = int(np.searchsorted(t.to_numpy(dtype="datetime64[ns]"), np.datetime64(onset_time), side="left"))
        onset_idx = max(0, min(onset_idx, n - 1))
        onset_score = float(s[onset_idx]) if np.isfinite(s[onset_idx]) else float("nan")

        events.append(
            {
                "onset_idx": onset_idx,
                "onset_time": pd.to_datetime(onset_time),
                "onset_score": onset_score,
                "peak_idx": int(peak_idx),
                "peak_time": pd.to_datetime(peak_time),
                "peak_score": float(peak_score),
                "region_start_idx": int(region_start_idx),
                "region_end_idx": int(region_end_idx),
                "candidate_start_idx": int(cand_start_idx),
                "candidate_end_idx": int(cand_end_idx),
                "theta_low": float(theta_low),
                "theta_high": float(theta_high),
                "D_thr": float(D_thr),
                "detection_reason": detection_reason,
                "is_boundary_event": bool(is_boundary_event),
            }
        )

    merged = _merge_events_keep_higher_peak(events, merge_within_minutes=int(merge_within_minutes))
    if not merged:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(merged)
    out = out.sort_values("onset_time").reset_index(drop=True)
    return out.loc[:, [c for c in cols if c in out.columns]]


def _theta_D_scalar_from_pool(
    pool: np.ndarray,
    d_threshold_mode: str,
    d_k: float,
    min_D_thr: float,
    d_quantile: float,
) -> float:
    """Single θ_D from finite D values (mad+ MAD or quantile)."""
    p = np.asarray(pool, dtype=float)
    p = p[np.isfinite(p)]
    if p.size == 0:
        return float(min_D_thr)
    mode = str(d_threshold_mode).lower()
    if mode == "quantile":
        return float(max(np.quantile(p, float(d_quantile)), float(min_D_thr)))
    med = float(np.median(p))
    mad = float(np.median(np.abs(p - med)))
    if (not np.isfinite(mad)) or mad == 0.0:
        return float(max(np.quantile(p, float(d_quantile)), float(min_D_thr)))
    thr = med + float(d_k) * mad
    return float(max(thr, float(min_D_thr)))


def _build_theta_D_per_index(
    D: np.ndarray,
    t: pd.Series,
    theta_d_per_day: bool,
    d_threshold_mode: str,
    d_k: float,
    min_D_thr: float,
    d_quantile: float,
) -> np.ndarray:
    """θ_D at each index: per-patient constant or per-calendar-day."""
    n = len(D)
    out = np.full(n, np.nan, dtype=float)
    if not theta_d_per_day:
        thr = _theta_D_scalar_from_pool(D, d_threshold_mode, d_k, min_D_thr, d_quantile)
        out[:] = thr
        return out
    days = pd.to_datetime(t, errors="coerce").dt.date
    for d in pd.unique(days):
        mask = (days == d).to_numpy()
        pool = D[mask]
        thr = _theta_D_scalar_from_pool(pool, d_threshold_mode, d_k, min_D_thr, d_quantile)
        out[mask] = thr
    return out


def _prominence_delta_S_series(s: np.ndarray, lookback_prom_min: int) -> np.ndarray:
    """ΔS[j] = max(S[j-Lb:j+1]) - min(...) with Lb = lookback_prom_min (inclusive span)."""
    n = len(s)
    lb = int(lookback_prom_min)
    out = np.full(n, np.nan, dtype=float)
    for j in range(n):
        j0 = max(0, j - lb)
        seg = s[j0 : j + 1]
        if np.any(np.isfinite(seg)):
            out[j] = float(np.nanmax(seg) - np.nanmin(seg))
    return out


def _build_theta_prom_per_index(
    delta_S: np.ndarray,
    t: pd.Series,
    theta_d_per_day: bool,
    prominence_quantile: float,
    theta_prom_fixed: float,
) -> np.ndarray:
    """θ_prom for quantile75 mode: Q of delta_S pool per patient or per day."""
    n = len(delta_S)
    out = np.full(n, np.nan, dtype=float)
    pool_all = delta_S[np.isfinite(delta_S)]
    if pool_all.size == 0:
        out[:] = float(theta_prom_fixed)
        return out
    q = float(prominence_quantile)
    if not theta_d_per_day:
        thr = float(np.quantile(pool_all, q))
        out[:] = max(thr, 1e-9)
        return out
    days = pd.to_datetime(t, errors="coerce").dt.date
    for d in pd.unique(days):
        mask = (days == d).to_numpy()
        pool = delta_S[mask]
        pool = pool[np.isfinite(pool)]
        if pool.size == 0:
            thr = float(theta_prom_fixed)
        else:
            thr = float(np.quantile(pool, q))
        out[mask] = max(thr, 1e-9)
    return out


def detect_meals_adaptive_rise_based(
    df: pd.DataFrame,
    time_col: str = "ts",
    score_col: str = "fusion_score",
    smooth_win: int = 7,
    L: int = 20,
    theta_d_per_day: bool = False,
    d_threshold_mode: str = "mad",
    d_k: float = 1.5,
    min_D_thr: float = 0.03,
    d_quantile: float = 0.85,
    persist_min: int = 4,
    lookback_prom_min: int = 90,
    prominence_mode: str = "fixed",
    theta_prom_fixed: float = 0.13,
    prominence_quantile: float = 0.75,
    theta_floor: float | None = 0.15,
    rise_eps: float = 0.02,
    merge_within_minutes: int = 60,
) -> pd.DataFrame:
    """
    Adaptive rise-based meal detection (local mean-shift D(t), prominence, onset backtrack).
    See project plan: theta_d_per_day also gates prominence Q75 pooling (patient vs day).
    """
    cols = [
        "onset_idx",
        "onset_time",
        "onset_score",
        "peak_idx",
        "peak_time",
        "peak_score",
        "anchor_idx",
        "D_thr",
        "adaptive_scope",
        "delta_S",
        "theta_prom",
        "prominence_mode",
        "d_threshold_mode",
        "persist_min",
        "L",
        "d_k",
        "theta_floor_pass",
    ]
    if df.empty:
        return pd.DataFrame(columns=cols)

    d = df[[time_col, score_col]].copy()
    d[time_col] = pd.to_datetime(d[time_col], errors="coerce")
    d = d.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    if d.empty:
        return pd.DataFrame(columns=cols)

    raw_s = pd.to_numeric(d[score_col], errors="coerce")
    score_smooth = smooth_score_minimal(raw_s, win=int(smooth_win))
    s = pd.to_numeric(score_smooth, errors="coerce").to_numpy(dtype=float)
    t = d[time_col]
    n = len(s)
    record_start = t.iloc[0]

    D = _compute_mean_shift_D(s, seg_min=int(L))
    theta_D = _build_theta_D_per_index(
        D, t, bool(theta_d_per_day), d_threshold_mode, d_k, min_D_thr, d_quantile
    )

    trig = np.zeros(n, dtype=bool)
    for i in range(n):
        if np.isfinite(D[i]) and np.isfinite(theta_D[i]) and D[i] > theta_D[i]:
            trig[i] = True

    runs = _find_contiguous_regions(trig)
    anchors: list[int] = []
    pm = int(persist_min)
    for rs, re in runs:
        if re - rs + 1 >= pm:
            anchors.append(int(re))

    delta_S = _prominence_delta_S_series(s, lookback_prom_min)
    scope = "day" if theta_d_per_day else "patient"
    prom_mode = str(prominence_mode).lower()
    if prom_mode == "quantile75":
        theta_prom_arr = _build_theta_prom_per_index(
            delta_S, t, bool(theta_d_per_day), prominence_quantile, theta_prom_fixed
        )
    else:
        theta_prom_arr = np.full(n, float(theta_prom_fixed), dtype=float)

    events: list[dict[str, Any]] = []
    adaptive_scope = scope

    for anchor in anchors:
        i = int(anchor)
        w0 = max(0, i - int(lookback_prom_min))
        Wslice = slice(w0, i + 1)
        seg = s[Wslice]
        if not np.any(np.isfinite(seg)):
            continue
        d_thr_i = float(theta_D[i])
        delta_i = float(delta_S[i]) if np.isfinite(delta_S[i]) else float("nan")

        if prom_mode == "fixed":
            thr_prom = float(theta_prom_fixed)
        else:
            thr_prom = float(theta_prom_arr[i]) if np.isfinite(theta_prom_arr[i]) else float(theta_prom_fixed)

        if not np.isfinite(delta_i) or delta_i < thr_prom:
            continue

        peak_rel = int(np.nanargmax(seg))
        peak_idx = w0 + peak_rel
        peak_score = float(s[peak_idx]) if np.isfinite(s[peak_idx]) else float("nan")

        floor_ok = True
        if theta_floor is not None:
            floor_ok = np.isfinite(peak_score) and peak_score >= float(theta_floor)
        if not floor_ok:
            continue

        i_base_rel = int(np.nanargmin(seg))
        i_base = w0 + i_base_rel
        s_min = float(s[i_base]) if np.isfinite(s[i_base]) else float("nan")
        onset_idx = i_base
        if np.isfinite(s_min):
            thr_rise = s_min + float(rise_eps)
            for k in range(i_base, i + 1):
                if np.isfinite(s[k]) and s[k] >= thr_rise:
                    onset_idx = k
                    break

        onset_time = max(t.iloc[onset_idx], record_start)
        onset_score = float(s[onset_idx]) if np.isfinite(s[onset_idx]) else float("nan")

        events.append(
            {
                "onset_idx": int(onset_idx),
                "onset_time": pd.to_datetime(onset_time),
                "onset_score": onset_score,
                "peak_idx": int(peak_idx),
                "peak_time": pd.to_datetime(t.iloc[peak_idx]),
                "peak_score": peak_score,
                "anchor_idx": int(anchor),
                "D_thr": d_thr_i,
                "adaptive_scope": adaptive_scope,
                "delta_S": delta_i,
                "theta_prom": thr_prom,
                "prominence_mode": prom_mode,
                "d_threshold_mode": str(d_threshold_mode).lower(),
                "persist_min": pm,
                "L": int(L),
                "d_k": float(d_k),
                "theta_floor_pass": bool(floor_ok),
            }
        )

    merged = _merge_events_keep_higher_peak(events, merge_within_minutes=int(merge_within_minutes))
    if not merged:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(merged)
    out = out.sort_values("onset_time").reset_index(drop=True)
    return out.loc[:, cols]


adaptive_rise_based_method = detect_meals_adaptive_rise_based


def _valley_rel_idx_rightmost_plateau(seg: np.ndarray, atol: float = 1e-9) -> int:
    """
    Relative index in seg for slope_prominence valley: global minimum; if several points tie
    (plateau), use the rightmost so ΔT is shorter and rise_slope larger.
    """
    seg = np.asarray(seg, dtype=float)
    if seg.size == 0:
        return 0
    vmin = float(np.nanmin(seg))
    if not np.isfinite(vmin):
        return int(np.nanargmin(seg))
    mask = np.isfinite(seg) & (np.abs(seg - vmin) <= atol)
    idxs = np.where(mask)[0]
    if idxs.size == 0:
        return int(np.nanargmin(seg))
    return int(idxs.max())


def _slope_regular_peak_row(
    p: int,
    s: np.ndarray,
    t: pd.Series,
    lb: int,
    min_dt: float,
    theta_prom: float,
    theta_slope: float,
) -> dict[str, Any]:
    """Same geometry as ``detect_meals_slope_prominence`` for a local-max index ``p`` (backward window)."""
    nan_row: dict[str, Any] = {
        "valley_idx": float("nan"),
        "valley_time": pd.NaT,
        "delta_S": float("nan"),
        "delta_T_min": float("nan"),
        "rise_slope_raw": float("nan"),
        "rise_slope": float("nan"),
        "pass_threshold": False,
    }
    w0 = max(0, p - lb)
    seg = s[w0 : p + 1]
    if not np.any(np.isfinite(seg)):
        return nan_row
    v_rel = _valley_rel_idx_rightmost_plateau(seg)
    v = w0 + v_rel
    sp = float(s[p]) if np.isfinite(s[p]) else float("nan")
    sv = float(s[v]) if np.isfinite(s[v]) else float("nan")
    if not (np.isfinite(sp) and np.isfinite(sv)):
        return nan_row
    delta_S = sp - sv
    delta_T_min = (t.iloc[p] - t.iloc[v]).total_seconds() / 60.0
    if delta_T_min <= min_dt:
        out = dict(nan_row)
        out["valley_idx"] = int(v)
        out["valley_time"] = pd.to_datetime(t.iloc[v])
        out["delta_S"] = float(delta_S)
        out["delta_T_min"] = float(delta_T_min)
        return out
    rs_raw = float(delta_S / delta_T_min)
    rise_slope = round(rs_raw, 4)
    pass_threshold = (
        delta_S >= float(theta_prom)
        and rise_slope >= float(theta_slope)
    )
    return {
        "valley_idx": int(v),
        "valley_time": pd.to_datetime(t.iloc[v]),
        "delta_S": float(delta_S),
        "delta_T_min": float(delta_T_min),
        "rise_slope_raw": float(rs_raw),
        "rise_slope": float(rise_slope),
        "pass_threshold": bool(pass_threshold),
    }


def _slope_boundary_accept_metrics(
    s: np.ndarray,
    t: pd.Series,
    lb: int,
    min_dt: float,
    theta_prom: float,
    theta_slope: float,
) -> tuple[int | None, dict[str, Any] | None]:
    """
    Mirror boundary branch of ``detect_meals_slope_prominence``.
    Returns (i_pk, metrics) only when the boundary candidate passes all thresholds (same as skip index).
    """
    finite_s = np.isfinite(s)
    if not np.any(finite_s):
        return None, None
    i_first = int(np.argmax(finite_s))
    t_first_score = t.iloc[i_first]
    win_end_boundary = t_first_score + pd.Timedelta(minutes=lb)
    idx_first = np.where((t >= t_first_score).to_numpy() & (t <= win_end_boundary).to_numpy())[0]
    if idx_first.size == 0 or not np.any(np.isfinite(s[idx_first])):
        return None, None
    i_pk = int(idx_first[int(np.nanargmax(s[idx_first]))])
    t_pk = t.iloc[i_pk]
    idx_fwd = np.where((t >= t_pk).to_numpy() & (t <= t_pk + pd.Timedelta(minutes=lb)).to_numpy())[0]
    if idx_fwd.size == 0:
        return None, None
    segf = s[idx_fwd]
    rel_v = _valley_rel_idx_rightmost_plateau(segf)
    v = int(idx_fwd[rel_v])
    sp = float(s[i_pk]) if np.isfinite(s[i_pk]) else float("nan")
    sv = float(s[v]) if np.isfinite(s[v]) else float("nan")
    if not (np.isfinite(sp) and np.isfinite(sv)):
        return None, None
    delta_S = sp - sv
    delta_T_min = (t.iloc[v] - t.iloc[i_pk]).total_seconds() / 60.0
    if delta_T_min <= min_dt:
        return None, None
    rs_raw = float(delta_S / delta_T_min)
    rise_slope = round(rs_raw, 4)
    if not (
        delta_S >= float(theta_prom)
        and rise_slope >= float(theta_slope)
    ):
        return None, None
    return i_pk, {
        "valley_idx": v,
        "valley_time": pd.to_datetime(t.iloc[v]),
        "delta_S": float(delta_S),
        "delta_T_min": float(delta_T_min),
        "rise_slope_raw": float(rs_raw),
        "rise_slope": float(rise_slope),
        "pass_threshold": True,
    }


_SLOPE_LOCAL_MAXIMA_COLS = [
    "peak_idx",
    "peak_time",
    "peak_score_smooth",
    "peak_score_raw",
    "valley_idx",
    "valley_time",
    "delta_S",
    "delta_T_min",
    "rise_slope_raw",
    "rise_slope",
    "pass_threshold",
    "pass_in_final",
]


def export_slope_local_maxima_csv(
    df: pd.DataFrame,
    out_csv: str | Path,
    time_col: str = "ts",
    score_col: str = "fusion_score",
    smooth_win: int = 7,
    lookback_min: int = 90,
    theta_prom: float = 0.15,
    theta_slope: float = 0.002,
    min_rise_minutes: float = 1e-6,
    final_events: pd.DataFrame | None = None,
) -> None:
    """
    Smooth score, list all 1D local maxima (same pre-filter as ``detect_meals_slope_prominence``),
    plus per-peak prominence / slope (aligned with detector) and threshold pass; optional
    ``pass_in_final`` when ``final_events`` contains merged ``peak_idx`` from the detector output.
    """
    p_out = Path(out_csv)
    p_out.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        pd.DataFrame(columns=_SLOPE_LOCAL_MAXIMA_COLS).to_csv(p_out, index=False)
        return

    d = df[[time_col, score_col]].copy()
    d[time_col] = pd.to_datetime(d[time_col], errors="coerce")
    d = d.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    if d.empty:
        pd.DataFrame(columns=_SLOPE_LOCAL_MAXIMA_COLS).to_csv(p_out, index=False)
        return

    raw_s = pd.to_numeric(d[score_col], errors="coerce")
    score_smooth = smooth_score_minimal(raw_s, win=int(smooth_win))
    s = pd.to_numeric(score_smooth, errors="coerce").to_numpy(dtype=float)
    t = d[time_col]
    raw_arr = pd.to_numeric(raw_s, errors="coerce").to_numpy(dtype=float)
    maxima, _ = detect_extrema_1d(s)
    lb = int(lookback_min)
    min_dt = float(min_rise_minutes)

    final_peak: set[int] = set()
    if final_events is not None and not final_events.empty and "peak_idx" in final_events.columns:
        final_peak = set(pd.to_numeric(final_events["peak_idx"], errors="coerce").dropna().astype(int).tolist())

    skip_pk, boundary_m = _slope_boundary_accept_metrics(
        s, t, lb, min_dt, theta_prom, theta_slope
    )
    maxima_set = {int(p) for p in maxima}

    rows_lm: list[dict[str, Any]] = []
    for p in maxima:
        p = int(p)
        if skip_pk is not None and p == skip_pk and boundary_m is not None:
            m = dict(boundary_m)
        else:
            m = _slope_regular_peak_row(p, s, t, lb, min_dt, theta_prom, theta_slope)
        rows_lm.append(
            {
                "peak_idx": p,
                "peak_time": pd.to_datetime(t.iloc[p]),
                "peak_score_smooth": float(s[p]) if np.isfinite(s[p]) else float("nan"),
                "peak_score_raw": float(raw_arr[p]) if p < len(raw_arr) and np.isfinite(raw_arr[p]) else float("nan"),
                "valley_idx": m["valley_idx"],
                "valley_time": m["valley_time"],
                "delta_S": m["delta_S"],
                "delta_T_min": m["delta_T_min"],
                "rise_slope_raw": m["rise_slope_raw"],
                "rise_slope": m["rise_slope"],
                "pass_threshold": m["pass_threshold"],
                "pass_in_final": bool(p in final_peak),
            }
        )

    if skip_pk is not None and boundary_m is not None and skip_pk not in maxima_set:
        m = dict(boundary_m)
        p = int(skip_pk)
        rows_lm.append(
            {
                "peak_idx": p,
                "peak_time": pd.to_datetime(t.iloc[p]),
                "peak_score_smooth": float(s[p]) if np.isfinite(s[p]) else float("nan"),
                "peak_score_raw": float(raw_arr[p]) if p < len(raw_arr) and np.isfinite(raw_arr[p]) else float("nan"),
                "valley_idx": m["valley_idx"],
                "valley_time": m["valley_time"],
                "delta_S": m["delta_S"],
                "delta_T_min": m["delta_T_min"],
                "rise_slope_raw": m["rise_slope_raw"],
                "rise_slope": m["rise_slope"],
                "pass_threshold": m["pass_threshold"],
                "pass_in_final": bool(p in final_peak),
            }
        )

    rows_lm.sort(key=lambda r: int(r["peak_idx"]))
    lm_df = pd.DataFrame(rows_lm, columns=_SLOPE_LOCAL_MAXIMA_COLS)
    lm_df.to_csv(p_out, index=False)


def _write_slope_candidate_quantiles_csv(
    prominences: list[float],
    slopes: list[float],
    out_csv: str | Path,
) -> None:
    """One row per metric; columns n and q10..q90, including q75 for detector thresholding."""
    qlevels = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9)
    rows: list[dict[str, Any]] = []
    for metric, arr in (("prominence", prominences), ("slope", slopes)):
        a = np.asarray(arr, dtype=float)
        a = a[np.isfinite(a)]
        row: dict[str, Any] = {"metric": metric, "n": int(a.size)}
        for q in qlevels:
            pct = int(round(q * 100))
            row[f"q{pct}"] = float(np.quantile(a, q)) if a.size > 0 else float("nan")
        rows.append(row)
    p_out = Path(out_csv)
    p_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(p_out, index=False)


_SLOPE_CANDIDATE_COLS = [
    "peak_idx",
    "peak_time",
    "peak_score_smooth",
    "valley_idx",
    "valley_time",
    "valley_score_smooth",
    "delta_S",
    "delta_T_min",
    "rise_slope_raw",
    "rise_slope",
    "is_boundary",
    "block_start",
    "block_end",
]


def _write_slope_candidate_details_csv(rows: list[dict[str, Any]], out_csv: str | Path) -> None:
    p_out = Path(out_csv)
    p_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=_SLOPE_CANDIDATE_COLS).to_csv(p_out, index=False)


def _slope_peak_in_gap_block(
    p: int,
    t: pd.Series,
    lb: int,
    gap_blocks: list[tuple[pd.Timestamp, pd.Timestamp]] | None,
) -> tuple[bool, int]:
    """Time-based lookback: peak index p valid only if [t-lb, t] lies in one gap_block."""
    if gap_blocks is None:
        return True, max(0, int(p) - int(lb))
    t_pk = pd.Timestamp(t.iloc[p])
    t_lb = t_pk - pd.Timedelta(minutes=int(lb))
    for b0, b1 in gap_blocks:
        if t_pk >= b0 and t_pk <= b1 and t_lb >= b0:
            idx = np.where((t >= t_lb).to_numpy() & (t <= t_pk).to_numpy())[0]
            if idx.size < 2:
                return False, 0
            diffs = t.iloc[idx].diff().dropna()
            if not diffs.empty and (diffs > pd.Timedelta(minutes=1)).any():
                return False, 0
            return True, int(idx[0])
    return False, 0


def detect_meals_slope_prominence(
    df: pd.DataFrame,
    time_col: str = "ts",
    score_col: str = "fusion_score",
    smooth_win: int = 7,
    lookback_min: int = 90,
    theta_prom: float = 0.15,
    theta_slope: float = 0.002,
    min_rise_minutes: float = 1e-6,
    onset_offset_min: int = 0,
    merge_within_minutes: int = 60,
    candidate_quantiles_csv: str | Path | None = None,
    candidate_details_csv: str | Path | None = None,
    prominence_mode: str = "fixed",
    prominence_quantile: float = 0.75,
    quantile_fit_days: set | None = None,
    gap_blocks: list[tuple[pd.Timestamp, pd.Timestamp]] | None = None,
    boundary_blocks: list[tuple[pd.Timestamp, pd.Timestamp]] | None = None,
    smooth_blocks: list[tuple[pd.Timestamp, pd.Timestamp]] | None = None,
    collect_only: bool = False,
) -> pd.DataFrame:
    """
    Local-peak detector: backward lookback [p-lookback,p] with valley at window minimum, prominence,
    slope, optional onset offset; merge by peak score. No absolute peak-score floor. Valley uses **rightmost** index among ties
    at the minimum (plateau) for ΔT and slope.

    Boundary / t0 peak: first finite score time **t_first**; **global argmax** in
    [t_first, t_first + lookback_min]; forward [t_peak, t_peak + lookback_min] with same
    **rightmost-minimum** valley. Onset: max(t_peak - lookback_min - onset_offset, record_start).
    If accepted, skip the local-maximum iteration at the same peak index.

    If ``candidate_quantiles_csv`` is set, writes quantiles 0.1..0.9 of all candidate
    prominences (delta_S) and slopes (delta_S/delta_T_min) over boundary + every local max
    that reaches a valid slope (before threshold filter).

    ``prominence_mode="quantile"``: set ``theta_prom`` to the ``prominence_quantile``
    (default 0.75) of all candidate prominences for this patient/series; ``theta_prom``
    is used as fallback when there are no candidates.
    """
    cols = [
        "onset_idx",
        "onset_time",
        "onset_score",
        "peak_idx",
        "peak_time",
        "peak_score",
        "valley_idx",
        "valley_time",
        "delta_S",
        "delta_T_min",
        "rise_slope",
        "theta_prom",
        "theta_slope",
        "lookback_min",
        "onset_offset_min",
        "merge_within_minutes",
    ]
    if df.empty:
        if candidate_quantiles_csv is not None:
            _write_slope_candidate_quantiles_csv([], [], candidate_quantiles_csv)
        if candidate_details_csv is not None:
            _write_slope_candidate_details_csv([], candidate_details_csv)
        return pd.DataFrame(columns=_SLOPE_CANDIDATE_COLS if collect_only else cols)

    d = df[[time_col, score_col]].copy()
    d[time_col] = pd.to_datetime(d[time_col], errors="coerce")
    d = d.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    if d.empty:
        if candidate_quantiles_csv is not None:
            _write_slope_candidate_quantiles_csv([], [], candidate_quantiles_csv)
        if candidate_details_csv is not None:
            _write_slope_candidate_details_csv([], candidate_details_csv)
        return pd.DataFrame(columns=_SLOPE_CANDIDATE_COLS if collect_only else cols)

    raw_s = pd.to_numeric(d[score_col], errors="coerce")
    if smooth_blocks is None:
        score_smooth = smooth_score_minimal(raw_s, win=int(smooth_win))
    else:
        score_smooth = smooth_score_by_blocks(raw_s, d[time_col], smooth_blocks, win=int(smooth_win), preserve_nan=True)
    s = pd.to_numeric(score_smooth, errors="coerce").to_numpy(dtype=float)
    t = d[time_col]
    n = len(s)
    record_start = t.iloc[0]
    lb = int(lookback_min)
    offm = int(onset_offset_min)
    merge_m = int(merge_within_minutes)
    min_dt = float(min_rise_minutes)

    maxima, _ = detect_extrema_1d(s)
    events: list[dict[str, Any]] = []
    cand_prom: list[float] = []
    cand_slope: list[float] = []
    candidate_rows: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    skip_boundary_peak_idxs: set[int] = set()

    def _append_event(
        *,
        p: int,
        v: int,
        sp: float,
        delta_S: float,
        delta_T_min: float,
        rise_slope: float,
        onset_time: pd.Timestamp,
        theta_prom_used: float,
    ) -> None:
        onset_idx = int(
            np.searchsorted(
                t.to_numpy(dtype="datetime64[ns]"),
                np.datetime64(pd.Timestamp(onset_time)),
                side="left",
            )
        )
        onset_idx = max(0, min(onset_idx, n - 1))
        onset_score = float(s[onset_idx]) if np.isfinite(s[onset_idx]) else float("nan")
        events.append(
            {
                "onset_idx": onset_idx,
                "onset_time": pd.to_datetime(onset_time),
                "onset_score": onset_score,
                "peak_idx": int(p),
                "peak_time": pd.to_datetime(t.iloc[p]),
                "peak_score": float(sp),
                "valley_idx": int(v),
                "valley_time": pd.to_datetime(t.iloc[v]),
                "delta_S": float(delta_S),
                "delta_T_min": float(delta_T_min),
                "rise_slope": float(rise_slope),
                "theta_prom": float(theta_prom_used),
                "theta_slope": float(theta_slope),
                "lookback_min": lb,
                "onset_offset_min": offm,
                "merge_within_minutes": merge_m,
            }
        )

    def _maybe_record_candidate(
        p_idx: int,
        v_idx: int,
        delta_S: float,
        delta_T_min: float,
        rs_raw: float,
        *,
        is_boundary: bool,
        block_start: pd.Timestamp | None = None,
        block_end: pd.Timestamp | None = None,
    ) -> None:
        if quantile_fit_days is not None and pd.Timestamp(t.iloc[p_idx]).date() not in quantile_fit_days:
            return
        cand_prom.append(float(delta_S))
        cand_slope.append(rs_raw)
        candidate_rows.append(
            {
                "peak_idx": int(p_idx),
                "peak_time": pd.to_datetime(t.iloc[p_idx]),
                "peak_score_smooth": float(s[p_idx]) if np.isfinite(s[p_idx]) else float("nan"),
                "valley_idx": int(v_idx),
                "valley_time": pd.to_datetime(t.iloc[v_idx]),
                "valley_score_smooth": float(s[v_idx]) if np.isfinite(s[v_idx]) else float("nan"),
                "delta_S": float(delta_S),
                "delta_T_min": float(delta_T_min),
                "rise_slope_raw": float(rs_raw),
                "rise_slope": float(round(rs_raw, 4)),
                "is_boundary": bool(is_boundary),
                "block_start": pd.to_datetime(block_start) if block_start is not None else pd.NaT,
                "block_end": pd.to_datetime(block_end) if block_end is not None else pd.NaT,
            }
        )

    def _record_boundary_candidate(block_start: pd.Timestamp | None, block_end: pd.Timestamp | None) -> None:
        if block_start is None or block_end is None:
            block_mask = np.ones(n, dtype=bool)
            record_floor = record_start
        else:
            b0 = pd.Timestamp(block_start)
            b1 = pd.Timestamp(block_end)
            block_mask = ((t >= b0).to_numpy() & (t <= b1).to_numpy())
            record_floor = b0
        finite_s = np.isfinite(s) & block_mask
        if not np.any(finite_s):
            return
        i_first = int(np.where(finite_s)[0][0])
        t_first_score = t.iloc[i_first]
        win_end_boundary = t_first_score + pd.Timedelta(minutes=lb)
        if block_end is not None:
            win_end_boundary = min(win_end_boundary, pd.Timestamp(block_end))
        idx_first = np.where(
            (t >= t_first_score).to_numpy()
            & (t <= win_end_boundary).to_numpy()
            & block_mask
        )[0]
        if idx_first.size == 0 or not np.any(np.isfinite(s[idx_first])):
            return
        i_pk = int(idx_first[int(np.nanargmax(s[idx_first]))])
        t_pk = t.iloc[i_pk]
        fwd_end = t_pk + pd.Timedelta(minutes=lb)
        if block_end is not None:
            fwd_end = min(fwd_end, pd.Timestamp(block_end))
        idx_fwd = np.where((t >= t_pk).to_numpy() & (t <= fwd_end).to_numpy() & block_mask)[0]
        if idx_fwd.size > 0:
            segf = s[idx_fwd]
            rel_v = _valley_rel_idx_rightmost_plateau(segf)
            v = int(idx_fwd[rel_v])
            sp = float(s[i_pk]) if np.isfinite(s[i_pk]) else float("nan")
            sv = float(s[v]) if np.isfinite(s[v]) else float("nan")
            if np.isfinite(sp) and np.isfinite(sv):
                delta_S = sp - sv
                delta_T_min = (t.iloc[v] - t.iloc[i_pk]).total_seconds() / 60.0
                if delta_T_min > min_dt:
                    rs_raw = float(delta_S / delta_T_min)
                    _maybe_record_candidate(
                        i_pk,
                        v,
                        delta_S,
                        delta_T_min,
                        rs_raw,
                        is_boundary=True,
                        block_start=block_start,
                        block_end=block_end,
                    )
                    rise_slope = round(rs_raw, 4)
                    onset_raw = t.iloc[i_pk] - pd.Timedelta(minutes=lb) - pd.Timedelta(minutes=offm)
                    onset_time = max(onset_raw, record_floor)
                    pending.append(
                        {
                            "p": i_pk,
                            "v": v,
                            "sp": sp,
                            "delta_S": delta_S,
                            "delta_T_min": delta_T_min,
                            "rise_slope": rise_slope,
                            "onset_time": pd.to_datetime(onset_time),
                            "is_boundary": True,
                        }
                    )

    if boundary_blocks is None:
        # Original behavior: one boundary candidate for the whole series.
        _record_boundary_candidate(None, None)
    else:
        for b0, b1 in boundary_blocks:
            _record_boundary_candidate(pd.Timestamp(b0), pd.Timestamp(b1))

    for p in maxima:
        p = int(p)
        ok_lb, w0 = _slope_peak_in_gap_block(p, t, lb, gap_blocks)
        if not ok_lb:
            continue
        seg = s[w0 : p + 1]
        if not np.any(np.isfinite(seg)):
            continue
        v_rel = _valley_rel_idx_rightmost_plateau(seg)
        v = w0 + v_rel
        sp = float(s[p]) if np.isfinite(s[p]) else float("nan")
        sv = float(s[v]) if np.isfinite(s[v]) else float("nan")
        if not (np.isfinite(sp) and np.isfinite(sv)):
            continue
        delta_S = sp - sv
        delta_T_min = (t.iloc[p] - t.iloc[v]).total_seconds() / 60.0
        if delta_T_min <= min_dt:
            continue
        rs_raw = float(delta_S / delta_T_min)
        _maybe_record_candidate(p, v, delta_S, delta_T_min, rs_raw, is_boundary=False)
        rise_slope = round(rs_raw, 4)

        onset_raw = t.iloc[v] - pd.Timedelta(minutes=offm)
        # onset_raw = t.iloc[p] - pd.Timedelta(minutes=lb) - pd.Timedelta(minutes=offm)
        onset_time = max(onset_raw, record_start)
        pending.append(
            {
                "p": p,
                "v": v,
                "sp": sp,
                "delta_S": delta_S,
                "delta_T_min": delta_T_min,
                "rise_slope": rise_slope,
                "onset_time": pd.to_datetime(onset_time),
                "is_boundary": False,
            }
        )

    if candidate_quantiles_csv is not None:
        _write_slope_candidate_quantiles_csv(cand_prom, cand_slope, candidate_quantiles_csv)
    if candidate_details_csv is not None:
        _write_slope_candidate_details_csv(candidate_rows, candidate_details_csv)

    if collect_only:
        return pd.DataFrame(candidate_rows, columns=_SLOPE_CANDIDATE_COLS)

    theta_prom_effective = float(theta_prom)
    if str(prominence_mode).lower() == "quantile" and cand_prom:
        theta_prom_effective = float(np.quantile(cand_prom, float(prominence_quantile)))

    if not np.isfinite(theta_prom_effective) or theta_prom_effective < 0:
        raise ValueError("Prominence threshold must be finite and non-negative.")

    def _accept_pending(c: dict[str, Any]) -> None:
        sp = float(c["sp"])
        delta_S = float(c["delta_S"])
        rise_slope = float(c["rise_slope"])
        if delta_S < theta_prom_effective or rise_slope < float(theta_slope):
            return
        p = int(c["p"])
        if c.get("is_boundary"):
            skip_boundary_peak_idxs.add(p)
        _append_event(
            p=p,
            v=int(c["v"]),
            sp=sp,
            delta_S=delta_S,
            delta_T_min=float(c["delta_T_min"]),
            rise_slope=rise_slope,
            onset_time=pd.to_datetime(c["onset_time"]),
            theta_prom_used=theta_prom_effective,
        )

    for c in pending:
        if c.get("is_boundary"):
            _accept_pending(c)
    for c in pending:
        if c.get("is_boundary"):
            continue
        if int(c["p"]) in skip_boundary_peak_idxs:
            continue
        _accept_pending(c)

    # merged = _merge_events_keep_higher_prominence_then_peak(events, merge_within_minutes=merge_m)
    if gap_blocks is None:
        merged = _merge_events_keep_higher_peak(events, merge_within_minutes=merge_m)
    else:
        merged = []
        for b0, b1 in gap_blocks:
            segment_events = [e for e in events if pd.Timestamp(b0) <= pd.Timestamp(e["onset_time"]) <= pd.Timestamp(b1)]
            merged.extend(_merge_events_keep_higher_peak(segment_events, merge_within_minutes=merge_m))
    if not merged:
        out = pd.DataFrame(columns=cols)
        out.attrs["theta_prom"] = theta_prom_effective
        return out
    out = pd.DataFrame(merged)
    out = out.sort_values("onset_time").reset_index(drop=True)
    out.attrs["theta_prom"] = theta_prom_effective
    return out.loc[:, cols]


slope_prominence_method = detect_meals_slope_prominence

def collect_meal_candidates(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """One pre-acceptance prominence per peak, with boundary geometry priority."""
    for name in ("theta_prom", "prominence_mode", "candidate_quantiles_csv",
                 "candidate_details_csv", "collect_only"):
        if name in kwargs:
            raise ValueError(f"{name} is an acceptance/export setting, not a candidate-pool option.")
    kwargs.setdefault("time_col", "ts")
    kwargs.setdefault("score_col", "score")
    candidates = detect_meals_slope_prominence(df, collect_only=True, **kwargs)
    if not candidates.empty:
        candidates = candidates.loc[np.isfinite(candidates["delta_S"]) & candidates["delta_S"].gt(0)].copy()
    if candidates.empty:
        return candidates
    return (
        candidates.sort_values("is_boundary", ascending=False, kind="stable")
        .drop_duplicates("peak_idx", keep="first")
        .sort_values("peak_idx", kind="stable")
        .reset_index(drop=True)
    )
