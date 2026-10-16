from __future__ import annotations

from dataclasses import dataclass
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
            onset_time = max(t_peak - lb, t_start)
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
import matplotlib.pyplot as plt


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