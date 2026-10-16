import numpy as np
import pandas as pd
from typing import Dict, List

# -------------------- Common utilities --------------------

def _ensure_minutely_series(s: pd.Series) -> pd.Series:
    if not isinstance(s.index, pd.DatetimeIndex):
        raise ValueError("merged.index 必须是 DatetimeIndex")
    s = s.sort_index().copy()
    s = s.asfreq('T')
    s = s.interpolate(limit=5, limit_direction='both').ffill().bfill()
    return s


def insulin_activity_from_iob(iob: pd.Series, dt_min: float = 1.0) -> pd.Series:
    iob = _ensure_minutely_series(iob.fillna(0.0))
    ia = (-(iob.diff()) / dt_min).fillna(0.0)
    ia = ia.clip(lower=0.0)
    return ia

# -------------------- Bergman model --------------------

def bergman_fx(x: np.ndarray, dt: float, u: float,
               Gb_mmol: float = 100/18,
               p1: float = 0.028, p2: float = 0.025, p3: float = 5e-5,
               tau_min: float = 10) -> np.ndarray:
    G, X, Ra = x
    dG = -p1 * (G - Gb_mmol) - X * G + Ra
    dX = -p2 * X + p3 * u
    dRa = -Ra / tau_min
    G_new = G + dG * dt
    X_new = X + dX * dt
    Ra_new = Ra + dRa * dt
    return np.array([G_new, X_new, Ra_new], dtype=float)


def bergman_hx(x: np.ndarray) -> np.ndarray:
    return np.array([x[0]], dtype=float)

# -------------------- Hovorka simplified model --------------------

def hovorka_fx(x: np.ndarray, u: float, dt: float,
               Gb_mmol: float,
               k_g: float, k_x: float, k_ix: float, tau_cho: float) -> np.ndarray:
    G, x1, x2, x3, Ra = x
    dG  = -k_g * (G - Gb_mmol) - (x1 + x2 + x3) * G + Ra
    dx1 = -k_x * x1 + k_ix * u
    dx2 = -k_x * x2 + k_ix * u
    dx3 = -k_x * x3 + k_ix * u
    dRa = -Ra / tau_cho
    return np.array([
        G  + dG  * dt,
        x1 + dx1 * dt,
        x2 + dx2 * dt,
        x3 + dx3 * dt,
        Ra + dRa * dt
    ], dtype=float)


def hovorka_hx(x: np.ndarray) -> np.ndarray:
    return np.array([x[0]], dtype=float)

# -------------------- Numeric Jacobian --------------------

def _num_jacobian(f, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    n = x.size
    f0 = f(x)
    m = f0.size
    J = np.zeros((m, n), dtype=float)
    for i in range(n):
        dx = np.zeros_like(x)
        dx[i] = eps
        J[:, i] = (f(x + dx) - f0) / eps
    return J

# -------------------- EKF (hand) --------------------

def _run_ekf_hand(idx: pd.DatetimeIndex, z_series: pd.Series, u_series: pd.Series,
                  x0: np.ndarray, P0: np.ndarray, Q: np.ndarray, R: np.ndarray,
                  fx_builder, hx, dt_min: float,
                  ra_idx: int, g_idx: int,
                  ra_thresh: float, g_thresh: float,
                  gap_allow_min: int, min_contiguous_min: int) -> Dict[str, object]:
    x = x0.copy()
    P = P0.copy()
    I = np.eye(len(x0), dtype=float)

    pred_G = []
    ra_list = []
    extra_tracks: List[float] = []
    meal_flags: List[pd.Timestamp] = []

    for ts in idx:
        z = np.array([z_series.loc[ts]], dtype=float)
        u = float(u_series.loc[ts])
        f = fx_builder(u)
        x_pred = f(x)
        F = _num_jacobian(f, x)
        P_pred = F @ P @ F.T + Q

        H = _num_jacobian(lambda xvec: hx(xvec), x_pred)
        y = z - hx(x_pred)
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.pinv(S)
        x = x_pred + K @ y
        P = (I - K @ H) @ P_pred

        pred_G.append(float(x[g_idx]))
        ra_list.append(float(x[ra_idx]))
        # 对于Hovorka，额外返回x1+x2+x3作为动作强度
        if len(x) == 5:
            extra_tracks.append(float(x[1] + x[2] + x[3]))
        else:
            extra_tracks.append(float(x[1]))

        if (x[ra_idx] > ra_thresh) and (x[g_idx] > g_thresh):
            meal_flags.append(ts)

    # 连续段聚合
    if len(meal_flags) > 0:
        timestamps = pd.Series(pd.to_datetime(meal_flags)).sort_values().reset_index(drop=True)
        diff = timestamps.diff().fillna(pd.Timedelta(seconds=0))
        gap = diff > pd.Timedelta(minutes=gap_allow_min)
        group_id = gap.cumsum()
        keep = timestamps.groupby(group_id).filter(lambda x: len(x) >= int(min_contiguous_min))
        if len(keep) > 0:
            start_times = (keep.groupby((diff > pd.Timedelta(minutes=gap_allow_min)).cumsum()).first()
                              .sort_values().to_list())
        else:
            start_times = []
    else:
        start_times = []

    return dict(
        events_time=start_times,
        pred_G=pd.Series(pred_G, index=idx, name='predicted_glucose'),
        ra_series=pd.Series(ra_list, index=idx, name='ra_estimate'),
        extra_series=pd.Series(extra_tracks, index=idx, name='x_action_or_X'),
    )

# -------------------- UKF (package) --------------------
try:
    from filterpy.kalman import UnscentedKalmanFilter as _UKF, MerweScaledSigmaPoints as _MSP
except Exception:
    _UKF = None; _MSP = None


def _run_ukf_pkg(idx: pd.DatetimeIndex, z_series: pd.Series, u_series: pd.Series,
                 x0: np.ndarray, P0: np.ndarray, Q: np.ndarray, R: np.ndarray,
                 fx_builder, hx, dt_min: float,
                 ra_idx: int, g_idx: int,
                 ra_thresh: float, g_thresh: float,
                 gap_allow_min: int, min_contiguous_min: int) -> Dict[str, object]:
    if _UKF is None or _MSP is None:
        raise ImportError("需要安装filterpy: pip install filterpy")

    points = _MSP(n=len(x0), alpha=0.1, beta=2.0, kappa=1)
    ukf = _UKF(dim_x=len(x0), dim_z=1, dt=dt_min, fx=lambda x, dt: x, hx=hx, points=points)
    ukf.x = x0.copy()
    ukf.P = P0.copy()
    ukf.Q = Q.copy()
    ukf.R = R.copy()

    pred_G = []
    ra_list = []
    extra_tracks: List[float] = []
    meal_flags: List[pd.Timestamp] = []

    for ts in idx:
        z = np.array([z_series.loc[ts]], dtype=float)
        u = float(u_series.loc[ts])
        ukf.fx = lambda x, dt, _u=u: fx_builder(_u)(x)
        ukf.predict()
        ukf.update(z)
        pred_G.append(float(ukf.x[g_idx]))
        ra_list.append(float(ukf.x[ra_idx]))
        if len(ukf.x) == 5:
            extra_tracks.append(float(ukf.x[1] + ukf.x[2] + ukf.x[3]))
        else:
            extra_tracks.append(float(ukf.x[1]))
        if (ukf.x[ra_idx] > ra_thresh) and (ukf.x[g_idx] > g_thresh):
            meal_flags.append(ts)

    if len(meal_flags) > 0:
        timestamps = pd.Series(pd.to_datetime(meal_flags)).sort_values().reset_index(drop=True)
        diff = timestamps.diff().fillna(pd.Timedelta(seconds=0))
        gap = diff > pd.Timedelta(minutes=gap_allow_min)
        group_id = gap.cumsum()
        keep = timestamps.groupby(group_id).filter(lambda x: len(x) >= int(min_contiguous_min))
        if len(keep) > 0:
            start_times = (keep.groupby((diff > pd.Timedelta(minutes=gap_allow_min)).cumsum()).first()
                              .sort_values().to_list())
        else:
            start_times = []
    else:
        start_times = []

    return dict(
        events_time=start_times,
        pred_G=pd.Series(pred_G, index=idx, name='predicted_glucose'),
        ra_series=pd.Series(ra_list, index=idx, name='ra_estimate'),
        extra_series=pd.Series(extra_tracks, index=idx, name='x_action_or_X'),
    )

# -------------------- EKF (package) --------------------
try:
    from filterpy.kalman import ExtendedKalmanFilter as _EKF
except Exception:
    _EKF = None


def _run_ekf_pkg(idx: pd.DatetimeIndex, z_series: pd.Series, u_series: pd.Series,
                 x0: np.ndarray, P0: np.ndarray, Q: np.ndarray, R: np.ndarray,
                 fx_builder, hx, dt_min: float,
                 ra_idx: int, g_idx: int,
                 ra_thresh: float, g_thresh: float,
                 gap_allow_min: int, min_contiguous_min: int) -> Dict[str, object]:
    if _EKF is None:
        raise ImportError("需要安装filterpy: pip install filterpy")

    ekf = _EKF(dim_x=len(x0), dim_z=1)
    ekf.x = x0.copy()
    ekf.P = P0.copy()
    ekf.Q = Q.copy()
    ekf.R = R.copy()

    pred_G = []
    ra_list = []
    extra_tracks: List[float] = []
    meal_flags: List[pd.Timestamp] = []

    for ts in idx:
        z = np.array([z_series.loc[ts]], dtype=float)
        u = float(u_series.loc[ts])

        fx = lambda xvec, dt: fx_builder(u)(xvec)
        FJ = lambda xvec, dt: _num_jacobian(lambda x_: fx_builder(u)(x_), xvec)
        HJ = lambda xvec: _num_jacobian(lambda x_: hx(x_), xvec)
        
        # filterpy 的包里predict没有fx，所以需要手动非线性预测
        # ekf.predict(fx=fx, FJacobian=FJ, args=(dt_min,))
        # ekf.update(z, HJacobian=HJ, hx=lambda xvec: hx(xvec))
        
        # 手动非线性预测
        F = FJ(ekf.x, dt_min)              # 观测点的雅可比
        x_pred = fx(ekf.x, dt_min)         # 非线性前向传播
        ekf.x = x_pred
        ekf.P = F @ ekf.P @ F.T + ekf.Q

        # 非线性更新（注意 Hx 的参数名）
        ekf.update(z, HJacobian=HJ, Hx=lambda xvec: hx(xvec))
        pred_G.append(float(ekf.x[g_idx]))
        ra_list.append(float(ekf.x[ra_idx]))
        if len(ekf.x) == 5:
            extra_tracks.append(float(ekf.x[1] + ekf.x[2] + ekf.x[3]))
        else:
            extra_tracks.append(float(ekf.x[1]))
        if (ekf.x[ra_idx] > ra_thresh) and (ekf.x[g_idx] > g_thresh):
            meal_flags.append(ts)

    if len(meal_flags) > 0:
        timestamps = pd.Series(pd.to_datetime(meal_flags)).sort_values().reset_index(drop=True)
        diff = timestamps.diff().fillna(pd.Timedelta(seconds=0))
        gap = diff > pd.Timedelta(minutes=gap_allow_min)
        group_id = gap.cumsum()
        keep = timestamps.groupby(group_id).filter(lambda x: len(x) >= int(min_contiguous_min))
        if len(keep) > 0:
            start_times = (keep.groupby((diff > pd.Timedelta(minutes=gap_allow_min)).cumsum()).first()
                              .sort_values().to_list())
        else:
            start_times = []
    else:
        start_times = []

    return dict(
        events_time=start_times,
        pred_G=pd.Series(pred_G, index=idx, name='predicted_glucose'),
        ra_series=pd.Series(ra_list, index=idx, name='ra_estimate'),
        extra_series=pd.Series(extra_tracks, index=idx, name='x_action_or_X'),
    )

# -------------------- UKF (hand) --------------------

def _merwe_weights(n: int, alpha=0.1, beta=2.0, kappa=1.0):
    lam = alpha**2 * (n + kappa) - n
    c = n + lam
    Wm = np.full(2 * n + 1, 1.0 / (2.0 * c))
    Wc = Wm.copy()
    Wm[0] = lam / c
    Wc[0] = lam / c + (1 - alpha**2 + beta)
    return lam, Wm, Wc


def _run_ukf_hand(idx: pd.DatetimeIndex, z_series: pd.Series, u_series: pd.Series,
                  x0: np.ndarray, P0: np.ndarray, Q: np.ndarray, R: np.ndarray,
                  fx_builder, hx, dt_min: float,
                  ra_idx: int, g_idx: int,
                  ra_thresh: float, g_thresh: float,
                  gap_allow_min: int, min_contiguous_min: int,
                  alpha: float = 0.1, beta: float = 2.0, kappa: float = 1.0) -> Dict[str, object]:
    n = len(x0)
    x = x0.copy()
    P = P0.copy()

    pred_G = []
    ra_list = []
    extra_tracks: List[float] = []
    meal_flags: List[pd.Timestamp] = []

    for ts in idx:
        z = np.array([z_series.loc[ts]], dtype=float)
        u = float(u_series.loc[ts])
        fx = fx_builder(u)

        # Sigma points
        lam, Wm, Wc = _merwe_weights(n, alpha, beta, kappa)
        c = n + lam
        # 数值稳定性的小抖动
        try:
            S = np.linalg.cholesky((c * P) + 1e-9 * np.eye(n))
        except np.linalg.LinAlgError:
            S = np.linalg.cholesky((c * (P + 1e-6 * np.eye(n))))
        Xi = np.zeros((n, 2 * n + 1))
        Xi[:, 0] = x
        for i in range(n):
            Xi[:, i + 1]     = x + S[:, i]
            Xi[:, i + 1 + n] = x - S[:, i]

        # Predict
        Xip = np.zeros_like(Xi)
        for j in range(2 * n + 1):
            Xip[:, j] = fx(Xi[:, j])
        x_pred = Xip @ Wm
        P_pred = Q.copy()
        for j in range(2 * n + 1):
            d = (Xip[:, j] - x_pred).reshape(-1, 1)
            P_pred += Wc[j] * (d @ d.T)

        # Measurement prediction
        Zsig = np.zeros((1, 2 * n + 1))
        for j in range(2 * n + 1):
            Zsig[:, j] = hx(Xip[:, j])
        z_pred = Zsig @ Wm
        Szz = R.copy()
        for j in range(2 * n + 1):
            dz = (Zsig[:, j] - z_pred).reshape(1, 1)
            Szz += Wc[j] * (dz @ dz.T)
        Pxz = np.zeros((n, 1))
        for j in range(2 * n + 1):
            dx = (Xip[:, j] - x_pred).reshape(-1, 1)
            dz = (Zsig[:, j] - z_pred).reshape(1, 1)
            Pxz += Wc[j] * (dx @ dz)

        # Update
        K = Pxz @ np.linalg.pinv(Szz)
        x = x_pred + (K @ (z - z_pred)).flatten()
        P = P_pred - K @ Szz @ K.T

        pred_G.append(float(x[g_idx]))
        ra_list.append(float(x[ra_idx]))
        if n == 5:
            extra_tracks.append(float(x[1] + x[2] + x[3]))
        else:
            extra_tracks.append(float(x[1]))
        if (x[ra_idx] > ra_thresh) and (x[g_idx] > g_thresh):
            meal_flags.append(ts)

    # 连续段聚合
    if len(meal_flags) > 0:
        print(f"meal_flags length: {len(meal_flags)}")
        print(f"meal_flags: {meal_flags}")
        timestamps = pd.Series(pd.to_datetime(meal_flags)).sort_values().reset_index(drop=True)
        diff = timestamps.diff().fillna(pd.Timedelta(seconds=0))
        gap = diff > pd.Timedelta(minutes=gap_allow_min)
        group_id = gap.cumsum()
        keep = timestamps.groupby(group_id).filter(lambda x: len(x) >= int(min_contiguous_min))
        if len(keep) > 0:
            start_times = (keep.groupby((diff > pd.Timedelta(minutes=gap_allow_min)).cumsum()).first()
                              .sort_values().to_list())
        else:
            start_times = []
    else:
        start_times = []

    return dict(
        events_time=start_times,
        pred_G=pd.Series(pred_G, index=idx, name='predicted_glucose'),
        ra_series=pd.Series(ra_list, index=idx, name='ra_estimate'),
        extra_series=pd.Series(extra_tracks, index=idx, name='x_action_or_X'),
    )

# -------------------- Public router --------------------

def physio_kf_method(
    merged: pd.DataFrame,
    model: str = 'bergman',      # 'bergman' | 'hovorka'
    filter_type: str = 'ukf',    # 'ekf' | 'ukf'
    impl: str = 'package',          # 'hand' | 'package'
    value_col: str = 'value',
    iob_col: str = 'iob',
    dose_col: str = 'dose',
    insulin_input: str = 'iob_activity',
    u_scale: float = 1.0,
    dt_min: float = 1.0,
    # bergman
    Gb_mmol: float = 100/18,
    # p1: float = 0.028, p2: float = 0.025, p3: float = 5e-5, tau_min: float = 10,
    p1: float = 0.028, p2: float = 0.025, p3: float = 5e-5, tau_min: float = 40,
    # hovorka
    k_g: float = 0.02, k_x: float = 0.02, k_ix: float = 1e-3, tau_cho: float = 40.0,
    # noises
    Q_diag_bergman: List[float] = [1.0, 0.01, 0.5],
    Q_diag_hovorka: List[float] = [0.05, 1e-4, 1e-4, 1e-4, 0.01],
    R_scalar: float = 5.0,
    # decision
    ra_thresh_mmol_per_min: float = 3/18,
    g_thresh_mmol: float = 100/18,
    gap_allow_min: int = 1,
    min_contiguous_min: int = 10
) -> Dict[str, object]:
    if not isinstance(merged.index, pd.DatetimeIndex):
        raise ValueError("merged.index 必须是 DatetimeIndex")
    if value_col not in merged.columns:
        raise ValueError(f"缺少列: {value_col}")

    z_series = _ensure_minutely_series(merged[value_col].astype(float))
    idx = z_series.index

    # if insulin_input == 'dose' and dose_col in merged.columns:
    #     u_series = merged[dose_col].reindex(idx).fillna(0.0).astype(float)
    if insulin_input == 'iob_activity' and iob_col in merged.columns:
        u_series = insulin_activity_from_iob(merged[iob_col].reindex(idx).astype(float))
    else:
        u_series = pd.Series(0.0, index=idx)
    u_series = (u_series * float(u_scale)).astype(float)

    if model == 'bergman':
        x0 = np.array([z_series.iloc[0], 0.0, 0.0], dtype=float)
        P0 = np.eye(3, dtype=float) * 5.0
        Q = np.diag(Q_diag_bergman).astype(float)
        R = np.array([[R_scalar]], dtype=float)
        ra_idx, g_idx = 2, 0
        fx_builder = lambda u: (lambda x: bergman_fx(x, dt=dt_min, u=u, Gb_mmol=Gb_mmol, p1=p1, p2=p2, p3=p3, tau_min=tau_min))
        hx = bergman_hx
    elif model == 'hovorka':
        x0 = np.array([z_series.iloc[0], 0.0, 0.0, 0.0, 0.0], dtype=float)
        P0 = np.diag([5.0, 1.0, 1.0, 1.0, 1.0]).astype(float)
        Q = np.diag(Q_diag_hovorka).astype(float)
        R = np.array([[R_scalar]], dtype=float)
        ra_idx, g_idx = 4, 0
        fx_builder = lambda u: (lambda x: hovorka_fx(x, u=u, dt=dt_min, Gb_mmol=Gb_mmol, k_g=k_g, k_x=k_x, k_ix=k_ix, tau_cho=tau_cho))
        hx = hovorka_hx
    else:
        raise ValueError("model must be 'bergman' or 'hovorka'")

    if filter_type == 'ekf' and impl == 'hand':
        out = _run_ekf_hand(idx, z_series, u_series, x0, P0, Q, R, fx_builder, hx, dt_min,
                             ra_idx, g_idx, ra_thresh_mmol_per_min, g_thresh_mmol,
                             gap_allow_min, min_contiguous_min)
    elif filter_type == 'ukf' and impl == 'package':
        out = _run_ukf_pkg(idx, z_series, u_series, x0, P0, Q, R, fx_builder, hx, dt_min,
                           ra_idx, g_idx, ra_thresh_mmol_per_min, g_thresh_mmol,
                           gap_allow_min, min_contiguous_min)
    elif filter_type == 'ekf' and impl == 'package':
        out = _run_ekf_pkg(idx, z_series, u_series, x0, P0, Q, R, fx_builder, hx, dt_min,
                           ra_idx, g_idx, ra_thresh_mmol_per_min, g_thresh_mmol,
                           gap_allow_min, min_contiguous_min)
    elif filter_type == 'ukf' and impl == 'hand':
        out = _run_ukf_hand(idx, z_series, u_series, x0, P0, Q, R, fx_builder, hx, dt_min,
                            ra_idx, g_idx, ra_thresh_mmol_per_min, g_thresh_mmol,
                            gap_allow_min, min_contiguous_min)
    else:
        raise ValueError("支持组合: EKF(hand/package) 或 UKF(hand/package)")

    # 统一返回
    return {
        'events_time': out['events_time'],
        'times': idx,
        'predicted_glucose': out['pred_G'],
        'ra_series': out['ra_series'],
        'extra_series': out['extra_series'],
    }


__all__ = ['meal_kalman_method'] 