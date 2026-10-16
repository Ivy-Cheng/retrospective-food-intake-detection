"""Exact equal weighting for the manuscript sensitivity configuration."""

def equal_weight_score(frame):
    """Use thirds with physiology; retain the original 0.5/0.5 CGM fallback."""
    stat = frame["pos_grad_energy_normalized"]
    trend = frame["trend_auc_medtronic_pos_normalized"]
    phys = frame["bergman_ra_post_iob_rise_len_min_normalized"]
    available = frame["bergman_ra_post_iob_rise_auc_normalized"].notna()
    equal = stat * (1.0 / 3.0) + trend * (1.0 / 3.0) + phys.fillna(0.0) * (1.0 / 3.0)
    fallback = stat * 0.5 + trend * 0.5
    return equal.where(available, fallback)
