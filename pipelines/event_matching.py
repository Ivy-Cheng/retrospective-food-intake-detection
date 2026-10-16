"""One-to-one temporal matching used by evaluation and figure classification."""
from __future__ import annotations

from datetime import datetime
from math import isfinite
from typing import Sequence


def match_event_times(
    reference_times: Sequence[datetime],
    detection_times: Sequence[datetime],
    window_min: float = 45,
) -> list[tuple[int, int]]:
    """Greedily accept candidate pairs in ascending absolute time difference.

    The inclusive ±window_min criterion is applied before assignment. Each
    reference and detection index is used at most once. Callers supply
    chronologically sorted times; equal-distance ties use the reference index,
    then the detection index, so assignment is deterministic.

    Return (reference_index, detection_index) pairs in acceptance order.
    This implements the manuscript rule, not maximum-cardinality matching.
    """
    seconds = float(window_min) * 60
    if not isfinite(seconds) or seconds < 0:
        raise ValueError("window_min must be finite and non-negative")
    candidates = sorted(
        (abs((detection - reference).total_seconds()), gi, di)
        for gi, reference in enumerate(reference_times)
        for di, detection in enumerate(detection_times)
        if abs((detection - reference).total_seconds()) <= seconds
    )
    used_reference: set[int] = set()
    used_detection: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _, gi, di in candidates:
        if gi in used_reference or di in used_detection:
            continue
        used_reference.add(gi)
        used_detection.add(di)
        matches.append((gi, di))
    return matches
