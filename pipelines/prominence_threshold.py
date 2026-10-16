"""Dataset-wide, unrounded prominence calibration and cached-score inputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from pipelines.fusion_refinement_new import detect_meals_slope_prominence


def fit_dataset_prominence_threshold(
    records: Mapping[str, pd.DataFrame] | None = None,
    *,
    candidate_exports: Mapping[str, pd.DataFrame] | None = None,
    quantile: float = 0.75,
) -> dict:
    """Pool the historical candidate-export rows exactly, without redefining them.

    No acceptance filtering, deduplication, zero removal or boundary-geometry
    substitution is performed here. Only non-finite delta_S values are excluded,
    matching the original prominence_distribution.py calculation. Preserving
    the historical population requires supplying the corresponding exports.
    """
    if not candidate_exports:
        raise ValueError("Supply historical candidate exports; score trajectories alone do not identify the historical candidate population.")
    if records is not None and set(records) != set(candidate_exports):
        raise ValueError("Full-fusion score records and candidate exports must identify exactly the same dataset.")
    if not np.isfinite(quantile) or not 0 <= quantile <= 1:
        raise ValueError("quantile must be finite and in [0, 1].")
    pooled, counts, digests = [], {}, {}
    for record_id, frame in candidate_exports.items():
        if "delta_S" not in frame:
            raise ValueError(f"Missing delta_S column in {record_id}.")
        values = pd.to_numeric(frame["delta_S"], errors="coerce").to_numpy()
        values = values[np.isfinite(values)]
        pooled.extend(values.tolist())
        counts[record_id] = len(values)
        digests[record_id] = hashlib.sha256(
            pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()
        ).hexdigest()
    if not pooled:
        raise ValueError("No finite candidate prominences in the supplied exports.")
    threshold = float(np.quantile(np.asarray(pooled), quantile, method="linear"))
    if threshold < 0:
        raise ValueError("Historical exports produce a negative prominence threshold.")
    return {
        "schema_version": 2,
        "quantile": float(quantile),
        "quantile_method": "linear",
        "threshold": threshold,
        "rounding": "none",
        "candidate_population": "historical_candidate_export_rows",
        "candidate_pool_rule": "Preserve all exported rows; pool all finite delta_S values without further filtering or deduplication.",
        "candidate_count": len(pooled),
        "record_ids": list(candidate_exports),
        "candidates_per_record": counts,
        "candidate_frame_sha256": digests,
        "calibration_evidence": "full_fusion",
        "peak_score_floor": None,
    }


def load_candidate_manifest(path: str | Path) -> dict[str, pd.DataFrame]:
    """Read the historical candidate exports explicitly named by a manifest."""
    path = Path(path).resolve()
    entries = json.loads(path.read_text(encoding="utf-8")).get("records", [])
    exports = {}
    for entry in entries:
        record_id = entry["record_id"]
        if record_id in exports:
            raise ValueError(f"Duplicate candidate record_id: {record_id}")
        if entry.get("evidence", "full_fusion") != "full_fusion":
            raise ValueError("Calibrate from full-fusion candidate exports.")
        if "candidate_file" not in entry:
            raise ValueError(f"Provide candidate_file for {record_id}.")
        candidate = Path(entry["candidate_file"]).expanduser()
        if not candidate.is_absolute():
            candidate = path.parent / candidate
        if entry.get("sha256") and hashlib.sha256(candidate.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Candidate checksum mismatch: {candidate}")
        exports[record_id] = pd.read_csv(candidate, float_precision="round_trip")
    if not exports:
        raise ValueError("No candidate exports in the manifest.")
    return exports


def read_dataset_threshold(path: str | Path) -> dict:
    metadata = json.loads(Path(path).read_text(encoding="utf-8"))
    threshold = metadata.get("threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("Threshold metadata must contain a numeric threshold.")
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("The fitted prominence threshold must be finite and non-negative.")
    if metadata.get("quantile") != 0.75 or metadata.get("rounding") != "none":
        raise ValueError("Expected an unrounded 75th-percentile calibration.")
    population = metadata.get("candidate_population")
    if population == "full_fusion_preacceptance_candidate_rows":
        if metadata.get("auto_pool_version") != 1:
            raise ValueError("Automatic candidate pooling requires auto_pool_version=1.")
    elif population != "historical_candidate_export_rows":
        raise ValueError("Expected historical exports or the explicitly selected versioned automatic pool.")
    return metadata


def options_from_context(context: dict) -> dict:
    options = {}
    if context.get("pool_days") is not None:
        options["quantile_fit_days"] = {pd.Timestamp(d).date() for d in context["pool_days"]}
    if context.get("gap_ranges") is not None:
        blocks = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in context["gap_ranges"]]
        options.update(gap_blocks=blocks, boundary_blocks=blocks, smooth_blocks=blocks)
    return options


def load_score_manifest(path: str | Path) -> tuple[dict, dict, dict]:
    """Read only the explicit score records named by a dataset manifest.

    score_file can be a full time/score trajectory or an end-anchored window
    table with a time_grid_file naming the original full record index.
    """
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    records, contexts, entries = {}, {}, {}
    for entry in manifest.get("records", []):
        record_id = entry["record_id"]
        if record_id in records:
            raise ValueError(f"Duplicate record_id: {record_id}")
        def resolve(name):
            p = Path(entry[name]).expanduser()
            return p if p.is_absolute() else path.parent / p
        score = pd.read_csv(resolve("score_file"), float_precision="round_trip")
        time_col = entry.get("time_column", "ts")
        score_col = entry.get("score_column", "score")
        t = pd.DatetimeIndex(pd.to_datetime(score[time_col], errors="raise"))
        if t.hasnans or t.duplicated().any():
            raise ValueError(f"Missing/duplicate score timestamps in {record_id}.")
        values = pd.Series(pd.to_numeric(score[score_col], errors="raise").to_numpy(), index=t)
        if "time_grid_file" in entry:
            grid_col = entry.get("time_grid_column", "ts")
            grid = pd.read_csv(resolve("time_grid_file"), usecols=[grid_col])
            t = pd.DatetimeIndex(pd.to_datetime(grid[grid_col], errors="raise"))
            if t.hasnans or t.duplicated().any():
                raise ValueError(f"Missing/duplicate full-record timestamps in {record_id}.")
            # Reindex to the original grid without filling absent minutes.
            values = values.reindex(t)
        records[record_id] = pd.DataFrame({"ts": t, "score": values.to_numpy()}).sort_values("ts").reset_index(drop=True)
        contexts[record_id] = entry.get("context", {})
        entries[record_id] = entry
    if not records:
        raise ValueError("The dataset manifest has no score records.")
    return records, contexts, entries


def detect_dataset_records(records: Mapping[str, pd.DataFrame], metadata: dict, contexts: Mapping[str, dict] | None = None) -> dict:
    """Apply exactly one supplied threshold to every record, including ablations."""
    threshold = float(metadata["threshold"])
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("Invalid dataset prominence threshold.")
    outputs = {}
    for record_id, frame in records.items():
        context = (contexts or {}).get(record_id, {})
        events = detect_meals_slope_prominence(
            frame, time_col="ts", score_col="score", theta_prom=threshold,
            theta_slope=0, prominence_mode="fixed",
            **options_from_context(context),
        )
        if context.get("output_days") is not None and not events.empty:
            keep = {pd.Timestamp(d).date() for d in context["output_days"]}
            events = events.loc[pd.to_datetime(events["onset_time"]).dt.date.isin(keep)].copy()
        outputs[record_id] = events
    return outputs
