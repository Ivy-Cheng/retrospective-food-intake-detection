"""Versioned CSV inputs shared by NHS and new datasets. Never writes input files."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd

SCHEMA_VERSION = 1

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_manifest(path):
    path = Path(path).expanduser().resolve()
    data = json.loads(path.read_text())
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Expected dataset schema_version=1.")
    if data.get("time_basis") != "local_naive":
        raise ValueError("Declare time_basis='local_naive'; timestamps must share a local clock.")
    if data.get("units") != {"glucose": "mmol/L", "bolus": "U"}:
        raise ValueError("Required units: glucose mmol/L; bolus U.")
    records = data.get("records", [])
    if not records:
        raise ValueError("Dataset manifest contains no records.")
    seen = set()
    for entry in records:
        rid = entry.get("record_id", "")
        if not isinstance(rid, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", rid):
            raise ValueError("record_id must use letters, digits, '_' or '-'.")
        if rid in seen:
            raise ValueError(f"Duplicate record_id: {rid}")
        seen.add(rid)
        if not entry.get("glucose_file"):
            raise ValueError(f"{rid}: glucose_file is required.")
        if entry.get("reference_file"):
            days = entry.get("monitoring_days")
            if isinstance(days, bool) or not isinstance(days, (int, float)) or not np.isfinite(days) or days <= 0:
                raise ValueError(f"{rid}: declare positive monitoring_days for evaluation.")
    data["_manifest_path"] = path
    return data

def resolve_file(manifest, name):
    p = Path(name).expanduser()
    return p.resolve() if p.is_absolute() else (manifest["_manifest_path"].parent / p).resolve()

def read_table(manifest, record, key, columns, *, optional=False):
    filename = record.get(key)
    if not filename:
        if optional:
            return pd.DataFrame({c: pd.Series(dtype="datetime64[ns]" if c == "ts" else float) for c in columns})
        raise ValueError(f"{record['record_id']}: missing {key}")
    path = resolve_file(manifest, filename)
    digest = record.get("sha256", {}).get(key)
    if digest and sha256(path) != digest:
        raise ValueError(f"Checksum mismatch: {path.name}")
    frame = pd.read_csv(path, float_precision="round_trip")
    if any(c not in frame for c in columns):
        raise ValueError(f"{record['record_id']}/{key}: required columns {columns}.")
    frame = frame[list(columns)].copy()
    frame["ts"] = pd.to_datetime(frame["ts"], format="ISO8601", errors="raise")
    if isinstance(frame["ts"].dtype, pd.DatetimeTZDtype):
        raise ValueError("Use aligned local timestamps without timezone offsets, as declared.")
    if frame["ts"].isna().any():
        raise ValueError("Missing timestamp.")
    # Existing preprocessing uses minute resolution. Reject subminute boluses rather than silently losing them.
    if key == "bolus_file" and (frame.ts != frame.ts.dt.floor("min")).any():
        raise ValueError("Bolus timestamps must be aligned to minutes.")
    for c in columns:
        if c == "ts":
            continue
        frame[c] = pd.to_numeric(frame[c], errors="raise")
        if not np.isfinite(frame[c].to_numpy(dtype=float)).all():
            raise ValueError(f"{key}: non-finite {c}")
        if c == "dose" and frame[c].lt(0).any():
            raise ValueError("Bolus doses must be non-negative.")
    return frame.sort_values("ts", kind="stable").reset_index(drop=True)

def load_record(manifest, record):
    glucose = read_table(manifest, record, "glucose_file", ["ts", "value"])
    if len(glucose) < 2:
        raise ValueError(f"{record['record_id']}: at least two glucose observations are required.")
    bolus = read_table(manifest, record, "bolus_file", ["ts", "dose"], optional=True)
    references = read_table(manifest, record, "reference_file", ["ts"], optional=True)
    return {"glucose": glucose, "bolus": bolus, "references": references}

def prepare_record(inputs):
    """Use the included NHS preprocessing and unchanged trend/duplicate/IOB calculations."""
    from pipelines.glucose_preprocessing import process_glucose_by_date
    from fusion_based import remove_glucose_duplicates, remove_bolus_duplicates, full_trend_indicators, minute_iob
    glucose = process_glucose_by_date(inputs["glucose"].copy(), strict=True)
    if glucose.empty:
        raise ValueError("No glucose observations remain after preprocessing.")
    glucose = remove_glucose_duplicates(glucose)
    bolus = remove_bolus_duplicates(inputs["bolus"].copy())
    merged = full_trend_indicators(glucose).sort_index()
    merged.index.name = "ts"
    # Preserve the original NHS reader's exact-timestamp bolus alignment.
    merged["dose"] = bolus.set_index("ts")["dose"].reindex(merged.index)
    merged["iob"] = minute_iob(merged, dia_hours=4)
    # Reference food events are deliberately absent from detector inputs.
    merged["carbs"] = np.nan
    return merged

def validate_dataset(manifest):
    rows = []
    for record in manifest["records"]:
        inputs = load_record(manifest, record)
        rows.append({"record_id": record["record_id"], **{k: len(v) for k, v in inputs.items()}})
    return rows
