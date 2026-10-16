# Standard input contract (schema version 1)

Reproduction starts from already prepared standard CSV files and a versioned manifest, with no sibling-project or OneDrive dependency. Here, prepared means arranged in the required file format: CGM values are raw observations, and signal preprocessing is performed by this repository. NHS raw-format conversion, date shifting and reference assembly are upstream and are not required steps in the reader release.

| File | Columns | Units / role |
|---|---|---|
| Glucose | ts,value | Raw observed CGM, mmol/L; required |
| Bolus | ts,dose | Insulin units (U); optional, including an empty header-only file |
| References | ts | Already prepared onsets; optional, evaluation only |

Example manifest:

    {
      "schema_version": 1,
      "dataset_id": "my_dataset",
      "time_basis": "local_naive",
      "units": {"glucose": "mmol/L", "bolus": "U"},
      "records": [{
        "record_id": "p01_session1",
        "participant_id": "p01",
        "session_id": "session1",
        "monitoring_days": 1.0,
        "glucose_file": "records/p01_session1/glucose.csv",
        "bolus_file": "records/p01_session1/bolus.csv",
        "reference_file": "records/p01_session1/references.csv"
      }]
    }

Paths are relative to the manifest. Record IDs are unique and use letters, digits, underscores and hyphens. Participant IDs group sessions for summary statistics; absent participant IDs mean one participant per record. Omit the bolus or reference file if unavailable.

Timestamps are ISO 8601 local recording times without offsets, e.g. 2000-01-01T19:02:00. All files for a record share that clock. No timezone/DST conversion is inferred. Bolus times must be minute-aligned. CGM follows the original minute-flooring preprocessing. Numeric values are finite; doses are non-negative. Floats use round-trip CSV precision.

Raw CGM duplicates remain in the bundle and are handled by original preprocessing/averaging. Identical bolus doses at an identical timestamp collapse; differing doses at that timestamp are summed. Bolus alignment and IOB retain the original NHS behavior on prepared CGM timestamps.

References never enter features, IOB, scores or calibration. Positive monitoring_days is required when reference_file is supplied; it is the explicit FP/day denominator, not calendar-date count. NHS uses one day per approximately 24-hour session. Record-level sha256 mappings may declare checksums keyed by glucose_file, bolus_file and reference_file.

## Preprocessing

The included glucose_preprocessing module preserves the original day-wise interpolation, gap marking and >216-minute day exclusion. Processing exceptions now fail explicitly rather than silently omit a day. Remaining prepared-CGM gaps >30 minutes, including breaks left by removed days, separate processing segments. Fusion excludes overlapping 90-minute windows and respects masks in physiology, trends, smoothing, boundary detection and merging. Baseline processes retained segments independently with its unchanged 125-sample cubic filter; shorter segments are skipped.

## NHS bundle

The external NHS bundle, when supplied under datasets/nhs_v1, contains 31 sessions from 16 participants; 17/aspart is excluded. One constant integer-day shift per participant anchors the earliest date at 2000-01-01, preserving time of day, all time intervals, measurement/dose values and duplicate observations. Raw headers, identifying strings and original dates are omitted.

NHS references are the authoritative supplied files already combining reports and confirmed scheduled meals upstream. Conversion shifts their dates but does not infer or add events.

Private source checksums and date-shift offsets remain in local provenance outside the reader release. They are not required runtime inputs. Clinical sharing permission remains unresolved; the local bundle is Git-ignored and absent from a GitHub checkout.

## Shared full-fusion calibration

All fusion-derived methods share the unrounded full-fusion pooled Q75, without reference labels.

- NHS paper mode declares calibration.mode = historical_candidate_exports and calibration.manifest_file. Preserve all finite delta_S rows, including zeros/duplicates, without acceptance filtering. The complete 31-session pool (594 rows) is used even for a one-record smoke run. Candidate timestamps are omitted from the bundle.
- New datasets omit calibration. Full-fusion scores generate pre-acceptance candidate rows, including segment-start boundary rows. Their pooled Q75 is reused for all methods. This explicitly approved population is versioned as full_fusion_preacceptance_candidate_rows / auto_pool_version=1, and is not silently substituted for NHS historical exports.
- --threshold-file explicitly reuses a previously fitted calibration artifact.

The generated candidate rows and threshold provenance are saved in each run. Empty pools fail because Q75 is undefined. Baseline-only runs do not require calibration.
