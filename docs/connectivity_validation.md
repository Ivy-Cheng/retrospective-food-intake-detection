# Connectivity validation — 2026-10-15

## Scope

The standardized raw-observation interface and unified experiment runner are connected. NHS source-format preparation was checked locally; its tools remain outside the reader release. This report records connectivity and input-integrity checks only. It does not establish reproduction of the manuscript's numerical results.

## Inputs and source protection

- Local NHS bundle: datasets/nhs_v1/manifest.json; 16 participants, 31 sessions.
- Preserved 9,827 raw glucose rows, 66 bolus rows and 113 prepared reference events; raw duplicate observations remain available to the original preprocessing.
- Preserved all 594 rows of the historical candidate-export calibration population.
- Dates were shifted by one constant integer-day offset per participant. Times of day, within-session intervals and inter-session spacing were preserved.
- Full-bundle checks confirmed measurement/dose values, reference events and calibration values against their source files, accounting for the date shift.
- All 93 source-file hashes were unchanged after preparation and validation.
- Identifying raw headers were omitted. Private provenance, source paths and date-shift keys are retained only in the Git-ignored audit directory.
- Clinical bundle files remain Git-ignored pending data-sharing permission. Date shifting is not a determination that public release is authorized.

## Executed checks

| Check | Result |
| --- | --- |
| Standard input/schema/checksum validation | All 31 NHS sessions passed; no cohort-wide algorithm run |
| Focused unit/regression tests | 29 passed |
| Source/cohort integrity check | 26 tracked source modules; 16 participants / 31 sessions passed |
| Synthetic end-to-end connectivity | One record, all eight configurations passed |
| NHS end-to-end connectivity | nhs_01_fiasp only, all eight configurations passed |
| Output generation | Event CSVs, evaluation summaries, Table III/IV/VI data, onset plots and example three-panel figures created |
| Input immutability during both runs | Passed |
| Example NHS figure inspection | Readable; generated successfully |

Eight configurations: baseline, full fusion, signal-only, physiology-only, Dexcom G7, equal weights, signal-leaning and physiology-leaning.

The synthetic run used automatic full-fusion pre-acceptance candidate collection, including boundary rows. The NHS run used the complete frozen historical pool. All fusion-derived configurations shared their respective full-fusion Q75 threshold.

## Local evidence

- results/connectivity_synthetic/run_manifest.json
- results/connectivity_nhs/run_manifest.json
- audit/nhs_preparation_provenance.json

Both run manifests record status=passed, validation_level=connectivity_only and paper_result_validation=false. Generated files and private audit evidence remain Git-ignored.

## Reader-release packaging update — 2026-10-16

The reader release starts from standard CSV files and a manifest. Two local NHS preparation/parser modules are no longer part of source integrity checks, leaving 24 tracked source modules. The 26-module count above records the original validation snapshot. Local historical targets and the empty test placeholder are no longer required. These packaging changes do not change detection calculations.

## Remaining release work

A full 31-session algorithm run, comparison against manuscript tables/figures and freezing of accepted numerical targets have not been performed in this stage, as requested. Clean-environment installation and final license/citation metadata remain separate release checks. The clinician adjudication dataset is outside this NHS workflow.

See reproduction.md for commands and data_format.md for the reusable input contract.
