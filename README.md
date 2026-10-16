# JBHI retrospective food-intake detection

Code for Context-aware and Fusion-based Retrospective Food Intake Detection for Diabetes Datasets.

The standard CSV input interface supports the NHS study and new datasets. A unified entry point runs baseline, full fusion, both ablations and all four trend/weight sensitivity configurations.

**Current validation:** synthetic data and one NHS session have passed connectivity testing. Full-cohort numerical reproduction and comparison with manuscript targets have not been performed in this stage. See [validation record](docs/connectivity_validation.md).

## Quick start

    pip install -r requirements.txt
    python examples/synthetic_demo.py --out-dir examples/generated/demo
    python reproduce.py --manifest examples/generated/demo/manifest.json --methods all --smoke --out-dir results/demo

The synthetic example includes a masked gap and exercises preprocessing, physiology/features, calibration, detection, evaluation, tables and plots. Its metrics are not paper results.

## Local NHS inputs

The date-shifted standard bundle is datasets/nhs_v1/manifest.json. It contains raw CGM/bolus observations, separately supplied references and historical calibration prominences. Clinical files are Git-ignored until sharing permission is confirmed.

    python reproduce.py --manifest datasets/nhs_v1/manifest.json --validate-inputs
    python reproduce.py --manifest datasets/nhs_v1/manifest.json --records nhs_01_fiasp --methods all --smoke --out-dir results/nhs_smoke

To run all included records later, omit --records and --smoke. That full run has not been executed in this connectivity stage.

## New datasets

Provide glucose CSV (ts,value; mmol/L), optional bolus CSV (ts,dose; U) and a manifest. Optional references are evaluation-only. Without historical artifacts, full-fusion pre-acceptance candidates provide the pooled Q75 shared by all methods.

See [input contract](docs/data_format.md), [run instructions](docs/reproduction.md), [dataset staging](datasets/README.md), [paper mapping](docs/paper_result_mapping.md) and [source provenance](docs/source_manifest.json).

Original module names and numerical calculations are retained where possible. New adapters and orchestration provide the runnable interface. Generated outputs remain Git-ignored; --check-layout verifies source integrity, not paper metrics.

The public target is the NHS primary study and sensitivity analyses. The separate clinician annotation dataset is not distributed here. LICENSE and CITATION.cff remain provisional release items.
