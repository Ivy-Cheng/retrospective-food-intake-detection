# Running the code

## Validation scope

The standard reader and eight-configuration entry point are connected. Connectivity tests cover synthetic data and one NHS session. Full 31-session algorithm reruns and comparison with manuscript targets have NOT been performed in this stage.

Start with already prepared standard CSV observations and a manifest, as specified in data_format.md. Raw CGM signal preprocessing is performed by this code. NHS raw-format conversion, date shifting and reference assembly occur upstream; the local preparation tools are not distributed or required.

Original OneDrive inputs remain read-only. Authorized NHS users must supply the prepared external bundle at datasets/nhs_v1; it is Git-ignored pending sharing permission and absent from a GitHub checkout.

## Environment

Install requirements.txt. The tested current environment is Python 3.12.7, NumPy 1.26.4, pandas 2.2.3, SciPy 1.14.1, FilterPy 1.4.5, matplotlib 3.9.2 and Plotly 5.24.1. These are not asserted to be historical research versions. Clean-install verification remains a release check.

## Input checks and connectivity

    python reproduce.py --manifest datasets/nhs_v1/manifest.json --validate-inputs
    python reproduce.py --manifest datasets/nhs_v1/manifest.json --plan

    python examples/synthetic_demo.py --out-dir examples/generated/demo
    python reproduce.py --manifest examples/generated/demo/manifest.json --methods all --smoke --out-dir results/smoke_synthetic

    python reproduce.py --manifest datasets/nhs_v1/manifest.json --records nhs_01_fiasp --methods all --smoke --out-dir results/smoke_nhs

Input validation reads all declared files/checksums without running algorithms. Smoke execution is restricted to at most two records and labeled connectivity-only. Single-participant SDs are undefined; blank SD cells are expected. These sample tables are not manuscript targets.

## Full experiment entry point (available, not run in this stage)

    python reproduce.py --manifest datasets/nhs_v1/manifest.json --methods all --out-dir results/nhs_run

Methods: baseline, fusion, signal_only, physiology_only, dexcom, equal_weights, signal_leaning, physiology_leaning. --methods accepts any subset. --threshold-file reuses calibration; --no-figures skips plots. Use a new output directory or explicit --overwrite.

Full-fusion features are computed once per record and reused. Weights come from configs/primary.json and configs/sensitivity.json, including exact thirds. Every fusion-derived method shares full-fusion calibration. New data use the same input contract without NHS-specific raw files or labels.

run_experiments.py exposes the same interface. The legacy fusion_main.py also accepts --dataset-manifest; its NHS reader defaults to the local standard bundle.

## Outputs

- scores/<record_id>: prepared data/physiology, component features and trajectory.
- calibration: threshold metadata and automatic candidate exports for new datasets.
- events/<method>/<record_id>.csv: detections.
- evaluation: per-session/participant metrics, Table III/IV/VI data and onset figures, when labels exist.
- figures: a three-panel example; 16/aspart is selected when present, otherwise the first selected record.
- run_manifest.json: execution scope, records/methods, calibration, hashes and environment.

Missing reference labels skip evaluation explicitly. Reference events are never passed into detection. Matching remains closest-pair-first within inclusive +/-45 minutes. Counts pool across sessions within participant; group summaries use macro means and sample SD. Monitoring days come from the manifest.

## Preserved decisions and remaining checks

Numerical-gradient signal energy, trapezoidal trend AUC, longest uninterrupted post-IOB Ra rise, higher-peak-score merging, 15-minute G7 and shared masked processing are retained. Scheduled-reference preparation remains upstream.

Historical aggregate targets are excluded from the reader release because they have not been accepted against the newly connected full workflow. No final numerical acceptance target is supplied yet. Full numerical reproduction, final manuscript synchronization, clean-install checking and license/citation decisions remain separate release tasks. The clinician reference dataset and its separate local audit are outside this NHS runner.

    python reproduce.py --check-layout
    python -m unittest discover -s tests -p 'test_*.py'

Source and focused regression checks are distinct from paper-result reproduction.
