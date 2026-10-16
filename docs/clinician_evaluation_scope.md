# Clinician evaluation scope

The public repository targets NHS primary/ablation/sensitivity runs. The clinician annotation dataset belongs to separate conference work and remains external.

A local masked rerun of baseline and fusion has been completed in experiments/clinician_evaluation_masked.py with results under the Git-ignored results/clinician_evaluation_masked directory. This audit script is local-only and excluded from the initial GitHub upload. It uses private local inputs/paths; it is separate from the new standard NHS runner and does not make that dataset publicly reproducible.

Do not redistribute clinician references or imply that they are supplied by this repository. If extending public scope later, first declare a permissible input contract, configurable paths and data access terms. The metrics express clinician-reference agreement rather than objective intake accuracy.
