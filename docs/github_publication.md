# Initial private GitHub snapshot

## Scope

Upload the primary NHS method implementation and reusable dataset interface as a private development snapshot. Numerical validation remains limited to the connectivity checks described in connectivity_validation.md.

Include:
- Core Python modules and pipelines, preserving existing source hashes.
- reproduce.py, run_experiments.py and prepare_nhs_dataset.py.
- configs, tests and the synthetic input generator.
- README, requirements, source provenance and workflow documentation.
- Historical aggregate targets in expected, explicitly retained as archived targets.
- datasets/README.md and empty results/plots placeholders.
- Provisional LICENSE and CITATION.cff, with no open-source license inferred.

Keep local:
- All clinical data bundles, references and historical candidate rows under datasets.
- Generated events, scores, evaluation tables and figures under results/plots.
- Private audit/provenance/date-shift files.
- experiments: local clinician and feature comparison scripts outside the primary workflow.
- Generated synthetic CSVs, caches, editor history and credentials.

The NHS bundle remains excluded even from the private repository while sharing permission is unresolved.

## Upload procedure

1. Confirm the exact GitHub owner and repository name/URL.
2. Verify the destination is private. For an existing repository, inspect its contents before integrating.
3. Initialize local Git if the destination is a new empty repository; review the upload allowlist and staged diff.
4. Commit the reviewed source snapshot, configure the confirmed remote, and push.
5. Verify remote visibility, branch and uploaded tree; confirm excluded clinical/local files are absent.

No full-cohort algorithm rerun or paper-result comparison is required for this packaging step. A future public release needs the separate checks recorded in reproduction.md.

## Use after cloning

Install requirements.txt and generate the synthetic input with examples/synthetic_demo.py. Run it through reproduce.py as described in README.md. The cloned repository does not contain NHS observations or clinician reference data.

The local clinician comparison scripts mentioned in clinician_evaluation_scope.md are intentionally outside this initial upload.

The destination has not yet been confirmed; no remote upload has been performed.
