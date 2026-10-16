#!/usr/bin/env python3
"""Run standard-input experiments or inspect source integrity.

--manifest delegates to run_experiments.py. Source checks do not run analyses.
Full numerical paper reproduction remains a separate check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def check_layout() -> list[str]:
    """Check tracked repository hashes and the frozen cohort scaffold."""
    manifest = json.loads((ROOT / "docs/source_manifest.json").read_text())
    cohort = json.loads((ROOT / "configs/nhs_cohort.json").read_text())
    errors = []
    for entry in manifest["copied_files"] + manifest.get("additional_sources", []) + [manifest["extracted_preprocessing"]]:
        target = ROOT / entry["repository_path"]
        if not target.is_file():
            errors.append(f"Missing original file: {entry['repository_path']}")
        elif hashlib.sha256(target.read_bytes()).hexdigest() != (entry.get("repository_sha256") or entry["sha256"]):
            errors.append(f"Tracked repository file changed: {entry['repository_path']}")
    sessions = [(s["participant_id"], s["variant"]) for s in cohort["included_sessions"]]
    if len(sessions) != 31 or len(set(sessions)) != 31:
        errors.append("The cohort must contain 31 distinct included sessions.")
    if len({pid for pid, _ in sessions}) != 16:
        errors.append("The cohort must contain 16 participants.")
    if ("17", "aspart") in sessions:
        errors.append("Participant 17/aspart must be excluded.")
    for filename in (
        "README.md", "LICENSE", "CITATION.cff", "requirements.txt", ".gitignore",
        "configs/primary.json", "configs/sensitivity.json",
        "docs/reproduction.md", "docs/data_format.md",
        "docs/paper_result_mapping.md", "docs/clinician_evaluation_scope.md",
        "pipelines/glucose_preprocessing.py", "examples/synthetic_demo.py",
        "expected/aggregate_metrics.json", "tests/test_reproduction_checks.py",
        "results/.gitkeep", "plots/.gitkeep",
    ):
        if not (ROOT / filename).is_file():
            errors.append(f"Missing scaffold file: {filename}")
    return errors


def show_plan() -> None:
    print("Status: standard inputs and all eight experiment configurations are connected.")
    print("1. Input checks: python reproduce.py --manifest <manifest.json> --validate-inputs")
    print("2. Connectivity: add --records <id> --methods all --smoke --out-dir <new-output>")
    print("3. Full-cohort numerical reproduction, clean installation and release metadata remain pending.")
    print("Details: docs/reproduction.md. No analyses run under --show-plan.")


def main() -> int:
    if "--manifest" in sys.argv[1:]:
        from run_experiments import main as run_main
        return run_main(sys.argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--show-plan", action="store_true", help="Show pending reproduction stages.")
    group.add_argument("--check-layout", action="store_true", help="Verify scaffold and tracked source hashes.")
    parser.add_argument("--manifest", help="Run experiments from a standard CSV manifest; use run_experiments.py --help for run options.")
    args = parser.parse_args()
    if args.check_layout:
        errors = check_layout()
        if errors:
            for error in errors:
                print(error)
            return 1
        manifest = json.loads((ROOT / "docs/source_manifest.json").read_text())
        tracked = len(manifest["copied_files"]) + len(manifest.get("additional_sources", [])) + 1
        print(f"Layout valid: {tracked} tracked source modules; 16 participants; 31 sessions.")
        print("This checks the scaffold only, not scientific reproduction.")
        return 0
    show_plan()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
