#!/usr/bin/env python3
"""Run the paper configurations from a versioned standard CSV input manifest."""
from __future__ import annotations
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd
from pipelines.dataset_io import load_manifest, load_record, prepare_record, validate_dataset, sha256, resolve_file

ROOT = Path(__file__).resolve().parent
METHODS = ("baseline", "fusion", "signal_only", "physiology_only", "dexcom", "equal_weights", "signal_leaning", "physiology_leaning")
PRIMARY = METHODS[:4]
SYSTEM = {"baseline": "baseline", "fusion": "fusion", "signal_only": "signal_based_only", "physiology_only": "physio_based_only"}
PRIMARY_DEFAULTS = json.loads((ROOT / "configs/primary.json").read_text())
SENSITIVITY_DEFAULTS = json.loads((ROOT / "configs/sensitivity.json").read_text())
WEIGHTS = {"fusion": tuple(PRIMARY_DEFAULTS["methods"]["fusion"]["weights"])}
WEIGHTS.update({entry["name"]: tuple(entry["weights"]) for entry in SENSITIVITY_DEFAULTS["scenarios"]})
FALLBACK = tuple(PRIMARY_DEFAULTS["methods"]["fusion"]["fallback_weights"])


def numeric_json(value):
    if isinstance(value, dict):
        return {k: numeric_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [numeric_json(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    return value

def build_method_score(features, method):
    stat = features["pos_grad_energy_normalized"]
    trend = features["trend_auc_dexcom_pos_normalized" if method == "dexcom" else "trend_auc_medtronic_pos_normalized"]
    phys = features["bergman_ra_post_iob_rise_len_min_normalized"]
    if method == "signal_only":
        return (stat + trend) / 2.
    if method == "physiology_only":
        # Preserve the archived physiology-only convention.
        return phys.fillna(0.).where(features["pos_grad_energy"].notna())
    available = features["bergman_ra_post_iob_rise_auc_normalized"].notna()
    w1, w2, w3 = WEIGHTS[method]
    return stat * np.where(available, w1, FALLBACK[0]) + trend * np.where(available, w2, FALLBACK[1]) + phys.fillna(0.) * np.where(available, w3, 0.)

def fit_auto(records, contexts, output):
    from pipelines.prominence_threshold import fit_dataset_prominence_threshold, options_from_context
    from pipelines.fusion_refinement_new import detect_meals_slope_prominence
    exports = {}
    for rid, frame in records.items():
        candidates = detect_meals_slope_prominence(
            frame, score_col="score", theta_prom=0., theta_slope=0.,
            collect_only=True, **options_from_context(contexts[rid]))
        exports[rid] = candidates
        candidates.to_csv(output / f"{rid}_full_fusion_candidates.csv", index=False)
    result = fit_dataset_prominence_threshold(records, candidate_exports=exports)
    result.update(candidate_population="full_fusion_preacceptance_candidate_rows", auto_pool_version=1,
                  candidate_pool_rule="All recorded full-fusion candidates before acceptance, including segment-start boundary rows; retain duplicate/zero rows.")
    return result

def evaluate_records(events, inputs, entries):
    from pipelines.event_matching import match_event_times
    rows, errors = [], []
    for rid, by_method in events.items():
        entry = entries[rid]
        if not entry.get("reference_file"):
            continue
        refs = inputs[rid]["references"].ts.tolist()
        for method, frame in by_method.items():
            col = "tmeal" if method == "baseline" else "onset_time"
            detected = pd.to_datetime(frame[col]).sort_values().tolist()
            pairs = match_event_times(refs, detected, window_min=45)
            signed = [(detected[j] - refs[i]).total_seconds()/60 for i,j in pairs]
            rows.append({"record_id": rid, "patient_id": str(entry.get("participant_id", rid)),
                         "variant_label": str(entry.get("session_id", entry.get("variant", rid))),
                         "system": SYSTEM.get(method, method), "method": method,
                         "TP": len(pairs), "FP": len(detected)-len(pairs), "FN": len(refs)-len(pairs),
                         "n_days": float(entry["monitoring_days"]), "reference_events": len(refs),
                         "mean_abs_onset_min": float(np.mean(np.abs(signed))) if signed else np.nan,
                         "median_abs_onset_min": float(np.median(np.abs(signed))) if signed else np.nan})
            for value in signed:
                errors.append({"record_id": rid, "patient_id": str(entry.get("participant_id", rid)),
                               "system": SYSTEM.get(method, method), "method": method,
                               "signed_onset_min": value, "abs_onset_min": abs(value)})
    session = pd.DataFrame(rows)
    errors = pd.DataFrame(errors, columns=["record_id","patient_id","system","method","signed_onset_min","abs_onset_min"])
    participant, aggregate = [], []
    if session.empty:
        return session, pd.DataFrame(), pd.DataFrame(), errors
    for (pid, method), group in session.groupby(["patient_id","method"], sort=False):
        tp, fp, fn = (int(group[c].sum()) for c in ["TP","FP","FN"])
        days = float(group.n_days.sum())
        matched = errors[(errors.patient_id == pid)&(errors.method == method)]
        participant.append({"patient_id": pid, "system": SYSTEM.get(method, method), "method": method,
                            "TP": tp, "FP": fp, "FN": fn, "n_days": days,
                            "sensitivity": tp/(tp+fn) if tp+fn else np.nan, "fp_per_day": fp/days,
                            "mean_abs_onset_min": matched.abs_onset_min.mean(),
                            "median_abs_onset_min": matched.abs_onset_min.median()})
    participant = pd.DataFrame(participant)
    for method, group in participant.groupby("method", sort=False):
        aggregate.append({"system": SYSTEM.get(method, method), "method": method,
                          "sensitivity_mean": group.sensitivity.mean(), "sensitivity_std": group.sensitivity.std(ddof=1),
                          "fp_per_day_mean": group.fp_per_day.mean(), "fp_per_day_std": group.fp_per_day.std(ddof=1),
                          "n_participants": len(group)})
    return session, participant, pd.DataFrame(aggregate), errors

def write_evaluation(output, events, inputs, entries, make_figures):
    from summary_table import build_summary_table
    session, participant, aggregate, errors = evaluate_records(events, inputs, entries)
    output.mkdir(parents=True, exist_ok=True)
    session.to_csv(output/"evaluation_by_session.csv", index=False)
    if session.empty:
        (output/"NO_REFERENCES.txt").write_text("No reference_file was supplied. Event detection ran without evaluation labels.\n")
        return
    participant.to_csv(output/"evaluation_by_participant_all_methods.csv", index=False)
    aggregate.to_csv(output/"evaluation_summary_all_methods.csv", index=False)
    primary = output/"primary"
    primary.mkdir(exist_ok=True)
    participant[participant.method.isin(PRIMARY)].to_csv(primary/"evaluation_by_participant.csv", index=False)
    aggregate[aggregate.method.isin(PRIMARY)].to_csv(primary/"evaluation_summary_by_system.csv", index=False)
    for rid in events:
        selected = errors[(errors.record_id == rid)&errors.method.isin(PRIMARY)]
        selected.to_csv(primary/f"onset_errors_{rid}.csv", index=False)
    table3 = aggregate[aggregate.method.isin(PRIMARY)]
    table4 = aggregate[aggregate.method.isin(["fusion","dexcom","equal_weights","signal_leaning","physiology_leaning"])]
    table3.to_csv(output/"table_iii.csv", index=False)
    table4.to_csv(output/"table_iv.csv", index=False)
    if not participant[participant.method.isin(PRIMARY)].empty:
        summary = build_summary_table(primary)
        summary.to_csv(output/"table_vi.csv", index=False)
    if make_figures and not errors[errors.method.isin(PRIMARY)].empty:
        from result_plot import main as plot_main
        plot_main(["--primary-dir", str(primary), "--formats", "png", "--dpi", "150"])

def write_example_figure(output, entries, raw, cache_paths, features, events, contexts):
    import importlib.util
    import matplotlib.pyplot as plt
    from pipelines.simple_heuristic_utils import filter_cgm_data
    rid = "nhs_16_aspart" if "nhs_16_aspart" in entries else next(iter(entries))
    if not {"baseline","fusion"} <= set(events[rid]):
        return
    name = "_paper_three_panel_display"
    spec = importlib.util.spec_from_file_location(name, ROOT/"results-display.py")
    display = importlib.util.module_from_spec(spec)
    sys.modules[name] = display
    spec.loader.exec_module(display)
    merged = pd.read_csv(cache_paths[rid]["merged_csv"],parse_dates=["ts"],float_precision="round_trip").set_index("ts")
    smooth_parts = []
    blocks = contexts[rid].get("gap_ranges") or [(merged.index.min(),merged.index.max())]
    for a,b in blocks:
        part = merged.loc[pd.Timestamp(a):pd.Timestamp(b)].reset_index()[["ts","value"]].dropna()
        if len(part)>=125:
            smooth_parts.append(filter_cgm_data(part,part.ts.dt.date.unique(),window_length=125,polyorder=3,plot=False))
    if not smooth_parts:
        return
    smooth = pd.concat(smooth_parts,ignore_index=True)
    refs = raw[rid]["references"].ts.tolist()
    baseline, fusion = events[rid]["baseline"], events[rid]["fusion"]
    context = display.PatientContext(
        patient_id=str(entries[rid].get("participant_id",rid)),variant_label=str(entries[rid].get("session_id",rid)),
        merged=merged,smooth_cgm=smooth,reference_meals=refs,scores=features[rid],
        baseline_detections=baseline,fusion_detections=fusion,
        baseline_class=display.classify_detections(refs,pd.to_datetime(baseline.tmeal).tolist()),
        fusion_class=display.classify_detections(refs,pd.to_datetime(fusion.onset_time).tolist()))
    figure = display.build_three_panel_figure(context)
    target=output/"figures";target.mkdir(exist_ok=True)
    figure.savefig(target/f"example_{rid}.png",dpi=150,bbox_inches="tight")
    plt.close(figure)


def run(manifest_path, output, methods, selected_ids=None, threshold_file=None, smoke=False, overwrite=False, make_figures=True):
    from fusion_main import run_pipeline
    from pipelines.fusion_refinement_new import detect_meals_slope_prominence
    from pipelines.prominence_threshold import load_candidate_manifest, fit_dataset_prominence_threshold, read_dataset_threshold, options_from_context
    from pipelines.simple_heuristic_utils import simple_heuristic
    manifest = load_manifest(manifest_path)
    all_ids = {r["record_id"] for r in manifest["records"]}
    if selected_ids and not set(selected_ids) <= all_ids:
        raise ValueError(f"Unknown record IDs: {set(selected_ids)-all_ids}")
    records = [r for r in manifest["records"] if not selected_ids or r["record_id"] in selected_ids]
    if smoke and len(records) > 2:
        raise ValueError("Connectivity smoke tests are limited to at most two records; select --records explicitly.")
    output = Path(output).resolve()
    input_root = manifest["_manifest_path"].parent
    if output == input_root or input_root in output.parents:
        raise ValueError("Output must be outside the input dataset directory.")
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError("Output is not empty; choose a new --out-dir or explicitly use --overwrite.")
    output.mkdir(parents=True, exist_ok=True)
    (output/"RUN_SCOPE.txt").write_text("CONNECTIVITY TEST ONLY. Not paper result reproduction or validation.\n" if smoke else "Experiment execution only; no comparison with paper target values is performed.\n")
    entries = {r["record_id"]:r for r in records}
    raw, prepared, features, contexts, trajectories, cache_paths = {}, {}, {}, {}, {}, {}
    hashes = {str(manifest["_manifest_path"]): sha256(manifest["_manifest_path"])}
    for number, entry in enumerate(records, 1):
        rid = entry["record_id"]
        print(f"[{number}/{len(records)}] Preprocess and score {rid}", flush=True)
        raw[rid] = load_record(manifest, entry)
        for key in ["glucose_file","bolus_file","reference_file"]:
            if entry.get(key):
                path = resolve_file(manifest, entry[key]); hashes[str(path)] = sha256(path)
        prepared[rid] = prepare_record(raw[rid])
        if len(prepared[rid]) < 91:
            raise ValueError(f"{rid}: fewer than 91 prepared CGM points.")
        if methods == ("baseline",):
            continue
        paths = run_pipeline("nhs", rid, variant_label="input", merged_input=prepared[rid],
                             window_length_min=int(PRIMARY_DEFAULTS["window_length_min"]), step_min=int(PRIMARY_DEFAULTS["step_min"]),
                             out_dir=str(output/"scores"/rid), scores_only=True, plot_results=False)
        cache_paths[rid] = paths
        features[rid] = pd.read_csv(paths["scores_csv"], parse_dates=["start","end"], float_precision="round_trip")
        contexts[rid] = json.loads(Path(paths["context_json"]).read_text())
        trajectory = pd.read_csv(paths["trajectory_csv"], parse_dates=["ts"], float_precision="round_trip")
        trajectories[rid] = trajectory
    calibration_dir = output/"calibration"; calibration_dir.mkdir(exist_ok=True)
    if methods == ("baseline",):
        metadata = {"threshold": None, "candidate_population": "not_applicable_baseline_only"}
    elif threshold_file:
        metadata = read_dataset_threshold(threshold_file)
        hashes[str(Path(threshold_file).resolve())] = sha256(threshold_file)
    elif manifest.get("calibration", {}).get("mode") == "historical_candidate_exports":
        path = resolve_file(manifest, manifest["calibration"]["manifest_file"])
        exports = load_candidate_manifest(path)
        if set(exports) != all_ids:
            raise ValueError("Historical calibration records must match the full dataset manifest, including records outside a smoke subset.")
        metadata = fit_dataset_prominence_threshold(candidate_exports=exports)
        hashes[str(path)] = sha256(path)
        for entry in json.loads(path.read_text())["records"]:
            file = (path.parent/entry["candidate_file"]).resolve(); hashes[str(file)] = sha256(file)
    else:
        metadata = fit_auto(trajectories, contexts, calibration_dir)
    metadata = dict(metadata)
    metadata["application_record_ids"] = list(entries)
    (calibration_dir/"dataset_threshold.json").write_text(json.dumps(numeric_json(metadata), indent=2)+"\n", encoding="utf-8")
    theta = metadata["threshold"]
    all_events = {}
    for rid in entries:
        all_events[rid] = {}
        for method in methods:
            target = output/"events"/method; target.mkdir(parents=True, exist_ok=True)
            if method == "baseline":
                source = prepared[rid].reset_index()[["ts","value"]].dropna()
                detected = simple_heuristic(source, source.ts.dt.date.unique(), mask_gaps=True, plot=False)
            else:
                score = build_method_score(features[rid], method)
                aligned = pd.Series(score.to_numpy(), index=features[rid].end).reindex(prepared[rid].index)
                frame = pd.DataFrame({"ts": prepared[rid].index, "score": aligned.to_numpy()})
                detected = detect_meals_slope_prominence(frame, score_col="score", theta_prom=theta, theta_slope=0,
                                                        **options_from_context(contexts[rid]))
            all_events[rid][method] = detected
            detected.to_csv(target/f"{rid}.csv", index=False)
            print(f"{rid}: {method} completed", flush=True)
    write_evaluation(output/"evaluation", all_events, raw, entries, make_figures)
    if make_figures and methods != ("baseline",):
        write_example_figure(output,entries,raw,cache_paths,features,all_events,contexts)
    for path, before in hashes.items():
        if sha256(path) != before:
            raise RuntimeError(f"Input changed during execution: {path}")
    report = {"status":"passed", "validation_level":"connectivity_only" if smoke else "execution_only",
              "paper_result_validation":False, "records":list(entries), "methods":list(methods),
              "shared_threshold":theta, "candidate_population":metadata["candidate_population"],
              "input_hashes_unchanged":True, "input_sha256":hashes,
              "environment":{"python":sys.version.split()[0], "packages":{name:version(name) for name in ["numpy","pandas","scipy","filterpy","matplotlib","plotly"]}},
              "configuration_sha256":{name:sha256(ROOT/"configs"/name) for name in ["primary.json","sensitivity.json"]},
              "outputs":{"events":"events/", "evaluation":"evaluation/", "calibration":"calibration/"}}
    (output/"run_manifest.json").write_text(json.dumps(numeric_json(report), indent=2)+"\n")
    print("Completed. No comparison with manuscript results was performed.", flush=True)
    return report

def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=ROOT/"results/study_run")
    parser.add_argument("--methods", nargs="+", choices=["all",*METHODS], default=["all"])
    parser.add_argument("--records", nargs="+")
    parser.add_argument("--threshold-file", type=Path)
    parser.add_argument("--validate-inputs", action="store_true", help="Validate all declared CSV inputs/checksums without running algorithms.")
    parser.add_argument("--plan", action="store_true", help="Display selected workflow without running algorithms.")
    parser.add_argument("--smoke", action="store_true", help="Label this as connectivity-only; restrict execution to <=2 records.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    return parser

def main(argv=None):
    a = build_parser().parse_args(argv)
    data = load_manifest(a.manifest)
    methods = METHODS if "all" in a.methods else tuple(dict.fromkeys(a.methods))
    if a.validate_inputs:
        print(json.dumps({"records":validate_dataset(data)}, indent=2))
        return 0
    if a.plan:
        chosen = [r["record_id"] for r in data["records"] if not a.records or r["record_id"] in a.records]
        print(json.dumps({"records":chosen,"methods":methods,"calibration":data.get("calibration",{"mode":"automatic_full_fusion_preacceptance_Q75"}),"paper_result_validation":False},indent=2))
        return 0
    run(a.manifest,a.out_dir,methods,a.records,a.threshold_file,a.smoke,a.overwrite,not a.no_figures)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
