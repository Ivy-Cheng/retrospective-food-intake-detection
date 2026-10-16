"""Historical export membership, unrounded pooling and no peak-score floor."""
from unittest import TestCase
import numpy as np
import pandas as pd
from pipelines.fusion_refinement_new import detect_meals_slope_prominence
from pipelines.prominence_threshold import fit_dataset_prominence_threshold, detect_dataset_records


def frame(values):
    return pd.DataFrame({"ts": pd.date_range("2000-01-01", periods=len(values), freq="min"), "score": values})


class ThresholdTests(TestCase):
    def test_pools_rows_not_record_quantiles(self):
        exports = {"a": pd.DataFrame({"delta_S": [.1, .2, .3]}), "b": pd.DataFrame({"delta_S": [.7]})}
        metadata = fit_dataset_prominence_threshold(candidate_exports=exports)
        self.assertEqual(metadata["threshold"], float(np.quantile([.1, .2, .3, .7], .75)))
        self.assertEqual(metadata["candidate_count"], 4)
        self.assertEqual(metadata["rounding"], "none")

    def test_preserves_duplicate_and_zero_export_rows(self):
        exports = {"a": pd.DataFrame({"peak_idx": [1, 1, 2, 3], "delta_S": [0, .2, .2, .4], "pass_threshold": [False, False, True, True]})}
        metadata = fit_dataset_prominence_threshold(candidate_exports=exports)
        self.assertEqual(metadata["candidate_count"], 4)
        self.assertEqual(metadata["threshold"], float(np.quantile([0, .2, .2, .4], .75)))

    def test_only_nonfinite_rows_are_excluded(self):
        metadata = fit_dataset_prominence_threshold(candidate_exports={"a": pd.DataFrame({"delta_S": [.1, np.nan, np.inf, .4]})})
        self.assertEqual(metadata["candidate_count"], 2)

    def test_missing_exports_or_mismatched_dataset_fail(self):
        with self.assertRaises(ValueError):
            fit_dataset_prominence_threshold({"a": frame([0, .2, 0])})
        with self.assertRaises(ValueError):
            fit_dataset_prominence_threshold({"a": frame([0, .2, 0])}, candidate_exports={"b": pd.DataFrame({"delta_S": [.2]})})

    def test_sub_012_peak_can_pass_prominence(self):
        events = detect_meals_slope_prominence(frame([0, .1, 0, 0, 0]), score_col="score", smooth_win=1, lookback_min=2, theta_prom=.05, theta_slope=0)
        self.assertEqual(len(events), 1)
        self.assertLess(float(events["peak_score"].iloc[0]), .12)
        self.assertNotIn("theta_peak", events.columns)

    def test_shared_threshold_is_not_rounded(self):
        records = {"a": frame([0]*12+[.4]*9+[0]*12), "b": frame([0]*12+[.3]*9+[0]*12)}
        metadata = fit_dataset_prominence_threshold(candidate_exports={"calibration": pd.DataFrame({"delta_S": [.1, .2, .25, .377158798819998]})})
        outputs = detect_dataset_records(records, metadata)
        self.assertTrue(all(not value.empty for value in outputs.values()))
        for value in outputs.values():
            self.assertTrue((value["theta_prom"] == metadata["threshold"]).all())
