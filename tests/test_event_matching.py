"""Matching regressions for the manuscript's globally ordered greedy rule."""
from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest import TestCase

import evaluation_new
import copy_evaluation
from pipelines.event_matching import match_event_times

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("figure_display_for_matching_test", ROOT / "results-display.py")
display = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = display
spec.loader.exec_module(display)


def times(*minutes):
    start = datetime(2000, 1, 1)
    return [start + timedelta(minutes=m) for m in minutes]


class MatchingTests(TestCase):
    def test_closest_pair_precedes_earliest_reference(self):
        self.assertEqual(match_event_times(times(0, 10), times(9, 30)), [(1, 0), (0, 1)])

    def test_one_detection_is_not_reused(self):
        self.assertEqual(match_event_times(times(0, 10), times(9)), [(1, 0)])

    def test_inclusive_window_and_outside_candidate(self):
        self.assertEqual(match_event_times(times(0, 200), times(-45, 245, 246)), [(0, 0), (1, 1)])

    def test_equal_distance_ties_are_deterministic(self):
        self.assertEqual(match_event_times(times(0), times(-10, 10)), [(0, 0)])

    def test_empty_inputs(self):
        self.assertEqual(match_event_times([], times(0)), [])
        self.assertEqual(match_event_times(times(0), []), [])

    def test_evaluators_and_figure_use_the_same_assignment(self):
        reference = times(0, 10)
        detection = times(9)
        for module in (evaluation_new, copy_evaluation):
            result = module.evaluate(reference, detection)
            self.assertEqual((result["TP"], result["FP"], result["FN"]), (1, 0, 1))
            self.assertEqual(result["onset_difference_min"], [-1])
        classified = display.classify_detections(reference, detection)
        self.assertEqual(classified.tp, detection)
        self.assertEqual(classified.fp, [])
        self.assertEqual(classified.fn, times(0))
