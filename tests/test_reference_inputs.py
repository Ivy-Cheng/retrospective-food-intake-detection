"""Check that supplied references are read without adding scheduled events."""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]


class Frame:
    """Minimal dataframe interface for testing reference IO without dependencies."""

    def __init__(self, times=None, has_ts=True):
        self.columns = ["ts"] if has_ts else []
        self.times = [] if times is None else times

    def __getitem__(self, key):
        if key != "ts":
            raise KeyError(key)
        return SimpleNamespace(tolist=lambda: list(self.times))


def isolated_reader(read_csv):
    # Load only the reader function, avoiding analysis/plotting import side effects.
    tree = ast.parse((ROOT / "evaluation_new.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "read_ground_truth")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[]))
    namespace = {"Path": Path, "pd": SimpleNamespace(read_csv=read_csv)}
    exec(compile(module, "<reference_reader_test>", "exec"), namespace)
    return namespace["read_ground_truth"]


class PreparedReferenceTests(TestCase):
    def test_reads_supplied_events_without_inserting_schedule(self):
        times = ["2000-01-01 18:57", "2000-01-01 21:23"]
        frame = Frame(times)
        read_csv = Mock(return_value=frame)
        actual, actual_frame = isolated_reader(read_csv)("01", "fiasp")
        self.assertEqual(actual, times)
        self.assertIs(actual_frame, frame)
        read_csv.assert_called_once_with(Path("results/evaluation/primary/ground_truth/ground_truth_01_fiasp.csv"))

    def test_accepts_explicit_prepared_reference_directory(self):
        read_csv = Mock(return_value=Frame([]))
        actual, _ = isolated_reader(read_csv)("17", "fiasp", reference_dir="external-references")
        self.assertEqual(actual, [])
        read_csv.assert_called_once_with(Path("external-references/ground_truth_17_fiasp.csv"))

    def test_missing_timestamp_column_is_rejected(self):
        read_csv = Mock(return_value=Frame(has_ts=False))
        with self.assertRaisesRegex(ValueError, "must contain a 'ts' column"):
            isolated_reader(read_csv)("01", "aspart")
