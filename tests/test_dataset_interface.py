"""Standard-input regressions, independent of clinical data."""
from pathlib import Path
import contextlib, io, json, tempfile, unittest
import numpy as np
import pandas as pd
from pipelines.dataset_io import load_manifest, load_record, prepare_record, sha256
from run_experiments import build_method_score, WEIGHTS

class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
        pd.DataFrame({"ts":["2000-01-01T00:00:00","2000-01-01T00:05:00"],"value":[6.,6.5]}).to_csv(self.root/"g.csv",index=False)
        self.data={"schema_version":1,"time_basis":"local_naive","units":{"glucose":"mmol/L","bolus":"U"},
                   "records":[{"record_id":"r1","glucose_file":"g.csv"}]}
    def tearDown(self):
        self.temp.cleanup()
    def manifest(self):
        p=self.root/"manifest.json";p.write_text(json.dumps(self.data));return load_manifest(p)
    def test_optional_bolus_and_labels(self):
        m=self.manifest();d=load_record(m,m["records"][0])
        self.assertEqual(len(d["glucose"]),2)
        self.assertTrue(d["bolus"].empty and d["references"].empty)
    def test_duplicate_ids_rejected(self):
        self.data["records"]*=2
        with self.assertRaisesRegex(ValueError,"Duplicate"): self.manifest()
    def test_units_must_be_declared(self):
        self.data["units"]["glucose"]="mg/dL"
        with self.assertRaisesRegex(ValueError,"units"): self.manifest()
    def test_changed_inputs_fail_checksum(self):
        self.data["records"][0]["sha256"]={"glucose_file":sha256(self.root/"g.csv")}
        m=self.manifest()
        with (self.root/"g.csv").open("a") as f: f.write("2000-01-01T00:10:00,7\n")
        with self.assertRaisesRegex(ValueError,"Checksum"): load_record(m,m["records"][0])
    def test_timezone_not_silently_converted(self):
        pd.DataFrame({"ts":["2000-01-01T00:00:00Z","2000-01-01T00:05:00Z"],"value":[6.,7.]}).to_csv(self.root/"g.csv",index=False)
        m=self.manifest()
        with self.assertRaisesRegex(ValueError,"local timestamps"): load_record(m,m["records"][0])
    def test_references_never_enter_detector_preparation(self):
        glucose=pd.DataFrame({"ts":pd.date_range("2000-01-01",periods=30,freq="5min"),"value":np.linspace(6,8,30)})
        empty=pd.DataFrame({"ts":pd.Series(dtype="datetime64[ns]"),"dose":pd.Series(dtype=float)})
        with contextlib.redirect_stdout(io.StringIO()):
            a=prepare_record({"glucose":glucose,"bolus":empty,"references":pd.DataFrame({"ts":[]})})
            b=prepare_record({"glucose":glucose,"bolus":empty,"references":pd.DataFrame({"ts":[pd.Timestamp("2000-01-01 01:00")]})})
        pd.testing.assert_frame_equal(a,b)


    def test_no_insulin_scoring_retains_signal_fallback(self):
        from fusion_main import run_pipeline
        glucose=pd.DataFrame({"ts":pd.date_range("2000-01-01",periods=61,freq="5min"),
                              "value":6+np.sin(np.arange(61)/6)})
        empty=pd.DataFrame({"ts":pd.Series(dtype="datetime64[ns]"),"dose":pd.Series(dtype=float)})
        with contextlib.redirect_stdout(io.StringIO()):
            merged=prepare_record({"glucose":glucose,"bolus":empty})
            paths=run_pipeline("nhs","unit",variant_label="no_insulin",merged_input=merged,
                               scores_only=True,plot_results=False,out_dir=str(self.root/"scored"))
        f=pd.read_csv(paths["scores_csv"],float_precision="round_trip")
        self.assertTrue(f.bergman_ra_post_iob_rise_auc_normalized.isna().all())
        np.testing.assert_allclose(f.score_6,(f.pos_grad_energy_normalized+f.trend_auc_medtronic_pos_normalized)/2,equal_nan=True)

class MethodTests(unittest.TestCase):
    def features(self):
        return pd.DataFrame({"pos_grad_energy":[1.,1.,np.nan],"pos_grad_energy_normalized":[1.,0.,np.nan],
            "trend_auc_medtronic_pos_normalized":[0.,1.,np.nan],"trend_auc_dexcom_pos_normalized":[1.,0.,np.nan],
            "bergman_ra_post_iob_rise_len_min_normalized":[0.,np.nan,np.nan],
            "bergman_ra_post_iob_rise_auc_normalized":[1.,np.nan,np.nan]})
    def test_exact_thirds_and_missing_physiology(self):
        out=build_method_score(self.features(),"equal_weights")
        self.assertEqual(WEIGHTS["equal_weights"],(1/3,1/3,1/3))
        self.assertEqual(out.iloc[0],1/3);self.assertEqual(out.iloc[1],.5)
        self.assertTrue(pd.isna(out.iloc[2]))
    def test_dexcom_uses_alternative_trend(self):
        self.assertAlmostEqual(build_method_score(self.features(),"dexcom").iloc[0],.6)
        self.assertAlmostEqual(build_method_score(self.features(),"fusion").iloc[0],.3)
    def test_physiology_only_does_not_unmask_invalid_windows(self):
        out=build_method_score(self.features(),"physiology_only")
        self.assertEqual(out.iloc[1],0.);self.assertTrue(pd.isna(out.iloc[2]))

if __name__=="__main__": unittest.main()
