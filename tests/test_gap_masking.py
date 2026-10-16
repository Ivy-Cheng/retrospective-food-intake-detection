"""Regression checks for excluding missing-CGM intervals."""
import unittest
import numpy as np
import pandas as pd
from pipelines.gap_masking import prepared_gap_intervals, split_processing_blocks
from pipelines.fusion_scoring import attach_physio_kf_per_ranges, build_fusion_results, safe_minmax_fit
from pipelines.simple_heuristic_utils import simple_heuristic

class GapMaskTests(unittest.TestCase):
    def test_short_gap_boundary_and_removed_day_break(self):
        d=pd.DataFrame({"ts":pd.to_datetime(["2024-01-01 00:00","2024-01-01 00:30","2024-01-01 01:01","2024-01-03 00:00"]),"value":[5,5,6,7]})
        gaps=prepared_gap_intervals(d)
        self.assertEqual(gaps,[(d.ts.iloc[1],d.ts.iloc[2]),(d.ts.iloc[2],d.ts.iloc[3])])

    def test_physiology_and_windows_do_not_cross_mask(self):
        idx=pd.date_range("2024-01-01",periods=400,freq="min")
        m=pd.DataFrame({"value":6+np.sin(np.arange(400)/30),"dose":np.nan,"iob":0.},index=idx)
        m.loc[idx[200]:idx[239],"value"]=np.nan
        gaps=prepared_gap_intervals(m)
        blocks=split_processing_blocks([(idx[0],idx[-1])],gaps)
        ra=attach_physio_kf_per_ranges(m,blocks)
        self.assertTrue(ra.loc[idx[200]:idx[239],"bergman_ra_series"].isna().all())
        self.assertTrue(ra.loc[idx[240]:idx[259],"medtronic_20m"].isna().all())
        scores=build_fusion_results(ra,processing_blocks=blocks)
        crossed=(scores.start<idx[240])&(scores.end>idx[199])
        self.assertTrue(scores.loc[crossed,"score_6"].isna().all())
        self.assertTrue(scores.loc[~crossed,"score_6"].notna().all())

    def test_constant_normalization_keeps_missing_windows_missing(self):
        result=safe_minmax_fit(pd.Series([0.,np.nan,0.]))
        self.assertTrue(pd.isna(result.iloc[1]))
        self.assertEqual(result.iloc[0],0.)

    def test_baseline_segments_and_no_gap_control(self):
        t=pd.date_range("2024-01-01",periods=1000,freq="min")
        d=pd.DataFrame({"ts":t,"value":8+5*np.sin(np.arange(1000)/40)})
        days=d.ts.dt.date.unique()
        old=simple_heuristic(d,days,mask_gaps=False)
        new=simple_heuristic(d,days,mask_gaps=True)
        pd.testing.assert_frame_equal(old,new)
        self.assertGreater(len(new),0)
        sparse=d.drop(index=range(450,500))
        masked=simple_heuristic(sparse,days)
        self.assertGreater(len(masked),0)
        self.assertFalse(((masked.tmin<t[500])&(masked.tmeal>t[449])).any())

if __name__=="__main__":
    unittest.main()
