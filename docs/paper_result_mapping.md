# Paper-to-code mapping

| Paper output | Runnable generation | Validation status |
|---|---|---|
| Figure 1 | Framework diagram; editable source not located | Diagram source remains a separate artifact |
| Table I | Trend definitions in fusion_based.py | G7 15-minute definition connected |
| Table II | run_experiments.py method selection | Four primary methods connected |
| Table III | evaluation/table_iii.csv | Connectivity tested; full numerical reproduction pending |
| Table IV | evaluation/table_iv.csv | All trend/weight settings connected; full numerical reproduction pending |
| Table V | Separate clinician masked audit | Outside public NHS data scope |
| Figure 2 | figures/example_<record_id>.png | Three-panel function connected; 16/aspart preferred when present |
| Figure 3 | evaluation/primary/onset/ | Onset summaries/plots connected |
| Table VI | evaluation/table_vi.csv | Participant summary connected |

Use the standard manifest entry point in docs/reproduction.md. Smoke output tables and figures demonstrate connectivity and are not manuscript acceptance targets. Historical aggregate targets are local-only and excluded from the reader release. Final numerical acceptance targets remain pending full-cohort validation. Reproduction starts with the standard CSV/manifest bundle; source-format NHS dataset preparation is upstream.
