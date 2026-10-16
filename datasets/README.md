# Dataset inputs

Reproduction starts from already prepared standard CSV files and a manifest. See the [input contract](../docs/data_format.md). The same interface is used for NHS and new datasets. CGM observations are raw; the repository performs signal preprocessing.

For authorized NHS reproduction, supply the prepared bundle at nhs_v1/manifest.json. It contains 31 sessions of date-shifted observations, finalized reference events and historical calibration prominences. This bundle is not included in the GitHub checkout. Clinical contents remain Git-ignored pending sharing permission; do not force-add them.

NHS raw-format conversion, date shifting and reference assembly are upstream dataset preparation. The corresponding local tools and private provenance are not required or distributed in the reader release. Removing headers and shifting dates does not establish disclosure approval.

For other datasets, prepare CSVs and a manifest according to the input contract. To generate a shareable synthetic example, use examples/synthetic_demo.py.
