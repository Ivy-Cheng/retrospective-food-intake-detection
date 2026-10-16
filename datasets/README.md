# Dataset inputs

See [input contract](../docs/data_format.md). The same manifest/CSV interface is used for NHS and new datasets.

The local nhs_v1 bundle contains 31 sessions of date-shifted raw observations, finalized reference events and historical calibration prominences. Clinical contents are Git-ignored pending sharing permission; do not force-add them. Removing headers and shifting dates does not establish disclosure approval. Private source paths and shift keys are kept only under the ignored audit directory.

Use prepare_nhs_dataset.py --help to stage a fresh bundle from read-only source roots; existing bundles are never overwritten. For a public synthetic input example, use examples/synthetic_demo.py.
