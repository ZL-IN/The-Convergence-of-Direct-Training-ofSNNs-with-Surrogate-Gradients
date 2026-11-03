# Contributing

Install the development, data and plotting extras, then run `pytest -q` and
`ruff check src tests`. Build the wheel and source archive with `python -m build`.
Tests use explicit synthetic tensors and temporary dataset fixtures; they do not download
benchmarks or claim benchmark performance.

For scientific changes, describe the affected forward equation, derivative/reset convention,
sampling independence, and metric definition. Add a numerical or end-to-end regression test
that exercises the behavior. Distinguish local sample counts from full-gradient replication.
Report the seed and actual data split for new experiments; do not tune on the test split.

Do not commit datasets, checkpoints, access tokens, private paths, machine logs, or code
without a redistribution license. Issues should include the command, environment versions,
and a small reproducible example with private information removed.
