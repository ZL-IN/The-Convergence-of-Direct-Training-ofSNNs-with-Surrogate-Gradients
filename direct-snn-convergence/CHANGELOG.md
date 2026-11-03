# Changelog

## 1.0.0 — 2026-11-04

- Independently implement the ATan/LocalZO runtime, recurrent LIF, reference architectures,
  task adapters, training, checkpointing, same-checkpoint diagnostics and plotting.
- Separate local derivative sample count from complete-gradient replication count.
- Use complete reset derivatives by default; label detached/identity-gradient comparisons.
- Keep continuous smooth activations without binary compression.
- Preserve fixed gradient coordinates and square the dataset-weighted mean discrepancy.
- Isolate probes from training state and random streams.
- Add actual compute metadata, strict checkpoint schemas, synthetic-data labels, numerical
  regression tests, installable packaging and CPU CI configuration.
- Document changes from anonymous supplementary experiments and the remaining benchmark
  reproduction work. Historical results and third-party framework code are not bundled.
