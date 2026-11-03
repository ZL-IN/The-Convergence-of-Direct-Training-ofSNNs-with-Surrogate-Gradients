# Validation record

Validated locally on 2026-11-04 with Python 3.12.14, PyTorch 2.7.1, NumPy 2.5.4,
torchvision 0.22.1, h5py 3.16.0 and Matplotlib 3.11.2 on macOS ARM64 / CPU.

- 41 pytest cases passed after the final source changes.
- Ruff static checks passed; source and tests formatted with Ruff.
- Editable package installation succeeded.
- Wheel and source distribution builds succeeded, including six preset JSON resources.
- Numerical smooth-gradient checks include recurrent resets and a two-layer network; singular
  zero margins in fractional-power interventions are not labeled as classical objective gradients.
- Monte Carlo tests cover normalized LocalZO means, nodewise variance reduction, and
  independent full-graph backward replication.
- Workflow tests cover training, same-checkpoint probes, delta sweeps, held-out evaluation,
  plot generation and versioned checkpoint round trips.
- A LocalZO training run is identical with and without the additional A/B/C probes under
  the tested CPU settings, verifying parameter/buffer/random-state isolation.
- The released SpikingJelly event-dataset constructor interfaces were inspected; complete
  event datasets were not downloaded or evaluated.

Not validated here: CUDA/Triton performance, distributed training (not implemented), complete
benchmark training, paper-score reproduction, equivalence to legacy checkpoints, or actual
GitHub-hosted CI execution. The CI workflow is configured for Python 3.11/3.12 on Linux CPU;
configuration is not a record of a remote CI run. Synthetic checks are not benchmark results.
