# Understanding the Convergence of Direct Training ofSNNs with Surrogate Gradients

# Direct SNN Convergence

Independent PyTorch reference implementation for **Understanding the Convergence of Direct
Training of SNNs with Surrogate Gradients** (NeurIPS 2026).

This release implements hard-forward surrogate gradients (SG), normalized ATan-matched LocalZO,
hard–smooth objective and backward-map diagnostics, same-checkpoint threshold interventions,
and frozen-checkpoint smoothing sweeps. It has a conventional installable package, six task
presets, numerical tests, and a CPU CI workflow.

**Release scope:** this is an independent public implementation, not a byte-for-byte release of
the anonymous supplementary repository. The old training framework is not redistributed.
Architectures, preprocessing, reset derivatives, and checkpoint formats differ in documented
ways. This release has not reproduced the paper's benchmark numbers; no datasets, historical
checkpoints, training curves, or claimed benchmark scores are bundled. See
[paper alignment](docs/PAPER_ALIGNMENT.md) before interpreting results.

## Install

Python 3.11 or 3.12 is recommended. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

For Linux CPU testing:

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[data,plots,dev]'
```

On Apple Silicon, install those PyTorch/torchvision versions from the ordinary PyPI index
without the CPU index option. The CPU implementation is tested; automatic device selection
chooses CUDA when available and otherwise CPU.

For Linux with a compatible NVIDIA driver and CUDA 12.8 wheels:

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[data,plots,dev]'
```

Event dataset adapters additionally need `python -m pip install -e '.[events]'`.
SpikingJelly is an optional dataset dependency; no private/unreleased neuron API is used.
The independent implementation does not require Lightning, Triton, CuPy, nvCOMP, or timm.

## Quick mechanics check

The following uses **explicit synthetic data** and checks the executable workflow. Its results
are not benchmark accuracy or evidence for the paper's reported experiments.

```bash
snn-convergence train --task scifar10 --synthetic --samples 16 \
  --width 4 --steps 4 --epochs 1 --batch-size 8 --device cpu \
  --probe-batches 1 --localzo-repeats 2 --groups --output runs/synthetic-check
snn-convergence evaluate --checkpoint runs/synthetic-check/best.pt --split test
```

The output directory must be new or empty. The command records `data_kind=synthetic`; this
label also appears in plots. No dataset is silently downloaded or replaced with synthetic data.

## Train on benchmarks

```bash
snn-convergence train --task scifar10 --data-root data/scifar10 --download \
  --device cuda --seed 1 --groups --output runs/scifar10-seed1
```

`--download` explicitly enables torchvision's CIFAR download. For the other tasks, prepare
the original data according to [the data guide](docs/DATA.md), then select `scifar100`,
`cifar10dvs`, `dvsgesture`, `shd`, or `tinyimagenet`. Each preset is a JSON resource in
`src/snn_convergence/presets/`; CLI options can override epochs, batch size, width, steps,
learning rate, and momentum.

Training selects `best.pt` using a validation split held out from the training data. The test
split is accessed only when explicitly evaluating it. Independent seed runs should use
different output directories. The reference implementation is single-device.

## LocalZO: two separate sample counts

`--local-samples m` averages **m local derivative samples at each spiking node**. This is the
construction in the supplied anonymous code. It is not an average of m full-network gradients.

`--gradient-replicates R` averages **R complete parameter gradients** for the same minibatch
and fixed forward graph, drawing new independent local factors in each backward evaluation.
BatchNorm buffers update once for the minibatch, not R times. These counts can be combined,
and both are recorded in metadata.

```bash
# Nodewise m=20; reproduces the estimator construction, not historical training results.
snn-convergence train --task scifar10 --data-root data/scifar10 \
  --estimator localzo --local-samples 20 --gradient-replicates 1 --output runs/nodewise20

# R=20 complete-gradient replicas, for the algorithmic form stated in the theorem.
snn-convergence train --task scifar10 --data-root data/scifar10 \
  --estimator localzo --local-samples 1 --gradient-replicates 20 --output runs/replicated20
```

For a fixed nodewise construction, independent complete-gradient replication reduces its
conditional variance by 1/R. The network variance of nodewise averaging is not asserted to be
the same constant divided by m. `--localzo-repeats` controls only Monte Carlo estimation in
diagnostic probes; it is separate from the training update.

## Frozen-checkpoint diagnostics

```bash
snn-convergence probe --checkpoint runs/scifar10-seed1/best.pt \
  --data-root data/scifar10 --groups --output analysis/interventions.csv

snn-convergence sweep --checkpoint runs/scifar10-seed1/best.pt \
  --data-root data/scifar10 --deltas 1 .5 .25 .125 .0625 .03125 \
  --localzo-repeats 1 --output analysis/delta-sweep.csv

snn-convergence plot analysis/delta-sweep.csv --x-axis delta \
  --metric objective_gap --output analysis/delta-gap.png
```

Every group uses the same parameters, evaluation minibatches, and evaluation-mode BatchNorm
buffers. The intervention is applied **during recurrent forward dynamics**, and its output is
written back to the membrane state before reset. It can therefore change later spikes and
firing rates. This is not post-processing of a fixed spike trace.

The original identity-gradient intervention convention is explicit. For intervened rows,
`smooth_gradient_is_objective_gradient=False` unless `--intervention-gradient exact` is chosen
the reset derivative is complete, and no singular zero margin is detected. Intervened backward-map comparisons are not silently
presented as exact gradients of the modified objective. At a zero margin, a non-unit power
intervention can be nonsmooth; exact mode uses a documented zero subgradient there.

## Mathematical conventions and files

- Hard spikes: `1[u > 0]`, including a silent spike at exactly zero margin.
- ATan smooth spike: `1/2 + atan(pi*alpha*u/(2*delta))/pi`.
- LocalZO noise: scaled Student-t with 3 degrees of freedom, normalized by its second moment.
- Reset: hard reset to zero; **full reset derivative by default**. The optional
  `--detach-reset` variant is explicitly labeled and does not yield the full smooth-objective gradient.
- `backward_mismatch_sq` squares the sample-weighted mean gradient difference, retaining
  all parameter coordinates. It is not the mean of per-batch squared differences.
- `tail_diagnostic` is the per-node average; `tail_sum_per_sample` preserves the graph-size factor.

Each run produces `metadata.json`, `training.csv`, `diagnostics.csv`, `best.pt`, and `last.pt`.
Metadata records actual wall-clock seconds, GPU-hours, device count, peak allocated CUDA
memory, seed, model specification, optimizer settings, and data kind. The timer covers training,
validation, probes, and checkpoint writing after dataset initialization. CPU runs report zero
GPU-hours. This records new runs; it cannot reconstruct historical compute usage.

Checkpoint loading uses `weights_only=True` and a versioned schema. Legacy Lightning
checkpoints are rejected rather than loaded with guessed key mappings.

## Verify and contribute

```bash
python -m pytest -q
ruff check src tests
python -m build
```

The tests include analytical/Monte Carlo correspondence, finite-difference smooth gradients,
recurrent interventions, all six reference architecture forward/backward paths, data indexing,
checkpoint round trips, CLI training/sweeps/plots, and probe isolation during LocalZO training.
CI is configured for Linux CPU on Python 3.11/3.12; CUDA benchmark reproduction is a separate
validation task. See [contributing](CONTRIBUTING.md).

## Citation and attribution

Use [CITATION.cff](CITATION.cff) and the [paper record](https://openreview.net/forum?id=QNkfQ8wsii).
The legacy supplementary experiments used the
[SNN gradient-checkpointing framework](https://github.com/AllenYolk/snn-gradient-checkpointing).
That framework motivated the migration; its implementation files are not bundled here.
Dependencies and original research are acknowledged in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
The independently authored files in this release are distributed under the [MIT license](LICENSE).
