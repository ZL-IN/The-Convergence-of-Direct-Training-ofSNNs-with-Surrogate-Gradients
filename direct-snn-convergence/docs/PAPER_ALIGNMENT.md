# Paper alignment and migration

This public release is an independently authored reference, prepared after auditing the
anonymous supplementary archive. It must not be described as having reproduced the paper
table without a separate benchmark run and comparison.

## What is retained

The hard-forward SG/LocalZO mechanism, normalized ATan-matched Student-t estimator,
hard-reset recurrence, smoothing width, threshold-tail definition, fixed-checkpoint comparison,
and the A/B/C window-power formula are implemented explicitly. Parameters alpha=2 and
delta=1 remain the defaults. The width of the intervention window is 2*delta and the powers
are log(1/4)/log(1/2), log(5/12)/log(1/2), and log(7/12)/log(1/2).

## Changes that matter scientifically

| Item | Anonymous supplement | Independent public reference |
|---|---|---|
| Implementation source | Adapted external training framework | New PyTorch implementation; no framework files vendored |
| Installation | Unavailable PyPI version 0.0.0.0.15; private PSN API use | Core depends on Torch/NumPy; optional event adapters use released SpikingJelly 0.0.0.0.14 |
| LocalZO m | Nodewise local averaging | Same nodewise form plus a separately named full-gradient replication count R |
| Reset derivative | Detached by default in experimental LIF | Complete by default; detached mode remains explicit |
| Compression | Runtime could switch to smooth activations after binary-compressor selection | No binary compression; continuous states remain continuous |
| Interventions | Snapshot mode reinjects margins into the recurrence | Recurrent reinjection retained; identity/exact derivative convention recorded |
| Figure-6 command | Guide lists independent A/B/C training presets | `probe --groups` evaluates REF/A/B/C on one frozen checkpoint and matched batches |
| Parameter gradients | Unused entries dropped | Unused entries represented by zeros in fixed coordinates |
| Gradient mismatch | Callback averages per-batch squared differences | Square of the sample-weighted mean difference over the stated probe set |
| Data selection | Several legacy loaders use official test data for validation | A separate validation subset selects checkpoints; test evaluation is explicit |
| SHD representation | Legacy binning drops unit 0 and reverses channel indices | Channels 0–699 preserved; half-open time bins [t,t+dt) |
| Checkpoints | Lightning/model-specific state_dicts | Versioned independent schema, strict loading |

For REF with a complete reset derivative, the smooth backward is the gradient of the smooth
evaluation objective. For identity-gradient A/B/C interventions, it is a prescribed backward
map; its discrepancy is labeled accordingly. The unchanged legacy Figure 6 is not automatically
certified by tests of the corrected implementation.

## Reference architectures

| Task | Public architecture | Boundary relative to the paper |
|---|---|---|
| Sequential CIFAR-10/100 | Six Conv1d-BN-LIF stages, spatial pooling, spiking hidden readout | Preserves the broad SCNN topology; training augmentation differs |
| DVSGesture | Seven SEW addition blocks and spatial pooling | Independently specified gesture reference; no weight-key compatibility |
| CIFAR10-DVS | Eight Conv2d-BN-LIF stages in four pooling stages | Global pooled readout; differs from legacy flattened VGG readout |
| SHD | Two hidden LIF layers, 700 input channels | **Does not implement the paper's dendritic DHLIF network**; shd_mlp is a reference baseline |
| TinyImageNet | SEW addition ResNet-34 with [3,4,6,3] blocks | Independent small-image stem and readout; legacy checkpoints incompatible |

The JSON presets retain relevant horizons, optimizer choices, learning rates, and epoch
budgets from the manuscript as starting settings. They are not a certificate of recipe or
accuracy equivalence. `temporal_ce` is the mean per-time-step cross entropy, without an
additional TET MSE regularizer. CIFAR augmentation is intentionally simple; legacy mixup,
cutmix, TrivialAugment, compression, CUDA kernels, PSN, learnable DHLIF, and LOMO are not
included. Full ImageNet and spiking transformers are outside this release.

## Mapping experiments

- SG/LocalZO: train matched seed/task configurations with the chosen nodewise sample counts;
  examine training.csv and localzo_alignment_cosine in diagnostics.csv. The diagnostic repeat
  count must be reported alongside local sample counts.
- Objective consistency: use hard_loss, smooth_loss and objective_gap from the same probe.
  No fitted curve is called a certified numerical upper bound.
- Frozen residual: `sweep` materializes the evaluation batches once and reuses both data and
  weights across deltas. The trained checkpoint is not asserted to be an exact F_delta minimizer.
- Controlled intervention: `probe --groups`; all rows share a checkpoint digest and batches.
  Use seeds 1/2/3 in separate runs for seed variability. Never combine separate trained A/B/C
  models and label them same-checkpoint interventions.
- Cross-model table: explicitly evaluate the desired split and collect new accuracy/tail
  measurements. Do not copy the historical paper numbers into this implementation's outputs.

## Migration procedure

Keep the anonymous archive and its historical logs/checkpoints privately for provenance.
Do not copy its source tree into this public repository. A legacy checkpoint cannot be
converted by renaming a prefix: architecture, reset conventions and task-specific neurons
need an explicit equivalence assessment. For now, train reference-v1 checkpoints anew.

Changing from detached to full reset changes training directions. Correcting the definition
of mean gradient mismatch can change the measured value even at a fixed checkpoint. Any new
curves used in the paper must identify the implementation and settings that generated them.
The code does not certify PL conditions, containment, or convergence of momentum/Adam recipes.
