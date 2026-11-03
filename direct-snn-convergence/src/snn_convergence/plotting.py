"""Plots from measured CSV rows only. Duplicate seeds and incompatible runs are rejected."""

import csv
import json
import math
from pathlib import Path
from collections import defaultdict

import numpy as np


def plot_csv(paths, output, metric="backward_mismatch_sq", x_axis="epoch"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped, seen, settings = defaultdict(list), set(), set()
    checkpoints = {}
    for path in paths:
        with Path(path).open(newline="") as source:
            for row in csv.DictReader(source):
                if metric not in row or x_axis not in row:
                    raise ValueError(f"missing {metric} or {x_axis} in {path}")
                group = row.get("group", "training")
                seed = row.get("seed", "")
                if not seed:
                    raise ValueError("every plotted run needs a seed")
                x, y = float(row[x_axis]), float(row[metric])
                if not math.isfinite(x) or not math.isfinite(y):
                    raise ValueError("non-finite plot input")
                key = (seed, group, x)
                if key in seen:
                    raise ValueError(f"duplicate seed/group/{x_axis}: {key}")
                seen.add(key)
                epoch = row.get("epoch", "")
                delta = row.get("delta", "")
                checkpoint_key = (seed, epoch, delta)
                digest = row.get("checkpoint_sha256")
                if digest:
                    previous = checkpoints.setdefault(checkpoint_key, digest)
                    if previous != digest:
                        raise ValueError("groups do not share the same checkpoint")
                runtime = json.loads(row.get("runtime", "{}"))
                settings.add(
                    (
                        row.get("task"),
                        row.get("data_kind"),
                        row.get("objective"),
                        row.get("model_spec"),
                        row.get("local_samples"),
                        row.get("gradient_replicates"),
                        runtime.get("alpha"),
                        runtime.get("detach_reset"),
                        None if x_axis == "delta" else runtime.get("delta"),
                    )
                )
                grouped[(group, x)].append(y)
    if len(settings) != 1 or not grouped:
        raise ValueError(
            "empty or incompatible runs; plot one matched experimental setting at a time"
        )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for group in sorted({g for g, _ in grouped}):
        xs = sorted(x for g, x in grouped if g == group)
        mean = np.array([np.mean(grouped[group, x]) for x in xs])
        std = np.array([np.std(grouped[group, x], ddof=0) for x in xs])
        axis.plot(xs, mean, label=group.upper())
        axis.fill_between(xs, mean - std, mean + std, alpha=0.18)
    axis.set(xlabel=x_axis, ylabel=metric)
    if x_axis == "delta":
        axis.set_xscale("log")
    kind = next(iter(settings))[1]
    axis.set_title(
        ("SYNTHETIC CHECK — " if kind == "synthetic" else "") + "mean ± population SD across seeds"
    )
    axis.tick_params(labelsize=12)
    axis.xaxis.label.set_size(14)
    axis.yaxis.label.set_size(14)
    axis.legend(fontsize=12)
    axis.grid(alpha=0.2)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output
