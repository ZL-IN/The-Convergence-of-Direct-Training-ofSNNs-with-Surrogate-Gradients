import argparse
import json
from pathlib import Path

import torch

from .data import TASKS
from .diagnostics import cached_batches, delta_sweep, probe
from .training import device_for, evaluate, load_checkpoint, make_loaders, train, write_csv


def parser():
    root = argparse.ArgumentParser(description="Independent direct-SNN convergence reference")
    commands = root.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="train and record matched diagnostics")
    training.add_argument("--task", choices=TASKS, default="scifar10")
    training.add_argument("--output", required=True, help="new/empty run directory")
    training.add_argument("--data-root", default="data/scifar10")
    training.add_argument("--download", action="store_true", help="explicit CIFAR download")
    training.add_argument("--device", default="auto")
    training.add_argument("--workers", type=int, default=0)
    training.add_argument("--threads", type=int, default=1)
    training.add_argument("--seed", type=int, default=1)
    for name in ("epochs", "batch-size", "steps", "width"):
        training.add_argument(f"--{name}", type=int)
    training.add_argument("--learning-rate", type=float)
    training.add_argument("--momentum", type=float)
    training.add_argument("--estimator", choices=("sg", "localzo"), default="sg")
    training.add_argument("--alpha", type=float, default=2.0)
    training.add_argument("--delta", type=float, default=1.0)
    training.add_argument("--local-samples", type=int, default=1)
    training.add_argument("--gradient-replicates", type=int, default=1)
    training.add_argument(
        "--detach-reset",
        action="store_true",
        help="legacy stop-gradient variant; smooth backward is not the full objective gradient",
    )
    training.add_argument("--groups", action="store_true", help="same-checkpoint REF/A/B/C probes")
    training.add_argument("--probe-batches", type=int, default=4)
    training.add_argument("--probe-every", type=int, default=1)
    training.add_argument("--localzo-repeats", type=int, default=8)
    training.add_argument(
        "--synthetic", action="store_true", help="explicitly labeled mechanics check"
    )
    training.add_argument("--samples", type=int, default=64)
    for name in ("probe", "sweep", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--checkpoint", required=True)
        command.add_argument("--data-root", default="data/scifar10")
        command.add_argument("--device", default="auto")
        command.add_argument("--threads", type=int, default=1)
        command.add_argument("--batch-size", type=int, default=16)
        command.add_argument("--workers", type=int, default=0)
        command.add_argument("--samples", type=int, default=64, help="synthetic checkpoints only")
        command.add_argument("--split", choices=("validation", "test"), default="validation")
        if name != "evaluate":
            command.add_argument("--output", required=True, help="new CSV path")
            command.add_argument("--probe-batches", type=int, default=4)
            command.add_argument("--local-samples", type=int, default=1)
            command.add_argument("--localzo-repeats", type=int, default=8)
            command.add_argument("--groups", action="store_true")
            command.add_argument(
                "--intervention-gradient", choices=("identity", "exact"), default="identity"
            )
        if name == "sweep":
            command.add_argument(
                "--deltas", type=float, nargs="+", default=[1, 0.5, 0.25, 0.125, 0.0625, 0.03125]
            )
    plotting = commands.add_parser("plot")
    plotting.add_argument("csv", nargs="+")
    plotting.add_argument("--output", required=True)
    plotting.add_argument("--metric", default="backward_mismatch_sq")
    plotting.add_argument("--x-axis", choices=("epoch", "delta"), default="epoch")
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "plot":
        from .plotting import plot_csv

        print(plot_csv(args.csv, args.output, args.metric, args.x_axis))
        return
    if args.threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(args.threads)
    if args.command == "train":
        train(args)
        return
    device = device_for(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    metadata = checkpoint["metadata"]
    loaders = make_loaders(
        metadata["task"],
        args.data_root,
        model.config,
        synthetic=metadata["data_kind"] == "synthetic",
        samples=args.samples,
        batch_size=args.batch_size,
        workers=args.workers,
        seed=metadata["seed"],
    )
    loader = loaders[1 if args.split == "validation" else 2]
    loss_kind = metadata["training"]["loss"]
    if args.command == "evaluate":
        print(
            json.dumps(
                evaluate(model, loader, device, loss_kind)
                | dict(split=args.split, data_kind=metadata["data_kind"])
            )
        )
        return
    path = Path(args.output)
    if path.exists():
        raise FileExistsError(path)
    batches = cached_batches(loader, args.probe_batches, device)
    options = dict(
        groups=args.groups,
        local_samples=args.local_samples,
        localzo_repeats=args.localzo_repeats,
        intervention_gradient=args.intervention_gradient,
        seed=metadata["seed"],
        loss_kind=loss_kind,
    )
    if args.command == "sweep":
        records = delta_sweep(model, batches, args.deltas, **options)
    else:
        records = probe(model, batches, **options)
    for row in records:
        row.update(
            epoch=checkpoint["epoch"],
            seed=metadata["seed"],
            task=metadata["task"],
            data_kind=metadata["data_kind"],
            split=args.split,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(path, records)
    print(path)
