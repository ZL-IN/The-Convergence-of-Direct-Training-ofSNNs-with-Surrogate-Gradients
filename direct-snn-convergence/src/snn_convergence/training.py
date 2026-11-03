"""Single-device reference training and versioned, weights-only-readable checkpoints."""

from dataclasses import asdict
import csv
import json
from pathlib import Path
import random
import platform
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import __version__
from .data import datasets, synthetic_dataset
from .diagnostics import cached_batches, estimate_gradient, probe
from .models import ModelConfig, build_model
from .objectives import objective
from .surrogates import Runtime


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, records):
    if not records:
        return
    fields = list(dict.fromkeys(key for row in records for key in row))
    with Path(path).open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow(
                {
                    k: json.dumps(v, sort_keys=True) if isinstance(v, dict) else v
                    for k, v in row.items()
                }
            )


def read_preset(task):
    path = Path(__file__).parent / "presets" / f"{task}.json"
    return json.loads(path.read_text())


def device_for(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("this release supports cpu and cuda devices")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def load_checkpoint(path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format") != "snn-convergence-reference-v1":
        raise ValueError(
            "not a reference-v1 checkpoint; legacy checkpoints need explicit migration"
        )
    model = build_model(ModelConfig(**payload["model_config"]), Runtime(**payload["runtime"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device), payload


def save_checkpoint(path, model, metadata, runtime, epoch):
    payload = dict(
        format="snn-convergence-reference-v1",
        version=__version__,
        model_config=asdict(model.config),
        runtime=asdict(runtime),
        state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
        metadata={k: v for k, v in metadata.items() if k != "status"},
        epoch=epoch,
    )
    temporary = Path(path).with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def make_loaders(
    task,
    root,
    config,
    *,
    synthetic=False,
    samples=64,
    batch_size=128,
    workers=0,
    seed=1,
    download=False,
):
    if synthetic:
        sources = [synthetic_dataset(config, samples, seed + i * 10_000) for i in range(3)]
    else:
        sources = datasets(task, root, config, seed=seed, download=download)
    loaders = []
    for i, source in enumerate(sources):
        loaders.append(
            DataLoader(
                source,
                batch_size=batch_size,
                shuffle=i == 0,
                num_workers=workers,
                drop_last=False,
                generator=torch.Generator().manual_seed(seed + i),
            )
        )
    return loaders


@torch.no_grad()
def evaluate(model, loader, device, loss_kind="ce"):
    model.eval()
    count = correct = 0
    loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(device, dtype=torch.float32), y.to(device)
        loss = objective(model, x, y, loss_kind)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite evaluation loss")
        count += len(y)
        loss_sum += len(y) * loss.item()
        correct += (model(x).argmax(-1) == y).sum().item()
    if count == 0:
        raise ValueError("empty evaluation data")
    return dict(loss=loss_sum / count, accuracy=100 * correct / count, samples=count)


def assign_gradient(model, gradient):
    offset = 0
    for parameter in model.parameters():
        if parameter.requires_grad:
            size = parameter.numel()
            parameter.grad = gradient[offset : offset + size].reshape_as(parameter).clone()
            offset += size
    if offset != gradient.numel():
        raise ValueError("parameter coordinates do not match the gradient vector")


def train(args):
    preset = read_preset(args.task)
    config = ModelConfig(
        **(
            preset["model"]
            | {k: v for k, v in {"steps": args.steps, "width": args.width}.items() if v is not None}
        )
    )
    settings = preset["training"] | {
        k: v
        for k, v in {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "momentum": args.momentum,
        }.items()
        if v is not None
    }
    if args.gradient_replicates < 1 or args.probe_every < 1 or args.probe_batches < 1:
        raise ValueError("replicate counts and probe intervals must be positive")
    if settings["epochs"] < 1 or settings["batch_size"] < 1 or args.samples < 2:
        raise ValueError("epochs/batch size must be positive, synthetic samples >= 2")
    if args.estimator == "sg" and args.gradient_replicates != 1:
        raise ValueError("SG has no perturbation randomness; use gradient_replicates=1")
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    device = device_for(args.device)
    seed_all(args.seed)
    runtime = Runtime(
        estimator=args.estimator,
        alpha=args.alpha,
        delta=args.delta,
        local_samples=args.local_samples,
        detach_reset=args.detach_reset,
    )
    model = build_model(config, runtime).to(device)
    training, validation, _ = make_loaders(
        args.task,
        args.data_root,
        config,
        synthetic=args.synthetic,
        samples=args.samples,
        batch_size=settings["batch_size"],
        workers=args.workers,
        seed=args.seed,
        download=args.download,
    )
    heldout = cached_batches(validation, args.probe_batches, device)
    if settings["optimizer"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=settings["learning_rate"],
            momentum=settings["momentum"],
            weight_decay=settings.get("weight_decay", 0.0),
        )
    if settings.get("scheduler", "cosine") == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, settings["step_size"], settings["gamma"]
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, settings["epochs"])
    metadata = dict(
        version=__version__,
        task=args.task,
        seed=args.seed,
        data_kind="synthetic" if args.synthetic else "benchmark",
        model=asdict(config),
        runtime=asdict(runtime),
        training=settings,
        gradient_replicates=args.gradient_replicates,
        device=str(device),
        torch_version=str(torch.__version__),
        numpy_version=np.__version__,
        python_version=platform.python_version(),
        status="running",
    )
    write_json(out / "metadata.json", metadata)
    rows, diagnostic_rows = [], []
    best = -1.0
    start = time.perf_counter()
    gpu = device.type == "cuda"
    if gpu:
        torch.cuda.reset_peak_memory_stats(device)
    try:
        for epoch in range(settings["epochs"]):
            model.train()
            loss_sum = count = 0
            for x, y in training:
                x, y = x.to(device, dtype=torch.float32), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = objective(model, x, y, settings["loss"])
                gradient = estimate_gradient(loss, model, args.gradient_replicates)
                if not torch.isfinite(loss) or not torch.isfinite(gradient).all():
                    raise FloatingPointError(
                        "non-finite loss/gradient; stopping before optimizer step"
                    )
                assign_gradient(model, gradient)
                optimizer.step()
                loss_sum += len(y) * loss.item()
                count += len(y)
            result = evaluate(model, validation, device, settings["loss"])
            rows.append(
                dict(
                    epoch=epoch,
                    seed=args.seed,
                    data_kind=metadata["data_kind"],
                    train_loss=loss_sum / count,
                    validation_loss=result["loss"],
                    validation_accuracy=result["accuracy"],
                    learning_rate=optimizer.param_groups[0]["lr"],
                )
            )
            if epoch % args.probe_every == 0:
                for row in probe(
                    model,
                    heldout,
                    groups=args.groups,
                    seed=args.seed,
                    local_samples=args.local_samples,
                    localzo_repeats=args.localzo_repeats,
                    loss_kind=settings["loss"],
                ):
                    diagnostic_rows.append(
                        row
                        | dict(
                            epoch=epoch,
                            seed=args.seed,
                            task=args.task,
                            data_kind=metadata["data_kind"],
                        )
                    )
                write_csv(out / "diagnostics.csv", diagnostic_rows)
            write_csv(out / "training.csv", rows)
            if result["accuracy"] > best:
                best = result["accuracy"]
                save_checkpoint(out / "best.pt", model, metadata, runtime, epoch)
            save_checkpoint(out / "last.pt", model, metadata, runtime, epoch)
            scheduler.step()
            print(json.dumps(rows[-1], allow_nan=False), flush=True)
        metadata["status"] = "complete"
    finally:
        if gpu:
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - start
        metadata.update(
            wall_seconds=seconds,
            gpu_hours=seconds / 3600 if gpu else 0.0,
            gpu_count=1 if gpu else 0,
            peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(device) if gpu else 0,
        )
        if metadata["status"] != "complete":
            metadata["status"] = "failed"
        write_json(out / "metadata.json", metadata)
    return out
