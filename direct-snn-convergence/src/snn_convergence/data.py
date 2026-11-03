"""Dataset adapters. Downloading is explicit; synthetic data are separately marked."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Subset


TASKS = ("scifar10", "scifar100", "cifar10dvs", "dvsgesture", "shd", "tinyimagenet")


def stratified_split(labels, fraction=0.1, seed=1):
    if not 0 < fraction < 1:
        raise ValueError("validation fraction must be in (0, 1)")
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    training, validation = [], []
    for value in np.unique(labels):
        indices = np.flatnonzero(labels == value)
        if len(indices) < 2:
            raise ValueError("each class needs at least two examples for a held-out split")
        rng.shuffle(indices)
        count = min(len(indices) - 1, max(1, round(len(indices) * fraction)))
        validation.extend(indices[:count].tolist())
        training.extend(indices[count:].tolist())
    return sorted(training), sorted(validation)


def bin_shd(times, units, steps, duration=1.0):
    times, units = np.asarray(times), np.asarray(units)
    if steps < 1 or duration <= 0 or times.shape != units.shape:
        raise ValueError("invalid SHD binning arguments")
    if np.any(~np.isfinite(times)):
        raise ValueError("SHD times must be finite")
    valid = (times >= 0) & (times < duration) & (units >= 0) & (units < 700)
    bins = np.floor(times[valid] * steps / duration).astype(np.int64)
    result = np.zeros((steps, 700), dtype=np.float32)
    result[bins, units[valid].astype(np.int64)] = 1.0
    return torch.from_numpy(result)


class SHD(Dataset):
    def __init__(self, path, steps):
        import h5py

        self.path, self.steps = str(path), steps
        with h5py.File(self.path, "r") as source:
            self.targets = np.asarray(source["labels"], dtype=np.int64)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        import h5py

        # Open in the worker that reads the item; do not share HDF5 handles across forks.
        with h5py.File(self.path, "r") as source:
            x = bin_shd(source["spikes/times"][index], source["spikes/units"][index], self.steps)
        return x, int(self.targets[index])


class TinyValidation(Dataset):
    def __init__(self, root, class_to_idx, transform):
        root = Path(root)
        annotations = root / "val_annotations.txt"
        if not annotations.is_file():
            raise FileNotFoundError(f"missing official TinyImageNet annotations: {annotations}")
        self.samples = []
        for row in annotations.read_text().splitlines():
            name, category, *_ = row.split()
            self.samples.append((root / "images" / name, class_to_idx[category]))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        from PIL import Image

        path, label = self.samples[index]
        with Image.open(path) as image:
            x = self.transform(image.convert("RGB"))
        return x, label


class Frames(Dataset):
    def __init__(self, source, size):
        self.source, self.size = source, size

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        x, label = self.source[index]
        x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
        x = torch.nn.functional.interpolate(x, (self.size, self.size), mode="nearest")
        return x, int(label)


def synthetic_dataset(config, samples, seed):
    from torch.utils.data import TensorDataset

    generator = torch.Generator().manual_seed(seed)
    architecture = config.architecture
    if architecture == "scnn":
        shape = (samples, 3, 32, config.steps)
    elif architecture == "shd_mlp":
        shape = (samples, config.steps, 700)
    elif architecture == "sew_resnet34":
        shape = (samples, 3, 16, 16)
    else:
        shape = (samples, config.steps, 2, 16, 16)
    x = torch.randn(shape, generator=generator)
    y = torch.randint(config.classes, (samples,), generator=generator)
    return TensorDataset(x, y)


def datasets(task, root, config, *, seed=1, download=False, validation_fraction=0.1):
    """Return train/validation/test; the test set is never used for checkpoint selection."""
    if task not in TASKS:
        raise ValueError(f"unknown task {task}")
    root = Path(root)
    if task == "shd":
        training = SHD(root / "shd_train.h5", config.steps)
        evaluation = training
        test = SHD(root / "shd_test.h5", config.steps)
        labels = training.targets
    elif task in {"cifar10dvs", "dvsgesture"}:
        if download:
            raise ValueError("prepare event datasets manually according to their access terms")
        options = dict(
            root=str(root), data_type="frame", frames_number=config.steps, split_by="number"
        )
        if task == "dvsgesture":
            from spikingjelly.datasets.dvs128_gesture import DVS128Gesture

            source = DVS128Gesture(train=True, **options)
            heldout = DVS128Gesture(train=False, **options)
        else:
            from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS

            source = CIFAR10DVS(**options)
            all_labels = [int(label) for _, label in source.samples]
            pool, test_indices = stratified_split(all_labels, 0.2, seed)
            heldout, source = Subset(source, test_indices), Subset(source, pool)
        # Read labels without materializing event-frame arrays.
        if isinstance(source, Subset):
            labels = [source.dataset.samples[i][1] for i in source.indices]
        else:
            labels = [label for _, label in source.samples]
        size = 128 if task == "dvsgesture" else 48
        training = evaluation = Frames(source, size)
        test = Frames(heldout, size)
    else:
        from torchvision import datasets as tv, transforms as tr

        if task.startswith("scifar"):
            mean, std = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            if task == "scifar100":
                mean, std = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            basic = [tr.Resize((32, config.steps)), tr.ToTensor(), tr.Normalize(mean, std)]
            augment = tr.Compose([tr.RandomHorizontalFlip(), *basic])
            transform = tr.Compose(basic)
            factory = tv.CIFAR10 if task == "scifar10" else tv.CIFAR100
            training = factory(str(root), train=True, transform=augment, download=download)
            evaluation = factory(str(root), train=True, transform=transform, download=False)
            test = factory(str(root), train=False, transform=transform, download=download)
        else:
            normalization = [
                tr.ToTensor(),
                tr.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
            training = tv.ImageFolder(
                str(root / "train"),
                tr.Compose(
                    [tr.RandomCrop(64, padding=4), tr.RandomHorizontalFlip(), *normalization]
                ),
            )
            transform = tr.Compose(normalization)
            evaluation = tv.ImageFolder(str(root / "train"), transform)
            test = TinyValidation(root / "val", training.class_to_idx, transform)
        labels = training.targets
    train_indices, validation_indices = stratified_split(labels, validation_fraction, seed)
    return Subset(training, train_indices), Subset(evaluation, validation_indices), test
