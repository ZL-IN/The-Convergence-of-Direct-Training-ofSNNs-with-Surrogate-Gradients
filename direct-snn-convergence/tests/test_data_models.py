import json

import h5py
import numpy as np
import pytest
import torch
from PIL import Image

from snn_convergence.data import (
    SHD,
    TinyValidation,
    bin_shd,
    stratified_split,
    synthetic_dataset,
    TASKS,
)
from snn_convergence.models import build_model, ModelConfig
from snn_convergence.training import read_preset


@pytest.mark.parametrize("task", TASKS)
def test_reference_architecture_forward_and_backward(task):
    spec = read_preset(task)["model"] | dict(width=2, steps=2)
    config = ModelConfig(**spec)
    model = build_model(config).eval()
    x, y = synthetic_dataset(config, 2, seed=1).tensors
    logits = model(x)
    assert logits.shape == (2, config.classes)
    torch.nn.functional.cross_entropy(logits, y).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_split_is_stratified_disjoint_and_repeatable():
    labels = [0] * 10 + [1] * 10
    train, val = stratified_split(labels, 0.2, seed=4)
    assert (train, val) == stratified_split(labels, 0.2, seed=4)
    assert set(train).isdisjoint(val)
    assert sorted(train + val) == list(range(20))
    assert np.bincount(np.array(labels)[val]).tolist() == [2, 2]


def test_shd_keeps_channel_zero_and_uses_half_open_bins():
    data = bin_shd(np.array([0.0, 0.01, 0.999, 1.0]), np.array([0, 1, 699, 2]), 100)
    assert data[0, 0] == 1 and data[1, 1] == 1 and data[99, 699] == 1
    assert data.sum() == 3


def test_shd_hdf5_reader(tmp_path):
    path = tmp_path / "sample.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("labels", data=np.array([3]))
        f.create_dataset("spikes/times", data=np.array([[0.0, 0.2]]))
        f.create_dataset("spikes/units", data=np.array([[0, 12]]))
    x, y = SHD(path, 10)[0]
    assert x.shape == (10, 700) and y == 3 and x.sum() == 2


def test_official_tinyimagenet_validation_layout(tmp_path):
    (tmp_path / "images").mkdir()
    Image.new("RGB", (64, 64)).save(tmp_path / "images" / "a.JPEG")
    (tmp_path / "val_annotations.txt").write_text("a.JPEG\tn0001\t0\t0\t63\t63\n")
    data = TinyValidation(tmp_path, {"n0001": 7}, lambda image: np.asarray(image))
    x, y = data[0]
    assert x.shape == (64, 64, 3) and y == 7


def test_all_presets_have_distinct_task_and_explicit_reference_status():
    for task in TASKS:
        preset = read_preset(task)
        assert preset["task"] == task and "independent reference" in preset["status"]
        assert json.loads(json.dumps(preset)) == preset
