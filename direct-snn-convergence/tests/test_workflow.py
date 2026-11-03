import csv
import json

import pytest
import torch

from snn_convergence.cli import main
from snn_convergence.diagnostics import state_digest
from snn_convergence.plotting import plot_csv
from snn_convergence.training import load_checkpoint


def train_command(output, *extra):
    return [
        "train",
        "--output",
        str(output),
        "--synthetic",
        "--epochs",
        "2",
        "--width",
        "2",
        "--steps",
        "2",
        "--samples",
        "8",
        "--batch-size",
        "4",
        "--probe-batches",
        "1",
        "--localzo-repeats",
        "1",
        "--device",
        "cpu",
        "--estimator",
        "localzo",
        "--local-samples",
        "2",
        *extra,
    ]


def test_full_cli_train_probe_sweep_evaluate_plot(tmp_path, capsys):
    run = tmp_path / "run"
    main(train_command(run, "--groups", "--gradient-replicates", "2"))
    meta = json.loads((run / "metadata.json").read_text())
    assert meta["status"] == "complete" and meta["data_kind"] == "synthetic"
    assert meta["wall_seconds"] > 0 and meta["gpu_hours"] == 0
    model, payload = load_checkpoint(run / "last.pt")
    assert payload["epoch"] == 1
    assert len(list(csv.DictReader((run / "diagnostics.csv").open()))) == 8
    sweep = tmp_path / "sweep.csv"
    main(
        [
            "sweep",
            "--checkpoint",
            str(run / "last.pt"),
            "--output",
            str(sweep),
            "--deltas",
            ".25",
            ".5",
            "--samples",
            "4",
            "--probe-batches",
            "1",
            "--localzo-repeats",
            "1",
        ]
    )
    rows = list(csv.DictReader(sweep.open()))
    assert len(rows) == 2 and rows[0]["hard_loss"] == rows[1]["hard_loss"]
    image = tmp_path / "sweep.png"
    main(
        [
            "plot",
            str(sweep),
            "--output",
            str(image),
            "--metric",
            "objective_gap",
            "--x-axis",
            "delta",
        ]
    )
    assert image.stat().st_size > 1000
    main(["evaluate", "--checkpoint", str(run / "best.pt"), "--samples", "4", "--split", "test"])
    assert '"data_kind": "synthetic"' in capsys.readouterr().out
    assert model.config.width == 2
    with pytest.raises(FileExistsError):
        main(train_command(run))


def test_group_probes_do_not_change_localzo_training_trajectory(tmp_path):
    main(train_command(tmp_path / "base"))
    main(train_command(tmp_path / "groups", "--groups"))
    first, _ = load_checkpoint(tmp_path / "base" / "last.pt")
    second, _ = load_checkpoint(tmp_path / "groups" / "last.pt")
    assert state_digest(first) == state_digest(second)


def test_legacy_checkpoint_is_rejected(tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save({"state_dict": {}}, path)
    with pytest.raises(ValueError, match="legacy"):
        load_checkpoint(path)


def test_plot_rejects_duplicate_seed_and_empty_input(tmp_path):
    path = tmp_path / "rows.csv"
    path.write_text("epoch,seed,group,backward_mismatch_sq,data_kind\n0,1,ref,1,synthetic\n")
    with pytest.raises(ValueError, match="duplicate"):
        plot_csv([path, path], tmp_path / "bad.png")
    with pytest.raises(ValueError, match="empty"):
        plot_csv([], tmp_path / "bad.png")
