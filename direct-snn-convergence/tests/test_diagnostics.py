import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from snn_convergence.diagnostics import probe, state_digest, flat_gradient, delta_sweep
from snn_convergence.models import ModelConfig, build_model
from snn_convergence.neurons import LIF, neuron_runtime
from snn_convergence.surrogates import Runtime


def fixture():
    torch.manual_seed(21)
    model = build_model(ModelConfig(classes=3, steps=3, width=2))
    batches = [
        (torch.randn(2, 3, 32, 3), torch.tensor([0, 1])),
        (torch.randn(1, 3, 32, 3), torch.tensor([2])),
    ]
    return model, batches


def test_probe_preserves_model_modes_rng_and_coordinates():
    model, batches = fixture()
    model.train()
    model.readout.eval()  # mixed module modes must be restored exactly
    modes = [m.training for m in model.modules()]
    state = state_digest(model)
    tensor_rng = torch.get_rng_state().clone()
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    rows = probe(model, batches, groups=True, localzo_repeats=2)
    assert state_digest(model) == state
    assert [m.training for m in model.modules()] == modes
    assert torch.equal(tensor_rng, torch.get_rng_state())
    assert random.getstate() == python_rng
    assert np.array_equal(np.random.get_state()[1], numpy_rng[1])
    assert {r["group"] for r in rows} == {"ref", "a", "b", "c"}
    assert len({r["checkpoint_sha256"] for r in rows}) == 1
    assert rows[0]["smooth_gradient_is_objective_gradient"]
    assert not rows[1]["smooth_gradient_is_objective_gradient"]  # identity intervention derivative
    assert all(0 <= r["tail_diagnostic"] <= 1 for r in rows)


def test_probe_squares_the_weighted_mean_gradient():
    model, batches = fixture()
    model.eval()
    hard, smooth = [], []
    for pulse, results in (("hard", hard), ("smooth", smooth)):
        with neuron_runtime(model, pulse=pulse):
            for x, y in batches:
                results.append(flat_gradient(F.cross_entropy(model(x), y), model).detach())
    expected = ((2 * hard[0] + hard[1]) / 3 - (2 * smooth[0] + smooth[1]) / 3).square().sum()
    measured = probe(model, batches, localzo_repeats=1)[0]
    np.testing.assert_allclose(
        measured["backward_mismatch_sq"], expected.item(), rtol=1e-5, atol=1e-9
    )


def test_delta_sweep_reuses_checkpoint_and_data():
    model, batches = fixture()
    rows = delta_sweep(model, batches, [0.5, 1.0], localzo_repeats=1)
    assert [r["delta"] for r in rows] == [0.5, 1.0]
    assert rows[0]["checkpoint_sha256"] == rows[1]["checkpoint_sha256"]
    assert rows[0]["hard_loss"] == rows[1]["hard_loss"]


def test_detached_reset_is_not_labeled_as_an_objective_gradient():
    model, batches = fixture()
    with neuron_runtime(model, detach_reset=True):
        assert not probe(model, batches, localzo_repeats=1)[0][
            "smooth_gradient_is_objective_gradient"
        ]


def test_runtime_restored_after_exception():
    model, _ = fixture()
    try:
        with neuron_runtime(model, pulse="smooth"):
            raise RuntimeError("test")
    except RuntimeError:
        pass
    assert all(m.runtime == Runtime() for m in model.modules() if hasattr(m, "runtime"))


def test_fractional_intervention_at_zero_is_not_labeled_differentiable():
    class EdgeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(2))
            self.neuron = LIF()

        def forward(self, x):
            return self.neuron((x * self.weight).unsqueeze(0)).squeeze(0)

    model = EdgeModel()
    rows = probe(
        model,
        [(torch.ones(2, 2), torch.tensor([0, 1]))],
        groups=True,
        intervention_gradient="exact",
        localzo_repeats=1,
    )
    assert rows[0]["smooth_gradient_is_objective_gradient"]
    assert not rows[-1]["smooth_gradient_is_objective_gradient"]
