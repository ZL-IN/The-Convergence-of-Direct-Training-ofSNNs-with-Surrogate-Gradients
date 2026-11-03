import math

import pytest
import torch
from torch import nn

from snn_convergence.diagnostics import flat_gradient, estimate_gradient
from snn_convergence.neurons import LIF, Intervention, neuron_runtime
from snn_convergence.surrogates import (
    Runtime,
    atan_derivative,
    localzo_derivative,
    smooth_spike,
    spike,
    threshold_tail,
)


@pytest.mark.parametrize("delta", [0.1, 0.5, 1.0, 2.0])
def test_matched_derivative_and_exact_scalar_tail(delta):
    x = torch.linspace(-3, 3, 101, dtype=torch.double, requires_grad=True)
    values = smooth_spike(x, delta=delta)
    derivative = torch.autograd.grad(values.sum(), x)[0]
    torch.testing.assert_close(derivative, atan_derivative(x, delta=delta))
    torch.testing.assert_close(
        ((x > 0).double() - values).abs(), threshold_tail(x, delta=delta) / 2
    )
    assert threshold_tail(torch.tensor([1e12])).item() > 0


def test_localzo_mean_matches_atan():
    torch.manual_seed(37)
    x = torch.tensor([-1.0, 0.0, 0.5, 2.0])
    empirical = localzo_derivative(x, samples=200_000)
    torch.testing.assert_close(empirical, atan_derivative(x), atol=0.025, rtol=0.03)


def test_local_average_variance_decreases():
    torch.manual_seed(41)
    x = torch.zeros(20_000)
    one = localzo_derivative(x, samples=1)
    sixteen = localzo_derivative(x, samples=16)
    ratio = (sixteen.var() / one.var()).item()
    assert 0.025 < ratio < 0.15


def test_smooth_lif_has_complete_gradient():
    neuron = LIF(runtime=Runtime(pulse="smooth")).double()
    x = torch.tensor([[0.8, 0.4], [0.6, 0.9], [0.3, 0.2]], dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(neuron, (x,), eps=1e-6, atol=1e-5)


def test_smooth_two_layer_gradient_including_resets():
    model = nn.Sequential(
        nn.Linear(2, 3),
        LIF(runtime=Runtime(pulse="smooth")),
        nn.Linear(3, 2),
        LIF(runtime=Runtime(pulse="smooth")),
    ).double()
    x = torch.rand(3, 2, 2, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(model, (x,), eps=1e-6, atol=1e-5)


def test_threshold_boundary_and_sg_derivative():
    x = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
    y = spike(x, Runtime())
    assert y.tolist() == [0.0, 0.0, 1.0]
    torch.testing.assert_close(torch.autograd.grad(y.sum(), x)[0], atan_derivative(x))


def test_state_reinjection_can_change_later_spikes():
    neuron = LIF(beta=0.5)
    x = torch.tensor([[0.5], [0.6]])
    assert neuron(x).flatten().tolist() == [0.0, 0.0]
    neuron.intervention = Intervention(gamma=2, radius=2)
    assert neuron(x).flatten().tolist() == [0.0, 1.0]


def test_intervention_preserves_current_sign_and_has_explicit_gradient():
    x = torch.tensor([-0.5, 0.0, 0.5, 3.0], dtype=torch.double, requires_grad=True)
    for mode in ("identity", "exact"):
        mapped = Intervention(2.0, 2.0, mode)(x)
        assert ((mapped > 0) == (x > 0)).all()
        gradient = torch.autograd.grad(mapped.sum(), x)[0]
        target = (
            torch.ones_like(x)
            if mode == "identity"
            else torch.tensor([0.5, 0.0, 0.5, 1.0], dtype=torch.double)
        )
        torch.testing.assert_close(gradient, target)


def test_unused_parameter_coordinates_are_retained():
    model = nn.ParameterList([nn.Parameter(torch.ones(2)), nn.Parameter(torch.ones(3))])
    assert flat_gradient(model[0].sum(), model).tolist() == [1, 1, 0, 0, 0]
    assert flat_gradient(model[1].sum(), model).tolist() == [0, 0, 1, 1, 1]


def test_graph_replication_resamples_backward_factors():
    torch.manual_seed(12)
    model = nn.Sequential(nn.Linear(2, 2), LIF(), nn.Linear(2, 1), LIF())
    x = torch.ones(2, 3, 2)
    expected = flat_gradient(model(x).sum(), model)
    with neuron_runtime(model, estimator="localzo", local_samples=1):
        loss = model(x).sum()
        mean = estimate_gradient(loss, model, replicates=3000)
    torch.testing.assert_close(mean, expected, atol=0.10, rtol=0.20)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(delta=0),
        dict(alpha=float("nan")),
        dict(local_samples=0),
        dict(local_samples=1.5),
        dict(pulse="smooth", estimator="localzo"),
    ],
)
def test_invalid_runtime(kwargs):
    with pytest.raises(ValueError):
        Runtime(**kwargs)


@pytest.mark.parametrize("kwargs", [dict(gamma=-1, radius=2), dict(gamma=math.inf, radius=2)])
def test_invalid_intervention(kwargs):
    with pytest.raises(ValueError):
        Intervention(**kwargs)
