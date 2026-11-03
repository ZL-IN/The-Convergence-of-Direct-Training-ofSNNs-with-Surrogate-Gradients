"""Finite-horizon hard-reset LIF with explicit reset and intervention derivatives."""

from contextlib import contextmanager
from dataclasses import dataclass, replace
import math

import torch
from torch import nn

from .surrogates import Runtime, spike


@dataclass(frozen=True)
class Intervention:
    gamma: float
    radius: float
    gradient: str = "identity"

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in (self.gamma, self.radius)):
            raise ValueError("gamma and radius must be finite and positive")
        if self.gradient not in {"identity", "exact"}:
            raise ValueError("gradient must be identity or exact")

    def __call__(self, u):
        magnitude = u.abs()
        positive = (magnitude / self.radius).clamp_min(torch.finfo(u.dtype).tiny)
        mapped = self.radius * positive.pow(self.gamma)
        mapped = torch.where(magnitude > 0, mapped, torch.zeros_like(mapped))
        modified = u.sign() * torch.where(magnitude <= self.radius, mapped, magnitude)
        if self.gradient == "identity":
            return u + (modified - u).detach()
        # At u=0 the exact derivative may be undefined; use the zero subgradient.
        return modified


class LIF(nn.Module):
    def __init__(self, beta=0.5, runtime=None):
        super().__init__()
        if not math.isfinite(beta) or not 0 <= beta < 1:
            raise ValueError("beta must be finite and in [0, 1)")
        self.beta = beta
        self.runtime = runtime or Runtime()
        self.intervention = None
        self.trace_enabled = False
        self.trace = None
        self.intervention_nondifferentiable = False

    def forward(self, current):
        if current.shape[0] < 1:
            raise ValueError("empty time axis")
        voltage = torch.zeros_like(current[0])
        self.intervention_nondifferentiable = False
        outputs, margins = [], []
        for value in current.unbind(0):
            membrane = self.beta * voltage + value
            margin = membrane - 1.0
            if self.intervention is not None:
                if (
                    self.intervention.gradient == "exact"
                    and self.intervention.gamma < 1
                    and (margin == 0).any().item()
                ):
                    self.intervention_nondifferentiable = True
                margin = self.intervention(margin)
                membrane = margin + 1.0  # state reinjection, including subsequent time steps
            firing = spike(margin, self.runtime)
            reset = firing.detach() if self.runtime.detach_reset else firing
            voltage = membrane * (1 - reset)
            outputs.append(firing)
            if self.trace_enabled:
                margins.append(margin.detach())
        result = torch.stack(outputs)
        self.trace = (torch.stack(margins), result.detach()) if self.trace_enabled else None
        return result


@contextmanager
def neuron_runtime(model, *, intervention=None, trace=False, **changes):
    modules = [m for m in model.modules() if isinstance(m, LIF)]
    states = [(m.runtime, m.intervention, m.trace_enabled) for m in modules]
    try:
        for m in modules:
            m.runtime = replace(m.runtime, **changes)
            m.intervention = intervention
            m.trace_enabled = trace
        yield
    finally:
        for m, (runtime, previous, enabled) in zip(modules, states):
            m.runtime, m.intervention, m.trace_enabled = runtime, previous, enabled
