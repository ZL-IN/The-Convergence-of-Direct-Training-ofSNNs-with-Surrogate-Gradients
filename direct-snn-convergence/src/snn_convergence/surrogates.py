"""ATan smoothing and its normalized, matched Student-t LocalZO estimator."""

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class Runtime:
    pulse: str = "hard"
    estimator: str = "sg"
    alpha: float = 2.0
    delta: float = 1.0
    local_samples: int = 1
    detach_reset: bool = False

    def __post_init__(self):
        if self.pulse not in {"hard", "smooth"}:
            raise ValueError("pulse must be hard or smooth")
        if self.estimator not in {"sg", "localzo"}:
            raise ValueError("estimator must be sg or localzo")
        if self.pulse == "smooth" and self.estimator != "sg":
            raise ValueError("smooth forward uses its matched derivative, not LocalZO")
        for name in ("alpha", "delta"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if type(self.local_samples) is not int or self.local_samples < 1:
            raise ValueError("local_samples must be a positive integer")


def smooth_spike(u, alpha=2.0, delta=1.0):
    return 0.5 + torch.atan(math.pi * alpha * u / (2 * delta)) / math.pi


def atan_derivative(u, alpha=2.0, delta=1.0):
    return alpha / (2 * delta * (1 + (math.pi * alpha * u / (2 * delta)).square()))


def threshold_tail(u, alpha=2.0, delta=1.0):
    # atan2 avoids catastrophic cancellation in 1 - 2*atan(x)/pi for large |u|.
    numerator = torch.full_like(u, 2 * delta)
    return (2 / math.pi) * torch.atan2(numerator, math.pi * alpha * u.abs())


def matched_noise(shape, alpha, *, device, dtype):
    # z = 2/(pi*alpha*sqrt(3))*t_3, with E[z^2] = 4/(pi^2*alpha^2).
    normal = torch.randn(shape, device=device, dtype=dtype)
    chi = torch.randn((3, *shape), device=device, dtype=dtype).square().sum(0)
    return (2 / (math.pi * alpha)) * normal / chi.clamp_min(torch.finfo(dtype).tiny).sqrt()


def localzo_derivative(u, alpha=2.0, delta=1.0, samples=1, chunk_size=16):
    if samples < 1 or chunk_size < 1:
        raise ValueError("samples and chunk_size must be positive")
    # Sampling and accumulation use at least float32 even for half-precision activations.
    dtype = torch.float64 if u.dtype == torch.float64 else torch.float32
    values = u.to(dtype)
    total = torch.zeros_like(values)
    moment = 4 / (math.pi**2 * alpha**2)
    for start in range(0, samples, chunk_size):
        count = min(chunk_size, samples - start)
        z = matched_noise((count, *u.shape), alpha, device=u.device, dtype=dtype)
        positive = (values.unsqueeze(0) + delta * z > 0).to(dtype)
        negative = (values.unsqueeze(0) - delta * z > 0).to(dtype)
        total += ((positive - negative) * z / (2 * delta * moment)).sum(0)
    return (total / samples).to(u.dtype)


class _HardSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, runtime):
        ctx.save_for_backward(u)
        ctx.runtime = runtime
        return (u > 0).to(u.dtype)

    @staticmethod
    def backward(ctx, output_gradient):
        (u,) = ctx.saved_tensors
        r = ctx.runtime
        if r.estimator == "sg":
            derivative = atan_derivative(u, r.alpha, r.delta)
        else:
            derivative = localzo_derivative(u, r.alpha, r.delta, r.local_samples)
        return output_gradient * derivative, None


def spike(u, runtime):
    if runtime.pulse == "smooth":
        return smooth_spike(u, runtime.alpha, runtime.delta)
    return _HardSpike.apply(u, runtime)
