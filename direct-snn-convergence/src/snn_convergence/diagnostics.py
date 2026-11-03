"""Matched-batch diagnostics with fixed coordinates, explicit scope, and RNG isolation."""

from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import random

import numpy as np
import torch
from torch.nn import functional as F

from .neurons import Intervention, LIF, neuron_runtime
from .objectives import objective
from .surrogates import threshold_tail


def flat_gradient(loss, model, *, retain_graph=False):
    parameters = [p for p in model.parameters() if p.requires_grad]
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=retain_graph)
    # Keep every parameter's coordinate, including unused parameters.
    return torch.cat(
        [
            (torch.zeros_like(p) if g is None else g).reshape(-1)
            for p, g in zip(parameters, gradients)
        ]
    )


def estimate_gradient(loss, model, replicates=1):
    if replicates < 1:
        raise ValueError("replicates must be positive")
    mean = None
    for index in range(replicates):
        gradient = flat_gradient(loss, model, retain_graph=index < replicates - 1).detach()
        mean = gradient / replicates if mean is None else mean + gradient / replicates
    return mean


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@contextmanager
def isolated_evaluation(model, seed=0):
    flags = [(m, m.training) for m in model.modules()]
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed % (2**32))
            model.eval()
            yield
    finally:
        for m, flag in flags:
            m.training = flag
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def cached_batches(loader, limit, device):
    if limit < 1:
        raise ValueError("probe batch limit must be positive")
    result = []
    for index, (x, y) in enumerate(loader):
        if index >= limit:
            break
        result.append((x.to(device=device, dtype=torch.float32), y.to(device)))
    if not result:
        raise ValueError("no probe samples")
    return result


def _trace_totals(model, alpha, delta):
    tail, fired, count = 0.0, 0.0, 0
    for layer in model.modules():
        if isinstance(layer, LIF) and layer.trace is not None:
            margin, firing = layer.trace
            tail += threshold_tail(margin, alpha, delta).sum().item()
            fired += firing.sum().item()
            count += margin.numel()
    if not count:
        raise ValueError("model has no traced spiking nodes")
    return tail, fired, count


def probe(
    model,
    batches,
    *,
    groups=False,
    local_samples=1,
    localzo_repeats=8,
    delta=None,
    intervention_gradient="identity",
    seed=1,
    loss_kind="ce",
):
    if localzo_repeats < 1 or local_samples < 1:
        raise ValueError("LocalZO sample and repeat counts must be positive")
    batches = list(batches)  # every group and delta uses exactly these tensors
    total = sum(len(y) for _, y in batches)
    if total == 0:
        raise ValueError("empty probe data")
    baseline = next(m.runtime for m in model.modules() if isinstance(m, LIF))
    delta = baseline.delta if delta is None else delta
    definitions = {"ref": None}
    if groups:
        import math

        definitions.update(
            {
                name: Intervention(
                    math.log(target) / math.log(0.5), 2 * delta, intervention_gradient
                )
                for name, target in (("a", 1 / 4), ("b", 5 / 12), ("c", 7 / 12))
            }
        )
    records = []
    checkpoint = state_digest(model)
    with isolated_evaluation(model, seed), torch.enable_grad():
        for group, intervention in definitions.items():
            regular_intervention = True
            hard_gradient = smooth_gradient = zo_gradient = None
            hard_loss = soft_loss = tail = fired = 0.0
            nodes = 0
            for x, y in batches:
                weight = len(y) / total
                with neuron_runtime(
                    model,
                    pulse="hard",
                    estimator="sg",
                    delta=delta,
                    intervention=intervention,
                    trace=True,
                ):
                    loss = objective(model, x, y, loss_kind)
                    gradient = flat_gradient(loss, model).detach()
                    hard_gradient = (
                        weight * gradient
                        if hard_gradient is None
                        else hard_gradient + weight * gradient
                    )
                    hard_loss += weight * loss.detach().item()
                    t, f, n = _trace_totals(model, baseline.alpha, delta)
                    tail, fired, nodes = tail + t, fired + f, nodes + n
                with neuron_runtime(
                    model, pulse="smooth", estimator="sg", delta=delta, intervention=intervention
                ):
                    loss = objective(model, x, y, loss_kind)
                    regular_intervention &= not any(
                        m.intervention_nondifferentiable
                        for m in model.modules()
                        if isinstance(m, LIF)
                    )
                    gradient = flat_gradient(loss, model).detach()
                    smooth_gradient = (
                        weight * gradient
                        if smooth_gradient is None
                        else smooth_gradient + weight * gradient
                    )
                    soft_loss += weight * loss.detach().item()
                with neuron_runtime(
                    model,
                    pulse="hard",
                    estimator="localzo",
                    delta=delta,
                    local_samples=local_samples,
                    intervention=intervention,
                ):
                    loss = objective(model, x, y, loss_kind)
                    gradient = estimate_gradient(loss, model, localzo_repeats)
                    zo_gradient = (
                        weight * gradient
                        if zo_gradient is None
                        else zo_gradient + weight * gradient
                    )
            mismatch = (hard_gradient - smooth_gradient).square().sum().item()
            cosine = F.cosine_similarity(
                zo_gradient.unsqueeze(0), hard_gradient.unsqueeze(0)
            ).item()
            values = [hard_loss, soft_loss, tail, fired, mismatch, cosine]
            if not np.isfinite(values).all():
                raise FloatingPointError(
                    "non-finite diagnostic; refusing to write a valid-looking row"
                )
            records.append(
                dict(
                    group=group,
                    checkpoint_sha256=checkpoint,
                    samples=total,
                    nodes_per_sample=nodes / total,
                    tail_diagnostic=tail / nodes,
                    tail_sum_per_sample=tail / total,
                    firing_rate=fired / nodes,
                    hard_loss=hard_loss,
                    smooth_loss=soft_loss,
                    objective_gap=abs(hard_loss - soft_loss),
                    backward_mismatch_sq=mismatch,
                    localzo_alignment_cosine=cosine,
                    local_samples=local_samples,
                    gradient_replicates=localzo_repeats,
                    smooth_gradient_is_objective_gradient=(
                        not baseline.detach_reset
                        and regular_intervention
                        and (intervention is None or intervention_gradient == "exact")
                    ),
                    intervention_gradient="none" if intervention is None else intervention_gradient,
                    runtime=asdict(baseline) | {"delta": delta},
                    delta=delta,
                    objective=loss_kind,
                    model_spec=model.specification() if hasattr(model, "specification") else {},
                )
            )
    if state_digest(model) != checkpoint:
        raise RuntimeError("diagnostic unexpectedly changed parameters or persistent buffers")
    return records


def delta_sweep(model, batches, deltas, **kwargs):
    batches = list(batches)
    deltas = list(deltas)
    if not deltas:
        raise ValueError("provide at least one delta")
    return [row for delta in deltas for row in probe(model, batches, delta=delta, **kwargs)]
