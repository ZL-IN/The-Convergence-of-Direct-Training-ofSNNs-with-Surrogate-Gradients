from torch.nn import functional as F


def objective(model, x, y, kind="ce"):
    if kind == "ce":
        return F.cross_entropy(model(x), y)
    if kind == "temporal_ce":
        logits = model(x, temporal=True)
        return F.cross_entropy(logits.flatten(0, 1), y.repeat(logits.shape[0]))
    raise ValueError(f"unknown objective: {kind}")
