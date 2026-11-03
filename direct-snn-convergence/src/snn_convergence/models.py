"""Independent reference architectures. Legacy state_dicts are not compatible."""

from dataclasses import dataclass, asdict

from torch import nn

from .neurons import LIF
from .surrogates import Runtime


@dataclass(frozen=True)
class ModelConfig:
    architecture: str = "scnn"
    classes: int = 10
    steps: int = 32
    width: int = 128
    beta: float = 0.5

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in (self.classes, self.steps, self.width)):
            raise ValueError("classes, steps and width must be positive integers")


class TimeApply(nn.Module):
    def __init__(self, operation):
        super().__init__()
        self.operation = operation

    def forward(self, x):
        t, b = x.shape[:2]
        y = self.operation(x.flatten(0, 1))
        return y.reshape(t, b, *y.shape[1:])


def conv_block(cin, cout, dimension, beta, runtime, stride=1):
    conv = nn.Conv1d if dimension == 1 else nn.Conv2d
    bn = nn.BatchNorm1d if dimension == 1 else nn.BatchNorm2d
    return nn.Sequential(
        TimeApply(
            nn.Sequential(conv(cin, cout, 3, stride=stride, padding=1, bias=False), bn(cout))
        ),
        LIF(beta, runtime),
    )


class SEWBlock(nn.Module):
    def __init__(self, cin, cout, stride, beta, runtime):
        super().__init__()
        self.branch = nn.Sequential(
            conv_block(cin, cout, 2, beta, runtime, stride),
            conv_block(cout, cout, 2, beta, runtime),
        )
        self.shortcut = (
            nn.Identity()
            if cin == cout and stride == 1
            else nn.Sequential(
                TimeApply(
                    nn.Sequential(
                        nn.Conv2d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm2d(cout)
                    )
                ),
                LIF(beta, runtime),
            )
        )

    def forward(self, x):
        return self.branch(x) + self.shortcut(x)


class ReferenceSNN(nn.Module):
    def __init__(self, config, runtime):
        super().__init__()
        self.config = config
        c, beta, width = config.classes, config.beta, config.width
        architecture = config.architecture
        if architecture == "scnn":
            blocks = []
            for index in range(6):
                if index == 3:
                    blocks.append(TimeApply(nn.AvgPool1d(2)))
                blocks.append(conv_block(3 if index == 0 else width, width, 1, beta, runtime))
            blocks += [
                TimeApply(
                    nn.Sequential(
                        nn.AdaptiveAvgPool1d(8), nn.Flatten(1), nn.Linear(width * 8, width * 2)
                    )
                ),
                LIF(beta, runtime),
            ]
            self.features = nn.Sequential(*blocks)
            self.readout = nn.Linear(width * 2, c)
        elif architecture == "shd_mlp":
            self.features = nn.Sequential(
                TimeApply(nn.Linear(700, width)),
                LIF(beta, runtime),
                TimeApply(nn.Linear(width, width)),
                LIF(beta, runtime),
            )
            self.readout = nn.Linear(width, c)
        elif architecture == "gesture7":
            blocks = [conv_block(2, width, 2, beta, runtime)]
            for _ in range(7):
                blocks += [
                    SEWBlock(width, width, 1, beta, runtime),
                    TimeApply(nn.MaxPool2d(2, ceil_mode=True)),
                ]
            self.features = nn.Sequential(
                *blocks, TimeApply(nn.AdaptiveAvgPool2d(1)), TimeApply(nn.Flatten(1))
            )
            self.readout = nn.Linear(width, c)
        elif architecture == "spiking_vgg":
            blocks, cin = [], 2
            for scale in (1, 2, 4, 8):
                cout = width * scale
                blocks += [
                    conv_block(cin, cout, 2, beta, runtime),
                    conv_block(cout, cout, 2, beta, runtime),
                    TimeApply(nn.AvgPool2d(2, ceil_mode=True)),
                ]
                cin = cout
            self.features = nn.Sequential(
                *blocks, TimeApply(nn.AdaptiveAvgPool2d(1)), TimeApply(nn.Flatten(1))
            )
            self.readout = nn.Linear(cin, c)
        elif architecture == "sew_resnet34":
            blocks, cin = [conv_block(3, width, 2, beta, runtime)], width
            for stage, count in enumerate((3, 4, 6, 3)):
                cout = width * (2**stage)
                for block in range(count):
                    stride = 2 if stage and block == 0 else 1
                    blocks.append(SEWBlock(cin, cout, stride, beta, runtime))
                    cin = cout
            self.features = nn.Sequential(
                *blocks, TimeApply(nn.AdaptiveAvgPool2d(1)), TimeApply(nn.Flatten(1))
            )
            self.readout = nn.Linear(cin, c)
        else:
            raise ValueError(f"unknown architecture: {architecture}")

    def forward(self, x, temporal=False):
        if self.config.architecture == "scnn":
            if x.ndim != 4 or x.shape[1:3] != (3, 32) or x.shape[3] != self.config.steps:
                raise ValueError("SCNN expects [batch, 3, 32, steps]")
            sequence = x.permute(3, 0, 1, 2)
        elif self.config.architecture == "shd_mlp":
            if x.ndim != 3 or x.shape[1:] != (self.config.steps, 700):
                raise ValueError("SHD expects [batch, steps, 700]")
            sequence = x.transpose(0, 1)
        elif self.config.architecture == "sew_resnet34":
            if x.ndim != 4 or x.shape[1] != 3:
                raise ValueError("SEW-ResNet expects [batch, 3, height, width]")
            sequence = x.unsqueeze(0).expand(self.config.steps, *x.shape)
        else:
            if x.ndim != 5 or x.shape[1:3] != (self.config.steps, 2):
                raise ValueError("event CNN expects [batch, steps, 2, height, width]")
            sequence = x.transpose(0, 1)
        logits = self.readout(self.features(sequence))
        return logits if temporal else logits.mean(0)

    def specification(self):
        return asdict(self.config)


def build_model(config=None, runtime=None):
    return ReferenceSNN(config or ModelConfig(), runtime or Runtime())
