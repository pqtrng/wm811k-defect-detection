"""Model architectures for wm811k pipeline.

Ported verbatim from notebooks/03_train.ipynb (cells: WaferCNN, BasicBlock,
WaferResNet18). Architectural decisions preserved exactly:
- Normalization (x / 2.0) lives INSIDE forward(), not in the dataset.
  Grad-CAM and serving must feed raw {0,1,2} tensors -- never pre-normalize.
- CIFAR-style stem in WaferResNet18: 3x3 stride-1 conv, no maxpool.
  The ImageNet stem (7x7 stride-2 + maxpool) downsamples 4x immediately,
  which is too aggressive for 64x64 inputs.
- The attribute name `layer4` is a public contract: Grad-CAM (T4) hooks it.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WaferCNN(nn.Module):
    """Small baseline CNN for 64x64 single channel wafer maps."""

    def __init__(self, num_classes: int = 8):
        super().__init__()
        self.features = nn.Sequential(
            # block 1: 1 -> 32, 64x64 -> 32x32
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # block 2: 32 -> 64, 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # block 3: 64 -> 128, 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 128x8x8 -> 128x1x1, robust to size changes
            nn.Flatten(),  # -> [N, 128]
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = x / 2.0  # normalize die values {0, 1, 2} -> {0, 0.5, 1.0}
        x = self.features(x)
        x = self.classifier(x)
        return x


class BasicBlock(nn.Module):
    """Standard ResNet BasicBlock: two 3x3 conv with a residual connection.
    The skip connection is the whole point of ResNet: it lets the gradient flow
    directly past the conv layers (identity path), so deep networks stay
    trainable instead of degrading. expansion=1 means output channels == `out_channels`.
    """

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        # First conv carries the stride: this is where spatial downsampling happens
        # at the start of stages 2/3/4 (stride=2). padding=1 keeps 3x3 size-preserving
        # when stride=1. bias=False because BN right after subtracts the mean,
        # which cancels any constant bias - BN beta does the shifting instead.
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=3, stride=1, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # downsample: a 1x1 conv (+BN) applied to the identity shortcut, used only
        # when the shortcut's shape doesn't match the main path - i.e. channels
        # changed or stride halved the spatial size. Without it, `out + identity`
        # crashes on shape mismatch. Built by _make_layer and passed in
        # (None for most blocks).
        self.downsample = downsample

    def forward(self, x):
        identity = x

        # Main path: conv -> BN -> ReLU, then conv -> BN (no ReLU yet)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Match the shortcut's shape to out if needed.
        if self.downsample is not None:
            identity = self.downsample(x)

        # The residual add, then ReLU (post-activation).
        # ReLU comes after the add so the identity signal passes through un-clipped.
        out += identity
        return F.relu(out)


class WaferResNet18(nn.Module):
    """ResNet-18 from scratch for 1-channel 64x64 wafer maps, 8 classes.
    Deviations from the ImageNet original:
        - Stem is a 3x3 stride-1 conv, no maxpool (ImageNet uses 7x7 stride-2
          + maxpool, which downsamples 4x immediately -- too aggressive for a
          64x64 input).
        - conv1 takes 1 input channel (wafer is single-channel), fc outputs
          8 classes. The core (BasicBlock, [2,2,2,2] layout, 64->128->256->512,
          stride-2 stage starts) is standard ResNet-18.
    """

    def __init__(self, num_classes: int = 8, in_channels: int = 1):
        super().__init__()
        # Running channel count while BUILDING the network. Starts at 64 because
        # that's how many channels exist right after the stem; _make_layer reads
        # and updates it.
        self.in_channels = 64

        # Stem: 1 -> 64 channels, keeps 64x64 (kernel 3 + padding 1, stride 1)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=64,
            kernel_size=3, stride=1, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)

        # 4 stages, each = num_blocks BasicBlock. Spatial 64->64->32->16->8
        self.layer1 = self._make_layer(out_channels=64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(out_channels=128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(out_channels=256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(out_channels=512, num_blocks=2, stride=2)

        # Head: pool each channel's 8x8 map to a single number, then classify.
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # 512x8x8 -> 512x1x1
        self.fc = nn.Linear(in_features=512, out_features=num_classes)  # 512 -> 8

        self._init_weights()

    def _make_layer(self, out_channels, num_blocks, stride):
        """Build one stage of num_blocks BasicBlock. Only the FIRST block may
        change shape (it carries the stride and/or the channel change), so only
        it may need a downsample on its shortcut. The remaining blocks are
        stride-1, same-channel -> no downsample.
        """
        downsample = None
        # A projection shortcut is needed only if the shortcut's shape won't
        # match the main path: either the spatial size shrinks (stride != 1) or
        # the channel count changes (self.in_channels != out_channels).
        if stride != 1 or self.in_channels != out_channels * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=out_channels * BasicBlock.expansion,
                    kernel_size=1, stride=stride, bias=False,
                ),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion),
            )

        # First block: uses the incoming self.in_channels, applies stride + downsample
        layers = [
            BasicBlock(
                in_channels=self.in_channels, out_channels=out_channels,
                stride=stride, downsample=downsample,
            )
        ]
        # After the first block the running channel count becomes out_channels
        self.in_channels = out_channels * BasicBlock.expansion
        # Remaining blocks: in == out channels, stride 1, no downsample
        for _ in range(1, num_blocks):
            layers.append(
                BasicBlock(in_channels=self.in_channels, out_channels=out_channels)
            )
        return nn.Sequential(*layers)

    def _init_weights(self):
        """Kaiming init for conv weights matches ReLU, standard BN init."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Same normalization as WaferCNN, so the architecture comparison stays clean
        x = x / 2.0  # discrete {0,1,2} -> {0, 0.5, 1.0}
        # Stem
        x = F.relu(self.bn1(self.conv1(x)))

        # 4 stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Head
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


_MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "cnn": WaferCNN,
    "resnet18": WaferResNet18,
}


def build_model(name: str, num_classes: int = 8) -> nn.Module:
    """Factory: map a config/CLI string to a model instance.

    Single source of truth for the name -> class mapping, so train.py,
    evaluate.py, gradcam.py and serve.py never duplicate an if/else chain.

    Args:
        name: one of "cnn", "resnet18" (case-insensitive).
        num_classes: pass config.num_classes from the caller; defaults to 8.

    Raises:
        ValueError: unknown name, with the list of valid options in the message
            so a CLI user can fix their --model flag immediately.
    """
    key = name.lower()
    if key not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model name {name!r}. Valid options: {sorted(_MODEL_REGISTRY)}"
        )
    return _MODEL_REGISTRY[key](num_classes=num_classes)
