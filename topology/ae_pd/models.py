"""Convolutional autoencoder for 2D scalar fields."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def BuildActivation(name: str) -> nn.Module:
    """Create activation layer from a short name."""

    if name.lower() == "relu":
        return nn.ReLU(inplace=True)
    if name.lower() == "silu":
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


def BuildDownBlock(inChannels: int, outChannels: int, activation: str) -> nn.Sequential:
    """Downsampling encoder block."""

    return nn.Sequential(
        nn.Conv2d(inChannels, outChannels, kernel_size=3, stride=2, padding=1),
        BuildActivation(activation),
        nn.Conv2d(outChannels, outChannels, kernel_size=3, stride=1, padding=1),
        BuildActivation(activation),
    )


def BuildUpBlock(inChannels: int, outChannels: int, activation: str) -> nn.Sequential:
    """Upsampling decoder block."""

    return nn.Sequential(
        nn.ConvTranspose2d(inChannels, outChannels, kernel_size=4, stride=2, padding=1),
        BuildActivation(activation),
        nn.Conv2d(outChannels, outChannels, kernel_size=3, stride=1, padding=1),
        BuildActivation(activation),
    )


class ConvAutoencoder(nn.Module):
    """Configurable 2D convolutional autoencoder."""

    def __init__(
        self,
        baseChannels: int = 32,
        numDown: int = 3,
        latentChannels: int = 128,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        if numDown < 1:
            raise ValueError("numDown must be >= 1")

        self.baseChannels = int(baseChannels)
        self.numDown = int(numDown)
        self.latentChannels = int(latentChannels)
        self.activation = activation

        encoderList: List[nn.Module] = []
        channelList: List[int] = []
        inChannels = 1
        for levelIdx in range(self.numDown):
            outChannels = self.baseChannels * (2**levelIdx)
            encoderList.append(BuildDownBlock(inChannels, outChannels, activation))
            channelList.append(outChannels)
            inChannels = outChannels
        self.encoder = nn.ModuleList(encoderList)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(inChannels, self.latentChannels, kernel_size=3, stride=1, padding=1),
            BuildActivation(activation),
            nn.Conv2d(self.latentChannels, inChannels, kernel_size=3, stride=1, padding=1),
            BuildActivation(activation),
        )

        decoderList: List[nn.Module] = []
        currentChannels = inChannels
        for levelIdx in reversed(range(self.numDown)):
            if levelIdx == 0:
                outChannels = self.baseChannels
            else:
                outChannels = channelList[levelIdx - 1]
            decoderList.append(BuildUpBlock(currentChannels, outChannels, activation))
            currentChannels = outChannels
        self.decoder = nn.ModuleList(decoderList)
        self.outputConv = nn.Conv2d(currentChannels, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, imageTensor: torch.Tensor) -> torch.Tensor:
        inputSpatialShape = imageTensor.shape[-2:]
        featureTensor = imageTensor
        for encoderBlock in self.encoder:
            featureTensor = encoderBlock(featureTensor)
        featureTensor = self.bottleneck(featureTensor)
        for decoderBlock in self.decoder:
            featureTensor = decoderBlock(featureTensor)
        reconTensor = self.outputConv(featureTensor)
        if reconTensor.shape[-2:] != inputSpatialShape:
            reconTensor = F.interpolate(
                reconTensor,
                size=inputSpatialShape,
                mode="bilinear",
                align_corners=False,
            )
        return reconTensor
