"""Model definitions for vector INR (Fourier-feature MLP and SIREN)."""

from __future__ import annotations

import math
from typing import Dict

import torch
from torch import nn


class FourierFeatures(nn.Module):
    """Fourier feature encoder for continuous coordinates."""

    def __init__(
        self,
        inputDim: int,
        numFrequencies: int,
        includeInput: bool = True,
    ) -> None:
        super().__init__()
        if inputDim <= 0:
            raise ValueError("inputDim must be > 0.")
        if numFrequencies < 0:
            raise ValueError("numFrequencies must be >= 0.")
        self.inputDim = inputDim
        self.numFrequencies = numFrequencies
        self.includeInput = includeInput
        frequencies = (2.0 ** torch.arange(numFrequencies, dtype=torch.float32)) * math.pi
        self.register_buffer("frequencies", frequencies, persistent=False)

    @property
    def outputDim(self) -> int:
        baseDim = self.inputDim if self.includeInput else 0
        return baseDim + (2 * self.inputDim * self.numFrequencies)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2 or inputs.shape[1] != self.inputDim:
            raise ValueError(
                f"FourierFeatures expects shape [N, {self.inputDim}], got {tuple(inputs.shape)}."
            )
        if self.numFrequencies == 0:
            if self.includeInput:
                return inputs
            return inputs.new_zeros((inputs.shape[0], 0))

        expanded = inputs[..., None] * self.frequencies[None, None, :]
        sinFeatures = torch.sin(expanded)
        cosFeatures = torch.cos(expanded)
        encoded = torch.cat([sinFeatures, cosFeatures], dim=-1).reshape(inputs.shape[0], -1)
        if self.includeInput:
            encoded = torch.cat([inputs, encoded], dim=1)
        return encoded


class Mlp(nn.Module):
    """Configurable ReLU MLP."""

    def __init__(
        self,
        inputDim: int,
        hiddenDim: int,
        hiddenLayers: int,
        outputDim: int = 2,
    ) -> None:
        super().__init__()
        if inputDim <= 0 or outputDim <= 0:
            raise ValueError("inputDim/outputDim must be > 0.")
        if hiddenDim <= 0:
            raise ValueError("hiddenDim must be > 0.")
        if hiddenLayers < 0:
            raise ValueError("hiddenLayers must be >= 0.")

        layers = []
        currentDim = inputDim
        for _ in range(hiddenLayers):
            layers.append(nn.Linear(currentDim, hiddenDim))
            layers.append(nn.ReLU())
            currentDim = hiddenDim
        layers.append(nn.Linear(currentDim, outputDim))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class SineLayer(nn.Module):
    """SIREN sine layer with recommended initialization."""

    def __init__(
        self,
        inputDim: int,
        outputDim: int,
        omega0: float = 30.0,
        isFirst: bool = False,
    ) -> None:
        super().__init__()
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.omega0 = float(omega0)
        self.isFirst = isFirst
        self.linear = nn.Linear(inputDim, outputDim)
        self.resetParameters()

    def resetParameters(self) -> None:
        with torch.no_grad():
            if self.isFirst:
                bound = 1.0 / float(self.inputDim)
            else:
                bound = math.sqrt(6.0 / float(self.inputDim)) / self.omega0
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega0 * self.linear(inputs))


class SirenMlp(nn.Module):
    """SIREN network with one final linear output layer."""

    def __init__(
        self,
        inputDim: int,
        hiddenDim: int,
        hiddenLayers: int,
        outputDim: int = 2,
        omega0: float = 30.0,
    ) -> None:
        super().__init__()
        if hiddenLayers < 1:
            raise ValueError("SIREN requires hiddenLayers >= 1.")

        layers = [SineLayer(inputDim, hiddenDim, omega0=omega0, isFirst=True)]
        for _ in range(hiddenLayers - 1):
            layers.append(SineLayer(hiddenDim, hiddenDim, omega0=omega0, isFirst=False))
        self.hiddenNetwork = nn.ModuleList(layers)
        self.finalLinear = nn.Linear(hiddenDim, outputDim)

        with torch.no_grad():
            bound = math.sqrt(6.0 / float(hiddenDim)) / float(omega0)
            self.finalLinear.weight.uniform_(-bound, bound)
            self.finalLinear.bias.uniform_(-bound, bound)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = inputs
        for layer in self.hiddenNetwork:
            features = layer(features)
        return self.finalLinear(features)


class FourierVectorInr(nn.Module):
    """INR using Fourier-encoded (x, y, t) + learned ensemble embedding."""

    def __init__(
        self,
        ensembleCount: int,
        embedDim: int,
        hiddenDim: int,
        hiddenLayers: int,
        numFrequencies: int,
    ) -> None:
        super().__init__()
        if ensembleCount <= 0:
            raise ValueError("ensembleCount must be > 0.")
        if embedDim <= 0:
            raise ValueError("embedDim must be > 0.")
        self.ensembleEmbedding = nn.Embedding(ensembleCount, embedDim)
        self.fourierEncoder = FourierFeatures(inputDim=3, numFrequencies=numFrequencies)
        self.mlp = Mlp(
            inputDim=self.fourierEncoder.outputDim + embedDim,
            hiddenDim=hiddenDim,
            hiddenLayers=hiddenLayers,
            outputDim=2,
        )

    def forward(self, coords: torch.Tensor, ensembleIndices: torch.Tensor) -> torch.Tensor:
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"Expected coords [N,3], got {tuple(coords.shape)}.")
        if ensembleIndices.ndim != 1 or ensembleIndices.shape[0] != coords.shape[0]:
            raise ValueError(
                "ensembleIndices must be 1D with same batch size as coords. "
                f"Got {tuple(ensembleIndices.shape)} for batch {coords.shape[0]}."
            )
        embedded = self.ensembleEmbedding(ensembleIndices)
        encodedCoords = self.fourierEncoder(coords)
        features = torch.cat([encodedCoords, embedded], dim=1)
        return self.mlp(features)


class SirenVectorInr(nn.Module):
    """INR using SIREN over raw (x, y, t) + learned ensemble embedding."""

    def __init__(
        self,
        ensembleCount: int,
        embedDim: int,
        hiddenDim: int,
        hiddenLayers: int,
        omega0: float = 30.0,
    ) -> None:
        super().__init__()
        if ensembleCount <= 0:
            raise ValueError("ensembleCount must be > 0.")
        if embedDim <= 0:
            raise ValueError("embedDim must be > 0.")
        self.ensembleEmbedding = nn.Embedding(ensembleCount, embedDim)
        self.siren = SirenMlp(
            inputDim=3 + embedDim,
            hiddenDim=hiddenDim,
            hiddenLayers=hiddenLayers,
            outputDim=2,
            omega0=omega0,
        )

    def forward(self, coords: torch.Tensor, ensembleIndices: torch.Tensor) -> torch.Tensor:
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"Expected coords [N,3], got {tuple(coords.shape)}.")
        if ensembleIndices.ndim != 1 or ensembleIndices.shape[0] != coords.shape[0]:
            raise ValueError(
                "ensembleIndices must be 1D with same batch size as coords. "
                f"Got {tuple(ensembleIndices.shape)} for batch {coords.shape[0]}."
            )
        embedded = self.ensembleEmbedding(ensembleIndices)
        features = torch.cat([coords, embedded], dim=1)
        return self.siren(features)


def buildVectorInrModel(
    modelName: str,
    ensembleCount: int,
    embedDim: int,
    hiddenDim: int,
    hiddenLayers: int,
    numFrequencies: int,
    sirenOmega0: float = 30.0,
) -> nn.Module:
    """Factory for supported INR models."""

    normalizedName = modelName.strip().lower()
    if normalizedName == "fourier":
        return FourierVectorInr(
            ensembleCount=ensembleCount,
            embedDim=embedDim,
            hiddenDim=hiddenDim,
            hiddenLayers=hiddenLayers,
            numFrequencies=numFrequencies,
        )
    if normalizedName == "siren":
        return SirenVectorInr(
            ensembleCount=ensembleCount,
            embedDim=embedDim,
            hiddenDim=hiddenDim,
            hiddenLayers=hiddenLayers,
            omega0=sirenOmega0,
        )
    raise ValueError(f"Unknown model '{modelName}'. Expected 'fourier' or 'siren'.")


def getModelMetadata(
    modelName: str,
    ensembleCount: int,
    embedDim: int,
    hiddenDim: int,
    hiddenLayers: int,
    numFrequencies: int,
    sirenOmega0: float,
) -> Dict[str, float]:
    """Serialize model configuration into checkpoint-friendly metadata."""

    return {
        "model": modelName,
        "ensembleCount": int(ensembleCount),
        "embedDim": int(embedDim),
        "hidden": int(hiddenDim),
        "layers": int(hiddenLayers),
        "freqs": int(numFrequencies),
        "sirenOmega0": float(sirenOmega0),
    }
