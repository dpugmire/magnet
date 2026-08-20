"""Loss functions for AE reconstruction and topology matching."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .topology import ComputePairedPersistenceImages


def ComputeReconLoss(reconTensor: torch.Tensor, targetTensor: torch.Tensor) -> torch.Tensor:
    """L1 reconstruction loss."""

    return F.l1_loss(reconTensor, targetTensor)


class PdLossWrapper:
    """CPU PD/PI loss wrapper.

    Notes:
    - PD/PI extraction is done with NumPy + Gudhi on detached tensors.
    - This term does not provide gradients back to the autoencoder output.
    """

    def __init__(
        self,
        pdDims: Sequence[int],
        pdDownsample: int,
        piRes: int,
        piSigma: float = 1.5,
        pdLossType: str = "huber",
        weightByPersistence: bool = True,
        minPersistence: float = 0.0,
    ) -> None:
        self.pdDims = [int(dimValue) for dimValue in pdDims]
        self.pdDownsample = int(pdDownsample)
        self.piRes = int(piRes)
        self.piSigma = float(piSigma)
        self.weightByPersistence = bool(weightByPersistence)
        self.minPersistence = float(minPersistence)
        pdLossTypeLower = pdLossType.lower()
        if pdLossTypeLower == "l2":
            self.criterion = nn.MSELoss(reduction="mean")
        elif pdLossTypeLower == "huber":
            self.criterion = nn.SmoothL1Loss(reduction="mean")
        else:
            raise ValueError(f"Unsupported pdLossType: {pdLossType}")
        self.pdLossType = pdLossTypeLower

    def ComputeLoss(
        self,
        reconBatch: torch.Tensor,
        targetBatch: torch.Tensor,
        maxBatchItems: int = 1,
    ) -> torch.Tensor:
        device = reconBatch.device
        batchSize = int(reconBatch.shape[0])
        useCount = max(0, min(batchSize, int(maxBatchItems)))
        if useCount == 0:
            return torch.zeros((), device=device, dtype=torch.float32)

        lossList = []
        for batchIdx in range(useCount):
            gtImage = np.asarray(
                targetBatch[batchIdx, 0].detach().cpu().numpy(),
                dtype=np.float32,
            )
            reconImage = np.asarray(
                reconBatch[batchIdx, 0].detach().cpu().numpy(),
                dtype=np.float32,
            )
            pairResult = ComputePairedPersistenceImages(
                gtImage=gtImage,
                reconImage=reconImage,
                pdDims=self.pdDims,
                pdDownsample=self.pdDownsample,
                piRes=self.piRes,
                piSigma=self.piSigma,
                weightByPersistence=self.weightByPersistence,
                minPersistence=self.minPersistence,
            )
            gtPiTensor = torch.from_numpy(pairResult["gtPi"]).to(device=device, dtype=torch.float32)
            reconPiTensor = torch.from_numpy(pairResult["reconPi"]).to(device=device, dtype=torch.float32)
            lossList.append(self.criterion(reconPiTensor, gtPiTensor))

        return torch.stack(lossList).mean()
