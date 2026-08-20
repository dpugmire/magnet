"""Loss functions for vector INR training."""

from __future__ import annotations

import torch


def velocityL1Loss(predVelocity: torch.Tensor, targetVelocity: torch.Tensor) -> torch.Tensor:
    """L1 reconstruction loss on normalized velocity channels."""

    if predVelocity.shape != targetVelocity.shape:
        raise ValueError(
            f"Velocity loss expects matching shapes, got {predVelocity.shape} vs {targetVelocity.shape}."
        )
    return torch.mean(torch.abs(predVelocity - targetVelocity))


def computeAutogradVorticity(
    predVelocity: torch.Tensor,
    coords: torch.Tensor,
    createGraph: bool = True,
) -> torch.Tensor:
    """Compute omega = d(vy)/dx - d(vx)/dy from model output via autograd."""

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must be [N,3], got {coords.shape}.")
    if predVelocity.ndim != 2 or predVelocity.shape[1] != 2:
        raise ValueError(f"predVelocity must be [N,2], got {predVelocity.shape}.")
    if not coords.requires_grad:
        raise ValueError("coords.requires_grad must be True to compute autograd vorticity.")

    gradOutputs = torch.ones_like(predVelocity[:, 0])
    vyGrad = torch.autograd.grad(
        outputs=predVelocity[:, 1],
        inputs=coords,
        grad_outputs=gradOutputs,
        create_graph=createGraph,
        retain_graph=True,
        only_inputs=True,
    )[0]
    vxGrad = torch.autograd.grad(
        outputs=predVelocity[:, 0],
        inputs=coords,
        grad_outputs=gradOutputs,
        create_graph=createGraph,
        retain_graph=True,
        only_inputs=True,
    )[0]
    dVyDx = vyGrad[:, 0]
    dVxDy = vxGrad[:, 1]
    omega = dVyDx - dVxDy
    return omega


def vorticityL1Loss(predOmega: torch.Tensor, targetOmega: torch.Tensor) -> torch.Tensor:
    """L1 loss for vorticity values."""

    if predOmega.shape != targetOmega.shape:
        raise ValueError(
            f"Vorticity loss expects matching shapes, got {predOmega.shape} vs {targetOmega.shape}."
        )
    return torch.mean(torch.abs(predOmega - targetOmega))
