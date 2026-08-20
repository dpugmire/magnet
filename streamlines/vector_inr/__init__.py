"""Vector INR package for ADIOS2 time-varying 2D vector fields."""

from .adios_dataset import MultiEnsembleDataset, NormalizationStats
from .models import buildVectorInrModel
from .train import TrainConfig, runTraining

__all__ = [
    "MultiEnsembleDataset",
    "NormalizationStats",
    "buildVectorInrModel",
    "TrainConfig",
    "runTraining",
]
