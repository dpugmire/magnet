"""Field-error metrics and CAESAR/direct-decoder error decomposition."""

from __future__ import annotations

from typing import Any

import numpy as np


def field_error_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, float]:
    reference_values = np.asarray(reference, dtype=np.float64)
    candidate_values = np.asarray(candidate, dtype=np.float64)
    if candidate_values.shape != reference_values.shape:
        raise ValueError(
            f"field shapes differ: {reference_values.shape} vs {candidate_values.shape}"
        )
    if reference_values.size == 0:
        raise ValueError("fields must not be empty")

    error = candidate_values - reference_values
    absolute_error = np.abs(error)
    rmse = float(np.sqrt(np.mean(error * error)))
    value_range = float(np.ptp(reference_values))
    reference_norm = float(np.linalg.norm(reference_values.reshape(-1)))
    candidate_flat = candidate_values.reshape(-1)
    reference_flat = reference_values.reshape(-1)
    if np.std(reference_flat) == 0.0 or np.std(candidate_flat) == 0.0:
        correlation = 1.0 if np.array_equal(reference_values, candidate_values) else 0.0
    else:
        correlation = float(np.corrcoef(reference_flat, candidate_flat)[0, 1])

    return {
        "rmse": rmse,
        "rangeNormalizedRmse": rmse / value_range if value_range > 0.0 else 0.0,
        "relativeL2": (
            float(np.linalg.norm(error.reshape(-1))) / reference_norm
            if reference_norm > 0.0
            else 0.0
        ),
        "meanAbsoluteError": float(np.mean(absolute_error)),
        "medianAbsoluteError": float(np.median(absolute_error)),
        "p95AbsoluteError": float(np.percentile(absolute_error, 95.0)),
        "p99AbsoluteError": float(np.percentile(absolute_error, 99.0)),
        "maximumAbsoluteError": float(np.max(absolute_error)),
        "correlation": correlation,
    }


def error_decomposition(
    raw_values: np.ndarray,
    caesar_values: np.ndarray,
    direct_values: np.ndarray,
) -> dict[str, Any]:
    """Separate compression, slice-decoder, and total end-to-end errors."""

    return {
        "compression": field_error_metrics(raw_values, caesar_values),
        "sliceDecoder": field_error_metrics(caesar_values, direct_values),
        "endToEnd": field_error_metrics(raw_values, direct_values),
    }


def staged_error_decomposition(
    raw_values: np.ndarray,
    base_values: np.ndarray,
    caesar_values: np.ndarray,
    direct_values: np.ndarray,
) -> dict[str, Any]:
    """Separate neural-base, residual, direct-decoder, and total errors."""

    return {
        "baseCompression": field_error_metrics(raw_values, base_values),
        "residualCorrection": field_error_metrics(base_values, caesar_values),
        "finalCompression": field_error_metrics(raw_values, caesar_values),
        "pointDecoderVsBase": field_error_metrics(base_values, direct_values),
        "pointDecoderVsFinal": field_error_metrics(caesar_values, direct_values),
        "endToEnd": field_error_metrics(raw_values, direct_values),
    }


def plane_decoder_error_decomposition(
    raw_values: np.ndarray,
    base_values: np.ndarray,
    caesar_values: np.ndarray,
    plane_values: np.ndarray,
) -> dict[str, Any]:
    """Separate CAESAR stages and plane-decoder errors."""

    return {
        "baseCompression": field_error_metrics(raw_values, base_values),
        "residualCorrection": field_error_metrics(base_values, caesar_values),
        "finalCompression": field_error_metrics(raw_values, caesar_values),
        "planeDecoderVsBase": field_error_metrics(base_values, plane_values),
        "planeDecoderVsFinal": field_error_metrics(caesar_values, plane_values),
        "endToEnd": field_error_metrics(raw_values, plane_values),
    }
