"""Calibrated logistic regression math helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping


def parse_float(value: object) -> float | None:
    """Parse finite float values from user/entity input."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(parsed):
        return None
    return parsed


def safe_sigmoid(x: float) -> float:
    """Numerically stable sigmoid implementation."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def logistic_probability(
    features: Mapping[str, float],
    coefficients: Mapping[str, float],
    intercept: float,
) -> tuple[float, float]:
    """Compute base logistic-regression probability and linear score."""
    score = intercept
    for feature_name, coefficient in coefficients.items():
        score += coefficient * features.get(feature_name, 0.0)
    return safe_sigmoid(score), score


def calibrated_probability(
    *,
    base_probability: float,
    calibration_slope: float,
    calibration_intercept: float,
) -> float:
    """Apply Platt-style calibration to a base probability."""
    clamped = min(max(base_probability, 1e-12), 1 - 1e-12)
    logit = math.log(clamped / (1.0 - clamped))
    calibrated_logit = calibration_slope * logit + calibration_intercept
    return safe_sigmoid(calibrated_logit)
