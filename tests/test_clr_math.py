from __future__ import annotations

import math

from custom_components.calibrated_logistic_regression.model import (
    calibrated_probability,
    logistic_probability,
    parse_float,
    safe_sigmoid,
)


def test_logistic_probability_known_coefficients() -> None:
    features = {"x1": 1.0, "x2": 3.0}
    coefficients = {"x1": 0.5, "x2": -0.2}
    intercept = -0.1

    p, score = logistic_probability(features, coefficients, intercept)

    expected_score = -0.1 + 0.5 * 1.0 + (-0.2) * 3.0
    expected_p = 1 / (1 + math.exp(-expected_score))
    assert score == expected_score
    assert math.isclose(p, expected_p, rel_tol=0.0, abs_tol=1e-15)


def test_calibrated_probability_platt_scaling() -> None:
    base_probability = 0.8

    calibrated = calibrated_probability(
        base_probability=base_probability,
        calibration_slope=0.5,
        calibration_intercept=-0.2,
    )

    logit = math.log(base_probability / (1 - base_probability))
    expected = 1 / (1 + math.exp(-(0.5 * logit - 0.2)))
    assert calibrated == expected


def test_safe_sigmoid_stable_at_extremes() -> None:
    assert safe_sigmoid(1e9) == 1.0
    assert safe_sigmoid(-1e9) == 0.0


def test_parse_float_rejects_non_finite_values() -> None:
    assert parse_float("1.25") == 1.25
    assert parse_float("nan") is None
    assert parse_float("inf") is None
    assert parse_float("not-a-number") is None
