from __future__ import annotations

import math
import sys
import types

homeassistant = types.ModuleType("homeassistant")
config_entries = types.ModuleType("homeassistant.config_entries")
core = types.ModuleType("homeassistant.core")
config_entries.ConfigEntry = object
core.HomeAssistant = object
sys.modules.setdefault("homeassistant", homeassistant)
sys.modules.setdefault("homeassistant.config_entries", config_entries)
sys.modules.setdefault("homeassistant.core", core)

from custom_components.calibrated_logistic_regression.lightgbm_inference import (
    LightGBMModelSpec,
    run_lightgbm_inference,
)


def test_lightgbm_inference_returns_unavailable_when_missing_features() -> None:
    result = run_lightgbm_inference(
        feature_values={"event_count": 4.0},
        missing_features=["on_ratio"],
        model=LightGBMModelSpec(
            feature_names=["event_count", "on_ratio"],
            model_payload={"intercept": -1.0, "weights": [0.4, 0.3]},
        ),
        threshold=70.0,
    )

    assert result.available is False
    assert result.native_value is None
    assert result.unavailable_reason == "missing_or_unmapped_features"


def test_lightgbm_inference_uses_payload_weights_for_probability() -> None:
    result = run_lightgbm_inference(
        feature_values={"event_count": 4.0, "on_ratio": 0.5},
        missing_features=[],
        model=LightGBMModelSpec(
            feature_names=["event_count", "on_ratio"],
            model_payload={"intercept": -1.0, "weights": [0.4, 0.3]},
        ),
        threshold=75.0,
    )

    expected_linear = -1.0 + 0.4 * 4.0 + 0.3 * 0.5
    expected_prob = 1.0 / (1.0 + math.exp(-expected_linear))

    assert result.available is True
    assert result.linear_score == expected_linear
    assert result.raw_probability == expected_prob
    assert result.native_value == expected_prob * 100.0
    assert result.is_above_threshold is False
    assert result.decision == "negative"
