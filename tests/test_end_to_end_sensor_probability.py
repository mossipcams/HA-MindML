from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

from homeassistant.core import State

from custom_components.calibrated_logistic_regression.sensor import (
    CalibratedLogisticRegressionSensor,
)


def test_config_entry_to_sensor_probability_smoke_path() -> None:
    hass = MagicMock()
    hass.states.get.side_effect = lambda entity_id: {
        "sensor.temperature": State("sensor.temperature", "21.5"),
        "sensor.humidity": State("sensor.humidity", "55"),
    }.get(entity_id)

    entry = MagicMock()
    entry.entry_id = "entry-smoke"
    entry.title = "Living Room Risk"
    entry.data = {
        "name": "Living Room Risk",
        "intercept": -8.0,
        "coefficients": {
            "sensor.temperature": 0.2,
            "sensor.humidity": 0.1,
        },
        "required_features": ["sensor.temperature", "sensor.humidity"],
        "calibration_slope": 1.0,
        "calibration_intercept": 0.0,
    }
    entry.options = {}

    sensor = CalibratedLogisticRegressionSensor(hass, entry)
    sensor._recompute_state(datetime.now())

    assert sensor.available is True
    assert sensor.native_value is not None
    assert 0.0 <= sensor.native_value <= 100.0
    attrs = sensor.extra_state_attributes
    assert attrs["missing_features"] == []
    assert attrs["feature_values"]["sensor.temperature"] == 21.5
