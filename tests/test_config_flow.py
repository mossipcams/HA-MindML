from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from custom_components.calibrated_logistic_regression.config_flow import (
    CalibratedLogisticRegressionConfigFlow,
    ClrOptionsFlow,
)


def test_user_step_shows_form_without_input() -> None:
    flow = CalibratedLogisticRegressionConfigFlow()
    flow.hass = MagicMock()
    flow._async_current_entries = MagicMock(return_value=[])

    result = asyncio.run(flow.async_step_user(None))

    assert result["type"] == "form"
    assert result["step_id"] == "user"


def test_user_step_rejects_invalid_coefficients() -> None:
    flow = CalibratedLogisticRegressionConfigFlow()
    flow.hass = MagicMock()
    flow._async_current_entries = MagicMock(return_value=[])

    result = asyncio.run(
        flow.async_step_user(
            {
                "name": "Kitchen CLR",
                "intercept": 0.0,
                "coefficients": "{bad json}",
                "required_features": "sensor.a,sensor.b",
                "calibration_slope": 1.0,
                "calibration_intercept": 0.0,
            }
        )
    )

    assert result["type"] == "form"
    assert result["errors"]["coefficients"] == "invalid_coefficients"


def test_user_step_creates_entry_for_valid_input() -> None:
    flow = CalibratedLogisticRegressionConfigFlow()
    flow.hass = MagicMock()
    flow._async_current_entries = MagicMock(return_value=[])

    result = asyncio.run(
        flow.async_step_user(
            {
                "name": "Kitchen CLR",
                "intercept": -0.1,
                "coefficients": '{"sensor.a": 0.5, "sensor.b": -0.2}',
                "required_features": "sensor.a, sensor.b",
                "calibration_slope": 1.3,
                "calibration_intercept": -0.1,
            }
        )
    )

    assert result["type"] == "create_entry"
    assert result["title"] == "Kitchen CLR"
    assert result["data"]["coefficients"]["sensor.a"] == 0.5
    assert result["data"]["required_features"] == ["sensor.a", "sensor.b"]


def test_user_step_aborts_duplicate_name() -> None:
    flow = CalibratedLogisticRegressionConfigFlow()
    flow.hass = MagicMock()
    existing = MagicMock()
    existing.data = {"name": "Kitchen CLR"}
    flow._async_current_entries = MagicMock(return_value=[existing])

    result = asyncio.run(
        flow.async_step_user(
            {
                "name": "Kitchen CLR",
                "intercept": 0.0,
                "coefficients": '{"sensor.a": 0.5}',
                "required_features": "sensor.a",
                "calibration_slope": 1.0,
                "calibration_intercept": 0.0,
            }
        )
    )

    assert result["type"] == "abort"
    assert result["reason"] == "already_configured"


def test_options_flow_updates_values() -> None:
    entry = MagicMock()
    entry.options = {}
    entry.data = {}

    flow = ClrOptionsFlow(entry)

    result = asyncio.run(
        flow.async_step_init(
            {
                "calibration_slope": 1.8,
                "calibration_intercept": -0.2,
            }
        )
    )

    assert result["type"] == "create_entry"
    assert result["data"]["calibration_slope"] == 1.8
    assert result["data"]["calibration_intercept"] == -0.2
