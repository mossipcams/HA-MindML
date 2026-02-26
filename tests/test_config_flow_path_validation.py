"""Tests for ML DB path validation in config and options flows."""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from custom_components.mindml.config_flow import (
    CalibratedLogisticRegressionConfigFlow,
    ClrOptionsFlow,
)


def test_user_step_shows_error_when_db_path_not_found() -> None:
    flow = CalibratedLogisticRegressionConfigFlow()
    flow.hass = MagicMock()
    flow._async_current_entries = MagicMock(return_value=[])

    result = asyncio.run(
        flow.async_step_user(
            {
                "name": "Test Sensor",
                "goal": "risk",
                "ml_db_path": "/nonexistent/path/that/does/not/exist/db.sqlite",
            }
        )
    )

    assert result["type"] == "form"
    assert result["step_id"] == "user"
    assert result["errors"]["ml_db_path"] == "db_not_found"


def test_options_model_shows_error_when_db_path_not_found() -> None:
    entry = MagicMock()
    entry.options = {}
    entry.data = {
        "required_features": ["sensor.a"],
        "feature_states": {"sensor.a": "22.5"},
        "feature_types": {"sensor.a": "numeric"},
        "state_mappings": {},
        "threshold": 50.0,
    }

    flow = ClrOptionsFlow(entry)
    flow.hass = MagicMock()

    result = asyncio.run(
        flow.async_step_model(
            {
                "ml_db_path": "/nonexistent/path/that/does/not/exist/db.sqlite",
                "ml_artifact_view": "vw_lightgbm_latest_model_artifact",
            }
        )
    )

    assert result["type"] == "form"
    assert result["step_id"] == "model"
    assert result["errors"]["ml_db_path"] == "db_not_found"
