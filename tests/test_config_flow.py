from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from custom_components.calibrated_logistic_regression.config_flow import (
    CalibratedLogisticRegressionConfigFlow,
    ClrOptionsFlow,
)


def _new_flow() -> CalibratedLogisticRegressionConfigFlow:
    flow = CalibratedLogisticRegressionConfigFlow()
    flow.hass = MagicMock()
    flow._async_current_entries = MagicMock(return_value=[])
    flow.hass.states.get.side_effect = lambda entity_id: {
        "sensor.a": MagicMock(state="22.5"),
        "binary_sensor.window": MagicMock(state="on"),
        "sensor.b": MagicMock(state="5"),
    }.get(entity_id)
    return flow


def test_wizard_happy_path_creates_entry() -> None:
    flow = _new_flow()

    user_result = asyncio.run(flow.async_step_user({"name": "Kitchen CLR", "goal": "risk"}))
    assert user_result["type"] == "form"
    assert user_result["step_id"] == "features"

    features_result = asyncio.run(
        flow.async_step_features(
            {"required_features": ["sensor.a", "binary_sensor.window"]}
        )
    )
    assert features_result["type"] == "form"
    assert features_result["step_id"] == "model"

    model_result = asyncio.run(
        flow.async_step_model(
            {
                "intercept": -0.2,
                "coefficients": '{"sensor.a": 0.5, "binary_sensor.window": 0.8}',
                "calibration_slope": 1.0,
                "calibration_intercept": 0.0,
            }
        )
    )
    assert model_result["type"] == "form"
    assert model_result["step_id"] == "preview"

    preview_result = asyncio.run(flow.async_step_preview({"confirm": True}))
    assert preview_result["type"] == "create_entry"
    assert preview_result["title"] == "Kitchen CLR"
    assert preview_result["data"]["goal"] == "risk"
    assert preview_result["data"]["feature_types"]["binary_sensor.window"] == "categorical"
    assert preview_result["data"]["state_mappings"]["binary_sensor.window"] == {
        "on": 1.0,
        "off": 0.0,
    }


def test_user_step_aborts_duplicate_name() -> None:
    flow = CalibratedLogisticRegressionConfigFlow()
    flow.hass = MagicMock()
    existing = MagicMock()
    existing.data = {"name": "Kitchen CLR"}
    flow._async_current_entries = MagicMock(return_value=[existing])

    result = asyncio.run(flow.async_step_user({"name": "Kitchen CLR", "goal": "risk"}))

    assert result["type"] == "abort"
    assert result["reason"] == "already_configured"


def test_features_step_infers_types_automatically() -> None:
    flow = _new_flow()
    asyncio.run(flow.async_step_user({"name": "Kitchen CLR", "goal": "risk"}))

    result = asyncio.run(flow.async_step_features({"required_features": ["sensor.a"]}))
    assert result["type"] == "form"
    assert result["step_id"] == "model"
    assert flow._draft["feature_types"]["sensor.a"] == "numeric"


def test_mappings_step_requires_manual_mapping_for_unknown_categorical() -> None:
    flow = _new_flow()
    flow.hass.states.get.side_effect = lambda entity_id: {
        "sensor.status_text": MagicMock(state="mystery"),
    }.get(entity_id)
    asyncio.run(flow.async_step_user({"name": "Kitchen CLR", "goal": "risk"}))
    features_result = asyncio.run(
        flow.async_step_features({"required_features": ["sensor.status_text"]})
    )
    assert features_result["step_id"] == "mappings"

    result = asyncio.run(flow.async_step_mappings({"state_mappings": "{}"}))

    assert result["type"] == "form"
    assert result["errors"]["state_mappings"] == "missing_categorical_mappings"

    model_result = asyncio.run(
        flow.async_step_mappings({"state_mappings": '{"sensor.status_text": {"mystery": 0.5}}'})
    )
    assert model_result["type"] == "form"
    assert model_result["step_id"] == "model"


def test_model_step_rejects_coefficient_mismatch() -> None:
    flow = _new_flow()
    asyncio.run(flow.async_step_user({"name": "Kitchen CLR", "goal": "risk"}))
    asyncio.run(flow.async_step_features({"required_features": ["sensor.a", "sensor.b"]}))

    result = asyncio.run(
        flow.async_step_model(
            {
                "intercept": 0.0,
                "coefficients": '{"sensor.a": 1.0}',
                "calibration_slope": 1.0,
                "calibration_intercept": 0.0,
            }
        )
    )

    assert result["type"] == "form"
    assert result["errors"]["coefficients"] == "coefficient_mismatch"


def test_options_flow_shows_management_menu() -> None:
    entry = MagicMock()
    entry.options = {}
    entry.data = {}

    flow = ClrOptionsFlow(entry)
    result = asyncio.run(flow.async_step_init())

    assert result["type"] == "menu"
