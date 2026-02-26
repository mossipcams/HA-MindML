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
    return flow


def test_wizard_happy_path_creates_entry_from_looped_feature_state_pairs() -> None:
    flow = _new_flow()

    user_result = asyncio.run(
        flow.async_step_user(
            {
                "name": "Kitchen MindML",
                "goal": "risk",
                "ml_db_path": "/tmp/ha_ml_data_layer.db",
            }
        )
    )
    assert user_result["type"] == "form"
    assert user_result["step_id"] == "features"

    first_pair = asyncio.run(
        flow.async_step_features(
            {
                "feature": "sensor.a",
                "state": "22.5",
                "threshold": 65.0,
            }
        )
    )
    assert first_pair["type"] == "form"
    assert first_pair["step_id"] == "feature_more"
    feature_more_keys = [str(k.schema) for k in first_pair["data_schema"].schema]
    assert "next_action" in feature_more_keys

    add_step = asyncio.run(flow.async_step_feature_more({"next_action": "add_feature"}))
    assert add_step["type"] == "form"
    assert add_step["step_id"] == "features"

    second_pair = asyncio.run(
        flow.async_step_features(
            {
                "feature": "binary_sensor.window",
                "state": "on",
                "threshold": 65.0,
            }
        )
    )
    assert second_pair["type"] == "form"
    assert second_pair["step_id"] == "feature_more"

    preview = asyncio.run(flow.async_step_feature_more({"next_action": "finish_features"}))
    assert preview["type"] == "form"
    assert preview["step_id"] == "preview"

    created = asyncio.run(flow.async_step_preview({"confirm": True}))
    assert created["type"] == "create_entry"
    assert created["title"] == "Kitchen MindML"
    assert created["data"]["required_features"] == ["sensor.a", "binary_sensor.window"]
    assert created["data"]["feature_states"] == {
        "sensor.a": "22.5",
        "binary_sensor.window": "on",
    }
    assert created["data"]["threshold"] == 65.0


def test_wizard_features_step_requires_feature_and_state() -> None:
    flow = _new_flow()
    asyncio.run(
        flow.async_step_user(
            {
                "name": "Kitchen MindML",
                "goal": "risk",
                "ml_db_path": "/tmp/ha_ml_data_layer.db",
            }
        )
    )

    result = asyncio.run(
        flow.async_step_features(
            {
                "feature": "sensor.a",
                "state": "",
                "threshold": 65.0,
            }
        )
    )
    assert result["type"] == "form"
    assert result["step_id"] == "features"
    assert result["errors"]["state"] == "required"


def test_user_step_aborts_duplicate_name() -> None:
    flow = CalibratedLogisticRegressionConfigFlow()
    flow.hass = MagicMock()
    existing = MagicMock()
    existing.data = {"name": "Kitchen MindML"}
    flow._async_current_entries = MagicMock(return_value=[existing])

    result = asyncio.run(
        flow.async_step_user(
            {
                "name": "Kitchen MindML",
                "goal": "risk",
                "ml_db_path": "/tmp/ha_ml_data_layer.db",
            }
        )
    )

    assert result["type"] == "abort"
    assert result["reason"] == "already_configured"


def test_options_flow_shows_management_menu() -> None:
    entry = MagicMock()
    entry.options = {}
    entry.data = {}

    flow = ClrOptionsFlow(entry)
    result = asyncio.run(flow.async_step_init())

    assert result["type"] == "menu"
    assert "model" in result["menu_options"]
    assert "feature_source" in result["menu_options"]
    assert "decision" in result["menu_options"]
    assert "features" in result["menu_options"]


def test_options_flow_decision_updates_threshold() -> None:
    entry = MagicMock()
    entry.options = {}
    entry.data = {"threshold": 50.0}

    flow = ClrOptionsFlow(entry)
    result = asyncio.run(flow.async_step_decision({"threshold": 72.5}))

    assert result["type"] == "create_entry"
    assert result["data"]["threshold"] == 72.5


def test_options_flow_features_updates_configuration_via_loop() -> None:
    entry = MagicMock()
    entry.options = {}
    entry.data = {
        "required_features": ["sensor.a"],
        "feature_states": {"sensor.a": "22.5"},
        "state_mappings": {},
        "threshold": 50.0,
    }

    flow = ClrOptionsFlow(entry)
    first_pair = asyncio.run(
        flow.async_step_features(
            {
                "feature": "sensor.a",
                "state": "23.0",
                "threshold": 65.0,
            }
        )
    )
    assert first_pair["type"] == "form"
    assert first_pair["step_id"] == "feature_more"

    asyncio.run(flow.async_step_feature_more({"next_action": "add_feature"}))
    asyncio.run(
        flow.async_step_features(
            {
                "feature": "binary_sensor.window",
                "state": "off",
                "threshold": 65.0,
            }
        )
    )
    updated = asyncio.run(flow.async_step_feature_more({"next_action": "finish_features"}))

    assert updated["type"] == "create_entry"
    assert updated["data"]["required_features"] == ["sensor.a", "binary_sensor.window"]
    assert updated["data"]["feature_states"] == {
        "sensor.a": "23.0",
        "binary_sensor.window": "off",
    }
    assert updated["data"]["threshold"] == 65.0


def test_user_step_allows_blank_ml_db_path_and_continues() -> None:
    flow = _new_flow()

    user_result = asyncio.run(
        flow.async_step_user(
            {
                "name": "Kitchen MindML",
                "goal": "risk",
                "ml_db_path": "",
            }
        )
    )

    assert user_result["type"] == "form"
    assert user_result["step_id"] == "features"


def test_user_step_blank_ml_db_path_uses_appdaemon_default() -> None:
    flow = _new_flow()

    asyncio.run(
        flow.async_step_user(
            {
                "name": "Kitchen MindML",
                "goal": "risk",
                "ml_db_path": "",
            }
        )
    )

    assert flow._draft["ml_db_path"] == "/homeassistant/appdaemon/ha_ml_data_layer.db"
