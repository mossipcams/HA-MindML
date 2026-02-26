from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from custom_components.mindml.const import DOMAIN
from custom_components.mindml.diagnostics import (
    async_get_config_entry_diagnostics,
)


def test_diagnostics_returns_structured_payload_with_redaction() -> None:
    hass = MagicMock()
    hass.data = {
        DOMAIN: {
            "entry-1": {
                "runtime": {
                    "status": "ok",
                    "last_error": None,
                }
            }
        }
    }
    entry = MagicMock()
    entry.entry_id = "entry-1"
    entry.title = "Kitchen MindML"
    entry.data = {
        "name": "Kitchen MindML",
        "ml_db_path": "/config/ha_ml_data_layer.db",
        "required_features": ["sensor.a"],
    }
    entry.options = {"threshold": 60.0}

    payload = asyncio.run(async_get_config_entry_diagnostics(hass, entry))

    assert payload["entry"]["entry_id"] == "entry-1"
    assert payload["entry"]["title"] == "Kitchen MindML"
    assert payload["config"]["data"]["ml_db_path"] == "**REDACTED**"
    assert payload["runtime"]["status"] == "ok"
