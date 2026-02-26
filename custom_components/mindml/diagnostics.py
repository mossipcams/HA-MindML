"""Diagnostics support for CLR integration."""

from __future__ import annotations

from typing import Any

try:
    from homeassistant.components.diagnostics import async_redact_data
except Exception:  # pragma: no cover - unit-test fallback
    async_redact_data = None

from .const import CONF_ML_DB_PATH, DOMAIN

REDACTED = "**REDACTED**"
SENSITIVE_KEYS = {CONF_ML_DB_PATH}


def _redact(payload: dict[str, Any]) -> dict[str, Any]:
    """Redact sensitive values in a diagnostics payload."""
    redacted: dict[str, Any] = {}
    for key, value in payload.items():
        if key in SENSITIVE_KEYS:
            redacted[key] = REDACTED
            continue
        if isinstance(value, dict):
            redacted[key] = _redact(value)
            continue
        redacted[key] = value
    return redacted


async def async_get_config_entry_diagnostics(hass: Any, config_entry: Any) -> dict[str, Any]:
    """Return config and runtime diagnostics for this integration entry."""
    domain_data = hass.data.get(DOMAIN, {}) if isinstance(getattr(hass, "data", None), dict) else {}
    entry_store = dict(domain_data.get(config_entry.entry_id, {}))
    runtime_data = dict(entry_store.get("runtime", {}))
    config_data = dict(config_entry.data)
    options_data = dict(config_entry.options)
    if callable(async_redact_data):
        config_data = async_redact_data(config_data, SENSITIVE_KEYS)
        options_data = async_redact_data(options_data, SENSITIVE_KEYS)
    else:
        config_data = _redact(config_data)
        options_data = _redact(options_data)

    return {
        "entry": {
            "entry_id": config_entry.entry_id,
            "title": config_entry.title,
        },
        "config": {
            "data": config_data,
            "options": options_data,
        },
        "runtime": runtime_data,
        "integration_data_keys": sorted(entry_store.keys()),
    }
