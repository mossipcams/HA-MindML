"""Diagnostics support for CLR integration."""

from __future__ import annotations

from typing import Any

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
    runtime_data = dict(domain_data.get(config_entry.entry_id, {}).get("runtime", {}))

    return {
        "entry": {
            "entry_id": config_entry.entry_id,
            "title": config_entry.title,
        },
        "config": {
            "data": _redact(dict(config_entry.data)),
            "options": _redact(dict(config_entry.options)),
        },
        "runtime": runtime_data,
    }
