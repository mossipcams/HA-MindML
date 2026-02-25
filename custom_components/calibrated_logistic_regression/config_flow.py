"""Config flow for Calibrated Logistic Regression."""

from __future__ import annotations

import json
from typing import Any

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.data_entry_flow import FlowResult

from .const import (
    CONF_CALIBRATION_INTERCEPT,
    CONF_CALIBRATION_SLOPE,
    CONF_COEFFICIENTS,
    CONF_INTERCEPT,
    CONF_NAME,
    CONF_REQUIRED_FEATURES,
    DEFAULT_CALIBRATION_INTERCEPT,
    DEFAULT_CALIBRATION_SLOPE,
    DOMAIN,
)


def _parse_coefficients(raw: str) -> dict[str, float] | None:
    """Parse coefficients JSON and coerce numeric values."""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None

    coefficients: dict[str, float] = {}
    for key, value in parsed.items():
        if not isinstance(key, str) or not key:
            return None
        try:
            coefficients[key] = float(value)
        except (TypeError, ValueError):
            return None
    return coefficients


def _parse_required_features(raw: str) -> list[str]:
    """Parse comma-separated list of required feature entity IDs."""
    return [item.strip() for item in raw.split(",") if item.strip()]


class CalibratedLogisticRegressionConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for the integration."""

    VERSION = 1

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Handle the user step."""
        errors: dict[str, str] = {}

        if user_input is not None:
            coefficients = _parse_coefficients(str(user_input[CONF_COEFFICIENTS]))
            if coefficients is None:
                errors[CONF_COEFFICIENTS] = "invalid_coefficients"
            required_features = _parse_required_features(
                str(user_input[CONF_REQUIRED_FEATURES])
            )
            if not required_features:
                errors[CONF_REQUIRED_FEATURES] = "required"

            name = str(user_input[CONF_NAME]).strip()
            if not name:
                errors[CONF_NAME] = "required"

            if not errors:
                for entry in self._async_current_entries():
                    if str(entry.data.get(CONF_NAME, "")).strip().casefold() == name.casefold():
                        return self.async_abort(reason="already_configured")

                return self.async_create_entry(
                    title=name,
                    data={
                        CONF_NAME: name,
                        CONF_INTERCEPT: float(user_input[CONF_INTERCEPT]),
                        CONF_COEFFICIENTS: coefficients,
                        CONF_REQUIRED_FEATURES: required_features,
                        CONF_CALIBRATION_SLOPE: float(user_input[CONF_CALIBRATION_SLOPE]),
                        CONF_CALIBRATION_INTERCEPT: float(
                            user_input[CONF_CALIBRATION_INTERCEPT]
                        ),
                    },
                )

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_NAME): str,
                    vol.Required(CONF_INTERCEPT, default=0.0): vol.Coerce(float),
                    vol.Required(CONF_COEFFICIENTS, default='{"sensor.example": 1.0}'): str,
                    vol.Required(CONF_REQUIRED_FEATURES, default="sensor.example"): str,
                    vol.Required(
                        CONF_CALIBRATION_SLOPE,
                        default=DEFAULT_CALIBRATION_SLOPE,
                    ): vol.Coerce(float),
                    vol.Required(
                        CONF_CALIBRATION_INTERCEPT,
                        default=DEFAULT_CALIBRATION_INTERCEPT,
                    ): vol.Coerce(float),
                }
            ),
            errors=errors,
        )

    @staticmethod
    def async_get_options_flow(config_entry):
        """Get options flow."""
        return ClrOptionsFlow(config_entry)


class ClrOptionsFlow(config_entries.OptionsFlow):
    """Handle options flow for calibration updates."""

    def __init__(self, config_entry) -> None:
        self._config_entry = config_entry

    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Manage options."""
        if user_input is not None:
            return self.async_create_entry(
                title="",
                data={
                    CONF_CALIBRATION_SLOPE: float(user_input[CONF_CALIBRATION_SLOPE]),
                    CONF_CALIBRATION_INTERCEPT: float(
                        user_input[CONF_CALIBRATION_INTERCEPT]
                    ),
                },
            )

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_CALIBRATION_SLOPE,
                        default=self._config_entry.options.get(
                            CONF_CALIBRATION_SLOPE,
                            DEFAULT_CALIBRATION_SLOPE,
                        ),
                    ): vol.Coerce(float),
                    vol.Required(
                        CONF_CALIBRATION_INTERCEPT,
                        default=self._config_entry.options.get(
                            CONF_CALIBRATION_INTERCEPT,
                            DEFAULT_CALIBRATION_INTERCEPT,
                        ),
                    ): vol.Coerce(float),
                }
            ),
        )
