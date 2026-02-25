"""Config flow for Calibrated Logistic Regression."""

from __future__ import annotations

import json
from typing import Any

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import selector

from .const import (
    CONF_CALIBRATION_INTERCEPT,
    CONF_CALIBRATION_SLOPE,
    CONF_COEFFICIENTS,
    CONF_FEATURE_TYPES,
    CONF_GOAL,
    CONF_INTERCEPT,
    CONF_NAME,
    CONF_REQUIRED_FEATURES,
    CONF_STATE_MAPPINGS,
    DEFAULT_CALIBRATION_INTERCEPT,
    DEFAULT_CALIBRATION_SLOPE,
    DEFAULT_GOAL,
    DOMAIN,
)
from .feature_mapping import (
    FEATURE_TYPE_CATEGORICAL,
    FEATURE_TYPE_NUMERIC,
    infer_feature_types_from_states,
    infer_state_mappings_from_states,
    parse_coefficients,
    parse_required_features,
    parse_state_mappings,
    validate_categorical_mappings,
)


def _build_user_schema() -> vol.Schema:
    return vol.Schema(
        {
            vol.Required(CONF_NAME): str,
            vol.Required(CONF_GOAL, default=DEFAULT_GOAL): selector.SelectSelector(
                selector.SelectSelectorConfig(
                    options=[
                        selector.SelectOptionDict(value="risk", label="Risk"),
                        selector.SelectOptionDict(
                            value="event_probability",
                            label="Event Probability",
                        ),
                        selector.SelectOptionDict(
                            value="success_probability",
                            label="Success Probability",
                        ),
                    ],
                    mode=selector.SelectSelectorMode.DROPDOWN,
                )
            ),
        }
    )


def _build_features_schema(default_features: list[str]) -> vol.Schema:
    return vol.Schema(
        {
            vol.Required(CONF_REQUIRED_FEATURES, default=default_features): selector.EntitySelector(
                selector.EntitySelectorConfig(multiple=True)
            ),
        }
    )


def _build_mappings_schema(default_mappings: str) -> vol.Schema:
    return vol.Schema(
        {
            vol.Optional(CONF_STATE_MAPPINGS, default=default_mappings): str,
        }
    )


def _build_model_schema(default_coefficients: str) -> vol.Schema:
    return vol.Schema(
        {
            vol.Required(CONF_INTERCEPT, default=0.0): vol.Coerce(float),
            vol.Required(CONF_COEFFICIENTS, default=default_coefficients): str,
            vol.Required(
                CONF_CALIBRATION_SLOPE,
                default=DEFAULT_CALIBRATION_SLOPE,
            ): vol.Coerce(float),
            vol.Required(
                CONF_CALIBRATION_INTERCEPT,
                default=DEFAULT_CALIBRATION_INTERCEPT,
            ): vol.Coerce(float),
        }
    )


def _build_preview_schema() -> vol.Schema:
    return vol.Schema(
        {
            vol.Required("confirm", default=True): bool,
        }
    )


class CalibratedLogisticRegressionConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for the integration."""

    VERSION = 1

    def __init__(self) -> None:
        self._draft: dict[str, Any] = {}

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Collect integration name and goal."""
        errors: dict[str, str] = {}
        if user_input is not None:
            name = str(user_input[CONF_NAME]).strip()
            goal = str(user_input[CONF_GOAL]).strip()
            if not name:
                errors[CONF_NAME] = "required"
            if not goal:
                errors[CONF_GOAL] = "required"

            if not errors:
                for entry in self._async_current_entries():
                    if str(entry.data.get(CONF_NAME, "")).strip().casefold() == name.casefold():
                        return self.async_abort(reason="already_configured")
                self._draft[CONF_NAME] = name
                self._draft[CONF_GOAL] = goal
                return await self.async_step_features()

        return self.async_show_form(
            step_id="user",
            data_schema=_build_user_schema(),
            errors=errors,
            description_placeholders={
                "goal_options": "risk, event_probability, success_probability",
            },
        )

    async def async_step_features(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> FlowResult:
        """Collect feature entity IDs."""
        errors: dict[str, str] = {}
        if user_input is not None:
            required_features = parse_required_features(user_input[CONF_REQUIRED_FEATURES])
            if not required_features:
                errors[CONF_REQUIRED_FEATURES] = "required"
            if not errors:
                self._draft[CONF_REQUIRED_FEATURES] = required_features
                observed_states: dict[str, str] = {}
                for feature in required_features:
                    state_obj = self.hass.states.get(feature)
                    observed_states[feature] = (
                        str(state_obj.state)
                        if state_obj is not None and hasattr(state_obj, "state")
                        else ""
                    )
                feature_types = infer_feature_types_from_states(observed_states)
                self._draft[CONF_FEATURE_TYPES] = {
                    feature: feature_types.get(feature, FEATURE_TYPE_NUMERIC)
                    for feature in required_features
                }

                inferred_state_mappings = infer_state_mappings_from_states(observed_states)
                self._draft[CONF_STATE_MAPPINGS] = {
                    feature: inferred_state_mappings[feature]
                    for feature in required_features
                    if feature in inferred_state_mappings
                    and self._draft[CONF_FEATURE_TYPES].get(feature)
                    == FEATURE_TYPE_CATEGORICAL
                }

                missing = validate_categorical_mappings(
                    feature_types=self._draft[CONF_FEATURE_TYPES],
                    state_mappings=self._draft[CONF_STATE_MAPPINGS],
                )
                if missing:
                    return await self.async_step_mappings()

                return await self.async_step_model()

        default_features = list(self._draft.get(CONF_REQUIRED_FEATURES, []))
        return self.async_show_form(
            step_id="features",
            data_schema=_build_features_schema(default_features),
            errors=errors,
            description_placeholders={
                "features_help": "Pick entities like sensor.temperature or binary_sensor.door.",
            },
        )

    async def async_step_mappings(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> FlowResult:
        """Collect categorical state mappings."""
        errors: dict[str, str] = {}
        feature_types = dict(self._draft.get(CONF_FEATURE_TYPES, {}))

        if user_input is not None:
            raw_input = user_input.get(CONF_STATE_MAPPINGS)
            if raw_input is None or str(raw_input).strip() == "":
                state_mappings = dict(self._draft.get(CONF_STATE_MAPPINGS, {}))
            else:
                state_mappings = parse_state_mappings(str(raw_input))
            if state_mappings is None:
                errors[CONF_STATE_MAPPINGS] = "invalid_state_mappings"
            else:
                missing = validate_categorical_mappings(
                    feature_types=feature_types,
                    state_mappings=state_mappings,
                )
                if missing:
                    errors[CONF_STATE_MAPPINGS] = "missing_categorical_mappings"

            if not errors:
                self._draft[CONF_STATE_MAPPINGS] = state_mappings
                return await self.async_step_model()

        default_mappings = json.dumps(self._draft.get(CONF_STATE_MAPPINGS, {}))
        return self.async_show_form(
            step_id="mappings",
            data_schema=_build_mappings_schema(default_mappings),
            errors=errors,
            description_placeholders={
                "mapping_example": '{"binary_sensor.window": {"on": 1, "off": 0}}',
            },
        )

    async def async_step_model(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> FlowResult:
        """Collect model coefficients and calibration params."""
        errors: dict[str, str] = {}
        required_features = set(self._draft.get(CONF_REQUIRED_FEATURES, []))

        if user_input is not None:
            coefficients = parse_coefficients(str(user_input[CONF_COEFFICIENTS]))
            if coefficients is None:
                errors[CONF_COEFFICIENTS] = "invalid_coefficients"
            elif set(coefficients) != required_features:
                errors[CONF_COEFFICIENTS] = "coefficient_mismatch"

            if not errors:
                self._draft[CONF_INTERCEPT] = float(user_input[CONF_INTERCEPT])
                self._draft[CONF_COEFFICIENTS] = coefficients
                self._draft[CONF_CALIBRATION_SLOPE] = float(user_input[CONF_CALIBRATION_SLOPE])
                self._draft[CONF_CALIBRATION_INTERCEPT] = float(
                    user_input[CONF_CALIBRATION_INTERCEPT]
                )
                return await self.async_step_preview()

        default_coefficients = json.dumps(
            {
                feature: 1.0
                for feature in self._draft.get(CONF_REQUIRED_FEATURES, [])
            }
        )
        return self.async_show_form(
            step_id="model",
            data_schema=_build_model_schema(default_coefficients),
            errors=errors,
            description_placeholders={
                "coefficients_example": '{"sensor.temp": 0.8, "binary_sensor.window": 1.2}',
            },
        )

    async def async_step_preview(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> FlowResult:
        """Show confirmation before creating entry."""
        if user_input is not None and bool(user_input.get("confirm")):
            return self.async_create_entry(
                title=str(self._draft[CONF_NAME]),
                data={
                    CONF_NAME: self._draft[CONF_NAME],
                    CONF_GOAL: self._draft[CONF_GOAL],
                    CONF_REQUIRED_FEATURES: self._draft[CONF_REQUIRED_FEATURES],
                    CONF_FEATURE_TYPES: self._draft[CONF_FEATURE_TYPES],
                    CONF_STATE_MAPPINGS: self._draft[CONF_STATE_MAPPINGS],
                    CONF_INTERCEPT: self._draft[CONF_INTERCEPT],
                    CONF_COEFFICIENTS: self._draft[CONF_COEFFICIENTS],
                    CONF_CALIBRATION_SLOPE: self._draft[CONF_CALIBRATION_SLOPE],
                    CONF_CALIBRATION_INTERCEPT: self._draft[CONF_CALIBRATION_INTERCEPT],
                },
            )

        return self.async_show_form(
            step_id="preview",
            data_schema=_build_preview_schema(),
            description_placeholders={
                "name": str(self._draft.get(CONF_NAME, "")),
                "goal": str(self._draft.get(CONF_GOAL, "")),
                "required_features": ", ".join(
                    self._draft.get(CONF_REQUIRED_FEATURES, [])
                ),
            },
        )

    @staticmethod
    def async_get_options_flow(config_entry):
        """Get options flow."""
        return ClrOptionsFlow(config_entry)


class ClrOptionsFlow(config_entries.OptionsFlow):
    """Handle options flow for post-setup configuration changes."""

    def __init__(self, config_entry) -> None:
        self._config_entry = config_entry

    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Show management sections."""
        return self.async_show_menu(
            step_id="init",
            menu_options=["features", "mappings", "calibration", "diagnostics"],
        )

    async def async_step_features(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> FlowResult:
        """Manage selected required features."""
        errors: dict[str, str] = {}
        if user_input is not None:
            required_features = parse_required_features(user_input[CONF_REQUIRED_FEATURES])
            if not required_features:
                errors[CONF_REQUIRED_FEATURES] = "required"
            else:
                return self.async_create_entry(
                    title="",
                    data={CONF_REQUIRED_FEATURES: required_features},
                )

        defaults = self._config_entry.options.get(
            CONF_REQUIRED_FEATURES,
            self._config_entry.data.get(CONF_REQUIRED_FEATURES, []),
        )
        return self.async_show_form(
            step_id="features",
            data_schema=_build_features_schema(defaults),
            errors=errors,
            description_placeholders={
                "features_help": "Pick entities to include as model features.",
            },
        )

    async def async_step_mappings(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> FlowResult:
        """Manage categorical mappings."""
        errors: dict[str, str] = {}
        if user_input is not None:
            state_mappings = parse_state_mappings(str(user_input.get(CONF_STATE_MAPPINGS, "{}")))
            if state_mappings is None:
                errors[CONF_STATE_MAPPINGS] = "invalid_state_mappings"
            else:
                return self.async_create_entry(
                    title="",
                    data={CONF_STATE_MAPPINGS: state_mappings},
                )

        defaults = self._config_entry.options.get(
            CONF_STATE_MAPPINGS,
            self._config_entry.data.get(CONF_STATE_MAPPINGS, {}),
        )
        return self.async_show_form(
            step_id="mappings",
            data_schema=_build_mappings_schema(json.dumps(defaults)),
            errors=errors,
        )

    async def async_step_calibration(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> FlowResult:
        """Manage calibration settings."""
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

        defaults_slope = self._config_entry.options.get(
            CONF_CALIBRATION_SLOPE,
            self._config_entry.data.get(
                CONF_CALIBRATION_SLOPE,
                DEFAULT_CALIBRATION_SLOPE,
            ),
        )
        defaults_intercept = self._config_entry.options.get(
            CONF_CALIBRATION_INTERCEPT,
            self._config_entry.data.get(
                CONF_CALIBRATION_INTERCEPT,
                DEFAULT_CALIBRATION_INTERCEPT,
            ),
        )
        return self.async_show_form(
            step_id="calibration",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_CALIBRATION_SLOPE, default=defaults_slope): vol.Coerce(float),
                    vol.Required(
                        CONF_CALIBRATION_INTERCEPT,
                        default=defaults_intercept,
                    ): vol.Coerce(float),
                }
            ),
        )

    async def async_step_diagnostics(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> FlowResult:
        """Read-only diagnostics hint step."""
        del user_input
        return self.async_show_form(
            step_id="diagnostics",
            data_schema=vol.Schema({}),
            description_placeholders={
                "configured_features": ", ".join(
                    self._config_entry.options.get(
                        CONF_REQUIRED_FEATURES,
                        self._config_entry.data.get(CONF_REQUIRED_FEATURES, []),
                    )
                )
            },
        )
