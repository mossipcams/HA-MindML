from __future__ import annotations

from custom_components.mindml.feature_mapping import (
    FEATURE_TYPE_CATEGORICAL,
    FEATURE_TYPE_NUMERIC,
    infer_feature_types_from_states,
    infer_state_mappings_from_states,
    parse_feature_types,
    parse_state_mappings,
    validate_categorical_mappings,
)


def test_parse_feature_types_requires_selected_features() -> None:
    parsed = parse_feature_types(
        '{"sensor.a": "numeric", "binary_sensor.window": "categorical"}',
        ["sensor.a", "binary_sensor.window"],
    )
    assert parsed == {"sensor.a": "numeric", "binary_sensor.window": "categorical"}


def test_parse_feature_types_rejects_unknown_values() -> None:
    assert parse_feature_types('{"sensor.a": "weird"}', ["sensor.a"]) is None


def test_parse_state_mappings_coerces_to_float() -> None:
    parsed = parse_state_mappings('{"binary_sensor.window": {"on": 1, "off": 0}}')
    assert parsed == {"binary_sensor.window": {"on": 1.0, "off": 0.0}}


def test_validate_categorical_mappings_reports_missing() -> None:
    missing = validate_categorical_mappings(
        feature_types={"sensor.a": "numeric", "binary_sensor.window": "categorical"},
        state_mappings={},
    )
    assert missing == ["binary_sensor.window"]


def test_infer_feature_types_from_states() -> None:
    feature_types = infer_feature_types_from_states(
        {
            "sensor.temperature": "21.3",
            "binary_sensor.window": "on",
            "person.matt": "home",
        }
    )
    assert feature_types["sensor.temperature"] == FEATURE_TYPE_NUMERIC
    assert feature_types["binary_sensor.window"] == FEATURE_TYPE_CATEGORICAL
    assert feature_types["person.matt"] == FEATURE_TYPE_CATEGORICAL


def test_infer_state_mappings_from_states_uses_common_defaults() -> None:
    mappings = infer_state_mappings_from_states(
        {
            "binary_sensor.window": "on",
            "person.matt": "away",
        }
    )
    assert mappings["binary_sensor.window"] == {"on": 1.0, "off": 0.0}
    assert mappings["person.matt"] == {"home": 1.0, "away": 0.0}
