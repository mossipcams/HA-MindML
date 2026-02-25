"""Parsing and validation helpers for feature typing and categorical mappings."""

from __future__ import annotations

import json
from typing import Final

from .model import parse_float

FEATURE_TYPE_NUMERIC: Final = "numeric"
FEATURE_TYPE_CATEGORICAL: Final = "categorical"
_VALID_FEATURE_TYPES: Final[frozenset[str]] = frozenset(
    {FEATURE_TYPE_NUMERIC, FEATURE_TYPE_CATEGORICAL}
)
_KNOWN_STATE_MAPPINGS: Final[dict[str, dict[str, float]]] = {
    "on": {"on": 1.0, "off": 0.0},
    "off": {"on": 1.0, "off": 0.0},
    "true": {"true": 1.0, "false": 0.0},
    "false": {"true": 1.0, "false": 0.0},
    "home": {"home": 1.0, "away": 0.0},
    "away": {"home": 1.0, "away": 0.0},
    "open": {"open": 1.0, "closed": 0.0},
    "closed": {"open": 1.0, "closed": 0.0},
}


def parse_required_features(raw: object) -> list[str]:
    """Parse required feature entity IDs from selector list or comma-separated string."""
    if isinstance(raw, list):
        parsed: list[str] = []
        for item in raw:
            if isinstance(item, str):
                cleaned = item.strip()
                if cleaned:
                    parsed.append(cleaned)
        return parsed

    if isinstance(raw, str):
        return [item.strip() for item in raw.split(",") if item.strip()]

    return []


def parse_coefficients(raw: str) -> dict[str, float] | None:
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


def parse_feature_types(
    raw: str,
    required_features: list[str],
) -> dict[str, str] | None:
    """Parse/validate feature types for selected features."""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None

    selected = set(required_features)
    parsed_keys = set(parsed.keys())
    if parsed_keys != selected:
        return None

    feature_types: dict[str, str] = {}
    for feature, feature_type in parsed.items():
        if not isinstance(feature, str) or feature not in selected:
            return None
        if not isinstance(feature_type, str):
            return None

        normalized = feature_type.strip().casefold()
        if normalized not in _VALID_FEATURE_TYPES:
            return None
        feature_types[feature] = normalized

    return feature_types


def parse_state_mappings(raw: str) -> dict[str, dict[str, float]] | None:
    """Parse nested JSON mapping for non-numeric state encoding."""
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None

    mappings: dict[str, dict[str, float]] = {}
    for entity_id, states in parsed.items():
        if not isinstance(entity_id, str) or not entity_id:
            return None
        if not isinstance(states, dict):
            return None

        per_state: dict[str, float] = {}
        for state_name, encoded_value in states.items():
            if not isinstance(state_name, str) or not state_name:
                return None
            try:
                per_state[state_name] = float(encoded_value)
            except (TypeError, ValueError):
                return None
        mappings[entity_id] = per_state

    return mappings


def validate_categorical_mappings(
    *,
    feature_types: dict[str, str],
    state_mappings: dict[str, dict[str, float]],
) -> list[str]:
    """Return sorted list of categorical features missing mapping tables."""
    missing = [
        feature
        for feature, feature_type in feature_types.items()
        if feature_type == FEATURE_TYPE_CATEGORICAL and not state_mappings.get(feature)
    ]
    missing.sort()
    return missing


def infer_feature_types_from_states(
    entity_states: dict[str, str],
) -> dict[str, str]:
    """Infer feature type from current observed states."""
    inferred: dict[str, str] = {}
    for entity_id, state in entity_states.items():
        if parse_float(state) is not None:
            inferred[entity_id] = FEATURE_TYPE_NUMERIC
        else:
            inferred[entity_id] = FEATURE_TYPE_CATEGORICAL
    return inferred


def infer_state_mappings_from_states(
    entity_states: dict[str, str],
) -> dict[str, dict[str, float]]:
    """Infer default mapping tables for categorical states."""
    mappings: dict[str, dict[str, float]] = {}
    for entity_id, raw_state in entity_states.items():
        normalized = raw_state.casefold()
        if normalized in _KNOWN_STATE_MAPPINGS:
            mappings[entity_id] = dict(_KNOWN_STATE_MAPPINGS[normalized])
    return mappings
