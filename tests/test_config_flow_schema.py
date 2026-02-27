from __future__ import annotations

from custom_components.mindml.config_flow import (
    _build_features_schema,
    _build_user_schema,
)


def test_user_schema_contains_name_goal_and_ml_settings() -> None:
    schema = _build_user_schema()
    keys = [str(k.schema) for k in schema.schema]
    assert "name" in keys
    assert "goal" in keys
    assert "ml_db_path" in keys
    assert "ml_artifact_view" in keys
    assert "ml_feature_source" in keys
    assert "ml_feature_view" in keys
    assert "bed_presence_entity" in keys


def test_features_schema_contains_required_features() -> None:
    schema = _build_features_schema(
        "sensor.a",
        "22",
        50.0,
    )
    keys = [str(k.schema) for k in schema.schema]
    assert "feature" in keys
    assert "state" in keys
    assert "threshold" in keys
    assert "state_mappings" not in keys


def test_features_schema_orders_state_fields_before_threshold() -> None:
    schema = _build_features_schema(
        "sensor.a",
        "22",
        50.0,
    )
    keys = [str(k.schema) for k in schema.schema]
    assert keys == ["feature", "state", "threshold"]
