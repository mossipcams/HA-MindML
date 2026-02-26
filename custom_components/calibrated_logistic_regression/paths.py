"""Path helpers for integration runtime defaults."""

from __future__ import annotations

from typing import Any

from .const import DEFAULT_ML_DB_FILENAME


def resolve_ml_db_path(hass: Any, configured_path: object) -> str:
    """Return user-configured ML DB path or a Home Assistant-relative default."""
    raw_path = str(configured_path or "").strip()
    if raw_path:
        return raw_path

    config = getattr(hass, "config", None)
    path_fn = getattr(config, "path", None)
    if callable(path_fn):
        return str(path_fn(DEFAULT_ML_DB_FILENAME))
    return f"/config/{DEFAULT_ML_DB_FILENAME}"
