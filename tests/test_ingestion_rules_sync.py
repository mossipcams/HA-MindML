from __future__ import annotations

import sqlite3
from pathlib import Path

from custom_components.mindml.ingestion_rules import sync_ingestion_rules


def _create_ingestion_rules_table(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE ingestion_rules (
                id INTEGER PRIMARY KEY,
                entity_id TEXT NOT NULL,
                state TEXT NOT NULL,
                source TEXT NOT NULL,
                updated_at_utc TEXT NOT NULL,
                UNIQUE(entity_id, state, source)
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def test_sync_ingestion_rules_replaces_source_rows_atomically(tmp_path: Path) -> None:
    db_path = tmp_path / "ha_ml_data_layer.db"
    _create_ingestion_rules_table(db_path)

    inserted = sync_ingestion_rules(
        db_path=str(db_path),
        source="mindml:entry-1",
        feature_states={"sensor.a": "on", "binary_sensor.window": "off"},
    )
    assert inserted == 2

    replaced = sync_ingestion_rules(
        db_path=str(db_path),
        source="mindml:entry-1",
        feature_states={"binary_sensor.window": "on"},
    )
    assert replaced == 1

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT entity_id, state, source
            FROM ingestion_rules
            ORDER BY entity_id ASC
            """
        ).fetchall()
    finally:
        conn.close()

    assert rows == [("binary_sensor.window", "on", "mindml:entry-1")]
