"""Helpers for syncing MindML-selected feature/state pairs to ingestion rules."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime


def sync_ingestion_rules(
    *,
    db_path: str,
    source: str,
    feature_states: dict[str, str],
) -> int:
    """Replace ingestion rules for a source with the provided feature-state pairs."""
    if not db_path:
        raise ValueError("ml_db_path is required")
    if not source:
        raise ValueError("source is required")

    conn = sqlite3.connect(db_path)
    try:
        now_utc = datetime.now(UTC).replace(microsecond=0).isoformat()
        conn.execute("DELETE FROM ingestion_rules WHERE source = ?", (source,))
        rows = [
            (entity_id, str(state), source, now_utc)
            for entity_id, state in sorted(feature_states.items())
            if str(entity_id).strip() and str(state).strip()
        ]
        if rows:
            conn.executemany(
                """
                INSERT INTO ingestion_rules(entity_id, state, source, updated_at_utc)
                VALUES (?, ?, ?, ?)
                """,
                rows,
            )
        conn.commit()
        return len(rows)
    finally:
        conn.close()
