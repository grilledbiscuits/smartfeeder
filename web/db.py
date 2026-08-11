"""SQLite access for the visits table.

This is the only module that talks SQL. The dashboard reads through
get_recent_visits(); the future feeder/inference process is expected to call
add_visit() once it has classified a visit and saved its media -- that's the
whole interface contract between the two sides.
"""

from __future__ import annotations

import sqlite3

from web.paths import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS visits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    species TEXT NOT NULL,
    confidence REAL NOT NULL,
    image_filename TEXT,
    video_filename TEXT,
    duration_seconds REAL
);
"""


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    # WAL so the feeder process can write while the dashboard reads.
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_connection()
    try:
        conn.execute(SCHEMA)
        conn.commit()
    finally:
        conn.close()


def add_visit(
    timestamp: str,
    species: str,
    confidence: float,
    image_filename: str | None = None,
    video_filename: str | None = None,
    duration_seconds: float | None = None,
) -> int:
    """Insert one visit record. Returns the new row's id.

    This is the function the feeder/inference pipeline should call -- not
    integrated into it yet, but this is the contract it will use.
    """
    conn = get_connection()
    try:
        cursor = conn.execute(
            """
            INSERT INTO visits
                (timestamp, species, confidence, image_filename, video_filename, duration_seconds)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (timestamp, species, confidence, image_filename, video_filename, duration_seconds),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_recent_visits(limit: int = 20) -> list[sqlite3.Row]:
    conn = get_connection()
    try:
        return conn.execute(
            "SELECT * FROM visits ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
    finally:
        conn.close()
