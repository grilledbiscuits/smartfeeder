"""Unified image manifest (SQLite), shared by every fetcher.

One row per image, regardless of source. SQLite rather than parquet as the
primary store because fetchers must be resumable and idempotent: they checkpoint
after every page, and a run that dies four hours in must pick up where it left
off. Parquet is written on demand for analysis (`export_parquet`).

Licence and attribution are recorded for EVERY image, without exception. CC-BY
and CC-BY-NC both require attribution on downstream use, so a row that cannot
say who made the image is a row we cannot legally use.

`observation_id` and `observer_id` are not decorative -- Phase 3 splits are
grouped on them. Multiple photos of the same individual in one burst straddling
a train/test boundary makes every metric fiction.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    image_id              TEXT PRIMARY KEY,   -- "{source}:{source_image_id}"
    source                TEXT NOT NULL,      -- inaturalist | gbif | wikimedia | flickr
    source_url            TEXT NOT NULL,      -- direct URL to the image bytes
    page_url              TEXT,               -- human-facing page, for attribution
    local_path            TEXT,               -- relative to repo root; NULL until downloaded
    sha256                TEXT,               -- of the downloaded bytes
    phash                 TEXT,               -- perceptual hash, Phase 3

    scientific_name       TEXT NOT NULL,
    common_name           TEXT,
    resolved_taxon_key    INTEGER,            -- GBIF usageKey
    inat_taxon_id         INTEGER,
    tier                  TEXT NOT NULL,

    observation_id        TEXT,               -- grouping key for splits
    observer_id           TEXT,               -- grouping key for splits
    sex_annotation        TEXT,               -- raw source value, NOT yet mapped
    life_stage_annotation TEXT,
    annotation_source     TEXT,

    license               TEXT NOT NULL,
    rights_holder         TEXT,
    attribution_text      TEXT,

    lat                   REAL,
    lon                   REAL,
    observed_date         TEXT,
    quality_grade         TEXT,

    width                 INTEGER,
    height                INTEGER,
    status                TEXT NOT NULL DEFAULT 'pending',
        -- pending | downloaded | failed | quarantined | duplicate
    status_detail         TEXT,
    split                 TEXT                -- train | val | test, set in Phase 3
);

CREATE INDEX IF NOT EXISTS idx_images_species ON images(scientific_name);
CREATE INDEX IF NOT EXISTS idx_images_status  ON images(status);
CREATE INDEX IF NOT EXISTS idx_images_obs     ON images(observation_id);
CREATE INDEX IF NOT EXISTS idx_images_phash   ON images(phash);
CREATE INDEX IF NOT EXISTS idx_images_split   ON images(split);

-- Resumability. One row per (source, species, query variant); `cursor` holds the
-- last id_above value consumed so a killed run resumes mid-species.
CREATE TABLE IF NOT EXISTS fetch_state (
    source       TEXT NOT NULL,
    species      TEXT NOT NULL,
    variant      TEXT NOT NULL DEFAULT 'default',
    cursor       TEXT,
    fetched      INTEGER NOT NULL DEFAULT 0,
    exhausted    INTEGER NOT NULL DEFAULT 0,
    updated_at   TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (source, species, variant)
);
"""


@dataclass
class ImageRecord:
    """One image, as a fetcher produces it (before download)."""

    image_id: str
    source: str
    source_url: str
    scientific_name: str
    tier: str
    license: str

    page_url: str | None = None
    local_path: str | None = None
    sha256: str | None = None
    phash: str | None = None
    common_name: str | None = None
    resolved_taxon_key: int | None = None
    inat_taxon_id: int | None = None
    observation_id: str | None = None
    observer_id: str | None = None
    sex_annotation: str | None = None
    life_stage_annotation: str | None = None
    annotation_source: str | None = None
    rights_holder: str | None = None
    attribution_text: str | None = None
    lat: float | None = None
    lon: float | None = None
    observed_date: str | None = None
    quality_grade: str | None = None
    width: int | None = None
    height: int | None = None
    status: str = "pending"
    status_detail: str | None = None
    split: str | None = None


class Manifest:
    """Thin wrapper over the SQLite store.

    Deliberately not an ORM: the access patterns are a handful of bulk upserts
    and aggregate queries, and the schema is the documentation.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        # WAL lets a long fetch keep writing while a query reads counts.
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Manifest:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- writes ----------------------------------------------------------------

    def upsert_images(self, records: list[ImageRecord]) -> int:
        """Insert new images, ignoring ones already present.

        Idempotent by design: re-running a fetcher over the same page must not
        duplicate rows or clobber a download that already succeeded.
        """
        if not records:
            return 0
        cols = [f.name for f in fields(ImageRecord)]
        sql = (
            f"INSERT INTO images ({','.join(cols)}) "
            f"VALUES ({','.join('?' * len(cols))}) "
            "ON CONFLICT(image_id) DO NOTHING"
        )
        rows = [tuple(getattr(r, c) for c in cols) for r in records]
        cur = self.conn.executemany(sql, rows)
        self.conn.commit()
        return cur.rowcount

    def mark(self, image_id: str, **updates: Any) -> None:
        if not updates:
            return
        sets = ",".join(f"{k}=?" for k in updates)
        self.conn.execute(
            f"UPDATE images SET {sets} WHERE image_id=?", (*updates.values(), image_id)
        )

    def commit(self) -> None:
        self.conn.commit()

    # -- fetch state -----------------------------------------------------------

    def get_state(self, source: str, species: str, variant: str = "default") -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM fetch_state WHERE source=? AND species=? AND variant=?",
            (source, species, variant),
        ).fetchone()

    def set_state(
        self,
        source: str,
        species: str,
        cursor: str | None,
        fetched: int,
        exhausted: bool = False,
        variant: str = "default",
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO fetch_state (source, species, variant, cursor, fetched, exhausted)
            VALUES (?,?,?,?,?,?)
            ON CONFLICT(source, species, variant) DO UPDATE SET
                cursor=excluded.cursor,
                fetched=excluded.fetched,
                exhausted=excluded.exhausted,
                updated_at=datetime('now')
            """,
            (source, species, variant, cursor, fetched, int(exhausted)),
        )
        self.conn.commit()

    # -- reads -----------------------------------------------------------------

    def count(self, **where: Any) -> int:
        sql = "SELECT COUNT(*) FROM images"
        params: list[Any] = []
        if where:
            sql += " WHERE " + " AND ".join(f"{k}=?" for k in where)
            params = list(where.values())
        return self.conn.execute(sql, params).fetchone()[0]

    def pending(self, limit: int | None = None) -> list[sqlite3.Row]:
        sql = "SELECT * FROM images WHERE status='pending'"
        if limit:
            sql += f" LIMIT {int(limit)}"
        return self.conn.execute(sql).fetchall()

    def iter_rows(self, where: str = "", params: tuple = ()) -> Iterator[sqlite3.Row]:
        sql = "SELECT * FROM images"
        if where:
            sql += f" WHERE {where}"
        yield from self.conn.execute(sql, params)

    def species_counts(self) -> list[sqlite3.Row]:
        """Per-species counts by status, for the data report."""
        return self.conn.execute(
            """
            SELECT scientific_name, tier,
                   COUNT(*) AS total,
                   SUM(status='downloaded') AS downloaded,
                   SUM(status='failed')     AS failed,
                   SUM(status='duplicate')  AS duplicate,
                   SUM(sex_annotation IS NOT NULL) AS sexed
            FROM images GROUP BY scientific_name, tier
            ORDER BY tier, scientific_name
            """
        ).fetchall()

    def sex_counts(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            """
            SELECT scientific_name, sex_annotation, COUNT(*) AS n
            FROM images
            WHERE status='downloaded'
            GROUP BY scientific_name, sex_annotation
            ORDER BY scientific_name, sex_annotation
            """
        ).fetchall()

    def export_parquet(self, path: Path) -> int:
        """Snapshot to parquet for analysis. SQLite remains the source of truth."""
        import pandas as pd

        df = pd.read_sql_query("SELECT * FROM images", self.conn)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        return len(df)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while data := fh.read(chunk):
            h.update(data)
    return h.hexdigest()


@contextlib.contextmanager
def open_manifest(path: Path) -> Iterator[Manifest]:
    m = Manifest(path)
    try:
        yield m
    finally:
        m.close()


def main() -> None:
    """Print manifest status: `uv run python -m birdcam.data.manifest`."""
    from birdcam.config import load_config

    cfg = load_config()
    path = cfg.path("manifest_db")
    if not path.is_file():
        print(f"No manifest at {path}. Run a fetcher first.")
        return

    with open_manifest(path) as m:
        rows = m.species_counts()
        if not rows:
            print("Manifest is empty.")
            return
        header = (
            f"{'species':<28} {'tier':<5} {'total':>7} {'ok':>7} "
            f"{'fail':>6} {'dup':>6} {'sexed':>7}"
        )
        print(header)
        print("-" * 74)
        for r in rows:
            print(
                f"{r['scientific_name']:<28} {r['tier']:<5} {r['total']:>7} "
                f"{r['downloaded'] or 0:>7} {r['failed'] or 0:>6} "
                f"{r['duplicate'] or 0:>6} {r['sexed'] or 0:>7}"
            )
        print(f"\ntotal rows: {m.count()}   downloaded: {m.count(status='downloaded')}")


if __name__ == "__main__":
    main()
