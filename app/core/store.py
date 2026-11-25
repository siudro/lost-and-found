"""Storage: a single SQLite database.

Chosen over a JSON pile because every question the chat asks is a filter
on structured fields, and SQLite answers those in microseconds with no
server, no credentials and no network. It also gives us the one property
the brief insists on: stable event ids and deduplicated reprocessing, via
primary keys rather than application logic.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

from .timeutil import iso_utc, parse_timestamp

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    source_id      TEXT PRIMARY KEY,
    camera_id      TEXT NOT NULL,
    venue_id       TEXT NOT NULL,
    kind           TEXT NOT NULL,
    clock          TEXT NOT NULL,
    timezone       TEXT NOT NULL,
    location_label TEXT,
    started_at     TEXT,
    duration_s     REAL,
    visibility     TEXT NOT NULL DEFAULT 'internal_only',
    media_path     TEXT,
    payload        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS windows (
    window_id    TEXT PRIMARY KEY,
    source_id    TEXT NOT NULL,
    window_start TEXT NOT NULL,
    window_end   TEXT NOT NULL,
    report_id    TEXT,
    report_date  TEXT,
    coverage     TEXT NOT NULL,
    scene_text   TEXT,
    payload      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS observations (
    observation_id TEXT PRIMARY KEY,
    source_id      TEXT NOT NULL,
    item_type      TEXT,
    primary_color  TEXT,
    is_worn        INTEGER NOT NULL DEFAULT 0,
    state          TEXT NOT NULL,
    onset_certainty TEXT,
    first_seen_at  TEXT,
    caption_text   TEXT,
    payload        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    event_id        TEXT PRIMARY KEY,
    observation_id  TEXT NOT NULL,
    source_id       TEXT NOT NULL,
    event_type      TEXT NOT NULL,
    state_at_event  TEXT NOT NULL,
    onset_certainty TEXT,
    event_time      TEXT,
    event_offset_s  REAL NOT NULL,
    window_id       TEXT NOT NULL,
    evidence_id     TEXT,
    is_worn         INTEGER NOT NULL DEFAULT 0,
    payload         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evidence (
    evidence_id  TEXT PRIMARY KEY,
    event_id     TEXT NOT NULL,
    source_id    TEXT NOT NULL,
    clip_start_s REAL NOT NULL,
    clip_end_s   REAL NOT NULL,
    seek_to_s    REAL NOT NULL,
    availability TEXT NOT NULL,
    player_ref   TEXT,
    payload      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS handovers (
    handover_id     TEXT PRIMARY KEY,
    intake_reference TEXT NOT NULL,
    recorded_by     TEXT NOT NULL,
    received_at     TEXT NOT NULL,
    status          TEXT NOT NULL,
    payload         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ingest_log (
    content_hash TEXT PRIMARY KEY,
    source_id    TEXT NOT NULL,
    window_id    TEXT NOT NULL,
    processed_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_time   ON events(event_time);
CREATE INDEX IF NOT EXISTS idx_events_state  ON events(state_at_event);
CREATE INDEX IF NOT EXISTS idx_events_source ON events(source_id);
CREATE INDEX IF NOT EXISTS idx_windows_start ON windows(window_start);
"""


@dataclass
class EventRow:
    event_id: str
    observation_id: str
    source_id: str
    event_type: str
    state_at_event: str
    onset_certainty: str | None
    event_time: datetime | None
    event_offset_s: float
    window_id: str
    evidence_id: str | None
    is_worn: bool
    payload: dict


class Store:
    """Thin wrapper over SQLite. No ORM: the queries are the interesting part."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def close(self) -> None:
        self._conn.close()

    def counts(self) -> dict[str, int]:
        tables = [
            "sources", "windows", "observations", "events",
            "evidence", "handovers",
        ]
        return {
            table: self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in tables
        }

    # -- writes ---------------------------------------------------------

    def upsert_source(self, record: dict) -> None:
        access = record.get("access", {})
        with self.transaction() as conn:
            conn.execute(
                """INSERT INTO sources (source_id, camera_id, venue_id, kind, clock,
                       timezone, location_label, started_at, duration_s, visibility,
                       media_path, payload)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(source_id) DO UPDATE SET payload=excluded.payload""",
                (
                    record["source_id"],
                    record["camera_id"],
                    record["venue_id"],
                    record["kind"],
                    record["clock"],
                    record["timezone"],
                    record.get("location_label"),
                    record.get("recording_started_at"),
                    record.get("duration_seconds"),
                    access.get("visibility", "internal_only"),
                    record.get("media_path"),
                    json.dumps(record),
                ),
            )

    def upsert_window(self, record: dict) -> bool:
        """Insert a window. Returns False when the same content was processed."""
        with self.transaction() as conn:
            content_hash = record.get("processing", {}).get("content_hash")
            if content_hash:
                seen = conn.execute(
                    "SELECT window_id FROM ingest_log WHERE content_hash = ?",
                    (content_hash,),
                ).fetchone()
                if seen and seen["window_id"] == record["window_id"]:
                    return False
                conn.execute(
                    """INSERT INTO ingest_log (content_hash, source_id, window_id, processed_at)
                       VALUES (?,?,?,?)
                       ON CONFLICT(content_hash) DO UPDATE SET processed_at=excluded.processed_at""",
                    (
                        content_hash,
                        record["source_id"],
                        record["window_id"],
                        iso_utc(datetime.now(timezone.utc)),
                    ),
                )
            conn.execute(
                """INSERT INTO windows (window_id, source_id, window_start, window_end,
                       report_id, report_date, coverage, scene_text, payload)
                   VALUES (?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(window_id) DO UPDATE SET
                       payload=excluded.payload, coverage=excluded.coverage,
                       scene_text=excluded.scene_text""",
                (
                    record["window_id"],
                    record["source_id"],
                    record["window_start"],
                    record["window_end"],
                    record.get("report_id"),
                    record.get("report_date"),
                    json.dumps(record.get("coverage", {})),
                    (record.get("scene_summary") or {}).get("text"),
                    json.dumps(record),
                ),
            )
        return True

    def upsert_observation(self, record: dict) -> None:
        attrs = record.get("attributes", {})
        with self.transaction() as conn:
            conn.execute(
                """INSERT INTO observations (observation_id, source_id, item_type,
                       primary_color, is_worn, state, onset_certainty, first_seen_at,
                       caption_text, payload)
                   VALUES (?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(observation_id) DO UPDATE SET
                       payload=excluded.payload, state=excluded.state""",
                (
                    record["observation_id"],
                    record["source_id"],
                    attrs.get("item_type"),
                    attrs.get("primary_color"),
                    1 if record.get("is_worn") else 0,
                    record["state"],
                    record.get("onset_certainty"),
                    record.get("first_seen_at"),
                    record.get("caption_text"),
                    json.dumps(record),
                ),
            )

    def upsert_event(self, record: dict) -> None:
        with self.transaction() as conn:
            conn.execute(
                """INSERT INTO events (event_id, observation_id, source_id, event_type,
                       state_at_event, onset_certainty, event_time, event_offset_s,
                       window_id, evidence_id, is_worn, payload)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(event_id) DO UPDATE SET payload=excluded.payload""",
                (
                    record["event_id"],
                    record["observation_id"],
                    record["source_id"],
                    record["event_type"],
                    record["state_at_event"],
                    record.get("onset_certainty"),
                    record.get("event_time"),
                    record["event_time_offset_s"],
                    record["window_id"],
                    record.get("evidence_id"),
                    1 if record.get("is_worn") else 0,
                    json.dumps(record),
                ),
            )

    def upsert_evidence(self, record: dict) -> None:
        with self.transaction() as conn:
            conn.execute(
                """INSERT INTO evidence (evidence_id, event_id, source_id,
                       clip_start_s, clip_end_s, seek_to_s, availability, player_ref, payload)
                   VALUES (?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(evidence_id) DO UPDATE SET
                       availability=excluded.availability""",
                (
                    record["evidence_id"],
                    record["event_id"],
                    record["source_id"],
                    record["clip_start_offset_s"],
                    record["clip_end_offset_s"],
                    record["seek_to_offset_s"],
                    record["availability"],
                    record.get("player_ref"),
                    json.dumps(record),
                ),
            )

    def add_handover(self, record: dict) -> None:
        """The only path by which a verified handover can exist."""
        with self.transaction() as conn:
            conn.execute(
                """INSERT INTO handovers (handover_id, intake_reference, recorded_by,
                       received_at, status, payload)
                   VALUES (?,?,?,?,?,?)
                   ON CONFLICT(handover_id) DO UPDATE SET payload=excluded.payload""",
                (
                    record["handover_id"],
                    record["intake_reference"],
                    record["recorded_by"],
                    record["received_at"],
                    record["verification"]["status"],
                    json.dumps(record),
                ),
            )

    # -- reads ----------------------------------------------------------

    def search_events(
        self,
        *,
        start: datetime,
        end: datetime,
        item_terms: Iterable[str] = (),
        colours: Iterable[str] = (),
        source_ids: Iterable[str] = (),
        exclude_worn: bool = True,
        limit: int = 50,
    ) -> list[EventRow]:
        """Filter events by time first, then by appearance."""
        clauses = ["event_time IS NOT NULL", "event_time >= ?", "event_time < ?"]
        params: list = [iso_utc(start), iso_utc(end)]

        if exclude_worn:
            clauses.append("is_worn = 0")

        items = [i for i in item_terms]
        if items:
            # An item whose type the detector could not determine (NULL) can
            # still be the jacket being asked about, so it stays in the
            # results. The ranking puts confirmed types first and the answer
            # says the type is unknown.
            marks = ",".join("?" for _ in items)
            clauses.append(
                f"observation_id IN (SELECT observation_id FROM observations "
                f"WHERE item_type IN ({marks}) OR item_type IS NULL)"
            )
            params.extend(items)

        colour_list = [c for c in colours]
        if colour_list:
            marks = ",".join("?" for _ in colour_list)
            clauses.append(
                f"observation_id IN (SELECT observation_id FROM observations "
                f"WHERE primary_color IN ({marks}))"
            )
            params.extend(colour_list)

        source_list = [s for s in source_ids]
        if source_list:
            marks = ",".join("?" for _ in source_list)
            clauses.append(f"source_id IN ({marks})")
            params.extend(source_list)

        sql = (
            "SELECT * FROM events WHERE " + " AND ".join(clauses)
            + " ORDER BY event_time ASC LIMIT ?"
        )
        params.append(limit)

        rows = self._conn.execute(sql, params).fetchall()
        return [
            EventRow(
                event_id=r["event_id"],
                observation_id=r["observation_id"],
                source_id=r["source_id"],
                event_type=r["event_type"],
                state_at_event=r["state_at_event"],
                onset_certainty=r["onset_certainty"],
                event_time=parse_timestamp(r["event_time"]) if r["event_time"] else None,
                event_offset_s=r["event_offset_s"],
                window_id=r["window_id"],
                evidence_id=r["evidence_id"],
                is_worn=bool(r["is_worn"]),
                payload=json.loads(r["payload"]),
            )
            for r in rows
        ]

    def coverage_between(self, start: datetime, end: datetime) -> list[dict]:
        """Which windows exist in a period, and which have holes."""
        rows = self._conn.execute(
            """SELECT * FROM windows
               WHERE window_start >= ? AND window_end <= ?
               ORDER BY window_start ASC""",
            (iso_utc(start), iso_utc(end)),
        ).fetchall()
        return [
            {
                "window_id": r["window_id"],
                "source_id": r["source_id"],
                "window_start": r["window_start"],
                "window_end": r["window_end"],
                "coverage": json.loads(r["coverage"]),
                "scene_text": r["scene_text"],
            }
            for r in rows
        ]

    def sources(self) -> list[dict]:
        rows = self._conn.execute("SELECT * FROM sources").fetchall()
        return [json.loads(r["payload"]) for r in rows]

    def observation(self, observation_id: str) -> dict | None:
        """The full observation record, for joining attributes onto an event."""
        row = self._conn.execute(
            "SELECT payload FROM observations WHERE observation_id = ?",
            (observation_id,),
        ).fetchone()
        return json.loads(row["payload"]) if row else None

    def evidence_for(self, evidence_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT payload FROM evidence WHERE evidence_id = ?", (evidence_id,)
        ).fetchone()
        return json.loads(row["payload"]) if row else None

    def handovers(self) -> list[dict]:
        rows = self._conn.execute("SELECT payload FROM handovers").fetchall()
        return [json.loads(r["payload"]) for r in rows]

    def events_for_window(self, window_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT payload FROM events WHERE window_id = ?", (window_id,)
        ).fetchall()
        return [json.loads(r["payload"]) for r in rows]

    def source(self, source_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT payload FROM sources WHERE source_id = ?", (source_id,)
        ).fetchone()
        return json.loads(row["payload"]) if row else None

    def observations_for_source(self, source_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT payload FROM observations WHERE source_id = ? ORDER BY first_seen_at",
            (source_id,),
        ).fetchall()
        return [json.loads(r["payload"]) for r in rows]

    def events_for_source(self, source_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT payload FROM events WHERE source_id = ? ORDER BY event_time",
            (source_id,),
        ).fetchall()
        return [json.loads(r["payload"]) for r in rows]

    def footage_spans(self) -> list[tuple[str, str]]:
        """(earliest window start, latest window end) per source, oldest first."""
        rows = self._conn.execute(
            """SELECT MIN(window_start) AS s, MAX(window_end) AS e FROM windows
               GROUP BY source_id ORDER BY s"""
        ).fetchall()
        return [(r["s"], r["e"]) for r in rows]
