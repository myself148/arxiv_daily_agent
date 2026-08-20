from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from core.contracts import PipelineState, RunRequest, RunSnapshot, RunStatus
from core.events import RunEvent


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RunStore:
    """SQLite run journal with stage-level checkpoints and event history."""

    def __init__(self, database_path: str) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(str(self.database_path), timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    current_stage TEXT,
                    request_json TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    completed_stages_json TEXT NOT NULL DEFAULT '[]',
                    result TEXT NOT NULL DEFAULT '',
                    error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS run_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    stage TEXT,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_run_events_run_id
                ON run_events(run_id, id);

                CREATE INDEX IF NOT EXISTS idx_runs_status
                ON runs(status, updated_at);
                """
            )

    def create_run(self, request: RunRequest, initial_state: PipelineState) -> RunSnapshot:
        request.validate()
        now = _utc_now()
        state = dict(initial_state)
        state.setdefault("run_id", request.run_id)
        state.setdefault("mode", request.mode)
        state.setdefault("provider_name", request.provider_name)

        try:
            with self._connection() as connection:
                connection.execute(
                    """
                    INSERT INTO runs (
                        run_id, mode, status, request_json, state_json,
                        completed_stages_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, '[]', ?, ?)
                    """,
                    (
                        request.run_id,
                        request.mode,
                        RunStatus.PENDING.value,
                        self._dump(request.to_dict()),
                        self._dump(state),
                        now,
                        now,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"Run '{request.run_id}' already exists") from exc
        return self.get_run(request.run_id)

    def mark_running(self, run_id: str) -> None:
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE runs
                SET status = ?, error = '', updated_at = ?
                WHERE run_id = ?
                """,
                (RunStatus.RUNNING.value, _utc_now(), run_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"Unknown run: {run_id}")

    def checkpoint(self, run_id: str, stage: str, state: PipelineState) -> RunSnapshot:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT completed_stages_json FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown run: {run_id}")

            completed_stages = list(json.loads(row["completed_stages_json"]))
            if stage not in completed_stages:
                completed_stages.append(stage)
            connection.execute(
                """
                UPDATE runs
                SET status = ?, current_stage = ?, state_json = ?,
                    completed_stages_json = ?, updated_at = ?
                WHERE run_id = ?
                """,
                (
                    RunStatus.RUNNING.value,
                    stage,
                    self._dump(dict(state)),
                    self._dump(completed_stages),
                    _utc_now(),
                    run_id,
                ),
            )
        return self.get_run(run_id)

    def complete(self, run_id: str, state: PipelineState, result: str) -> RunSnapshot:
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE runs
                SET status = ?, current_stage = NULL, state_json = ?,
                    result = ?, error = '', updated_at = ?
                WHERE run_id = ?
                """,
                (
                    RunStatus.COMPLETED.value,
                    self._dump(dict(state)),
                    result,
                    _utc_now(),
                    run_id,
                ),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"Unknown run: {run_id}")
        return self.get_run(run_id)

    def fail(
        self,
        run_id: str,
        state: PipelineState,
        error: str,
        *,
        current_stage: Optional[str],
    ) -> RunSnapshot:
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE runs
                SET status = ?, current_stage = ?, state_json = ?,
                    error = ?, updated_at = ?
                WHERE run_id = ?
                """,
                (
                    RunStatus.FAILED.value,
                    current_stage,
                    self._dump(dict(state)),
                    error,
                    _utc_now(),
                    run_id,
                ),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"Unknown run: {run_id}")
        return self.get_run(run_id)

    def append_event(self, event: RunEvent) -> None:
        if not event.run_id:
            return
        with self._connection() as connection:
            exists = connection.execute(
                "SELECT 1 FROM runs WHERE run_id = ?",
                (event.run_id,),
            ).fetchone()
            if exists is None:
                return
            connection.execute(
                """
                INSERT INTO run_events (
                    run_id, event_type, stage, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    event.run_id,
                    event.event_type.value,
                    event.stage,
                    self._dump(event.payload),
                    event.created_at,
                ),
            )

    def get_run(self, run_id: str) -> RunSnapshot:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown run: {run_id}")
        return self._to_snapshot(row)

    def list_runs(self, *, limit: int = 20) -> List[RunSnapshot]:
        if limit <= 0:
            return []
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._to_snapshot(row) for row in rows]

    def list_events(self, run_id: str) -> List[Dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT event_type, stage, payload_json, created_at
                FROM run_events
                WHERE run_id = ?
                ORDER BY id
                """,
                (run_id,),
            ).fetchall()
        return [
            {
                "event_type": row["event_type"],
                "stage": row["stage"],
                "payload": json.loads(row["payload_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    @staticmethod
    def _dump(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _to_snapshot(row: sqlite3.Row) -> RunSnapshot:
        return RunSnapshot(
            request=RunRequest.from_dict(json.loads(row["request_json"])),
            status=RunStatus(row["status"]),
            state=json.loads(row["state_json"]),
            completed_stages=list(json.loads(row["completed_stages_json"])),
            current_stage=row["current_stage"],
            result=row["result"],
            error=row["error"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
