from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from threading import Event
from typing import Callable, Iterator, List, Optional, Set
from uuid import uuid4

from core.contracts import RunRequest


class CronField:
    def __init__(self, expression: str, minimum: int, maximum: int, *, dow: bool = False) -> None:
        self.expression = expression.strip()
        self.minimum = minimum
        self.maximum = maximum
        self.is_wildcard = self.expression.startswith("*")
        self.values = self._parse(self.expression, dow=dow)

    def _parse(self, expression: str, *, dow: bool) -> Set[int]:
        if not expression:
            raise ValueError("Cron field must not be empty")
        values: Set[int] = set()
        for token in expression.split(","):
            token = token.strip()
            if not token:
                raise ValueError(f"Invalid cron field: {expression}")
            base, separator, step_text = token.partition("/")
            step = int(step_text) if separator else 1
            if step <= 0:
                raise ValueError("Cron step must be greater than 0")

            if base == "*":
                start, end = self.minimum, self.maximum
            elif "-" in base:
                start_text, end_text = base.split("-", 1)
                start, end = int(start_text), int(end_text)
            else:
                start = int(base)
                end = self.maximum if separator else start

            if start < self.minimum or end > self.maximum or start > end:
                raise ValueError(
                    f"Cron value '{token}' is outside {self.minimum}-{self.maximum}"
                )
            for value in range(start, end + 1, step):
                values.add(0 if dow and value == 7 else value)

        if not values:
            raise ValueError(f"Cron field produced no values: {expression}")
        return values

    def matches(self, value: int) -> bool:
        return value in self.values


class CronExpression:
    """Five-field cron expression supporting lists, ranges and step values."""

    def __init__(self, expression: str) -> None:
        fields = expression.split()
        if len(fields) != 5:
            raise ValueError("Cron expression must have five fields: minute hour day month weekday")
        self.expression = " ".join(fields)
        self.minute = CronField(fields[0], 0, 59)
        self.hour = CronField(fields[1], 0, 23)
        self.day = CronField(fields[2], 1, 31)
        self.month = CronField(fields[3], 1, 12)
        self.weekday = CronField(fields[4], 0, 7, dow=True)

    def matches(self, value: datetime) -> bool:
        cron_weekday = (value.weekday() + 1) % 7
        day_match = self.day.matches(value.day)
        weekday_match = self.weekday.matches(cron_weekday)
        if self.day.is_wildcard or self.weekday.is_wildcard:
            calendar_day_matches = day_match and weekday_match
        else:
            calendar_day_matches = day_match or weekday_match
        return (
            self.minute.matches(value.minute)
            and self.hour.matches(value.hour)
            and self.month.matches(value.month)
            and calendar_day_matches
        )

    def next_after(self, value: datetime) -> datetime:
        candidate = value.replace(second=0, microsecond=0) + timedelta(minutes=1)
        max_minutes = 366 * 24 * 60 * 6
        for _ in range(max_minutes):
            if self.matches(candidate):
                return candidate
            candidate += timedelta(minutes=1)
        raise ValueError(f"Could not resolve a future time for cron expression: {self.expression}")


@dataclass(frozen=True)
class Schedule:
    schedule_id: str
    name: str
    cron: str
    mode: str
    query: str
    max_results: int
    rag_enabled: bool
    provider_name: Optional[str]
    enabled: bool
    next_run_at: str
    last_run_at: str = ""
    last_run_id: str = ""
    last_error: str = ""
    created_at: str = ""


class SchedulerStore:
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
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schedules (
                    schedule_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    cron TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    query TEXT NOT NULL,
                    max_results INTEGER NOT NULL,
                    rag_enabled INTEGER NOT NULL,
                    provider_name TEXT,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    next_run_at TEXT NOT NULL,
                    last_run_at TEXT NOT NULL DEFAULT '',
                    last_run_id TEXT NOT NULL DEFAULT '',
                    last_error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                )
                """
            )

    def add(
        self,
        *,
        name: str,
        cron: str,
        mode: str,
        query: str,
        max_results: int,
        rag_enabled: bool,
        provider_name: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> Schedule:
        expression = CronExpression(cron)
        request = RunRequest(
            mode=mode,
            query=query,
            max_results=max_results,
            rag_enabled=rag_enabled,
            provider_name=provider_name,
        )
        request.validate()
        current = now or datetime.now()
        schedule_id = uuid4().hex
        next_run_at = expression.next_after(current).isoformat(timespec="seconds")
        created_at = current.isoformat(timespec="seconds")
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO schedules (
                    schedule_id, name, cron, mode, query, max_results,
                    rag_enabled, provider_name, enabled, next_run_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                """,
                (
                    schedule_id,
                    name.strip() or schedule_id,
                    expression.expression,
                    mode,
                    query,
                    max_results,
                    int(rag_enabled),
                    provider_name,
                    next_run_at,
                    created_at,
                ),
            )
        return self.get(schedule_id)

    def get(self, schedule_id: str) -> Schedule:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM schedules WHERE schedule_id = ?",
                (schedule_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown schedule: {schedule_id}")
        return self._to_schedule(row)

    def list(self) -> List[Schedule]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM schedules ORDER BY created_at, schedule_id"
            ).fetchall()
        return [self._to_schedule(row) for row in rows]

    def due(self, now: Optional[datetime] = None) -> List[Schedule]:
        current = (now or datetime.now()).isoformat(timespec="seconds")
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM schedules
                WHERE enabled = 1 AND next_run_at <= ?
                ORDER BY next_run_at
                """,
                (current,),
            ).fetchall()
        return [self._to_schedule(row) for row in rows]

    def set_enabled(self, schedule_id: str, enabled: bool, *, now: Optional[datetime] = None) -> None:
        schedule = self.get(schedule_id)
        next_run_at = schedule.next_run_at
        if enabled:
            next_run_at = CronExpression(schedule.cron).next_after(
                now or datetime.now()
            ).isoformat(timespec="seconds")
        with self._connection() as connection:
            connection.execute(
                "UPDATE schedules SET enabled = ?, next_run_at = ? WHERE schedule_id = ?",
                (int(enabled), next_run_at, schedule_id),
            )

    def remove(self, schedule_id: str) -> None:
        with self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM schedules WHERE schedule_id = ?",
                (schedule_id,),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"Unknown schedule: {schedule_id}")

    def mark_finished(
        self,
        schedule: Schedule,
        *,
        finished_at: datetime,
        run_id: str = "",
        error: str = "",
    ) -> None:
        next_run_at = CronExpression(schedule.cron).next_after(finished_at)
        with self._connection() as connection:
            connection.execute(
                """
                UPDATE schedules
                SET last_run_at = ?, last_run_id = ?, last_error = ?, next_run_at = ?
                WHERE schedule_id = ?
                """,
                (
                    finished_at.isoformat(timespec="seconds"),
                    run_id,
                    error,
                    next_run_at.isoformat(timespec="seconds"),
                    schedule.schedule_id,
                ),
            )

    @staticmethod
    def _to_schedule(row: sqlite3.Row) -> Schedule:
        return Schedule(
            schedule_id=row["schedule_id"],
            name=row["name"],
            cron=row["cron"],
            mode=row["mode"],
            query=row["query"],
            max_results=int(row["max_results"]),
            rag_enabled=bool(row["rag_enabled"]),
            provider_name=row["provider_name"],
            enabled=bool(row["enabled"]),
            next_run_at=row["next_run_at"],
            last_run_at=row["last_run_at"],
            last_run_id=row["last_run_id"],
            last_error=row["last_error"],
            created_at=row["created_at"],
        )


RunCallback = Callable[[RunRequest], str]


class SchedulerService:
    """Invokes Python pipelines directly; it never launches an opaque shell script."""

    def __init__(
        self,
        store: SchedulerStore,
        run_callback: RunCallback,
        *,
        poll_seconds: float = 30.0,
    ) -> None:
        if poll_seconds <= 0:
            raise ValueError("poll_seconds must be greater than 0")
        self.store = store
        self.run_callback = run_callback
        self.poll_seconds = poll_seconds

    def run_pending(self, now: Optional[datetime] = None) -> List[str]:
        current = now or datetime.now()
        run_ids: List[str] = []
        for schedule in self.store.due(current):
            request = RunRequest(
                mode=schedule.mode,
                query=schedule.query,
                max_results=schedule.max_results,
                rag_enabled=schedule.rag_enabled,
                provider_name=schedule.provider_name,
            )
            try:
                run_id = self.run_callback(request)
                run_ids.append(run_id)
                self.store.mark_finished(
                    schedule,
                    finished_at=datetime.now(),
                    run_id=run_id,
                )
            except Exception as exc:  # noqa: BLE001 - one job must not stop the scheduler
                self.store.mark_finished(
                    schedule,
                    finished_at=datetime.now(),
                    run_id=request.run_id,
                    error=str(exc),
                )
        return run_ids

    def serve(self, stop_event: Optional[Event] = None) -> None:
        stop = stop_event or Event()
        while not stop.is_set():
            self.run_pending()
            stop.wait(self.poll_seconds)
