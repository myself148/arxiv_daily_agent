from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from threading import RLock
from typing import Any, Callable, Dict, List, Optional


class EventType(str, Enum):
    RUN_STARTED = "run.started"
    RUN_RESUMED = "run.resumed"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"
    STAGE_STARTED = "stage.started"
    STAGE_COMPLETED = "stage.completed"
    STAGE_FAILED = "stage.failed"
    PROVIDER_FALLBACK = "provider.fallback"
    TOOL_STARTED = "tool.started"
    TOOL_COMPLETED = "tool.completed"
    TOOL_FAILED = "tool.failed"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class RunEvent:
    event_type: EventType
    run_id: str
    stage: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)


EventSubscriber = Callable[[RunEvent], None]


class EventBus:
    """Small synchronous event bus used by runtime, tools and persistence."""

    def __init__(self) -> None:
        self._subscribers: List[EventSubscriber] = []
        self._lock = RLock()

    def subscribe(self, subscriber: EventSubscriber) -> Callable[[], None]:
        with self._lock:
            self._subscribers.append(subscriber)

        def unsubscribe() -> None:
            with self._lock:
                if subscriber in self._subscribers:
                    self._subscribers.remove(subscriber)

        return unsubscribe

    def publish(self, event: RunEvent) -> None:
        with self._lock:
            subscribers = list(self._subscribers)

        for subscriber in subscribers:
            try:
                subscriber(event)
            except Exception:  # noqa: BLE001 - observers must not break a run
                logging.exception("Run event subscriber failed for %s", event.event_type.value)
