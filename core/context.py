from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterator, Optional


@dataclass(frozen=True)
class ExecutionContext:
    run_id: str = ""
    provider_name: Optional[str] = None
    event_bus: Optional[Any] = None


_CURRENT_CONTEXT: ContextVar[ExecutionContext] = ContextVar(
    "arxiv_agent_execution_context",
    default=ExecutionContext(),
)


def get_execution_context() -> ExecutionContext:
    return _CURRENT_CONTEXT.get()


@contextmanager
def use_execution_context(context: ExecutionContext) -> Iterator[None]:
    token = _CURRENT_CONTEXT.set(context)
    try:
        yield
    finally:
        _CURRENT_CONTEXT.reset(token)
