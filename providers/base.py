from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderResult:
    content: str
    provider_name: str
    model_name: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    raw_message: Optional[Any] = None


class ChatProvider(ABC):
    """Stable model provider contract used by the runtime."""

    name: str
    model_name: str

    @abstractmethod
    def complete(
        self,
        messages: Sequence[Any],
        *,
        operation_name: str,
        tool_schemas: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> ProviderResult:
        raise NotImplementedError
