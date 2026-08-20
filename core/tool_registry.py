from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional

from core.context import get_execution_context
from core.events import EventBus, EventType, RunEvent


ToolHandler = Callable[..., Any]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    handler: ToolHandler
    input_schema: Dict[str, Any] = field(default_factory=dict)
    expose_to_model: bool = False

    def model_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema or {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": True,
                },
            },
        }


class ToolRegistry:
    """Registry for model-callable tools and pipeline-only services."""

    def __init__(self, event_bus: Optional[EventBus] = None) -> None:
        self._definitions: Dict[str, ToolDefinition] = {}
        self._event_bus = event_bus

    def register(self, definition: ToolDefinition, *, replace: bool = False) -> None:
        if definition.name in self._definitions and not replace:
            raise ValueError(f"Tool or service '{definition.name}' is already registered")
        self._definitions[definition.name] = definition

    def get(self, name: str) -> ToolDefinition:
        try:
            return self._definitions[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool or service: {name}") from exc

    def names(self) -> List[str]:
        return sorted(self._definitions)

    def model_schemas(self) -> List[Dict[str, Any]]:
        return [
            definition.model_schema()
            for definition in self._definitions.values()
            if definition.expose_to_model
        ]

    def invoke(
        self,
        name: str,
        arguments: Optional[Mapping[str, Any]] = None,
        *,
        run_id: Optional[str] = None,
    ) -> Any:
        definition = self.get(name)
        resolved_run_id = run_id if run_id is not None else get_execution_context().run_id
        payload = {"tool": name}
        self._publish(EventType.TOOL_STARTED, resolved_run_id, payload)

        try:
            result = definition.handler(**dict(arguments or {}))
        except Exception as exc:
            self._publish(
                EventType.TOOL_FAILED,
                resolved_run_id,
                {"tool": name, "error": str(exc)},
            )
            raise

        self._publish(
            EventType.TOOL_COMPLETED,
            resolved_run_id,
            {"tool": name, "result_type": type(result).__name__},
        )
        return result

    def _publish(self, event_type: EventType, run_id: str, payload: Dict[str, Any]) -> None:
        if self._event_bus is None:
            return
        self._event_bus.publish(
            RunEvent(event_type=event_type, run_id=run_id, payload=payload)
        )
