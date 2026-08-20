from __future__ import annotations

import json
from typing import Any, List, Optional, Sequence

from core.context import get_execution_context
from core.tool_registry import ToolRegistry
from providers.router import ProviderRouter


class ModelToolLoop:
    """Owns only the model -> tool -> model conversation loop."""

    def __init__(
        self,
        provider_router: ProviderRouter,
        tool_registry: Optional[ToolRegistry] = None,
        *,
        max_iterations: int = 8,
    ) -> None:
        if max_iterations <= 0:
            raise ValueError("max_iterations must be greater than 0")
        self.provider_router = provider_router
        self.tool_registry = tool_registry or ToolRegistry()
        self.max_iterations = max_iterations

    def run(
        self,
        messages: Sequence[Any],
        *,
        operation_name: str,
        provider_name: Optional[str] = None,
    ) -> str:
        history: List[Any] = list(messages)
        selected_provider = provider_name or get_execution_context().provider_name

        for _ in range(self.max_iterations):
            result = self.provider_router.complete(
                history,
                operation_name=operation_name,
                provider_name=selected_provider,
                tool_schemas=self.tool_registry.model_schemas(),
            )
            if not result.tool_calls:
                return result.content.strip()

            history.append(
                result.raw_message
                if result.raw_message is not None
                else {
                    "role": "assistant",
                    "content": result.content,
                    "tool_calls": [call.__dict__ for call in result.tool_calls],
                }
            )
            for tool_call in result.tool_calls:
                tool_result = self.tool_registry.invoke(
                    tool_call.name,
                    tool_call.arguments,
                )
                history.append(self._tool_message(tool_call.id, tool_call.name, tool_result))

        raise RuntimeError(
            f"Model/tool loop exceeded {self.max_iterations} iterations for '{operation_name}'"
        )

    @staticmethod
    def _tool_message(call_id: str, name: str, result: Any) -> Any:
        content = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        try:
            from langchain_core.messages import ToolMessage

            return ToolMessage(content=content, tool_call_id=call_id, name=name)
        except ImportError:
            return {
                "role": "tool",
                "tool_call_id": call_id,
                "name": name,
                "content": content,
            }
