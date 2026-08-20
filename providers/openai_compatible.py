from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from providers.base import ChatProvider, ProviderResult, ToolCall


@dataclass(frozen=True)
class OpenAICompatibleSettings:
    name: str
    model_name: str
    api_key: str
    base_url: str
    temperature: float
    timeout: int
    max_retries: int
    retry_base_delay: float
    retry_max_delay: float


class OpenAICompatibleProvider(ChatProvider):
    def __init__(self, settings: OpenAICompatibleSettings) -> None:
        self.settings = settings
        self.name = settings.name
        self.model_name = settings.model_name
        self._model: Optional[Any] = None

    def _get_model(self) -> Any:
        if self._model is None:
            if not self.settings.api_key:
                raise ValueError("OPENAI_API_KEY is missing. Please set it in your .env file.")

            from langchain_openai import ChatOpenAI

            self._model = ChatOpenAI(
                model=self.settings.model_name,
                temperature=self.settings.temperature,
                api_key=self.settings.api_key,
                base_url=self.settings.base_url,
                timeout=self.settings.timeout,
                max_retries=0,
            )
        return self._model

    def complete(
        self,
        messages: Sequence[Any],
        *,
        operation_name: str,
        tool_schemas: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> ProviderResult:
        # Keep provider construction import-light: the OpenAI SDK is needed only
        # when this concrete provider is actually invoked.
        from tools.llm_utils import run_with_retry

        model = self._get_model()
        invoker = model.bind_tools(list(tool_schemas)) if tool_schemas else model
        response = run_with_retry(
            lambda: invoker.invoke(list(messages)),
            operation_name=f"{operation_name} via {self.name}/{self.model_name}",
            max_retries=self.settings.max_retries,
            base_delay=self.settings.retry_base_delay,
            max_delay=self.settings.retry_max_delay,
        )

        content = response.content
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)

        return ProviderResult(
            content=content.strip(),
            provider_name=self.name,
            model_name=self.model_name,
            tool_calls=self._parse_tool_calls(getattr(response, "tool_calls", None)),
            raw_message=response,
        )

    @staticmethod
    def _parse_tool_calls(raw_calls: Optional[Sequence[Any]]) -> List[ToolCall]:
        calls: List[ToolCall] = []
        for index, raw_call in enumerate(raw_calls or [], start=1):
            if isinstance(raw_call, dict):
                name = str(raw_call.get("name", ""))
                call_id = str(raw_call.get("id") or f"tool-call-{index}")
                arguments = raw_call.get("args", raw_call.get("arguments", {}))
            else:
                name = str(getattr(raw_call, "name", ""))
                call_id = str(getattr(raw_call, "id", "") or f"tool-call-{index}")
                arguments = getattr(raw_call, "args", {})

            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"value": arguments}
            if name:
                calls.append(ToolCall(id=call_id, name=name, arguments=dict(arguments or {})))
        return calls
