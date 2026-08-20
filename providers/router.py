from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from core.context import get_execution_context
from core.events import EventBus, EventType, RunEvent
from providers.base import ChatProvider, ProviderResult
from providers.openai_compatible import (
    OpenAICompatibleProvider,
    OpenAICompatibleSettings,
)


class ProviderRouter:
    """Selects a provider and falls back through an ordered provider chain."""

    def __init__(
        self,
        *,
        default_provider: str,
        fallback_providers: Optional[Sequence[str]] = None,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        self.default_provider = default_provider
        self.fallback_providers = list(fallback_providers or [])
        self._providers: Dict[str, ChatProvider] = {}
        self._event_bus = event_bus

    def register(self, provider: ChatProvider, *, replace: bool = False) -> None:
        if provider.name in self._providers and not replace:
            raise ValueError(f"Provider '{provider.name}' is already registered")
        self._providers[provider.name] = provider

    def names(self) -> List[str]:
        return sorted(self._providers)

    def complete(
        self,
        messages: Sequence[Any],
        *,
        operation_name: str,
        provider_name: Optional[str] = None,
        tool_schemas: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> ProviderResult:
        selected = provider_name or self.default_provider
        if selected not in self._providers:
            available = ", ".join(self.names()) or "none"
            raise KeyError(f"Unknown provider '{selected}'. Available providers: {available}")

        route = [selected]
        route.extend(name for name in self.fallback_providers if name not in route)
        errors: List[str] = []

        for index, name in enumerate(route):
            provider = self._providers.get(name)
            if provider is None:
                errors.append(f"{name}: provider is not registered")
                continue

            try:
                return provider.complete(
                    messages,
                    operation_name=operation_name,
                    tool_schemas=tool_schemas,
                )
            except Exception as exc:  # noqa: BLE001 - fallback boundary
                errors.append(f"{name}: {exc}")
                next_name = route[index + 1] if index + 1 < len(route) else None
                if next_name is not None:
                    self._publish_fallback(name, next_name, exc)

        detail = "; ".join(errors)
        raise RuntimeError(f"All model providers failed for '{operation_name}': {detail}")

    def _publish_fallback(self, failed: str, fallback: str, exc: Exception) -> None:
        if self._event_bus is None:
            return
        context = get_execution_context()
        self._event_bus.publish(
            RunEvent(
                event_type=EventType.PROVIDER_FALLBACK,
                run_id=context.run_id,
                payload={
                    "failed_provider": failed,
                    "fallback_provider": fallback,
                    "error": str(exc),
                },
            )
        )


def build_provider_router(config: Any, event_bus: Optional[EventBus] = None) -> ProviderRouter:
    provider_name = str(getattr(config, "provider_name", "primary"))
    fallback_models = tuple(getattr(config, "llm_fallback_models", ()))
    fallback_names = [f"{provider_name}:{model}" for model in fallback_models]
    router = ProviderRouter(
        default_provider=provider_name,
        fallback_providers=fallback_names,
        event_bus=event_bus,
    )

    common = {
        "api_key": config.api_key,
        "base_url": config.base_url,
        "temperature": config.temperature,
        "timeout": config.llm_timeout,
        "max_retries": config.llm_max_retries,
        "retry_base_delay": config.llm_retry_base_delay,
        "retry_max_delay": config.llm_retry_max_delay,
    }
    router.register(
        OpenAICompatibleProvider(
            OpenAICompatibleSettings(
                name=provider_name,
                model_name=config.model_name,
                **common,
            )
        )
    )

    for name, model_name in zip(fallback_names, fallback_models):
        router.register(
            OpenAICompatibleProvider(
                OpenAICompatibleSettings(
                    name=name,
                    model_name=model_name,
                    **common,
                )
            )
        )

    return router
