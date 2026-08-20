from core.context import ExecutionContext, use_execution_context
from core.events import EventBus, EventType
from core.tool_registry import ToolDefinition, ToolRegistry
from providers.base import ChatProvider, ProviderResult, ToolCall
from providers.router import ProviderRouter
from runtime.model_loop import ModelToolLoop


class StubProvider(ChatProvider):
    def __init__(self, name, responses):
        self.name = name
        self.model_name = f"{name}-model"
        self.responses = list(responses)
        self.calls = 0

    def complete(self, messages, *, operation_name, tool_schemas=None):
        self.calls += 1
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_provider_router_falls_back_and_emits_event():
    events = []
    event_bus = EventBus()
    event_bus.subscribe(events.append)
    router = ProviderRouter(
        default_provider="primary",
        fallback_providers=["backup"],
        event_bus=event_bus,
    )
    primary = StubProvider("primary", [TimeoutError("timed out")])
    backup = StubProvider(
        "backup",
        [ProviderResult("recovered", "backup", "backup-model")],
    )
    router.register(primary)
    router.register(backup)

    with use_execution_context(ExecutionContext(run_id="run-1")):
        result = router.complete([], operation_name="summary")

    assert result.content == "recovered"
    assert primary.calls == 1
    assert backup.calls == 1
    assert events[0].event_type is EventType.PROVIDER_FALLBACK
    assert events[0].run_id == "run-1"
    assert events[0].payload["fallback_provider"] == "backup"


def test_model_tool_loop_executes_registered_tool_then_returns_answer():
    calls = []
    event_bus = EventBus()
    event_bus.subscribe(calls.append)
    registry = ToolRegistry(event_bus)
    registry.register(
        ToolDefinition(
            name="math.double",
            description="Double an integer.",
            input_schema={
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
            },
            expose_to_model=True,
            handler=lambda value: value * 2,
        )
    )
    provider = StubProvider(
        "primary",
        [
            ProviderResult(
                "",
                "primary",
                "primary-model",
                tool_calls=[ToolCall("call-1", "math.double", {"value": 21})],
            ),
            ProviderResult("42", "primary", "primary-model"),
        ],
    )
    router = ProviderRouter(default_provider="primary")
    router.register(provider)
    loop = ModelToolLoop(router, registry, max_iterations=3)

    with use_execution_context(ExecutionContext(run_id="run-2")):
        answer = loop.run([{"role": "user", "content": "double 21"}], operation_name="math")

    assert answer == "42"
    assert provider.calls == 2
    assert [event.event_type for event in calls] == [
        EventType.TOOL_STARTED,
        EventType.TOOL_COMPLETED,
    ]
    assert all(event.run_id == "run-2" for event in calls)
