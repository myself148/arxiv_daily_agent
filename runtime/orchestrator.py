from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Mapping, Optional, Union

from core.context import ExecutionContext, use_execution_context
from core.contracts import (
    PipelineState,
    RunRequest,
    RunSnapshot,
    RunStatus,
    StageResult,
    StageStatus,
)
from core.events import EventBus, EventType, RunEvent
from runtime.run_store import RunStore


StageOutput = Union[StageResult, Mapping[str, Any], None]
StageHandler = Callable[[PipelineState], StageOutput]


@dataclass(frozen=True)
class StageDefinition:
    name: str
    handler: StageHandler


class PipelineOrchestrator:
    """Runs business stages and persists a checkpoint after each completed stage."""

    def __init__(
        self,
        *,
        run_store: RunStore,
        stages: List[StageDefinition],
        event_bus: Optional[EventBus] = None,
    ) -> None:
        if not stages:
            raise ValueError("At least one stage is required")
        names = [stage.name for stage in stages]
        if len(names) != len(set(names)):
            raise ValueError("Stage names must be unique")

        self.run_store = run_store
        self.stages = list(stages)
        self.event_bus = event_bus or EventBus()
        self.event_bus.subscribe(self.run_store.append_event)

    def execute(
        self,
        request: Optional[RunRequest] = None,
        *,
        initial_state: Optional[PipelineState] = None,
        resume_run_id: Optional[str] = None,
    ) -> RunSnapshot:
        if resume_run_id:
            snapshot = self.run_store.get_run(resume_run_id)
            if snapshot.status is RunStatus.COMPLETED:
                return snapshot
            resolved_request = snapshot.request
            state: PipelineState = dict(snapshot.state)
            completed_stages = set(snapshot.completed_stages)
            self._publish(EventType.RUN_RESUMED, resolved_request.run_id)
        else:
            if request is None:
                raise ValueError("request is required for a new run")
            request.validate()
            resolved_request = request
            state = dict(initial_state or {})
            state.update(
                {
                    "run_id": request.run_id,
                    "mode": request.mode,
                    "query": request.query,
                    "max_results": request.max_results,
                    "rag_enabled": request.rag_enabled,
                    "provider_name": request.provider_name,
                }
            )
            self.run_store.create_run(request, state)
            completed_stages = set()
            self._publish(EventType.RUN_STARTED, request.run_id)

        self.run_store.mark_running(resolved_request.run_id)
        current_stage: Optional[str] = None
        context = ExecutionContext(
            run_id=resolved_request.run_id,
            provider_name=resolved_request.provider_name,
            event_bus=self.event_bus,
        )

        try:
            with use_execution_context(context):
                for stage in self.stages:
                    if stage.name in completed_stages:
                        continue

                    current_stage = stage.name
                    self._publish(
                        EventType.STAGE_STARTED,
                        resolved_request.run_id,
                        stage=stage.name,
                    )
                    output = stage.handler(state)
                    result = self._normalize_stage_output(stage.name, output)
                    if result.status is StageStatus.FAILED:
                        raise RuntimeError(result.message or f"Stage '{stage.name}' failed")

                    state.update(result.updates)
                    self.run_store.checkpoint(resolved_request.run_id, stage.name, state)
                    completed_stages.add(stage.name)
                    self._publish(
                        EventType.STAGE_COMPLETED,
                        resolved_request.run_id,
                        stage=stage.name,
                        payload={"updated_fields": sorted(result.updates)},
                    )

                result_text = str(state.get("final_report", ""))
                snapshot = self.run_store.complete(
                    resolved_request.run_id,
                    state,
                    result_text,
                )
                self._publish(EventType.RUN_COMPLETED, resolved_request.run_id)
                return snapshot
        except Exception as exc:
            self.run_store.fail(
                resolved_request.run_id,
                state,
                str(exc),
                current_stage=current_stage,
            )
            self._publish(
                EventType.STAGE_FAILED,
                resolved_request.run_id,
                stage=current_stage,
                payload={"error": str(exc)},
            )
            self._publish(
                EventType.RUN_FAILED,
                resolved_request.run_id,
                payload={"error": str(exc)},
            )
            raise

    def _publish(
        self,
        event_type: EventType,
        run_id: str,
        *,
        stage: Optional[str] = None,
        payload: Optional[dict] = None,
    ) -> None:
        self.event_bus.publish(
            RunEvent(
                event_type=event_type,
                run_id=run_id,
                stage=stage,
                payload=payload or {},
            )
        )

    @staticmethod
    def _normalize_stage_output(stage: str, output: StageOutput) -> StageResult:
        if isinstance(output, StageResult):
            if output.stage != stage:
                raise ValueError(
                    f"Stage result '{output.stage}' does not match executing stage '{stage}'"
                )
            return output
        return StageResult(
            stage=stage,
            status=StageStatus.COMPLETED,
            updates=dict(output or {}),
        )
