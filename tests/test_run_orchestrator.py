import pytest

from core.contracts import RunRequest, RunStatus
from core.events import EventBus
from runtime.orchestrator import PipelineOrchestrator, StageDefinition
from runtime.run_store import RunStore


def test_failed_run_resumes_after_last_completed_stage(tmp_path):
    store = RunStore(str(tmp_path / "runs.db"))
    first_stage_calls = []

    def research(state):
        first_stage_calls.append(state["run_id"])
        return {"papers": [{"title": "test"}]}

    def failing_review(_state):
        raise RuntimeError("temporary provider failure")

    request = RunRequest(
        mode="graph",
        query="cat:cs.CV",
        max_results=1,
        run_id="recoverable-run",
    )
    first_attempt = PipelineOrchestrator(
        run_store=store,
        event_bus=EventBus(),
        stages=[
            StageDefinition("research", research),
            StageDefinition("review", failing_review),
            StageDefinition("report", lambda _state: {"final_report": "done"}),
        ],
    )

    with pytest.raises(RuntimeError, match="temporary provider failure"):
        first_attempt.execute(request)

    failed = store.get_run(request.run_id)
    assert failed.status is RunStatus.FAILED
    assert failed.completed_stages == ["research"]
    assert failed.current_stage == "review"
    assert failed.state["papers"] == [{"title": "test"}]

    resumed = PipelineOrchestrator(
        run_store=store,
        event_bus=EventBus(),
        stages=[
            StageDefinition("research", research),
            StageDefinition("review", lambda _state: {"reviews": ["ok"]}),
            StageDefinition("report", lambda _state: {"final_report": "done"}),
        ],
    ).execute(resume_run_id=request.run_id)

    assert resumed.status is RunStatus.COMPLETED
    assert resumed.result == "done"
    assert resumed.completed_stages == ["research", "review", "report"]
    assert first_stage_calls == [request.run_id]
    assert [event["event_type"] for event in store.list_events(request.run_id)] == [
        "run.started",
        "stage.started",
        "stage.completed",
        "stage.started",
        "stage.failed",
        "run.failed",
        "run.resumed",
        "stage.started",
        "stage.completed",
        "stage.started",
        "stage.completed",
        "run.completed",
    ]
