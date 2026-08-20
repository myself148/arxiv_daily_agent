from datetime import datetime

from runtime.scheduler import CronExpression, SchedulerService, SchedulerStore


def test_cron_expression_supports_ranges_lists_and_steps():
    expression = CronExpression("*/15 8-9 * * 1-5")

    assert expression.matches(datetime(2026, 8, 20, 8, 30))
    assert not expression.matches(datetime(2026, 8, 20, 10, 30))
    assert not expression.matches(datetime(2026, 8, 23, 8, 30))
    assert expression.next_after(datetime(2026, 8, 20, 9, 59)) == datetime(
        2026, 8, 21, 8, 0
    )


def test_scheduler_invokes_pipeline_with_persisted_request(tmp_path):
    store = SchedulerStore(str(tmp_path / "scheduler.db"))
    schedule = store.add(
        name="daily graph report",
        cron="0 8 * * *",
        mode="graph",
        query="cat:cs.CV",
        max_results=2,
        rag_enabled=False,
        provider_name="primary",
        now=datetime(2026, 8, 20, 7, 59),
    )
    received = []

    def run_pipeline(request):
        received.append(request)
        return request.run_id

    run_ids = SchedulerService(store, run_pipeline).run_pending(
        datetime(2026, 8, 20, 8, 0)
    )

    assert len(run_ids) == 1
    assert received[0].query == "cat:cs.CV"
    assert received[0].max_results == 2
    assert received[0].rag_enabled is False
    assert received[0].provider_name == "primary"
    updated = store.get(schedule.schedule_id)
    assert updated.last_run_id == run_ids[0]
    assert updated.last_error == ""


def test_scheduler_records_run_id_when_pipeline_fails(tmp_path):
    store = SchedulerStore(str(tmp_path / "scheduler.db"))
    schedule = store.add(
        name="failing report",
        cron="0 8 * * *",
        mode="single",
        query="cat:cs.CV",
        max_results=1,
        rag_enabled=False,
        now=datetime(2026, 8, 20, 7, 59),
    )
    expected_run_ids = []

    def fail(request):
        expected_run_ids.append(request.run_id)
        raise RuntimeError("pipeline failed")

    assert SchedulerService(store, fail).run_pending(datetime(2026, 8, 20, 8, 0)) == []
    updated = store.get(schedule.schedule_id)
    assert updated.last_run_id == expected_run_ids[0]
    assert updated.last_error == "pipeline failed"
