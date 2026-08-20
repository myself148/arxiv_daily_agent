import argparse
from typing import Optional

from agent import ArxivAgent
from config import APP_CONFIG
from core.contracts import RunRequest
from graph_agent import run_graph_pipeline
from runtime.scheduler import SchedulerService, SchedulerStore


def execute_request(request: RunRequest) -> str:
    if request.mode == "single":
        snapshot = ArxivAgent().run_pipeline(
            query=request.query,
            max_results=request.max_results,
            provider_name=request.provider_name,
            run_id=request.run_id,
        )
    else:
        snapshot = run_graph_pipeline(
            query=request.query,
            max_results=request.max_results,
            rag_enabled=request.rag_enabled,
            provider_name=request.provider_name,
            run_id=request.run_id,
        )
    return snapshot.run_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage the in-process ArXiv Agent scheduler.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    add = subparsers.add_parser("add", help="Add a scheduled pipeline run.")
    add.add_argument("--name", required=True)
    add.add_argument("--cron", required=True, help='Five-field cron, for example "0 8 * * *".')
    add.add_argument("--mode", choices=("single", "graph"), default="graph")
    add.add_argument("--query", default=APP_CONFIG.arxiv_query)
    add.add_argument("--max-results", type=int, default=APP_CONFIG.arxiv_max_results)
    add.add_argument("--provider", default=None)
    add.add_argument("--no-rag", action="store_true")

    subparsers.add_parser("list", help="List schedules.")
    remove = subparsers.add_parser("remove", help="Remove a schedule.")
    remove.add_argument("schedule_id")
    enable = subparsers.add_parser("enable", help="Enable a schedule.")
    enable.add_argument("schedule_id")
    disable = subparsers.add_parser("disable", help="Disable a schedule.")
    disable.add_argument("schedule_id")
    subparsers.add_parser("run-pending", help="Run all jobs currently due, then exit.")
    subparsers.add_parser("serve", help="Run the scheduler service until interrupted.")
    return parser


def main(argv: Optional[list] = None) -> None:
    args = build_parser().parse_args(argv)
    store = SchedulerStore(APP_CONFIG.scheduler_store_path)

    if args.command == "add":
        schedule = store.add(
            name=args.name,
            cron=args.cron,
            mode=args.mode,
            query=args.query,
            max_results=args.max_results,
            rag_enabled=not args.no_rag,
            provider_name=args.provider,
        )
        print(f"Added {schedule.schedule_id}: next run {schedule.next_run_at}")
        return

    if args.command == "list":
        for schedule in store.list():
            status = "enabled" if schedule.enabled else "disabled"
            print(
                f"{schedule.schedule_id}  {status:8}  {schedule.cron:15}  "
                f"{schedule.mode:6}  next={schedule.next_run_at}  {schedule.name}"
            )
        return

    if args.command == "remove":
        store.remove(args.schedule_id)
        print(f"Removed {args.schedule_id}")
        return

    if args.command in {"enable", "disable"}:
        store.set_enabled(args.schedule_id, args.command == "enable")
        print(f"{args.command.title()}d {args.schedule_id}")
        return

    service = SchedulerService(
        store,
        execute_request,
        poll_seconds=APP_CONFIG.scheduler_poll_seconds,
    )
    if args.command == "run-pending":
        run_ids = service.run_pending()
        print(f"Completed {len(run_ids)} run(s): {', '.join(run_ids)}")
        return

    print("Scheduler service started. Press Ctrl+C to stop.")
    try:
        service.serve()
    except KeyboardInterrupt:
        print("Scheduler service stopped.")


if __name__ == "__main__":
    main()
