import argparse

from agent import run_single_agent
from config import APP_CONFIG
from graph_agent import run_graph_agent
from runtime.run_store import RunStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the ArXiv Daily Agent project.")
    parser.add_argument(
        "--mode",
        choices=("single", "graph"),
        default="graph",
        help="Choose the single-agent or multi-agent workflow.",
    )
    parser.add_argument(
        "--query",
        default=None,
        help='Override the ArXiv query, for example: cat:cs.CV AND "object detection"',
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=None,
        help="Override how many papers should be processed.",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="Select a configured provider by name for this run.",
    )
    parser.add_argument(
        "--resume-run",
        default=None,
        metavar="RUN_ID",
        help="Resume a failed/interrupted run from its last completed stage.",
    )
    parser.add_argument(
        "--list-runs",
        action="store_true",
        help="List recent persisted runs and exit.",
    )
    rag_group = parser.add_mutually_exclusive_group()
    rag_group.add_argument(
        "--rag",
        dest="rag_enabled",
        action="store_true",
        default=None,
        help="Enable retrieval-augmented context in graph mode.",
    )
    rag_group.add_argument(
        "--no-rag",
        dest="rag_enabled",
        action="store_false",
        help="Disable retrieval-augmented context in graph mode.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_store = RunStore(APP_CONFIG.run_store_path)

    if args.list_runs:
        for snapshot in run_store.list_runs():
            stage = snapshot.current_stage or "-"
            print(
                f"{snapshot.run_id}  {snapshot.status.value:9}  "
                f"{snapshot.request.mode:6}  stage={stage}  updated={snapshot.updated_at}"
            )
        return

    mode = args.mode
    if args.resume_run:
        mode = run_store.get_run(args.resume_run).request.mode

    if mode == "single":
        run_single_agent(
            query=args.query,
            max_results=args.max_results,
            provider_name=args.provider,
            resume_run_id=args.resume_run,
        )
        return

    run_graph_agent(
        query=args.query,
        max_results=args.max_results,
        rag_enabled=args.rag_enabled,
        provider_name=args.provider,
        resume_run_id=args.resume_run,
    )


if __name__ == "__main__":
    main()
