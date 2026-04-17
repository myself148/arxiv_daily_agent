import argparse

from agent import run_single_agent
from graph_agent import run_graph_agent


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
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.mode == "single":
        run_single_agent(query=args.query, max_results=args.max_results)
        return

    run_graph_agent(query=args.query, max_results=args.max_results)


if __name__ == "__main__":
    main()
