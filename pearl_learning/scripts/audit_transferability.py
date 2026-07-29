"""Create a leakage-safe meta-training coverage report for frozen tasks."""
from __future__ import annotations

import argparse

from pearl_learning.src.casebook import load_casebook
from pearl_learning.src.io import write_json
from pearl_learning.src.taskbook import load_taskbook
from pearl_learning.src.transferability import transferability_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--taskbook", required=True)
    parser.add_argument("--casebook-root", required=True)
    parser.add_argument("--candidate-split", choices=["meta_validation", "meta_test_template", "meta_test_logical"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--similarity-temperature", type=float, default=0.5)
    parser.add_argument("--include-hidden-rules", action="store_true", help="oracle/offline explanation only; not deployment-legal")
    args = parser.parse_args()
    taskbook = load_taskbook(args.taskbook)
    all_tasks = [task for tasks in taskbook.values() for task in tasks]
    casebooks = {task.task_id: load_casebook(task, args.casebook_root) for task in all_tasks}
    report = transferability_report(
        taskbook,
        casebooks,
        candidate_split=args.candidate_split,
        include_hidden_rules=args.include_hidden_rules,
        similarity_temperature=args.similarity_temperature,
    )
    write_json(args.output, report)
    print(f"transferability diagnostic: {args.output}")


if __name__ == "__main__":
    main()
