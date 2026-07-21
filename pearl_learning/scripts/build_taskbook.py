from __future__ import annotations
import argparse
from pearl_learning.src.casebook import build_casebook, save_casebook
from pearl_learning.src.io import read_config
from pearl_learning.src.taskbook import build_taskbook, save_taskbook


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); args = parser.parse_args()
    cfg = read_config(args.config); taskbook = build_taskbook(cfg); digest = save_taskbook(taskbook, cfg["project"]["output_root"] + "/taskbooks")
    for tasks in taskbook.values():
        for task in tasks: save_casebook(task, build_casebook(task, cfg), cfg["project"]["output_root"])
    print(f"wrote frozen taskbook sha256={digest} and replayable casebooks")


if __name__ == "__main__": main()
