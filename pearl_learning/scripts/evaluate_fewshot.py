from __future__ import annotations
import argparse
import torch
from pathlib import Path
from pearl_learning.src.checkpoint import load_checkpoint
from pearl_learning.src.evaluator import evaluate_fewshot
from pearl_learning.src.io import read_config, write_json
from pearl_learning.src.pearl_agent import PEARLAgent
from pearl_learning.src.taskbook import build_taskbook


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--checkpoint", required=True); parser.add_argument("--split", choices=["meta_test_template", "meta_test_logical", "meta_validation"], required=True); parser.add_argument("--query-cases", type=int); args = parser.parse_args()
    cfg = read_config(args.config); device = torch.device("cuda" if torch.cuda.is_available() and cfg["experiment"].get("device") != "cpu" else "cpu"); agent = PEARLAgent(int(cfg["environment"]["observation_dim"]), int(cfg["environment"]["action_dim"]), cfg, device); load_checkpoint(args.checkpoint, agent, device)
    root = Path(cfg["project"]["output_root"]) / "final_eval" / args.split
    result = evaluate_fewshot(agent, cfg, build_taskbook(cfg)[args.split], args.split, args.query_cases, root)
    write_json(root / "pearl_fewshot.json", result)
    print(f"no-gradient few-shot result: {root / 'pearl_fewshot.json'}")


if __name__ == "__main__": main()
