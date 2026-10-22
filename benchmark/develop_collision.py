"""Small, development-only fixed-budget validation before final protocol freezing."""

from dataclasses import asdict
import hashlib
import json

from controllers.highway.catalog import SUTS
from methods.baselines.fsbo.method import CONFIG as FSBO_CONFIG
from methods.baselines.pooled_ft.method import CONFIG as POOLED_CONFIG
from methods.gcrt.config import CONFIG

from .build_collision_models import model_path
from methods.baselines.response_baselines.collision import METHODS
from .collision_runs import MAIN_METHODS, run_task
from .collect import require_learning_freeze
from .construction import load_construction
from .executions import ExecutionBank
from .protocol import DATA_ROOT, DEFAULT, ROOT, method_contract
from .scenarios import DOMAINS


OUTPUT = ROOT / "results/benchmark/collision/development"
SETTINGS = {"candidate_count": 128, "budget": 32, "initial_count": 8, "seeds": [11, 23]}
REPRESENTATIVE_SCENARIOS = {"rule": "front_brake", "optimization": "cut_out",
                            "search": "cut_in", "learning": "lane_change_rear"}


def main():
    require_learning_freeze()
    model_hashes = {}
    for kind in DOMAINS:
        for method in ("history_response", "fsbo", "pooled_ft"):
            path = model_path(method, "all", kind)
            model_hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    rows = []
    for kind in DOMAINS:
        history, _ = load_construction(kind)
        for sut in SUTS:
            if sut.role != "development":
                continue
            bank = ExecutionBank(DATA_ROOT / "physical/development" / kind / sut.sut_id)
            bank.candidates = bank.candidates[:SETTINGS["candidate_count"]]
            methods = ["history_response", *METHODS]
            if kind == REPRESENTATIVE_SCENARIOS[sut.family]:
                methods += ["no_response", "fixed_weights", "no_target", "greedy"]
            for seed in SETTINGS["seeds"]:
                for method in methods:
                    checkpoint = (model_path("history_response", "all", kind) if method in MAIN_METHODS else
                                  model_path(method, "all", kind) if method in ("fsbo", "pooled_ft") else None)
                    used_history = history if method not in ("random", "space_filling", "gp_collision") else ()
                    metadata = {"target_id": sut.sut_id, "lineage": sut.lineage, "family": sut.family,
                                "scenario": kind, "condition": "development", "method": method,
                                "environment_seed": DEFAULT.environment_seeds[0], "algorithm_seed": seed}
                    destination = OUTPUT / sut.sut_id / kind / f"{method}_{seed}.json"
                    output = run_task(bank, used_history, method=method, algorithm_seed=seed,
                                      budget=SETTINGS["budget"], initial_count=SETTINGS["initial_count"],
                                      destination=destination, metadata=metadata, checkpoint=checkpoint,
                                      horizon=DEFAULT.horizon)
                    rows.append({"path": str(destination), **metadata, **output["metrics"]})
                    print("development", len(rows), "/ 288", sut.sut_id, kind, method, flush=True)
    if len(rows) != 288 or any(row["queries"] != SETTINGS["budget"] for row in rows):
        raise ValueError("Development validation did not complete its registered comparison matrix")
    report = {"status": "complete", "settings": SETTINGS, "runs": rows,
              "model_hashes": model_hashes, "source_contract": method_contract(),
              "configuration": {"main": asdict(CONFIG), "fsbo": FSBO_CONFIG, "pooled_ft": POOLED_CONFIG},
              "data_role": "development_only"}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
