"""Complete the three required evidence groups using the already collected raw pools."""

import subprocess
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from .discovery_config import read_json, write_json
from .independent_preparation import OUTPUT, validate_preparation
from .protocol import ROOT


def run_stage(module, *arguments):
    log = (
        OUTPUT
        / "workflow_logs"
        / (module.rsplit(".", 1)[-1] + ("_" + "_".join(arguments) if arguments else "") + ".log")
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    print("workflow_stage", module, *arguments, "log", str(log.relative_to(ROOT)), flush=True)
    with log.open("ab", buffering=0) as stream:
        stream.write(("\n" + datetime.now(timezone.utc).isoformat() + "\n").encode())
        subprocess.run(
            [sys.executable, "-u", "-B", "-m", module, *arguments],
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


def finish_main():
    run_stage("experiments.independent_models", "main")
    run_stage("experiments.independent_models", "priors")


def main():
    validate_preparation()
    if any(
        read_json(OUTPUT / name)["status"] != "complete"
        for name in (
            "policy_training.json",
            "nonlearning_qualification.json",
            "interaction_construction.json",
            "target_bank_index.json",
        )
    ):
        raise ValueError(
            "Reuse the completed, registered driving systems and raw pools; do not recollect them"
        )
    path = OUTPUT / "reduced_workflow.json"
    state = {
        "status": "construction_active",
        "completed": [],
        "errors": {},
        "unique_runs": 1880,
        "reused_references": 600,
        "equivalent_queries": 240640,
        "new_physical_calls": 0,
        "previous_workflow": "workflow.json",
    }
    write_json(path, state)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(finish_main): "main_models_and_priors",
            pool.submit(
                run_stage, "experiments.independent_models", "classifiers"
            ): "required_classifiers",
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
                state["completed"].append(name)
            except Exception:
                state["errors"][name] = traceback.format_exc()
                state["status"] = "construction_failed"
            write_json(path, state)
    if state["errors"]:
        raise RuntimeError("Inspect the recorded required-model construction failure")
    state["status"] = "independent_validation_active"
    for stage in ("freeze", "banks", "first", "audit_first", "run", "audit", "report"):
        state["stage"] = stage
        write_json(path, state)
        try:
            run_stage("experiments.independent_validation", stage)
        except Exception:
            state.update(status="independent_validation_failed", error=traceback.format_exc())
            write_json(path, state)
            raise
    state.update(status="complete_awaiting_result_and_visual_review", stage="review")
    write_json(path, state)
    return state


if __name__ == "__main__":
    main()
