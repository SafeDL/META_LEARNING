"""Complete the frozen study, supplying missing derived checkpoint fields."""
from methods.history_guided_testing.io import read_json, write_json
from research.risk_conditioned_response_testing import (
    audit_confirmation, evaluate_confirmation, report_confirmation,
)
from research.risk_conditioned_response_testing.confirmation import verify_lock

from .config import CURRENT_CONFIRMATION, RESULTS


def complete_checkpoints():
    protocol = read_json(CURRENT_CONFIRMATION / "protocol.json")
    verify_lock(protocol)
    completed = 0
    for profile in protocol["profiles"]:
        for replicate in range(protocol["replicates_per_profile"]):
            folder = CURRENT_CONFIRMATION / profile["name"] / f"pool_{replicate}"
            for method in protocol["methods"]:
                for seed in protocol["seeds"]:
                    path = folder / "selection" / f"{method}_{seed}.json"
                    result = read_json(path)
                    if len(result["curve"]) != protocol["budget"]:
                        raise ValueError(f"Incomplete discovery curve: {path}")
                    changed = False
                    for count in (10, 30, 150):
                        key, value = f"F{count}", result["curve"][count - 1]
                        if key in result and result[key] != value:
                            raise ValueError(f"Checkpoint differs: {path}, {key}")
                        if key not in result:
                            result[key] = value
                            changed = True
                    if changed:
                        write_json(path, result)
                    completed += 1
    write_json(RESULTS / "confirmation_execution.json", {
        "frozen_confirmation": str(CURRENT_CONFIRMATION),
        "selection_runs": completed,
        "source_edits": False,
        "output_additions": ["F10", "F30", "F150"],
        "reason": "Frozen evaluator requires checkpoints omitted by frozen selector writer",
        "selected_indices_observations_and_original_metrics_unchanged": True,
    })


def main():
    complete_checkpoints()
    evaluate_confirmation.main()
    audit_confirmation.main()
    report_confirmation.main()
    protocol = read_json(CURRENT_CONFIRMATION / "protocol.json")
    verify_lock(protocol)
    print("FROZEN CONFIRMATION FINISHED WITHOUT SOURCE CHANGES", flush=True)


if __name__ == "__main__":
    main()
