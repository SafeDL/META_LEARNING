"""Verify physical experiment comparability and register development inputs."""
from dataclasses import asdict

import numpy as np

from methods.history_guided_testing.config import BOUNDS, MEASUREMENT
from methods.history_guided_testing.io import read_json, write_json
from research.behavior_response_testing.confirmation import (
    pool_scenes, profiles, verify_lock,
)

from .config import (BUDGET, COHORT, EXPLOITATION_ROOTS,
                     INTERIOR_QUADRATURE_POINTS, LEARNING_RATE, METHODS,
                     QUERY_COUNT, RESULTS, SEEDS, SUPPORT_COUNTS, TEMPLATES,
                     TRAINING_STEPS, UNCERTAINTY_ROOTS, VALIDATION_INTERVAL,
                     WEIGHT_DECAY, split_names)


def main():
    source = read_json(COHORT / "protocol.json")
    verify_lock(source)
    generated = [asdict(profile) for profile in profiles()]
    if source["profiles"] != generated:
        raise ValueError("Development SUTs differ from the measured cohort")
    names = {profile["name"] for profile in generated}
    split = split_names()
    flattened = [name for group in split.values() for name in group]
    if len(set(flattened)) != len(flattened) or set(flattened) != names:
        raise ValueError("Development split duplicates or omits systems")
    banks, family_settings = [], {}
    for profile_index, profile in enumerate(generated):
        for replicate in range(2):
            folder = COHORT / profile["name"] / f"pool_{replicate}"
            scenarios = read_json(folder / "scenarios.json")
            if scenarios != pool_scenes(profile_index, replicate):
                raise ValueError(f"Scenario settings differ: {folder}")
            with np.load(folder / "responses.npz") as bank:
                x = bank["x"]
                if not np.array_equal(x, [s["numeric_input"] for s in scenarios]):
                    raise ValueError(f"Physical bank coordinates differ: {folder}")
                if (x.shape != (2048, 5) or bank["risk"].shape != (2048,)
                        or bank["collision"].shape != (2048,)
                        or bank["collision"].dtype != np.bool_
                        or not np.all(np.isfinite(bank["risk"]))
                        or np.any((bank["risk"] < 0) | (bank["risk"] > 1))):
                    raise ValueError(f"Invalid physical bank: {folder}")
                for family in (0, 1):
                    if np.sum(x[:, 4] == family) != 1024:
                        raise ValueError(f"Scenario families unbalanced: {folder}")
                    exemplar = scenarios[family * 1024]
                    family_settings[TEMPLATES[family]] = {
                        "fixed_context": exemplar["fixed_context"],
                        "active_parameter_names": list(exemplar["active_parameters"]),
                        "physical_parameter_bounds": BOUNDS[family],
                        "simulator_seed": exemplar["simulator_seed"],
                    }
                banks.append({"profile": profile["name"],
                              "replicate": replicate, "records": len(x)})
    protocol = {
        "role": "Development using already disclosed physical data; not blind confirmation",
        "source_cohort": str(COHORT),
        "profiles": generated,
        "split": split,
        "physical_banks_checked": banks,
        "physical_records_reused": sum(bank["records"] for bank in banks),
        "new_physical_measurements": 0,
        "simulator": "existing project highway-env; metadrive Python environment",
        "scenario_templates": list(TEMPLATES),
        "normalized_coordinate_family": {"0": TEMPLATES[0], "1": TEMPLATES[1]},
        "physical_parameter_bounds": BOUNDS,
        "risk_measurement": MEASUREMENT,
        "risk_working_model": {
            "observation": "clip(latent Gaussian risk plus working noise, 0, 1)",
            "boundary_likelihood": "Gaussian lower/upper tail mass, not point density",
            "filter": "Gaussian moment matching per behavior with low-rank covariance factors",
            "approximations": ["Gaussian filtering after censoring", "logistic-normal readout moments"],
            "common_to_all_new_training_and_selection_controls": True,
            "physical_risk_measurement_changed": False,
        },
        "scenario_family_settings": family_settings,
        "collision_definition": "original simulator ego_collision; unchanged",
        "methods": list(METHODS),
        "seeds": list(SEEDS),
        "budget": BUDGET,
        "support_counts": SUPPORT_COUNTS,
        "query_count": QUERY_COUNT,
        "training": {"steps": TRAINING_STEPS,
                     "validation_interval": VALIDATION_INTERVAL,
                     "lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
                     "trainable": "same collision head and residual slope in both modes",
                     "frozen": "risk representation, risk head, behavior grid, GP"},
        "selector_feedback": "only the risk from each queried scene",
        "target_parameters_and_unqueried_outcomes": "parent evaluator only",
        "training_decoder_from_all_48_profiles_used": False,
        "development_holdout_runs": 12 * 2 * len(SEEDS) * len(METHODS),
        "primary_metrics": ["normalized_area", "recall"],
        "lookahead": {
            "continuation": "one hypothetical bounded risk and remaining batch",
            "exploitation_roots": EXPLOITATION_ROOTS,
            "uncertainty_roots": UNCERTAINTY_ROOTS,
            "interior_quadrature_points": INTERIOR_QUADRATURE_POINTS,
            "endpoint_atoms": "exact predictive mass at 0 and 1",
            "root_screening": "failure probability and approximate failure-probability variance; greedy always included",
            "constraint": "terminal score at least the same-approximation greedy root score",
            "information_value": "gain from reselecting/reordering relative to the fixed pre-observation continuation; remove probability drift on fixed choices",
            "coherent_posterior_equivalence": "equals raw expected continuation if posterior class probabilities satisfy the tower property",
            "unconstrained_control": "same area objective and roots with terminal constraint removed",
        },
    }
    write_json(RESULTS / "protocol.json", protocol)
    print("DEVELOPMENT INPUTS CHECKED", len(banks),
          protocol["physical_records_reused"],
          {role: len(group) for role, group in split.items()}, flush=True)


if __name__ == "__main__":
    main()
