"""Run the frozen independent confirmation experiment."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np

from highway_sim_env.data.response_bank import ResponseBank
from highway_sim_env.mining.low_rank_prior import LowRankPrior
from highway_sim_env.mining.mining import diagnostic_mining, random_mining
from methods.function_posterior_search.benchmark import (
    CONFIRMATION_SEEDS,
    HETEROGENEITY_LEVELS,
    MODES,
    REGIME_COUNTS,
    SCENARIO_COUNT,
    SOURCE_NAMES,
    TARGET_COVERAGES,
    generate_confirmation_scenarios,
    load_or_build_bank,
    physical_episode_count,
    release_manifest,
    target_releases,
)
from methods.detour_fusion.fusion import (
    detour_guided_diagnostic_mining,
    detour_static_mining,
    hierarchy_prior,
)
from methods.function_conditioned_routing.config import RoutingExperimentConfig
from methods.function_conditioned_routing.experiment import severity_response
from methods.function_conditioned_routing.routing import functional_routed_mining
from replications.adate_highway_env.adate.mixture_selector import MixtureSelector
from replications.detour_highway_env.detour.features import encode_scenarios
from replications.detour_highway_env.detour.history import scenario_specs, source_history

from .search import (
    ADAPTIVE_METHOD,
    BUDGETS,
    NOISE_SCALES,
    run_adaptive_campaign,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results" / "method_chains" / "function_posterior_search" / "confirmation"
BANK_DIR = OUTPUT / "banks"
SUPPORT_BUDGET = 10
TOTAL_BUDGET = 50
RANDOM_REPEATS = 20
BOOTSTRAP_SAMPLES = 10000
METHODS = (
    "Random",
    "DETOUR",
    "Mining",
    "Mining-DETOUR",
    "AdaTE Global",
    "Function-Conditioned Mining",
    ADAPTIVE_METHOD,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _adate_indices(sources: np.ndarray, target: np.ndarray) -> np.ndarray:
    selector = MixtureSelector(sources, TOTAL_BUDGET, "sequential")
    while True:
        chosen = selector.propose()
        if chosen is None:
            break
        selector.observe(chosen, float(target[chosen]))
    return np.asarray(selector.selected, dtype=int)


def _select_noise(
    sources: np.ndarray,
    source_events: np.ndarray,
    modes: np.ndarray,
) -> tuple[float, list[dict[str, object]]]:
    rows = []
    for pseudo_target in range(len(sources)):
        training = np.arange(len(sources)) != pseudo_target
        truth = source_events[pseudo_target]
        for noise_scale in NOISE_SCALES:
            selected, _ = run_adaptive_campaign(
                sources[training],
                source_events[training],
                modes,
                sources[pseudo_target],
                noise_scale,
            )
            recalls = [
                truth[selected[:budget]].sum() / max(1, truth.sum())
                for budget in BUDGETS
            ]
            rows.append({
                "pseudo_target_sut": SOURCE_NAMES[pseudo_target],
                "noise_scale": noise_scale,
                "mean_checkpoint_recall": float(np.mean(recalls)),
                "target_data_used": False,
            })
    scores = {
        scale: np.mean([
            row["mean_checkpoint_recall"] for row in rows
            if row["noise_scale"] == scale
        ])
        for scale in NOISE_SCALES
    }
    selected = max(scores, key=lambda scale: (scores[scale], scale))
    for row in rows:
        row["selected_noise_scale"] = selected
    return selected, rows


def _record_rows(
    seed: int,
    target_name: str,
    coverage: str,
    heterogeneity: str,
    method: str,
    queried: np.ndarray,
    collisions: np.ndarray,
    near_misses: np.ndarray,
    runtime: float,
    repeat: int = 0,
) -> list[dict[str, object]]:
    critical = collisions | near_misses
    critical_total = int(critical.sum())
    collision_total = int(collisions.sum())
    near_total = int(near_misses.sum())
    rows = []
    for budget in BUDGETS:
        prefix = queried[:budget]
        critical_found = int(critical[prefix].sum())
        collision_found = int(collisions[prefix].sum())
        near_found = int(near_misses[prefix].sum())
        oracle_recall = min(1.0, budget / critical_total) if critical_total else 0.0
        recall = critical_found / critical_total if critical_total else 0.0
        rows.append({
            "seed": seed,
            "target_sut": target_name,
            "coverage": coverage,
            "heterogeneity": heterogeneity,
            "method": method,
            "repeat": repeat,
            "budget": budget,
            "critical_events_available": critical_total,
            "critical_events_found": critical_found,
            "critical_recall": recall,
            "collision_events_available": collision_total,
            "collision_events_found": collision_found,
            "collision_recall": collision_found / collision_total if collision_total else 0.0,
            "near_misses_available": near_total,
            "near_misses_found": near_found,
            "near_miss_recall": near_found / near_total if near_total else 0.0,
            "oracle_recall": oracle_recall,
            "oracle_regret": oracle_recall - recall,
            "selection_runtime_seconds": runtime,
            "queried_indices": ";".join(map(str, prefix)),
        })
    return rows


def evaluate_bank(
    bank: ResponseBank,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Evaluate all methods on independent targets with fixed source histories."""
    config = RoutingExperimentConfig(
        num_anchors=SCENARIO_COUNT,
        support_budget=SUPPORT_BUDGET,
        total_budget=TOTAL_BUDGET,
        random_repeats=RANDOM_REPEATS,
        seed=seed,
    )
    source_indices = [bank.index_of(name) for name in SOURCE_NAMES]
    targets = target_releases()
    responses = severity_response(bank, list(range(len(bank.sut_names))))
    events = bank.collisions | bank.near_misses
    sources = responses[source_indices]
    source_events = events[source_indices]
    prior = LowRankPrior.fit(sources, config.prior_rank)
    noise_scale, tuning = _select_noise(sources, source_events, bank.modes)
    for row in tuning:
        row["seed"] = seed
    features = encode_scenarios(scenario_specs(bank))
    history = source_history(bank, targets[0].name, "collision", SOURCE_NAMES)
    hierarchy = hierarchy_prior(
        encode_scenarios(tuple(item.scenario for item in history)),
        features,
        np.asarray([item.failed for item in history], dtype=bool),
        seed,
    )
    rows: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []
    random_seeds = np.random.SeedSequence(seed).spawn(len(targets))
    for target_order, release in enumerate(targets):
        index = bank.index_of(release.name)
        target = responses[index]
        collisions, near_misses = bank.collisions[index], bank.near_misses[index]
        started = perf_counter()
        detour = detour_static_mining(hierarchy, collisions, near_misses, TOTAL_BUDGET)
        elapsed = perf_counter() - started
        rows.extend(_record_rows(seed, release.name, release.coverage,
                                 release.heterogeneity, "DETOUR",
                                 detour.queried_indices, collisions, near_misses, elapsed))
        started = perf_counter()
        mining = diagnostic_mining(
            prior, target, collisions, near_misses, SUPPORT_BUDGET, TOTAL_BUDGET,
        )
        elapsed = perf_counter() - started
        rows.extend(_record_rows(seed, release.name, release.coverage,
                                 release.heterogeneity, "Mining",
                                 mining.queried_indices, collisions, near_misses, elapsed))
        started = perf_counter()
        fused = detour_guided_diagnostic_mining(
            prior, hierarchy, target, collisions, near_misses,
            SUPPORT_BUDGET, TOTAL_BUDGET, config.hierarchy_weight,
        )
        elapsed = perf_counter() - started
        rows.extend(_record_rows(seed, release.name, release.coverage,
                                 release.heterogeneity, "Mining-DETOUR",
                                 fused.queried_indices, collisions, near_misses, elapsed))
        started = perf_counter()
        adate = _adate_indices(sources, target)
        elapsed = perf_counter() - started
        rows.extend(_record_rows(seed, release.name, release.coverage,
                                 release.heterogeneity, "AdaTE Global", adate,
                                 collisions, near_misses, elapsed))
        started = perf_counter()
        routed = functional_routed_mining(
            "Function-Conditioned Mining",
            sources,
            bank.modes,
            prior,
            target,
            collisions,
            near_misses,
            config,
            functional=True,
            trusted=True,
            support_policy="coverage",
        )
        elapsed = perf_counter() - started
        rows.extend(_record_rows(seed, release.name, release.coverage,
                                 release.heterogeneity, "Function-Conditioned Mining",
                                 routed.trace.queried_indices, collisions, near_misses, elapsed))
        started = perf_counter()
        posterior, audit = run_adaptive_campaign(
            sources,
            source_events,
            bank.modes,
            target,
            noise_scale,
        )
        elapsed = perf_counter() - started
        rows.extend(_record_rows(seed, release.name, release.coverage,
                                 release.heterogeneity, ADAPTIVE_METHOD, posterior,
                                 collisions, near_misses, elapsed))
        decisions.extend({
            "seed": seed,
            "target_sut": release.name,
            "coverage": release.coverage,
            "heterogeneity": release.heterogeneity,
            "noise_scale": noise_scale,
            **decision,
        } for decision in audit)
        rng = np.random.default_rng(random_seeds[target_order])
        for repeat in range(RANDOM_REPEATS):
            started = perf_counter()
            random = random_mining(collisions, near_misses, TOTAL_BUDGET, rng)
            elapsed = perf_counter() - started
            rows.extend(_record_rows(seed, release.name, release.coverage,
                                     release.heterogeneity, "Random",
                                     random.queried_indices, collisions, near_misses,
                                     elapsed, repeat))
        print(f"seed {seed} / {release.name}: complete", flush=True)
    return rows, tuning, decisions


def _unit_metrics(rows: list[dict[str, object]]) -> dict[tuple[object, ...], dict[str, float]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (row["seed"], row["target_sut"], row["coverage"],
               row["heterogeneity"], row["method"], row["budget"])
        grouped[key].append(row)
    metrics = {}
    for key, records in grouped.items():
        metrics[key] = {
            name: float(np.mean([float(row[name]) for row in records]))
            for name in (
                "critical_recall", "collision_recall", "near_miss_recall",
                "critical_events_found", "oracle_regret", "selection_runtime_seconds",
            )
        }
    return metrics


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metrics = _unit_metrics(rows)
    summaries = []
    scopes = [("overall", "all", "all")]
    scopes += [("coverage", value, "all") for value in TARGET_COVERAGES]
    scopes += [("heterogeneity", "all", value) for value in HETEROGENEITY_LEVELS]
    scopes += [("cell", coverage, heterogeneity)
               for coverage in TARGET_COVERAGES
               for heterogeneity in HETEROGENEITY_LEVELS]
    for scope, coverage, heterogeneity in scopes:
        for method in METHODS:
            for budget in BUDGETS:
                selected = [value for key, value in metrics.items()
                            if key[4] == method and key[5] == budget
                            and (coverage == "all" or key[2] == coverage)
                            and (heterogeneity == "all" or key[3] == heterogeneity)]
                row: dict[str, object] = {
                    "scope": scope,
                    "coverage": coverage,
                    "heterogeneity": heterogeneity,
                    "method": method,
                    "budget": budget,
                    "seed_target_units": len(selected),
                }
                for name in selected[0]:
                    values = np.asarray([item[name] for item in selected])
                    row[f"mean_{name}"] = float(values.mean())
                    row[f"std_{name}"] = float(values.std(ddof=1))
                summaries.append(row)
    return summaries


def _hierarchical_interval(
    differences: dict[int, np.ndarray],
    rng: np.random.Generator,
) -> tuple[float, float]:
    seeds = np.asarray(sorted(differences))
    samples = np.empty(BOOTSTRAP_SAMPLES)
    for iteration in range(BOOTSTRAP_SAMPLES):
        drawn_seeds = rng.choice(seeds, len(seeds), replace=True)
        values = []
        for seed in drawn_seeds:
            targets = differences[int(seed)]
            values.extend(rng.choice(targets, len(targets), replace=True))
        samples[iteration] = np.mean(values)
    return tuple(np.quantile(samples, (0.025, 0.975)))


def paired_comparisons(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metrics = _unit_metrics(rows)
    rng = np.random.default_rng(CONFIRMATION_SEEDS[0])
    comparisons = []
    scopes = [("overall", "all", "all")]
    scopes += [("coverage", value, "all") for value in TARGET_COVERAGES]
    scopes += [("heterogeneity", "all", value) for value in HETEROGENEITY_LEVELS]
    for scope, coverage, heterogeneity in scopes:
        for reference in METHODS[:-1]:
            for budget in BUDGETS:
                by_seed: dict[int, list[float]] = defaultdict(list)
                for key, proposed in metrics.items():
                    seed, target, cov, hetero, method, row_budget = key
                    if method != ADAPTIVE_METHOD or row_budget != budget:
                        continue
                    if coverage != "all" and cov != coverage:
                        continue
                    if heterogeneity != "all" and hetero != heterogeneity:
                        continue
                    baseline = metrics[seed, target, cov, hetero, reference, budget]
                    by_seed[int(seed)].append(
                        proposed["critical_recall"] - baseline["critical_recall"]
                    )
                arrays = {seed: np.asarray(values) for seed, values in by_seed.items()}
                all_differences = np.concatenate(list(arrays.values()))
                low, high = _hierarchical_interval(arrays, rng)
                comparisons.append({
                    "scope": scope,
                    "coverage": coverage,
                    "heterogeneity": heterogeneity,
                    "method": ADAPTIVE_METHOD,
                    "reference": reference,
                    "budget": budget,
                    "mean_recall_difference": float(all_differences.mean()),
                    "hierarchical_bootstrap_low": float(low),
                    "hierarchical_bootstrap_high": float(high),
                    "wins": int(np.sum(all_differences > 1e-12)),
                    "ties": int(np.sum(np.abs(all_differences) <= 1e-12)),
                    "losses": int(np.sum(all_differences < -1e-12)),
                    "seed_target_units": len(all_differences),
                })
    return comparisons


def summarize_subgroups(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Audit function and scenario-regime recalls without changing selection."""
    units: dict[tuple[object, ...], list[float]] = defaultdict(list)
    available: dict[tuple[object, ...], int] = {}
    subgroup_sizes: dict[tuple[int, str, str], int] = {}
    banks = {
        seed: ResponseBank.load(BANK_DIR / f"response_bank_{seed}.npz")
        for seed in CONFIRMATION_SEEDS
    }
    regimes = {
        seed: generate_confirmation_scenarios(seed)[3]
        for seed in CONFIRMATION_SEEDS
    }
    release_lookup = {release.name: release for release in target_releases()}
    for row in rows:
        seed = int(row["seed"])
        bank = banks[seed]
        release = release_lookup[str(row["target_sut"])]
        target_index = bank.index_of(release.name)
        truth = bank.collisions[target_index] | bank.near_misses[target_index]
        queried = np.asarray(str(row["queried_indices"]).split(";"), dtype=int)
        groups = [("mode", mode, np.asarray(bank.modes) == mode) for mode in MODES]
        groups += [
            ("regime", regime, regimes[seed] == regime)
            for regime in REGIME_COUNTS
        ]
        for kind, value, mask in groups:
            total = int(truth[mask].sum())
            key = (
                seed,
                release.name,
                release.coverage,
                release.heterogeneity,
                row["method"],
                int(row["budget"]),
                kind,
                value,
            )
            found = int(truth[queried][mask[queried]].sum())
            units[key].append(found / total if total else np.nan)
            available[key] = total
            subgroup_sizes[seed, kind, value] = int(mask.sum())
    unit_values = {
        key: float(np.nanmean(values)) if not np.all(np.isnan(values)) else np.nan
        for key, values in units.items()
    }
    summary = []
    for kind, values in (("mode", MODES), ("regime", tuple(REGIME_COUNTS))):
        for value in values:
            for method in METHODS:
                for budget in BUDGETS:
                    keys = [
                        key for key in unit_values
                        if key[4] == method and key[5] == budget
                        and key[6] == kind and key[7] == value
                    ]
                    recalls = np.asarray([unit_values[key] for key in keys])
                    event_counts = np.asarray([available[key] for key in keys])
                    sizes = np.asarray([
                        subgroup_sizes[int(key[0]), kind, value] for key in keys
                    ])
                    valid = np.isfinite(recalls)
                    summary.append({
                        "subgroup": kind,
                        "value": value,
                        "method": method,
                        "budget": budget,
                        "seed_target_units": len(keys),
                        "evaluable_units": int(valid.sum()),
                        "mean_event_count": float(event_counts.mean()),
                        "mean_event_prevalence": float(np.mean(event_counts / sizes)),
                        "mean_critical_recall": (
                            float(recalls[valid].mean()) if valid.any() else np.nan
                        ),
                    })
    return summary


def _write_design_files() -> None:
    _write_csv(OUTPUT / "target_manifest.csv", release_manifest())
    scenario_rows = []
    for seed in CONFIRMATION_SEEDS:
        anchors, modes, controls, regimes = generate_confirmation_scenarios(seed)
        scenario_rows.extend({
            "seed": seed,
            "scenario_index": index,
            "mode": mode,
            "regime": regime,
            "initial_gap": anchor[0],
            "relative_speed": anchor[1],
            "timing": control[0],
            "intensity": control[1],
        } for index, (anchor, mode, control, regime) in enumerate(
            zip(anchors, modes, controls, regimes, strict=True)
        ))
    _write_csv(OUTPUT / "scenario_manifest.csv", scenario_rows)


def _lookup(
    summary: list[dict[str, object]],
    method: str,
    budget: int,
    coverage: str = "all",
    heterogeneity: str = "all",
) -> dict[str, object]:
    return next(row for row in summary if row["method"] == method
                and row["budget"] == budget and row["coverage"] == coverage
                and row["heterogeneity"] == heterogeneity)


def _recall_text(value: object) -> str:
    number = float(value)
    return "n/a" if not np.isfinite(number) else f"{number:.4f}"


def write_report(
    summary: list[dict[str, object]],
    comparisons: list[dict[str, object]],
    records: list[dict[str, object]],
    subgroup_summary: list[dict[str, object]],
) -> None:
    critical = {}
    for coverage in TARGET_COVERAGES:
        values = [int(row["critical_events_available"]) for row in records
                  if row["coverage"] == coverage and row["method"] == ADAPTIVE_METHOD
                  and row["budget"] == 50]
        critical[coverage] = (np.mean(values), np.mean(values) / SCENARIO_COUNT)
    lines = [
        "# Independent confirmation of Function Posterior Search",
        "",
        "This is a frozen physical confirmation experiment. It uses five previously unused "
        "scenario seeds, 600 scenarios per seed, six functions, and 18 targets arranged as "
        "a 3 (source coverage) x 3 (functional heterogeneity) x 2 factorial. No target outcome "
        "was used to define a controller, scenario, source-only hyperparameter, or baseline.",
        "",
        "Primary endpoint: target-macro critical-event Recall@50 for adaptive-support posterior "
        "search versus Function-Conditioned Mining. Random uses 20 repeats; all other methods "
        "are deterministic. Collision OR near miss is the critical-event oracle.",
        "",
        "## Overall results",
        "",
        "| Method | B=10 | B=20 | B=30 | B=50 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        values = [_lookup(summary, method, budget)["mean_critical_recall"] for budget in BUDGETS]
        lines.append(f"| {method} | " + " | ".join(f"{value:.4f}" for value in values) + " |")
    lines += ["", "## Recall@50 by source coverage", "",
              "| Coverage | Function-Conditioned Mining | Posterior search | Difference |",
              "| --- | ---: | ---: | ---: |"]
    for coverage in TARGET_COVERAGES:
        baseline = _lookup(summary, "Function-Conditioned Mining", 50, coverage=coverage)
        proposed = _lookup(summary, ADAPTIVE_METHOD, 50, coverage=coverage)
        difference = proposed["mean_critical_recall"] - baseline["mean_critical_recall"]
        lines.append(
            f"| {coverage} | {baseline['mean_critical_recall']:.4f} | "
            f"{proposed['mean_critical_recall']:.4f} | {difference:+.4f} |"
        )
    lines += ["", "## Recall@50 by functional heterogeneity", "",
              "| Heterogeneity | Function-Conditioned Mining | Posterior search | Difference |",
              "| --- | ---: | ---: | ---: |"]
    for heterogeneity in HETEROGENEITY_LEVELS:
        baseline = _lookup(summary, "Function-Conditioned Mining", 50,
                           heterogeneity=heterogeneity)
        proposed = _lookup(summary, ADAPTIVE_METHOD, 50,
                           heterogeneity=heterogeneity)
        difference = proposed["mean_critical_recall"] - baseline["mean_critical_recall"]
        lines.append(
            f"| {heterogeneity} | {baseline['mean_critical_recall']:.4f} | "
            f"{proposed['mean_critical_recall']:.4f} | {difference:+.4f} |"
        )
    primary = next(row for row in comparisons if row["scope"] == "overall"
                   and row["reference"] == "Function-Conditioned Mining"
                   and row["budget"] == 50)
    lines += [
        "",
        "## Confirmatory comparison",
        "",
        f"Adaptive posterior search minus Function-Conditioned Mining at Recall@50: "
        f"{primary['mean_recall_difference']:+.4f}, hierarchical 95% bootstrap interval "
        f"[{primary['hierarchical_bootstrap_low']:+.4f}, "
        f"{primary['hierarchical_bootstrap_high']:+.4f}], with "
        f"{primary['wins']} wins, {primary['ties']} ties, and {primary['losses']} losses "
        f"over {primary['seed_target_units']} seed-target units.",
        "",
        "## Descriptive function audit at Recall@50",
        "",
        "| Function | Event prevalence | Function-Conditioned Mining | Posterior search |",
        "| --- | ---: | ---: | ---: |",
    ]
    for mode in MODES:
        baseline = next(
            row for row in subgroup_summary
            if row["subgroup"] == "mode" and row["value"] == mode
            and row["method"] == "Function-Conditioned Mining" and row["budget"] == 50
        )
        proposed = next(
            row for row in subgroup_summary
            if row["subgroup"] == "mode" and row["value"] == mode
            and row["method"] == ADAPTIVE_METHOD and row["budget"] == 50
        )
        lines.append(
            f"| {mode} | {baseline['mean_event_prevalence']:.1%} | "
            f"{_recall_text(baseline['mean_critical_recall'])} | "
            f"{_recall_text(proposed['mean_critical_recall'])} |"
        )
    lines += [
        "",
        "## Descriptive scenario-regime audit at Recall@50",
        "",
        "| Regime | Event prevalence | Function-Conditioned Mining | Posterior search |",
        "| --- | ---: | ---: | ---: |",
    ]
    for regime in REGIME_COUNTS:
        baseline = next(
            row for row in subgroup_summary
            if row["subgroup"] == "regime" and row["value"] == regime
            and row["method"] == "Function-Conditioned Mining" and row["budget"] == 50
        )
        proposed = next(
            row for row in subgroup_summary
            if row["subgroup"] == "regime" and row["value"] == regime
            and row["method"] == ADAPTIVE_METHOD and row["budget"] == 50
        )
        lines.append(
            f"| {regime} | {baseline['mean_event_prevalence']:.1%} | "
            f"{_recall_text(baseline['mean_critical_recall'])} | "
            f"{_recall_text(proposed['mean_critical_recall'])} |"
        )
    lines += [
        "",
        "## Event prevalence and interpretation",
        "",
    ]
    for coverage, (count, prevalence) in critical.items():
        lines.append(f"- {coverage}: mean {count:.1f} critical events per 600 scenarios "
                     f"({prevalence:.1%}).")
    lines += [
        "",
        "Exact coverage is the method-favourable mechanism check; interpolated coverage tests "
        "model misspecification inside the source hull; unseen coverage is an adverse negative "
        "control. Global targets are another negative control. Broad superiority is supported "
        "only if the overall paired interval excludes zero without relying solely on the exact "
        "or fully heterogeneous cells.",
        "",
        "The gain is not uniform across the scenario space. Posterior search is weaker in the "
        "core regime and stronger at boundary/outside conditions; it also trades recall among "
        "functions. The evidence therefore supports stress-oriented prioritization, not a claim "
        "that every function or operating region improves. Overall critical-event prevalence "
        "is moderate rather than ultra-rare.",
        "",
        "The simulator remains a straight-road highway-env harness with scheduled traffic and "
        "does not establish real-world safety or certification. Passing scenarios use their "
        "existing three-vehicle implementation; timing and intensity are stored but do not alter "
        "that mode's fixed schedule.",
    ]
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(workers: int = 12) -> Path:
    """Build all physical banks, run baselines, and write formal evidence."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    BANK_DIR.mkdir(parents=True, exist_ok=True)
    _write_design_files()
    rows: list[dict[str, object]] = []
    tuning: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []
    for seed in CONFIRMATION_SEEDS:
        print(f"seed {seed}: building or loading physical bank", flush=True)
        bank = load_or_build_bank(BANK_DIR / f"response_bank_{seed}.npz", seed, workers)
        result, source_rows, audit = evaluate_bank(bank, seed)
        rows.extend(result)
        tuning.extend(source_rows)
        decisions.extend(audit)
    summary = summarize(rows)
    comparisons = paired_comparisons(rows)
    subgroup_summary = summarize_subgroups(rows)
    for filename, data in (
        ("records.csv", rows),
        ("summary.csv", summary),
        ("subgroup_summary.csv", subgroup_summary),
        ("paired_comparisons.csv", comparisons),
        ("source_selection.csv", tuning),
        ("decisions.csv", decisions),
    ):
        _write_csv(OUTPUT / filename, data)
    manifest = {
        "status": "frozen independent confirmation",
        "scenario_seeds": list(CONFIRMATION_SEEDS),
        "scenario_count_per_seed": SCENARIO_COUNT,
        "modes": list(MODES),
        "regimes_per_mode": REGIME_COUNTS,
        "source_suts": list(SOURCE_NAMES),
        "target_count": len(target_releases()),
        "target_factorial": {
            "coverage": list(TARGET_COVERAGES),
            "heterogeneity": list(HETEROGENEITY_LEVELS),
            "replicates": 2,
        },
        "budgets": list(BUDGETS),
        "support_budget": SUPPORT_BUDGET,
        "random_repeats": RANDOM_REPEATS,
        "noise_scales": list(NOISE_SCALES),
        "hyperparameter_visibility": "source releases only; one selection per scenario seed",
        "target_visibility": (
            "only selected target responses; full labels used for offline scoring"
        ),
        "event_oracle": "collision OR near_miss",
        "primary_endpoint": (
            "target-macro critical-event Recall@50: adaptive posterior search minus "
            "Function-Conditioned Mining"
        ),
        "bootstrap": (
            "10000 paired hierarchical resamples: scenario seed, then targets within seed"
        ),
        "physical_execution": "all distinct controller-module and scenario pairs in highway-env",
        "new_simulator_banks": len(CONFIRMATION_SEEDS),
        "physical_episodes_per_seed": physical_episode_count(),
        "total_new_physical_episodes": (
            len(CONFIRMATION_SEEDS) * physical_episode_count()
        ),
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )
    write_report(summary, comparisons, rows, subgroup_summary)
    return OUTPUT


if __name__ == "__main__":
    print(run())
