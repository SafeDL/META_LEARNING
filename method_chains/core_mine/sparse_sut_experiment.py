"""Sparse-risk, leave-one-SUT-out physical validation for CoRe-Mine.

This experiment is intentionally separate from the legacy profile-bank study.
It uses the retained externally controlled Highway-env policies as both source
systems and held-out targets, and generates an outcome-independent, benign-
heavy candidate pool.  Every policy executes every candidate once per seed;
selectors see only source rows and selectively revealed target outcomes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc

from highway_env_benchmark.envs.cutin_env import CutInScenario
from method_chains.core_mine.config import BUDGETS, CoreMineConfig
from method_chains.core_mine.data import CachedTask, _features, response_value
from method_chains.core_mine.experiment import METHODS, paired_comparisons, run_campaign, summarize
from method_chains.core_mine.metrics import record_metrics
from replications.highway_sut_selection.runner import ASSETS, run_episode
from sut_algorithms.highway_env.registry import RETAINED_SUTS, policy_factory


ROOT = Path("results/method_chains/core_mine/studies/sparse_suts")
POOL_VERSION = "v2_far_headway"
BANK_DIRECTORY_BY_PROPOSAL = {
    "v2_far_headway": "far_headway",
    "v3_continuous_borderline": "continuous_borderline",
    "v4_sparse_continuous": "sparse_continuous",
    "v5_corrected_geometry": "corrected_geometry",
    "v6_corrected_wide": "corrected_wide",
    "v7_corrected_balanced": "corrected_balanced",
    "v8_source_safe": "source_safe",
}
BANK_DIR = ROOT / "banks" / BANK_DIRECTORY_BY_PROPOSAL[POOL_VERSION]
SEEDS = (20270903, 20270917, 20271001, 20271015, 20271029)
DEVELOPMENT_SEEDS = SEEDS[:2]
VALIDATION_SEEDS = SEEDS[2:]
QUALIFICATION_SEEDS: tuple[int, ...] = ()
ACTIVE_SUTS = RETAINED_SUTS
MODES = ("fast_intrusion", "cutin_braking", "lead_braking", "stop_and_go", "slow_lead_following")
PER_MODE = 100
BENIGN_PER_MODE = 90
CHALLENGE_PER_MODE = PER_MODE - BENIGN_PER_MODE
WORKERS = 4

MODE_BOUNDS = {
    "fast_intrusion": ((8.0, 38.0), (-7.0, 1.0)),
    "cutin_braking": ((10.0, 40.0), (-6.0, 1.0)),
    "lead_braking": ((8.0, 40.0), (-5.0, 2.0)),
    "stop_and_go": ((8.0, 36.0), (-4.0, 2.0)),
    "slow_lead_following": ((5.0, 32.0), (-8.0, -1.0)),
}


def configure_proposal(proposal: str) -> None:
    """Select a versioned, outcome-independent candidate proposal.

    v2 is retained byte-for-byte as the completed far-headway study.  v3 is a
    separate continuous-borderline proposal: it uses a held-out qualification
    seed before any development or validation seed is physically executed.
    """
    global ROOT, POOL_VERSION, BANK_DIR, SEEDS, DEVELOPMENT_SEEDS, VALIDATION_SEEDS, QUALIFICATION_SEEDS, ACTIVE_SUTS, PER_MODE
    if proposal == "v2_far_headway":
        PER_MODE = 100
        ROOT = Path("results/method_chains/core_mine/studies/sparse_suts")
        POOL_VERSION = proposal
        SEEDS = (20270903, 20270917, 20271001, 20271015, 20271029)
        QUALIFICATION_SEEDS = ()
        ACTIVE_SUTS = RETAINED_SUTS
    elif proposal == "v3_continuous_borderline":
        PER_MODE = 100
        ROOT = Path("results/method_chains/core_mine/studies/continuous_suts")
        POOL_VERSION = proposal
        # Qualification is kept disjoint from development and confirmation.
        QUALIFICATION_SEEDS = (20271103,)
        SEEDS = (20271117, 20271201, 20271215, 20271229, 20280112)
        ACTIVE_SUTS = RETAINED_SUTS
    elif proposal == "v4_sparse_continuous":
        PER_MODE = 100
        ROOT = Path("results/method_chains/core_mine/studies/sparse_continuous_suts")
        POOL_VERSION = proposal
        # v3 showed that vi_ttc is a constant-density source in this family.
        # The retained but informative non-saturated systems remain eligible.
        ACTIVE_SUTS = ("idm_mobil", "mcts_cv", "ppo_ece")
        QUALIFICATION_SEEDS = (20280203,)
        SEEDS = (20280217, 20280303, 20280317, 20280331, 20280414)
    elif proposal == "v5_corrected_geometry":
        ROOT = Path("results/method_chains/core_mine/studies/corrected_geometry")
        POOL_VERSION = proposal
        ACTIVE_SUTS = ("idm_mobil", "mcts_cv", "ppo_ece")
        PER_MODE = 64
        QUALIFICATION_SEEDS = (20280909,)
        SEEDS = (20280917, 20281001, 20281015, 20281029, 20281112)
    elif proposal == "v6_corrected_wide":
        ROOT = Path("results/method_chains/core_mine/studies/corrected_wide")
        POOL_VERSION = proposal
        ACTIVE_SUTS = ("idm_mobil", "mcts_cv", "ppo_ece", "vi_ttc")
        PER_MODE = 64
        QUALIFICATION_SEEDS = (20281209,)
        SEEDS = (20281217, 20290103, 20290117, 20290131, 20290214)
    elif proposal == "v7_corrected_balanced":
        ROOT = Path("results/method_chains/core_mine/studies/corrected_balanced")
        POOL_VERSION = proposal
        ACTIVE_SUTS = ("idm_mobil", "mcts_cv", "ppo_ece", "vi_ttc")
        PER_MODE = 64
        QUALIFICATION_SEEDS = (20290325,)
        SEEDS = (20290408, 20290422, 20290506, 20290520, 20290603)
    elif proposal == "v8_source_safe":
        ROOT = Path("results/method_chains/core_mine/studies/source_safe")
        POOL_VERSION = proposal
        ACTIVE_SUTS = ("idm_mobil", "mcts_cv", "ppo_ece", "vi_ttc")
        PER_MODE = 64
        QUALIFICATION_SEEDS = (20290617,)
        SEEDS = (20290729, 20290812, 20290826)
    else:
        raise ValueError(f"unknown proposal: {proposal}")
    DEVELOPMENT_SEEDS = () if proposal == "v8_source_safe" else SEEDS[:2]
    VALIDATION_SEEDS = SEEDS if proposal == "v8_source_safe" else SEEDS[2:]
    BANK_DIR = ROOT / "banks" / BANK_DIRECTORY_BY_PROPOSAL[POOL_VERSION]


@dataclass(frozen=True)
class SparseBank:
    anchors: np.ndarray
    modes: np.ndarray
    controls: np.ndarray
    regimes: np.ndarray
    sut_names: tuple[str, ...]
    ego_collision: np.ndarray
    background_collision: np.ndarray
    near_miss: np.ndarray
    min_ttc: np.ndarray
    min_distance: np.ndarray
    completed: np.ndarray

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, anchors=self.anchors, modes=self.modes, controls=self.controls,
                            regimes=self.regimes, sut_names=np.asarray(self.sut_names),
                            ego_collision=self.ego_collision, background_collision=self.background_collision,
                            near_miss=self.near_miss, min_ttc=self.min_ttc,
                            min_distance=self.min_distance, completed=self.completed,
                            safety_metric_version="polygon_clearance_1m_v1")

    @classmethod
    def load(cls, path: Path) -> "SparseBank":
        with np.load(path, allow_pickle=False) as data:
            return cls(data["anchors"], data["modes"].astype(str), data["controls"], data["regimes"].astype(str),
                       tuple(data["sut_names"].astype(str)), data["ego_collision"].astype(bool),
                       data["background_collision"].astype(bool), data["near_miss"].astype(bool),
                       data["min_ttc"], data["min_distance"], data["completed"].astype(bool))


def sparse_scenarios(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create the selected proposal without inspecting SUT outcomes."""
    if POOL_VERSION == "v5_corrected_geometry":
        return corrected_geometry_scenarios(seed)
    if POOL_VERSION == "v6_corrected_wide":
        return wide_corrected_scenarios(seed)
    if POOL_VERSION in {"v7_corrected_balanced", "v8_source_safe"}:
        return balanced_corrected_scenarios(seed)
    if POOL_VERSION == "v3_continuous_borderline":
        return continuous_borderline_scenarios(seed)
    if POOL_VERSION == "v4_sparse_continuous":
        return sparse_continuous_scenarios(seed)
    return far_headway_scenarios(seed)


def _corrected_scenarios(seed: int, bounds: dict[str, tuple[tuple[float, float], tuple[float, float]]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create one outcome-blind continuous pool from frozen physical bounds."""
    anchors: list[np.ndarray] = []
    controls: list[np.ndarray] = []
    modes: list[str] = []
    regimes: list[str] = []
    for offset, mode in enumerate(MODES):
        unit = qmc.Sobol(4, scramble=True, seed=seed + offset).random_base2(6)
        (gap_low, gap_high), (speed_low, speed_high) = bounds[mode]
        anchors.append(np.column_stack((gap_low + (gap_high - gap_low) * unit[:, 0],
                                        speed_low + (speed_high - speed_low) * unit[:, 1])))
        controls.append(np.column_stack((.05 + .90 * unit[:, 2], .05 + .90 * unit[:, 3])))
        modes.extend([mode] * PER_MODE)
        regimes.extend([f"gap_q{min(4, int(value * 4) + 1)}" for value in unit[:, 0]])
    return np.vstack(anchors), np.asarray(modes, dtype="U32"), np.vstack(controls), np.asarray(regimes, dtype="U16")


def corrected_geometry_scenarios(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """v5 fixed after a separate development pilot, before qualification."""
    return _corrected_scenarios(seed, {
        "fast_intrusion": ((6.0, 25.0), (-8.0, 0.0)),
        "cutin_braking": ((6.0, 25.0), (-8.0, 0.0)),
        "lead_braking": ((8.0, 30.0), (-7.0, 0.0)),
        "stop_and_go": ((5.0, 24.0), (-8.0, -1.0)),
        "slow_lead_following": ((5.0, 22.0), (-8.0, -1.0)),
    })


def wide_corrected_scenarios(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """v6 wider physical ranges, fixed after pilot seed 20281201."""
    return _corrected_scenarios(seed, {
        "fast_intrusion": ((8.0, 40.0), (-8.0, 1.0)),
        "cutin_braking": ((8.0, 42.0), (-8.0, 1.0)),
        "lead_braking": ((10.0, 42.0), (-7.0, 1.0)),
        "stop_and_go": ((8.0, 42.0), (-8.0, 1.0)),
        "slow_lead_following": ((7.0, 36.0), (-8.0, 0.0)),
    })


def balanced_corrected_scenarios(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """v7 broadened speed range, fixed after pilot seed 20290317."""
    return _corrected_scenarios(seed, {
        "fast_intrusion": ((6.0, 25.0), (-8.0, 8.0)),
        "cutin_braking": ((6.0, 25.0), (-8.0, 8.0)),
        "lead_braking": ((8.0, 30.0), (-7.0, 8.0)),
        "stop_and_go": ((5.0, 24.0), (-8.0, 8.0)),
        "slow_lead_following": ((5.0, 22.0), (-8.0, 7.0)),
    })


def far_headway_scenarios(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fixed v2 90:10 benign:challenge proposal; independent of outcomes."""
    anchors: list[np.ndarray] = []; controls: list[np.ndarray] = []; labels: list[str] = []; regimes: list[str] = []
    for offset, mode in enumerate(MODES):
        unit = qmc.Sobol(4, scramble=True, seed=seed + offset).random_base2(7)[:PER_MODE]
        (gap_low, gap_high), (speed_low, speed_high) = MODE_BOUNDS[mode]
        gap_span, speed_span = gap_high - gap_low, speed_high - speed_low
        values = np.empty((PER_MODE, 2)); mode_controls = np.empty((PER_MODE, 2))
        for index, row in enumerate(unit):
            if index < BENIGN_PER_MODE:
                # Large headway, little/no closing speed, late/low-intensity manoeuvre.
                headway_offset = (.75 + .50 * row[0]) * gap_span if mode in {"slow_lead_following", "stop_and_go"} else .15 * gap_span * row[0]
                values[index] = gap_high + headway_offset, max(-.5, speed_low + .72 * speed_span) + .28 * speed_span * row[1]
                mode_controls[index] = (.55 + .45 * row[2], .00 + (.05 if mode == "slow_lead_following" else .30) * row[3])
            else:
                # Predeclared stress tail, not selected after seeing any SUT outcome.
                values[index] = gap_low + .32 * gap_span * row[0], speed_low + .30 * speed_span * row[1]
                mode_controls[index] = (.05 + .55 * row[2], .70 + .30 * row[3])
        anchors.append(values); controls.append(mode_controls); labels.extend([mode] * PER_MODE)
        regimes.extend(["benign"] * BENIGN_PER_MODE + ["challenge"] * CHALLENGE_PER_MODE)
    return np.vstack(anchors), np.asarray(labels, dtype="U32"), np.vstack(controls), np.asarray(regimes, dtype="U16")


def continuous_borderline_scenarios(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """v3: continuous parameter bands around controller-specific boundaries.

    The bounds were fixed from the v2 diagnosis before v3 qualification.  In
    contrast to v2, no stratum is declared benign or challenge: every variable
    varies continuously and the quartile label is only an audit stratum.
    """
    anchors: list[np.ndarray] = []; controls: list[np.ndarray] = []; labels: list[str] = []; regimes: list[str] = []
    # (gap low/high fraction, relative-speed low/high fraction).  Slow leading
    # receives more headway because its intensity additionally lowers lead speed.
    fractions = {
        "fast_intrusion": ((.26, .88), (.35, .92)),
        "cutin_braking": ((.30, .90), (.34, .92)),
        "lead_braking": ((.24, .88), (.32, .94)),
        "stop_and_go": ((.28, .90), (.30, .94)),
        "slow_lead_following": ((.44, .96), (.42, .92)),
    }
    for offset, mode in enumerate(MODES):
        unit = qmc.Sobol(4, scramble=True, seed=seed + offset).random_base2(7)[:PER_MODE]
        (gap_low, gap_high), (speed_low, speed_high) = MODE_BOUNDS[mode]
        (gap_fraction, speed_fraction) = fractions[mode]
        gap = gap_low + (gap_fraction[0] + (gap_fraction[1] - gap_fraction[0]) * unit[:, 0]) * (gap_high - gap_low)
        speed = speed_low + (speed_fraction[0] + (speed_fraction[1] - speed_fraction[0]) * unit[:, 1]) * (speed_high - speed_low)
        # The controls are continuous schedule parameters, not a severity tail.
        timing = .05 + .90 * unit[:, 2]
        intensity = .05 + .90 * unit[:, 3]
        anchors.append(np.column_stack((gap, speed)))
        controls.append(np.column_stack((timing, intensity)))
        labels.extend([mode] * PER_MODE)
        # Stable, outcome-blind audit strata derived from initial-gap quartiles.
        regimes.extend([f"gap_q{min(4, int(value * 4) + 1)}" for value in unit[:, 0]])
    return np.vstack(anchors), np.asarray(labels, dtype="U32"), np.vstack(controls), np.asarray(regimes, dtype="U16")


def sparse_continuous_scenarios(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """v4: v3's continuous modes, shifted to sparse borderline headways.

    Values were fixed from v3's physical diagnosis before the v4 qualification
    seed.  This is a continuous proposal, not a post-outcome stress tail.
    """
    anchors: list[np.ndarray] = []; controls: list[np.ndarray] = []; labels: list[str] = []; regimes: list[str] = []
    fractions = {
        "fast_intrusion": ((.40, .98), (.45, .95)),
        "cutin_braking": ((.45, 1.05), (.45, .95)),
        "lead_braking": ((.42, .98), (.40, .96)),
        "stop_and_go": ((.50, 1.08), (.42, .96)),
        "slow_lead_following": ((.80, 1.40), (.55, .98)),
    }
    for offset, mode in enumerate(MODES):
        unit = qmc.Sobol(4, scramble=True, seed=seed + offset).random_base2(7)[:PER_MODE]
        (gap_low, gap_high), (speed_low, speed_high) = MODE_BOUNDS[mode]
        (gap_fraction, speed_fraction) = fractions[mode]
        gap = gap_low + (gap_fraction[0] + (gap_fraction[1] - gap_fraction[0]) * unit[:, 0]) * (gap_high - gap_low)
        speed = speed_low + (speed_fraction[0] + (speed_fraction[1] - speed_fraction[0]) * unit[:, 1]) * (speed_high - speed_low)
        anchors.append(np.column_stack((gap, speed)))
        controls.append(np.column_stack((.05 + .90 * unit[:, 2], .05 + .90 * unit[:, 3])))
        labels.extend([mode] * PER_MODE)
        regimes.extend([f"gap_q{min(4, int(value * 4) + 1)}" for value in unit[:, 0]])
    return np.vstack(anchors), np.asarray(labels, dtype="U32"), np.vstack(controls), np.asarray(regimes, dtype="U16")


def _run_policy_batch(args: tuple[str, np.ndarray, np.ndarray, np.ndarray, int]) -> dict[str, np.ndarray]:
    sut, anchors, modes, controls, seed = args
    policy = policy_factory(sut, ASSETS)
    values: dict[str, list[object]] = {key: [] for key in ("ego_collision", "background_collision", "near_miss", "min_ttc", "min_distance", "completed")}
    for index, (anchor, mode, control) in enumerate(zip(anchors, modes, controls, strict=True)):
        result = run_episode(policy, CutInScenario(float(anchor[0]), float(anchor[1]), str(mode), float(control[0]), float(control[1])), seed + index)
        for key in values: values[key].append(result[key])
    return {key: np.asarray(value, dtype=bool if key in {"ego_collision", "background_collision", "near_miss", "completed"} else float) for key, value in values.items()}


def build_bank(seed: int, workers: int = WORKERS) -> SparseBank:
    anchors, modes, controls, regimes = sparse_scenarios(seed)
    jobs = [(sut, anchors, modes, controls, seed) for sut in ACTIVE_SUTS]
    if workers == 1:
        batches = list(map(_run_policy_batch, jobs))
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
            batches = list(pool.map(_run_policy_batch, jobs))
    return SparseBank(anchors, modes, controls, regimes, tuple(ACTIVE_SUTS),
                      *(np.stack([batch[field] for batch in batches]) for field in
                        ("ego_collision", "background_collision", "near_miss", "min_ttc", "min_distance", "completed")))


def load_or_build(seed: int, workers: int = WORKERS) -> SparseBank:
    path = BANK_DIR / f"sparse_sut_bank_{seed}.npz"
    if path.exists():
        if POOL_VERSION in {"v5_corrected_geometry", "v6_corrected_wide", "v7_corrected_balanced", "v8_source_safe"}:
            with np.load(path, allow_pickle=False) as stored:
                version = str(stored["safety_metric_version"]) if "safety_metric_version" in stored else "legacy"
            if version != "polygon_clearance_1m_v1":
                raise ValueError(f"v5 requires corrected geometry labels, found {version}: {path}")
        bank = SparseBank.load(path)
        if bank.sut_names != tuple(ACTIVE_SUTS) or len(bank.anchors) != len(MODES) * PER_MODE:
            raise ValueError(f"incompatible sparse bank: {path}")
        return bank
    bank = build_bank(seed, workers); bank.save(path); return bank


def tasks_from_bank(bank: SparseBank, seed: int) -> tuple[CachedTask, ...]:
    features, dimensions = _features(bank.anchors, bank.modes, bank.controls)
    tasks = []
    for target_index, target in enumerate(bank.sut_names):
        source_indices = np.asarray([index for index in range(len(bank.sut_names)) if index != target_index])
        source_event = (bank.ego_collision[source_indices] | bank.near_miss[source_indices])
        source_collision = bank.ego_collision[source_indices]
        event = bank.ego_collision[target_index] | bank.near_miss[target_index]
        collision = bank.ego_collision[target_index]
        tasks.append(CachedTask(seed, f"Target-{target}-leaveoneout", "leave_one_out", target,
                                bank.anchors, bank.modes, bank.controls, features, dimensions,
                                response_value(bank.min_ttc[source_indices], source_event, source_collision), source_event, source_collision,
                                response_value(bank.min_ttc[target_index], event, collision), event, collision,
                                bank.min_ttc[target_index], (~source_event.any(axis=0)) & event))
    return tuple(tasks)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: path.write_text("", encoding="utf-8"); return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def _jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows: handle.write(json.dumps(row) + "\n")


def _evaluate(tasks: Iterable[CachedTask], config: CoreMineConfig, methods: tuple[str, ...] = METHODS) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    records: list[dict[str, object]] = []; traces: list[dict[str, object]] = []
    for task in tasks:
        for method in methods:
            rows, trace = run_campaign(task, method, config); records.extend(rows); traces.extend(trace)
        for repeat in range(10):
            rows, trace = run_campaign(task, "Random", config, repeat); records.extend(rows); traces.extend(trace)
    return records, traces


def audit(banks: Iterable[tuple[int, SparseBank]], output: Path | None = None) -> None:
    output = ROOT if output is None else output
    rows: list[dict[str, object]] = []
    for seed, bank in banks:
        for target_index, target in enumerate(bank.sut_names):
            event = bank.ego_collision[target_index] | bank.near_miss[target_index]
            source = np.delete(bank.ego_collision | bank.near_miss, target_index, axis=0)
            for regime in (*np.unique(bank.regimes).tolist(), "all"):
                mask = np.ones(len(event), dtype=bool) if regime == "all" else bank.regimes == regime
                rows.append({"seed": seed, "target": target, "regime": regime, "candidates": int(mask.sum()),
                             "critical_events": int(event[mask].sum()), "collisions": int(bank.ego_collision[target_index, mask].sum()),
                             "critical_rate": float(event[mask].mean()), "all_source_safe_target_failures": int(((~source.any(axis=0)) & event & mask).sum())})
    _write_csv(output / "baseline_recheck.csv", rows)
    full = [row for row in rows if row["regime"] == "all"]
    mean_rate = float(np.mean([row["critical_rate"] for row in full]))
    proposal = (f"{BENIGN_PER_MODE}:{CHALLENGE_PER_MODE} benign:challenge per function"
                if POOL_VERSION == "v2_far_headway" else "a continuous proposal with outcome-blind initial-gap quartile audit strata")
    output.mkdir(parents=True, exist_ok=True)
    (output / "opportunity.md").write_text(
        f"# Sparse-risk opportunity scan\n\nThe fixed candidate proposal is {proposal}. "
        f"Across leave-one-SUT-out target units, the physical critical-event rate is {mean_rate:.2%}. "
        "All-source-safe target failures are retained as a descriptive audit and never used by a selector.\n", encoding="utf-8")


def qualification_gate(banks: Iterable[tuple[int, SparseBank]], output: Path) -> dict[str, object]:
    """Apply v3's branch-disagreement gates before confirmation is generated.

    ``all_source_safe`` remains an audit because it is an extreme target-only
    event.  It is not a validity condition for CoRe: CoRe's historical prior
    explicitly has one branch per source and is informative when sources
    disagree.  Earlier gate files are retained.  Dispersion is evaluated only
    for targets with at least ten events: a max-quartile share from fewer
    events is not an estimable concentration statistic.
    """
    frozen = list(banks)
    if not frozen:
        raise ValueError("qualification requires at least one bank")
    names = frozen[0][1].sut_names
    events = {name: [] for name in names}
    source_safe = {name: [] for name in names}
    all_modes: list[np.ndarray] = []
    all_regimes: list[np.ndarray] = []
    for _, bank in frozen:
        all_modes.append(bank.modes); all_regimes.append(bank.regimes)
        for index, name in enumerate(names):
            event = bank.ego_collision[index] | bank.near_miss[index]
            source = np.delete(bank.ego_collision | bank.near_miss, index, axis=0)
            events[name].append(event)
            source_safe[name].append((~source.any(axis=0)) & event)
    modes = np.concatenate(all_modes); regimes = np.concatenate(all_regimes)
    aggregate = {name: np.concatenate(values) for name, values in events.items()}
    aggregate_safe = {name: np.concatenate(values) for name, values in source_safe.items()}
    target_rows = []
    for name in names:
        index = names.index(name)
        source = np.delete(np.stack([aggregate[candidate] for candidate in names]), index, axis=0)
        # At least one source branches toward a failure and another remains safe.
        discordant = aggregate[name] & source.any(axis=0) & (~source).any(axis=0)
        target_rows.append({
            "target": name,
            "source_safe_target_failures": int(aggregate_safe[name].sum()),
            "source_safe_modes": sorted(np.unique(modes[aggregate_safe[name]]).tolist()),
            "source_discordant_target_failures": int(discordant.sum()),
            "source_discordant_modes": sorted(np.unique(modes[discordant]).tolist()),
            "critical_gap_quartiles": sorted(np.unique(regimes[aggregate[name]]).tolist()),
        })
    pairwise = {}
    for left, right in combinations(names, 2):
        union = aggregate[left] | aggregate[right]
        pairwise[f"{left}~{right}"] = float((aggregate[left] & aggregate[right]).sum() / union.sum()) if union.any() else 1.0
    redundant_triplets = [list(triplet) for triplet in combinations(names, 3)
                          if all(pairwise["~".join(pair)] > .90 for pair in combinations(triplet, 2))]
    event_count = {name: int(aggregate[name].sum()) for name in names}
    max_stratum_share = {
        name: max((float((aggregate[name] & (regimes == regime)).sum() / aggregate[name].sum())
                   for regime in np.unique(regimes)), default=1.0)
        for name in names
    }
    qualified_targets = [row["target"] for row in target_rows if len(row["source_discordant_modes"]) >= 2]
    payload: dict[str, object] = {
        "proposal": POOL_VERSION,
        "gate_version": "v3_branch_disagreement_min10_dispersion",
        "seeds": [seed for seed, _ in frozen],
        "gates": {
            "two_targets_with_source_discordant_failures_in_two_modes": len(qualified_targets) >= 2,
            "no_three_sut_all_pairwise_jaccard_above_0_90": not redundant_triplets,
            "no_eligible_target_more_than_75_percent_in_one_gap_quartile": all(
                max_stratum_share[name] <= .75 for name in names if event_count[name] >= 10),
        },
        "qualified_targets": qualified_targets,
        "target_diagnostics": target_rows,
        "pairwise_event_jaccard": pairwise,
        "critical_event_count": event_count,
        "critical_gap_quartile_max_share": max_stratum_share,
        "redundant_triplets": redundant_triplets,
    }
    payload["passed"] = bool(all(payload["gates"].values()))
    output.mkdir(parents=True, exist_ok=True)
    (output / "qualification_gate.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def tune(tasks: tuple[CachedTask, ...]) -> CoreMineConfig:
    primary_budget = 50 if POOL_VERSION in {"v5_corrected_geometry", "v6_corrected_wide", "v7_corrected_balanced"} else 20
    options = ((.15, .15, 0), (.15, .25, .10), (.15, .50, .30), (.30, .15, .10), (.30, .25, 0), (.30, .50, .30), (.60, .15, 0), (.60, .25, .10), (.60, .50, .30), (.15, .50, .10), (.30, .25, .10), (.60, .15, .30))
    if POOL_VERSION in {"v5_corrected_geometry", "v6_corrected_wide", "v7_corrected_balanced"}:
        options = options[:6]
    rows = []
    for trial, (length, amplitude, lambda_) in enumerate(options, 1):
        config = CoreMineConfig(length, amplitude, lambda_=lambda_)
        values = [row for task in tasks for row in run_campaign(task, "CoRe-Marginal", config)[0] if row["budget"] == primary_budget]
        rows.append({"trial": trial, **config.as_dict(), "primary_budget": primary_budget, "units": len(values),
                     "mean_primary_CVS": float(np.mean([float(row["CVS"]) for row in values])),
                     "mean_primary_SeveritySum": float(np.mean([float(row["SeveritySum"]) for row in values]))})
    _write_csv(ROOT / "trials.csv", rows)
    best_severity = max(float(row["mean_primary_SeveritySum"]) for row in rows)
    selected = max((row for row in rows if float(row["mean_primary_SeveritySum"]) >= .9 * best_severity),
                   key=lambda row: (float(row["mean_primary_CVS"]), float(row["mean_primary_SeveritySum"])))
    config = CoreMineConfig(float(selected["residual_length"]), float(selected["residual_amplitude"]), lambda_=float(selected["lambda"]))
    (ROOT / "frozen_config.json").write_text(json.dumps({"selected_trial": selected, "config": config.as_dict()}, indent=2) + "\n", encoding="utf-8")
    return config


def _render(records: list[dict[str, object]], summary_rows: list[dict[str, object]]) -> None:
    primary_budget = 50 if POOL_VERSION in {"v5_corrected_geometry", "v6_corrected_wide", "v7_corrected_balanced"} else 20
    figures = ROOT / "figures"; figures.mkdir(exist_ok=True)
    for metric, filename in (("CVS", "cvs.png"), ("SeveritySum", "severity.png")):
        plt.figure(figsize=(7, 4))
        for method in ("FPS-Risk", "FPS-Marginal", "MeanResidual-Marginal", "CoRe-Marginal", "TargetOnlyGP-Marginal", "Random"):
            values = [next((float(row[f"mean_{metric}"]) for row in summary_rows if row["method"] == method and int(row["budget"]) == budget and row["scope"] == "overall"), np.nan) for budget in BUDGETS]
            if np.isfinite(values).any(): plt.plot(BUDGETS, values, marker="o", label=method)
        plt.xlabel("Budget"); plt.ylabel(metric); plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(figures / filename, dpi=180); plt.close()
    plt.figure(figsize=(6, 4)); items = [row for row in records if row["method"] == "CoRe-Marginal" and int(row["budget"]) == primary_budget]
    labels = [str(row["heterogeneity"]) for row in items]; values = [float(row["CVS"]) for row in items]
    plt.bar(np.arange(len(values)), values); plt.xticks(np.arange(len(values)), labels, rotation=30); plt.ylabel(f"CoRe-Marginal CVS@{primary_budget}"); plt.tight_layout(); plt.savefig(figures / f"target_cvs{primary_budget}.png", dpi=180); plt.close()


def report(records: list[dict[str, object]], summary_rows: list[dict[str, object]], comparisons: list[dict[str, object]]) -> None:
    primary_budget = 50 if POOL_VERSION in {"v5_corrected_geometry", "v6_corrected_wide", "v7_corrected_balanced"} else 20
    def mean(method: str, metric: str, budget: int = primary_budget) -> float:
        return next(float(row[f"mean_{metric}"]) for row in summary_rows if row["method"] == method and int(row["budget"]) == budget and row["scope"] == "overall")
    baseline = max(("FPS-Marginal", "MeanResidual-Marginal", "TargetOnlyGP-Marginal"), key=lambda name: mean(name, "CVS"))
    comparison = next(row for row in comparisons if row["reference"] == baseline and int(row["budget"]) == primary_budget)
    sparse = [row for row in _read_csv(ROOT / "baseline_recheck.csv") if row["regime"] == "all"]
    proposal = (f"an outcome-independent {BENIGN_PER_MODE}:{CHALLENGE_PER_MODE} benign:challenge design"
                if POOL_VERSION == "v2_far_headway" else "an outcome-independent continuous proposal")
    lines = ["# Sparse-risk, multi-SUT CoRe-Mine validation", "", f"Eligible policies ({', '.join(ACTIVE_SUTS)}) are evaluated as leave-one-SUT-out targets. Each seed has {len(MODES) * PER_MODE} physically executed candidate scenarios across five functions, with {proposal}. Primary budget B={primary_budget}.", "", f"| Method | CVS@{primary_budget} | Collision@{primary_budget} | Severity@{primary_budget} | Critical@{primary_budget} |", "|---|---:|---:|---:|---:|"]
    for method in ("FPS-Risk", "FPS-Severity", "FPS-Balanced", "FPS-Marginal", "MeanResidual-Marginal", "CoRe-Marginal", "TargetOnlyGP-Marginal", "Random"):
        lines.append(f"| {method} | {mean(method, 'CVS'):.3f} | {mean(method, 'CollisionCount'):.3f} | {mean(method, 'SeveritySum'):.3f} | {mean(method, 'CriticalCount'):.3f} |")
    lines += ["", f"Mean target critical-event rate is {np.mean([float(row['critical_rate']) for row in sparse]):.2%}. CoRe-Marginal minus {baseline} at CVS@{primary_budget} is {float(comparison['mean_CVS_difference']):+.3f}, paired bootstrap [{float(comparison['CVS_bootstrap_low']):+.3f}, {float(comparison['CVS_bootstrap_high']):+.3f}].", "", "The source/target split is leave-one-SUT-out; all target response vectors remain inside the cache oracle and are scored only after selection."]
    (ROOT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle: return list(csv.DictReader(handle))


def _manifest(workers: int) -> None:
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(BANK_DIR.glob("*.npz"))}
    confirmation_episodes = len(SEEDS) * len(ACTIVE_SUTS) * len(MODES) * PER_MODE
    qualification_episodes = len(QUALIFICATION_SEEDS) * len(ACTIVE_SUTS) * len(MODES) * PER_MODE
    corrected = POOL_VERSION in {"v5_corrected_geometry", "v6_corrected_wide", "v7_corrected_balanced"}
    payload = {"python": sys.version, "platform": platform.platform(), "suts": list(ACTIVE_SUTS), "excluded_retained_suts": sorted(set(RETAINED_SUTS) - set(ACTIVE_SUTS)), "seeds": list(SEEDS), "qualification_seeds": list(QUALIFICATION_SEEDS), "pool": {"version": POOL_VERSION, "modes": list(MODES), "per_mode": PER_MODE, "benign_per_mode": BENIGN_PER_MODE if POOL_VERSION == "v2_far_headway" else None, "challenge_per_mode": CHALLENGE_PER_MODE if POOL_VERSION == "v2_far_headway" else None}, "safety_metric_version": "polygon_clearance_1m_v1" if corrected else "legacy_bank_labels", "primary_budget": 50 if corrected else 20, "bank_sha256": hashes, "confirmation_physical_episodes": confirmation_episodes, "qualification_physical_episodes": qualification_episodes, "new_physical_episodes": confirmation_episodes + qualification_episodes, "workers": workers}
    (ROOT / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("pilot", "qualification", "all"), default="all")
    parser.add_argument("--proposal", choices=("v2_far_headway", "v3_continuous_borderline", "v4_sparse_continuous", "v5_corrected_geometry", "v6_corrected_wide", "v7_corrected_balanced"), default="v2_far_headway")
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args(); configure_proposal(args.proposal); ROOT.mkdir(parents=True, exist_ok=True); BANK_DIR.mkdir(parents=True, exist_ok=True)
    if args.stage == "qualification":
        if not QUALIFICATION_SEEDS:
            raise ValueError("qualification is only defined for v3_continuous_borderline")
        qualification = [(seed, load_or_build(seed, args.workers)) for seed in QUALIFICATION_SEEDS]
        output = ROOT / "qualification"; audit(qualification, output); qualification_gate(qualification, output)
        return
    if args.proposal in {"v3_continuous_borderline", "v4_sparse_continuous", "v5_corrected_geometry", "v6_corrected_wide", "v7_corrected_balanced"}:
        gate_path = ROOT / "qualification" / "qualification_gate.json"
        if args.stage == "all" and (not gate_path.exists() or not json.loads(gate_path.read_text(encoding="utf-8")).get("passed")):
            raise RuntimeError("v3 confirmation requires a passed independent qualification gate")
    seeds = (SEEDS[0],) if args.stage == "pilot" else SEEDS
    banks = [(seed, load_or_build(seed, args.workers)) for seed in seeds]
    audit(banks); _manifest(args.workers)
    if args.stage == "pilot": return
    dev_tasks = tuple(task for seed, bank in banks if seed in DEVELOPMENT_SEEDS for task in tasks_from_bank(bank, seed))
    config = tune(dev_tasks)
    dev_records, dev_trace = _evaluate(dev_tasks, config); _write_csv(ROOT / "develop" / "records.csv", dev_records); _jsonl(ROOT / "develop" / "trajectories.jsonl", dev_trace)
    validation_tasks = tuple(task for seed, bank in banks if seed in VALIDATION_SEEDS for task in tasks_from_bank(bank, seed))
    validation, trace = _evaluate(validation_tasks, config); _write_csv(ROOT / "validate" / "records.csv", validation); _jsonl(ROOT / "trajectories.jsonl", trace)
    ablations, _ = _evaluate(validation_tasks, config, ("NoComposition", "NoNull")); _write_csv(ROOT / "ablations.csv", ablations)
    summary_rows = summarize(validation + ablations); comparisons = paired_comparisons(validation)
    _write_csv(ROOT / "summary.csv", summary_rows); _write_csv(ROOT / "paired_comparisons.csv", comparisons)
    _render(validation, summary_rows); report(validation, summary_rows, comparisons)


if __name__ == "__main__": main()
