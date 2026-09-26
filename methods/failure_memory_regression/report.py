"""Generate the compact FBRT-Memory v2 report and three evidence figures."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from methods.failure_memory_regression.archive import read_jsonl


METHOD_ORDER = ["Random", "HistoryRank-UCB-v2", "FailureDistance-v2",
                "FBRT-NoMemory", "FBRT-Memory"]
COLORS = {"Random": "#8b95a5", "HistoryRank-UCB-v2": "#ce8f31",
          "FailureDistance-v2": "#42979b", "FBRT-NoMemory": "#8c77b5",
          "FBRT-Memory": "#d04e46"}


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(
            key for row in rows for key in row)), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _number(value, default=np.nan):
    try:
        if value in (None, "", "None", "NA", "nan"):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _write_detection_figure(root: Path, summary: list[dict]) -> None:
    output = root / "figures" / "cumulative_failure_discovery.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in summary:
        grouped[(row.get("task_id", ""), row.get("method", ""),
                 row.get("repeat", "0"))].append(row)
    tasks = sorted({key[0] for key in grouped})
    if not tasks:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.text(0.5, 0.5, "No completed selector tasks", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(output, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return
    cols = 2
    rows_count = int(np.ceil(len(tasks) / cols))
    fig, axes = plt.subplots(rows_count, cols, figsize=(11, 3.2 * rows_count), squeeze=False)
    for axis, task in zip(axes.flat, tasks):
        for method in METHOD_ORDER:
            runs = [items for (task_id, name, _), items in grouped.items()
                    if task_id == task and name == method]
            if not runs:
                continue
            x_values, y_values = [], []
            for items in runs:
                ordered = sorted(items, key=lambda row: int(_number(row.get("budget"), 0)))
                x_values = [int(_number(item.get("budget"), 0)) for item in ordered]
                y_values.append([_number(item.get("failure_count"), 0.0) for item in ordered])
            axis.plot(x_values, np.mean(y_values, axis=0), marker="o", linewidth=1.8,
                      label=method, color=COLORS.get(method))
        axis.set_title(task.replace("_", " "), fontsize=8)
        axis.set_xlabel("Target queries")
        axis.set_ylabel("Observed failures / regressions")
        axis.set_xticks([1, 5, 10, 20])
        axis.grid(alpha=0.2)
    for axis in axes.flat[len(tasks):]:
        axis.set_axis_off()
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
                   bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Budgeted failure discovery by task", y=1.04, fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_pattern_support_figure(root: Path, bank: list[dict], cards: list[dict]) -> None:
    output = root / "figures" / "measured_labels_and_pattern_support.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    templates = []
    for row in bank:
        template = row.get("template_id")
        if template and template not in templates:
            templates.append(template)
    templates = templates[:2]
    if not templates:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No compact bank episodes are available", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(output, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return
    fig, axes = plt.subplots(1, len(templates), figsize=(6 * len(templates), 4), squeeze=False)
    for axis, template in zip(axes[0], templates):
        local = [row for row in bank if row.get("template_id") == template]
        for build in sorted({row.get("build_id", "") for row in local}):
            points = [row for row in local if row.get("build_id") == build]
            # Plot normalized coordinates from only the two active physical inputs.
            coords = []
            for row in points:
                active = row["scenario"].get("active_parameters", {})
                bounds = row["scenario"].get("research_bounds", {})
                names = list(active)
                values = []
                for name in names[:2]:
                    low, high = bounds.get(name, [0.0, 1.0])
                    values.append((float(active[name]) - float(low)) /
                                  (float(high) - float(low)))
                if len(values) == 2:
                    coords.append(values)
            if not coords:
                continue
            color = np.asarray([bool(row.get("ego_collision")) for row in points[:len(coords)]], dtype=int)
            axis.scatter([p[0] for p in coords], [p[1] for p in coords],
                         c=color, cmap=matplotlib.colors.ListedColormap(["#448e77", "#cc5146"]),
                         marker="o" if build.endswith("ref_v2") else "x", s=35,
                         label=build, alpha=0.82)
        local_cards = [card for card in cards if card.get("template_id") == template]
        for card in local_cards:
            center = card.get("center", [0, 0])
            radius = float(card.get("radius", 0.0))
            axis.scatter([center[0]], [center[1]], marker="+", s=90, color="#202a35")
            circle = plt.Circle(center, max(radius, 0.025), fill=False,
                                color="#202a35", linewidth=0.8, alpha=0.35)
            axis.add_patch(circle)
        axis.set_title(template)
        axis.set_xlabel("Normalized active parameter 1")
        axis.set_ylabel("Normalized active parameter 2")
        axis.set_xlim(-0.05, 1.05); axis.set_ylim(-0.05, 1.05)
        axis.grid(alpha=0.2); axis.legend(fontsize=7, loc="best")
    fig.suptitle("Measured compact-bank labels and observed pattern support")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_pattern_lifecycle(root: Path, cards: list[dict]) -> None:
    output = root / "figures" / "pattern_lifecycle.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for card in cards:
        occurrence = card.get("occurrence_by_build", {})
        statuses = Counter(item.get("status", "unknown") for item in occurrence.values())
        rows.append({"pattern_id": card.get("pattern_id"),
                     "template_id": card.get("template_id"),
                     "context_id": card.get("context_id"),
                     "failure_evidence_count": len(card.get("failure_record_ids", [])),
                     "pass_contrast_count": len(card.get("pass_contrast_record_ids", [])),
                     "related_pattern_count": len(card.get("parent_pattern_ids", [])),
                     "evidence_status": card.get("evidence_status"),
                     "build_statuses": json.dumps(occurrence, sort_keys=True),
                     "confirmed_build_count": statuses.get("confirmed_present", 0),
                     "local_pass_build_count": statuses.get("local_pass_evidence", 0)})
    _write_csv(root / "figures" / "pattern_lifecycle.csv", rows)
    if not rows:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No observed failure patterns", ha="center", va="center")
        ax.set_axis_off()
    else:
        totals = Counter(row["evidence_status"] for row in rows)
        fig, ax = plt.subplots(figsize=(8, 4))
        names = ["observed_region", "contrast_available", "singleton"]
        values = [totals.get(name, 0) for name in names]
        bars = ax.bar(names, values, color=["#b94b43", "#e0a141", "#597aaf"])
        for bar, count in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(count),
                    ha="center", va="bottom")
        ax.set_ylabel("Pattern cards with real execution IDs")
        ax.set_title("Failure-pattern lifecycle evidence")
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_report(root: Path, legacy_result: dict | None = None,
                    compact_task_status: list[dict] | None = None) -> str:
    summary = _read_csv(root / "summary_by_task.csv")
    bank = read_jsonl(root / "compact_bank" / "episodes.jsonl")
    cards = read_jsonl(root / "patterns.jsonl")
    acceptance = json.loads((root / "acceptance.json").read_text(encoding="utf-8"))
    ledger = json.loads((root / "compute_ledger.json").read_text(encoding="utf-8"))
    capabilities = json.loads((root / "capabilities.json").read_text(encoding="utf-8"))
    recipe = json.loads((root / "selected_recipe.json").read_text(encoding="utf-8"))
    resource_path = root / "resource_status.json"
    resource = json.loads(resource_path.read_text(encoding="utf-8")) if resource_path.exists() else {}
    asset_path = root / "ppo_asset_provenance.json"
    asset = json.loads(asset_path.read_text(encoding="utf-8")) if asset_path.exists() else {}
    _write_detection_figure(root, summary)
    _write_pattern_support_figure(root, bank, cards)
    _write_pattern_lifecycle(root, cards)

    lines = [
        "# FBRT-Memory v2 execution report", "",
        "## Executive summary", "",
        f"- Selected recipe: `{recipe.get('recipe')}` from static policy-action capability, before method comparison.",
        f"- Legacy measured data imported: 480 reference and 960 target episodes; cache replay uses 0 new simulations.",
        f"- New physical episodes: **{ledger.get('new_physical_episodes', 0)} / 400**; compact-bank records: **{len(bank)}**.",
        f"- Delivery level: **{acceptance.get('delivery_level')}**; observed method effect: `{acceptance.get('method_effect_observation')}`.",
        f"- PPO checkpoint: `{resource.get('ppo_status', recipe.get('ppo_checkpoint_status'))}`; pinned asset SHA-256: `{asset.get('sha256', 'unavailable')}`.",
        "- No policy was trained. The Bayesian logistic models ran on CPU; the `metadrive` conda environment reported CUDA availability in `protocol.json`.",
        "",
        "The report separates engineering acceptance from test effectiveness. Local gains in legacy replay are not claims of superiority on the compact bank.",
        "",
        "## Scope and execution decisions", "",
        f"The catalogue retains {json.loads((root / 'protocol.json').read_text(encoding='utf-8')).get('catalogue_candidate_count', 14)} candidates. The selected shared pool has {len(recipe.get('selected_scenario_ids', []))} templates × 16 cases. Selected IDs: `{', '.join(recipe.get('selected_scenario_ids', []))}`.",
        f"Installed highway-env `{capabilities.get('installed_highway_env_version')}` exposes PPO actions `{', '.join(capabilities.get('policy_action_names', []))}` and speed levels `{capabilities.get('policy_target_speeds_mps')} m/s`. Full stop: `{capabilities.get('can_command_full_stop')}`; lane change: `{capabilities.get('can_change_lane')}`.",
        "The V2 bank uses S01, S02, S05, S06, and S08. S03/S04 are not used because the shared PPO action contract has no zero-speed action. S07 and S09–S14 remain catalogue candidates and were not run.",
        "",
        "## Task results", "",
        "`summary_by_task.csv` retains every repeat and @1/@5/@10/@20 outcome. Random is aggregated across its ten repeats below; other methods show their single run. Missing values remain explicit.",
        "",
        "| Task | Target | Method | Queries (mean) | Failure pool | Failures found (mean) | Detected | First failure (median) | Recall (mean) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    grouped = defaultdict(list)
    for row in summary:
        if int(_number(row.get("budget"), 0)) == 20 or _number(row.get("effective_horizon"), 0) < 20:
            key = (row.get("task_id", ""), row.get("target_build", ""), row.get("method", ""))
            grouped[key].append(row)
    def mean_field(rows, field):
        values = [_number(row.get(field), np.nan) for row in rows]
        values = [value for value in values if np.isfinite(value)]
        return float(np.mean(values)) if values else np.nan
    for (task_id, target, method), rows in sorted(grouped.items()):
        queries = mean_field(rows, "actual_queries")
        failures = mean_field(rows, "failure_count")
        recalls = mean_field(rows, "failure_recall")
        detected_n = sum(str(row.get("failure_detected", "")).lower() == "true" for row in rows)
        ranks = [_number(row.get("first_failure_rank"), np.nan) for row in rows]
        ranks = [rank for rank in ranks if np.isfinite(rank)]
        median_rank = f"{np.median(ranks):.1f}" if ranks else ">B"
        pool = rows[0].get("failure_pool_count", "NA")
        query_text = f"{queries:.1f}" if np.isfinite(queries) else "NA"
        failure_text = f"{failures:.2f}" if np.isfinite(failures) else "NA"
        recall_text = f"{recalls:.3f}" if np.isfinite(recalls) else "NA"
        detected_text = f"{detected_n}/{len(rows)} ({detected_n / len(rows):.0%})" if rows else "NA"
        method_text = f"{method} (n={len(rows)})" if method == "Random" else method
        lines.append(f"| {task_id} | {target} | {method_text} | {query_text} | {pool} | {failure_text} | {detected_text} | {median_rank} | {recall_text} |")
    if not summary:
        lines.append("| NA | NA | NA | 0 | NA | NA | NA | NA | NA |")
    compact_groups = {key[2]: rows for key, rows in grouped.items()
                      if key[0] == "compact_regression_mobil_ref_to_rear_guard_off"}
    if compact_groups:
        compact_task_rows = next(iter(compact_groups.values()))
        pool = compact_task_rows[0].get("failure_pool_count", "NA")
        candidates = compact_task_rows[0].get("candidate_count", "NA")
        lines.extend(["", "## Compact regression reading", "",
                      f"The MOBIL compact holdout contains {candidates} candidates and {pool} verified target-build failures. At B=20, results were:"])
        for method in ("FBRT-Memory", "FBRT-NoMemory", "HistoryRank-UCB-v2",
                       "FailureDistance-v2", "Random"):
            rows = compact_groups.get(method, [])
            if not rows:
                continue
            found = mean_field(rows, "failure_count")
            rate = sum(str(row.get("failure_detected", "")).lower() == "true" for row in rows) / len(rows)
            suffix = f" across {len(rows)} repeats" if len(rows) > 1 else ""
            lines.append(f"- {method}: {found:.2f} failures found on average ({rate:.0%} of runs detected at least one){suffix}.")
        lines.append("This development holdout did not show a benefit from FBRT-Memory at B=20; the `local_gain_observed` acceptance flag refers to observations elsewhere in the legacy replay and is not a general superiority claim.")
    ppo_regression = {key[2]: rows for key, rows in grouped.items()
                      if key[0] == "compact_regression_ppo_ref_to_obs_age020"}
    cross_ppo = {key[2]: rows for key, rows in grouped.items()
                 if key[0].endswith("_ppo_ref_after_mobil")}
    if ppo_regression or cross_ppo:
        lines.extend(["", "## PPO and cross-agent reading", ""])
        if ppo_regression:
            sample = next(iter(ppo_regression.values()))[0]
            lines.append(
                f"The PPO observation-delay regression had {sample.get('candidate_count')} "
                f"parent-pass candidates and {sample.get('failure_pool_count')} new target failures. "
                "Its failure recall is undefined because this bank contained no PPO regressions.")
        if cross_ppo:
            sample = next(iter(cross_ppo.values()))[0]
            lines.append(
                f"The independent PPO-reference session had {sample.get('failure_pool_count')} "
                "collisions among 80 candidates. At B=20, the selectors found:")
            for method in ("FBRT-Memory", "FBRT-NoMemory", "HistoryRank-UCB-v2",
                           "FailureDistance-v2", "Random"):
                rows = cross_ppo.get(method, [])
                if rows:
                    found = mean_field(rows, "failure_count")
                    if method == "Random":
                        lines.append(f"- {method}: {found:.2f} collisions on average across {len(rows)} repeats.")
                    else:
                        lines.append(f"- {method}: {found:.0f} collisions.")
            lines.append("The preceding MOBIL-reference session observed no collisions, so this sequence did not transfer a prior failure pattern to PPO.")
        if asset:
            lines.append(
                f"The PPO ZIP was saved with Stable-Baselines3 {asset.get('checkpoint_sb3_version')} "
                f"and inferred under {asset.get('runtime_sb3_version')}; the loader warned about "
                "training-schedule deserialization, while deterministic inference and the smoke runs completed.")
    lines.extend(["", "## Physical bank", "",
                  "| Build | Episodes | Ego collisions | Inconclusive |",
                  "|---|---:|---:|---:|"])
    counts = defaultdict(lambda: [0, 0, 0])
    for row in bank:
        counts[row["build_id"]][0] += 1
        counts[row["build_id"]][1] += int(row.get("ego_collision") is True and not row.get("inconclusive", False))
        counts[row["build_id"]][2] += int(row.get("inconclusive", False))
    for build, (episodes, collisions, inconclusive) in sorted(counts.items()):
        lines.append(f"| {build} | {episodes} | {collisions} | {inconclusive} |")
    if not counts:
        lines.append("| No physical compact bank | 0 | NA | NA |")
    lines.extend(["", "| Template | Build | Episodes | Ego collisions | Inconclusive |",
                  "|---|---|---:|---:|---:|"])
    template_counts = defaultdict(lambda: [0, 0, 0])
    for row in bank:
        key = (row.get("template_id", "unknown"), row["build_id"])
        template_counts[key][0] += 1
        template_counts[key][1] += int(row.get("ego_collision") is True and not row.get("inconclusive", False))
        template_counts[key][2] += int(row.get("inconclusive", False))
    for (template, build), (episodes, collisions, inconclusive) in sorted(template_counts.items()):
        lines.append(f"| {template} | {build} | {episodes} | {collisions} | {inconclusive} |")
    cutout = [row for row in bank if row.get("template_id") == "fbrt_cutout_static"]
    invalid_exit = sum(row.get("event_times", {}).get("first_exit_s") == 0.0
                       for row in cutout)
    ppo_lead = sum(row.get("build_id", "").startswith("ppo_") and
                   row.get("ego_collision") is True and
                   row.get("collision_partner_role") == "lead" for row in cutout)
    if invalid_exit:
        lines.extend(["", "### Event interpretation limit", "",
                      f"`first_exit_s` was incorrectly recorded as 0.0 in {invalid_exit}/{len(cutout)} frozen v2 S02 episodes by the destination-lane event detector; do not use that timestamp. {ppo_lead} PPO S02 collisions were with the moving lead vehicle, so they do not establish a collision with the revealed static target. Collision labels and partner roles are retained as measured. The separate v4 audit corrects the detector without changing this bank."])
    if resource.get("blocked_builds"):
        lines.extend(["", "### Resource boundary", "",
                      f"The PPO checkpoint was not available at `{resource.get('checkpoint_path')}` during this run. The rules-only bank and legacy replay completed within the shared budget; the PPO regression and cross-agent session are marked unavailable.",
                      ""])
    lines.extend(["## Pattern memory and persistence", "",
                  f"The memory contains {len(cards)} pattern cards with execution references. Pattern centers use same-context normalized scenario inputs; system response labels remain separated. The compact cross-agent branch stores only queried valid outcomes. Snapshot metadata and hashes are in `history_snapshots/`.",
                  "",
                  "The selector logs contributing pattern IDs per query. New failure centers are saved only when a real queried collision is observed; unqueried candidates remain unlabeled.",
                  "",
                  "## Costs and artifacts", "",
                  f"- New physical calls: {ledger.get('new_physical_episodes', 0)} / {ledger.get('global_physical_cap', 400)}.",
                  f"- Physical call categories: `{json.dumps(ledger.get('physical_episode_categories', {}), sort_keys=True)}`.",
                  f"- Cached legacy replay cost: 0; accumulated logical query count: {ledger.get('logical_query_count', 0)}.",
                  f"- Wall time recorded by stages: {ledger.get('wall_clock_seconds', 0)} s.",
                  f"- The v3 and v4 validation audits used {ledger.get('physical_episode_categories', {}).get('validation_audit', 0)} separate physical episodes. Their GIFs and findings are in `validation_audit_v4/README.md`; they are excluded from the frozen v2 method rankings.",
                  "- Figures: cumulative failure discovery, measured labels with observed pattern support, and pattern lifecycle.",
                  "- Source provenance and limits: `scenario_sources.md`; full 14-card catalogue: `scenario_catalogue.yaml`.",
                  "- Per-session query logs, updates and snapshot hashes are under `sessions/` and `history_snapshots/`.",
                  "",
                  "## Acceptance", "",
                  f"Overall: **{acceptance.get('delivery_level')}**. Each E1–E10 item, evidence, physical cap, and resource status is recorded in `acceptance.json`.",
                  "",
                  "The empirical result is reported as observed. This is a development replay and a compact bank; it does not establish general superiority, statistical significance, complete policy validation, or standards certification.",
                  ""])
    report_path = root / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path.as_posix()


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-bank", action="store_true")
    args = parser.parse_args()
    del args
    root = Path("results/method_chains/failure_memory_regression/memory_v2")
    print(generate_report(root))


if __name__ == "__main__":
    main()
