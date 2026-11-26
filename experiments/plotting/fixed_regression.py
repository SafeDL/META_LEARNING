"""Plot collision discovery and paired version changes from audited records."""

import argparse
import csv
from collections import defaultdict
from math import ceil

from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from experiments.regression.fixed_inputs import AUC_WINDOW, H0_COUNTS, METHODS, NEW_WINDOW, SUITE_SIZE, VERSIONS, methods_for, output_root, query_directory, version_directory
from experiments.protocol import ROOT
from experiments.plotting.style import STYLE, audit_figure, plt
from experiments.regression.protocol import INITIAL_COUNT
from experiments.regression.preflight import DQN_CONTINUAL, POLICIES
from experiments.storage import read_json

SCENES = dict(zip(H0_COUNTS, ("Following brake", "Restricted cut-in",
                            "Cut-out reveal", "Side encroachment")))
LINES = {
    "full": ("Ours method", "#D04444", "-"),
    "detour": ("DETOUR (adapted)", "#7958A0", ":"),
    "birchler": ("Birchler greedy", "#41826C", "--"),
    "sdc_scissor": ("SDC-Scissor (adapted)", "#216998", "-."),
    "spectre": ("SPECTRE (adapted)", "#A46616", (0, (5, 2, 1, 2, 1, 2))),
}
Q90_MARKERS = {"full": "o", "detour": "^", "birchler": "s", "sdc_scissor": "D", "spectre": "v"}
TESTED_VERSIONS = VERSIONS[1:4]
DESTINATION = ROOT / "figures/policy_regression"


def save(fig, axes, filename, policy):
    fig.canvas.draw()
    audit_figure(fig, list(axes), filename, report_root=output_root(policy) / "reports/figure_quality")
    folder = {DQN_CONTINUAL: "dqn/continual_learning"}
    destination = DESTINATION / folder.get(policy, policy)
    destination.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination / f"{filename}.png", dpi=600)
    plt.close(fig)


def panel_style(ax, letter):
    ax.annotate(letter, (0, 1), xycoords="axes fraction", xytext=(0, 5),
                textcoords="offset points", fontweight="bold", va="bottom")
    ax.grid(axis="y", color="#E4E4E4", linewidth=0.45)
    ax.set_axisbelow(True)


def method_handles(methods=METHODS, *, mark_q90=False):
    return [Line2D([], [], label=LINES[method][0], color=LINES[method][1], ls=LINES[method][2],
                   marker=Q90_MARKERS[method] if mark_q90 else None,
                   markersize=3.4, markerfacecolor="none",
                   lw=1.65 if method == "full" else 1.3) for method in methods]


def recall(curves, totals, *, methods=None, policy="ppo"):
    methods = methods_for(policy) if methods is None else methods
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(3, 4, sharex=True, sharey=True, figsize=(7.16, 5.6))
        fig.subplots_adjust(left=.085, right=.974, top=.925, bottom=.205 if len(methods) == 5 else .17,
                            wspace=.29, hspace=.32)
        for row, version in enumerate(TESTED_VERSIONS):
            for col, (kind, name) in enumerate(SCENES.items()):
                ax = axes[row, col]
                panel_style(ax, chr(97 + row * len(SCENES) + col))
                if row == 0:
                    ax.annotate(name, (0, 1), xycoords="axes fraction", xytext=(12, 5),
                                textcoords="offset points", va="bottom")
                if col == 0:
                    ax.set_ylabel(f"{version} recall")
                count = totals.get((kind, version))
                ax.set(xlim=(0, SUITE_SIZE), ylim=(0, 1.03), xticks=(0, 500, 1000, 1500, 2000),
                       yticks=(0, .5, .9))
                group = curves.get(kind, {}).get(version, {})
                if count is None or not all(method in group for method in methods):
                    ax.text(.5, .5, "Unavailable", transform=ax.transAxes, ha="center")
                    continue
                if count == 0:
                    ax.text(.5, .5, "0 collisions", transform=ax.transAxes, ha="center")
                    continue
                ax.axhline(.9, color="#909090", linewidth=.6, linestyle="--")
                for method in reversed(methods):
                    cumulative = group[method]
                    _, color, line = LINES[method]
                    ax.step(range(len(cumulative) + 1), [0, *(n / count for n in cumulative)],
                            where="post", color=color, ls=line, lw=1.55 if method == "full" else 1.15)
                    q90 = next(q for q, hits in enumerate(cumulative, 1) if 10 * hits >= 9 * count)
                    ax.plot(q90, cumulative[q90 - 1] / count, marker=Q90_MARKERS[method],
                            markersize=2.8 if method == "full" else 4.2, markeredgewidth=.85,
                            color=color, markerfacecolor="none", zorder=4, linestyle="none")
        fig.supxlabel("Paid queries", y=.131 if len(methods) == 5 else .095, fontsize=9)
        handles = method_handles(methods, mark_q90=True)
        if len(handles) == 5:
            handles = [handles[i] for i in (0, 3, 1, 4, 2)]  # Matplotlib fills columns first.
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .015),
                   frameon=False, ncols=3 if len(methods) == 5 else len(methods), columnspacing=1.3)
        save(fig, axes.flat, "collision_recall", policy)


def dqn_query_savings(reports, policy=DQN_CONTINUAL, *, filename="method_contrasts.csv"):
    with (reports / filename).open(encoding="utf-8-sig", newline="") as stream:
        saved = {(row["kind"], row["version"], row["comparator"]): int(row["q90_queries_saved"])
                 for row in csv.DictReader(stream)}
    controls = [method for method in methods_for(policy) if method != "full"]
    # Zero is neutral. A negative difference means the comparator needed fewer queries.
    low = min(0, min(saved.values()))
    high = max(0, max(saved.values()))
    limit = max(1, ceil(max(-low, high) / 100)) * 100
    palette = ["#F4F8FA", "#20546F"] if low == 0 else ["#A34836", "#F4F8FA", "#20546F"]
    cmap = LinearSegmentedColormap.from_list("paired_queries", palette)
    scale = Normalize(0 if low == 0 else -limit, limit)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(controls), sharey=True, figsize=(7.16, 2.95))
        fig.subplots_adjust(left=.18, right=.985, top=.85, bottom=.31, wspace=.15)
        for i, (ax, method) in enumerate(zip(axes, controls, strict=True)):
            values = [[saved[kind, version, method] for version in TESTED_VERSIONS] for kind in SCENES]
            cells = ax.pcolormesh(values, cmap=cmap, norm=scale, edgecolors="white", linewidth=.8)
            ax.invert_yaxis()
            ax.set(xticks=[j + .5 for j in range(len(TESTED_VERSIONS))], xticklabels=TESTED_VERSIONS,
                   yticks=[j + .5 for j in range(len(SCENES))], yticklabels=list(SCENES.values()))
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            label_name = LINES[method][0].replace(" (adapted)", "")
            for label, offset, weight in ((chr(97 + i), 0, "bold"), (label_name, 12, "normal")):
                ax.annotate(label, (0, 1), xycoords="axes fraction", xytext=(offset, 5),
                            textcoords="offset points", weight=weight, va="bottom")
            for row, rounds in enumerate(values):
                for col, value in enumerate(rounds):
                    r, g, b, _ = cmap(scale(value))
                    ink = "white" if .2126 * r + .7152 * g + .0722 * b < .49 else "#152A35"
                    ax.text(col + .5, row + .5, f"{value:,}", ha="center", va="center", color=ink)
        bar = fig.colorbar(cells, cax=fig.add_axes([.30, .155, .54, .032]), orientation="horizontal")
        bar.outline.set_visible(False)
        bar.set_label("Queries saved by Ours method at 90% collision recall")
        save(fig, axes, "q90_queries_saved", policy)


def version_rates(versions, policy="ppo"):
    series = defaultdict(dict)
    for row in versions:
        series[row["kind"]][row["version"]] = float(row["collision_rate"]) * 100
    styles = (("#426C91", "o"), ("#B44341", "s"), ("#41826C", "^"), ("#845D93", "D"))
    if policy == DQN_CONTINUAL:
        dqn_version_rates(series, styles, policy)
        return
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.16, 3.0))
        fig.subplots_adjust(left=.11, right=.98, top=.94, bottom=.28)
        ax.grid(axis="y", color="#E4E4E4", linewidth=.45)
        ax.set_axisbelow(True)
        for (kind, name), (color, marker) in zip(SCENES.items(), styles):
            ax.plot(VERSIONS, [series[kind].get(v, float("nan")) for v in VERSIONS], label=name,
                    color=color, marker=marker, linewidth=1.35, markersize=4)
        family = "PPO" if policy == "ppo" else "DQN"
        ax.set(ylabel="Collision rate (%) ↓", xlabel=f"{family} version", ylim=(0, 100))
        fig.legend(loc="lower center", bbox_to_anchor=(.5, .01), ncols=4, frameon=False)
        save(fig, [ax], "version_collision_rates", policy)


def dqn_version_rates(series, styles, policy):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, sharex=True, figsize=(7.16, 3.15))
        fig.subplots_adjust(left=.105, right=.975, top=.9, bottom=.27, wspace=.42)
        for ax, letter in zip(axes, "ab", strict=True):
            panel_style(ax, letter)
        levels, changes = axes
        for (kind, label), (color, marker) in zip(SCENES.items(), styles, strict=True):
            values = [series[kind].get(v, float("nan")) for v in VERSIONS]
            levels.plot(VERSIONS, values, label=label, color=color, marker=marker,
                        linewidth=1.35, markersize=4)
            deltas = [value - values[0] for value in values]
            if any(abs(delta) > 1e-12 for delta in deltas):
                changes.plot(VERSIONS, deltas, color=color, marker=marker, linewidth=1.35, markersize=4)
        levels.set(ylabel="Collision rate (%) ↓", ylim=(0, 100))
        changes.axhline(0, color="#909090", ls="--", lw=.65, zorder=0)
        changes.set_ylabel("Collision rate change from h0 (pp)")
        changes.margins(y=.2)
        fig.supxlabel("DQN version", y=.111, fontsize=9)
        fig.legend(handles=levels.lines, loc="lower center", bbox_to_anchor=(.5, .01), ncols=4, frameon=False)
        save(fig, axes, "version_collision_rates", policy)


def count_matrix(ax, columns, keys, colors, values, *, highlight_max=False):
    ax.set(xlim=(-.5, len(columns) - .5), ylim=(len(SCENES) - .5, -.5),
           yticks=range(len(SCENES)), yticklabels=list(SCENES.values()),
           xticks=range(len(columns)), xticklabels=[label for _, label in columns])
    ax.xaxis.tick_top()
    ax.tick_params(axis="both", length=0, pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    width = .88 / len(keys)
    for row, kind in enumerate(SCENES):
        ax.axhline(row + .5, color="#DBE0E4", linewidth=.55)
        for col, (column, _) in enumerate(columns):
            counts = [values.get((kind, column, key)) for key in keys]
            emphasis = max((n for n in counts if n is not None), default=0) if highlight_max else 0
            for i, (color, count) in enumerate(zip(colors, counts, strict=True)):
                center = col - .44 + (i + .5) * width
                if count:
                    ax.add_patch(Rectangle((center - .45 * width, row - .34), .9 * width, .68,
                                           facecolor=color, alpha=.09, linewidth=0))
                weight = "bold" if count and count == emphasis else "normal"
                ax.text(center, row, "—" if count is None else f"{count:,}", va="center", ha="center",
                        color=color if count else "#636971", fontweight=weight)
    ax.axhline(-.5, color="#DBE0E4", linewidth=.55, clip_on=False)


def collision_transitions(versions, policy="ppo"):
    keys = ("removed_collisions", "new_collisions")
    colors = ("#30776D", "#A63A3A")
    columns = [(after, f"{before} → {after}") for before, after in zip(VERSIONS, VERSIONS[1:])]
    counts = {(row["kind"], row["version"], key): int(row[key])
              for row in versions if row["version"] != "h0" for key in keys}
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.16, 2.28))
        fig.subplots_adjust(left=.245, right=.985, top=.815, bottom=.265)
        count_matrix(ax, columns, keys, colors, counts)
        handles = [Line2D([], [], marker="s", ls="", ms=5, color=color, label=label)
                   for color, label in zip(colors, ("Resolved collisions", "New collisions"))]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .045),
                   frameon=False, ncols=2)
        save(fig, [ax], "collision_transitions", policy)


def query_outcomes(curves, metrics, policy="ppo"):
    """Accumulate newly introduced collisions from each paid query prefix."""
    outcomes = {}
    methods = methods_for(policy)
    for kind in SCENES:
        for previous, version in zip(VERSIONS, TESTED_VERSIONS):
            if not all((kind, version, method) in metrics for method in methods):
                continue
            window = AUC_WINDOW if policy == "ppo" else min(len(curves[kind][version][m]) for m in methods)
            predecessor = read_json(version_directory(kind, "truth", previous, policy) / "sealed.json")["feedback"]
            common = None
            for method in methods:
                if policy == DQN_CONTINUAL:
                    window = len(curves[kind][version][method])
                path = query_directory(kind, version, method, policy) / "result.json.gz"
                rows = read_json(path)["query_ledger"][:window]
                prefix = [(row["index"], row["feedback"]) for row in rows[:INITIAL_COUNT]]
                if common is None:
                    common = prefix
                if (prefix != common or len(rows) != window
                        or [row["call"] for row in rows] != list(range(1, window + 1))):
                    raise ValueError(f"Invalid paid query order for {kind}, {version}, {method}")
                new_collisions = [0]
                cumulative = 0
                for row, total in zip(rows, curves[kind][version][method][:window], strict=True):
                    prior = predecessor[row["index"]]
                    current = row["feedback"]
                    if prior["scenario_id"] != current["scenario_id"]:
                        raise ValueError(f"Mismatched coordinate across versions in {path}")
                    collision = current["collision"]
                    new_collisions.append(new_collisions[-1] + int(collision and not prior["collision"]))
                    cumulative += int(collision)
                    if cumulative != total:
                        raise ValueError(f"Paid collision sequence differs from audited curve in {path}")
                if new_collisions[NEW_WINDOW] != metrics[kind, version, method]:
                    raise ValueError(f"Paid query outcomes differ from audited counts in {path}")
                outcomes[kind, version, method] = new_collisions
    return outcomes


def failure_response_evolution(outcomes, policy="ppo", *, versions=None):
    if policy == DQN_CONTINUAL:
        return continual_collision_progress(outcomes, versions)
    if policy != "ppo":
        dqn_new_collision_progress(outcomes, policy)
        return
    counts = {key: values[NEW_WINDOW] for key, values in outcomes.items()}
    methods = methods_for(policy)
    examples = [("cut_in_onset", "R1"), ("cut_in_onset", "R2")]
    maximum = max((outcomes[(*kv, m)][-1] for kv in examples for m in methods), default=0)
    shading = "#E6EBEE"
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(7.16, 4.8))
        grid = fig.add_gridspec(2, max(1, len(examples)), left=.245, right=.978, top=.865, bottom=.245,
                               height_ratios=(1., 1.63), wspace=.24, hspace=.57)
        table = fig.add_subplot(grid[0, :])
        table.annotate("a", (0, 1), xycoords="axes fraction", xytext=(0, 20),
                       textcoords="offset points", va="bottom", fontweight="bold")
        table.annotate(f"New collisions by query {NEW_WINDOW}", (0, 1), xycoords="axes fraction",
                       xytext=(13, 20), textcoords="offset points", va="bottom")
        count_matrix(table, [(version, version) for version in TESTED_VERSIONS],
                     methods, [LINES[method][1] for method in methods], counts, highlight_max=True)
        table.tick_params(axis="x", pad=4)
        axes = [table]
        for i, (kind, version) in enumerate(examples):
            ax = fig.add_subplot(grid[1, i], sharey=axes[1] if i else None)
            axes.append(ax)
            panel_style(ax, "bc"[i])
            ax.annotate(f"{SCENES[kind]} · {version}", (0, 1), xycoords="axes fraction",
                        xytext=(13, 5), textcoords="offset points", va="bottom")
            ax.axvspan(0, INITIAL_COUNT, color=shading, linewidth=0)
            ax.axvline(NEW_WINDOW, color="#929AA1", linewidth=.7, ls="--", zorder=1)
            for method in reversed(methods):
                _, color, line = LINES[method]
                hits = outcomes[kind, version, method]
                ax.step(range(len(hits)), hits, where="post", color=color, linestyle=line,
                        linewidth=1.65 if method == "full" else 1.3)
            ax.set(xlim=(0, AUC_WINDOW), ylim=(-2, (maximum // 25 + 1) * 25),
                   xticks=range(0, AUC_WINDOW + 1, NEW_WINDOW), xlabel="Paid queries")
            ax.tick_params(axis="y", labelleft=i == 0)
            if i == 0:
                ax.set_ylabel("Cumulative\nnew collisions")
        shared = Rectangle((0, 0), 1, 1, facecolor=shading, linewidth=0, label="Shared initialization")
        handles = method_handles(methods)
        handles = [handles[0], handles[3], handles[1], handles[4], handles[2], shared]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .012),
                   frameon=False, ncols=3, columnspacing=1.3, handletextpad=.5)
        save(fig, axes, "failure_response_evolution", policy)


def dqn_new_collision_progress(outcomes, policy):
    tasks = [(kind, version) for kind in SCENES for version in TESTED_VERSIONS
             if any(outcomes.get((kind, version, method), [0])[-1] for method in METHODS)]
    if not tasks:
        return
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(len(tasks), 1, squeeze=False, figsize=(7.16, 3.25 + 1.9 * (len(tasks) - 1)))
        axes = axes[:, 0]
        fig.subplots_adjust(left=.123, right=.975, top=.875, bottom=.29, hspace=.42)
        shading = "#E6EBEE"
        for i, (ax, (kind, version)) in enumerate(zip(axes, tasks, strict=True)):
            if len(tasks) > 1:
                panel_style(ax, chr(97 + i))
                offset = 13
            else:
                ax.grid(axis="y", color="#E4E4E4", linewidth=.45)
                ax.set_axisbelow(True)
                offset = 0
            ax.annotate(f"{SCENES[kind]} · {version}", (0, 1), xycoords="axes fraction",
                        xytext=(offset, 5), textcoords="offset points", va="bottom")
            ax.axvspan(0, INITIAL_COUNT, color=shading, linewidth=0)
            for window in (NEW_WINDOW, AUC_WINDOW):
                ax.axvline(window, color="#929AA1", linewidth=.65, ls="--", zorder=1)
            maximum = max(outcomes[kind, version, method][-1] for method in METHODS)
            for method in reversed(METHODS):
                _, color, line = LINES[method]
                hits = outcomes[kind, version, method]
                ax.step(range(len(hits)), hits, where="post", color=color, linestyle=line,
                        linewidth=1.65 if method == "full" else 1.3)
            tick_step = max(1, (maximum + 6) // 7)
            ax.set(xlim=(0, len(hits) - 1), ylim=(-.15, maximum + .35),
                   yticks=range(0, maximum + 1, tick_step),
                   xlabel="Paid queries", ylabel="Cumulative new collisions")
        fig.legend(handles=method_handles(), loc="lower center", bbox_to_anchor=(.5, .064),
                   frameon=False, ncols=4, columnspacing=1.3, handletextpad=.5)
        shared = Rectangle((0, 0), 1, 1, facecolor=shading, linewidth=0, label="Shared initialization")
        fig.legend(handles=[shared], loc="lower center", bbox_to_anchor=(.5, .004), frameon=False)
        save(fig, axes, "failure_response_evolution", policy)


def continual_collision_progress(outcomes, versions):
    """Show early counts for all tasks and recall wherever new collisions exist."""
    methods = methods_for(DQN_CONTINUAL)
    introduced = {(r["kind"], r["version"]): int(r["new_collisions"])
                  for r in versions if r["version"] != "h0"}
    tasks = [(kind, version) for kind in SCENES for version in TESTED_VERSIONS if introduced[kind, version]]
    columns = min(2, len(tasks)) or 1
    curve_rows = ceil(len(tasks) / columns)
    counts = {key: hits[NEW_WINDOW] for key, hits in outcomes.items()}
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(7.16, 2.95 + 1.9 * curve_rows))
        grid = fig.add_gridspec(1 + curve_rows, columns, left=.245, right=.98,
                                top=1 - .68 / fig.get_figheight(), bottom=1.16 / fig.get_figheight(),
                                hspace=.65, wspace=.23)
        table = fig.add_subplot(grid[0, :])
        table.annotate("a", (0, 1), xycoords="axes fraction", xytext=(0, 20), textcoords="offset points",
                       va="bottom", fontweight="bold")
        table.annotate("New collisions by query 100", (0, 1), xycoords="axes fraction", xytext=(13, 20),
                       textcoords="offset points", va="bottom")
        count_matrix(table, [(v, v) for v in TESTED_VERSIONS], methods, [LINES[m][1] for m in methods],
                     counts, highlight_max=True)
        table.tick_params(axis="x", pad=4)
        axes = [table]
        for i, (kind, version) in enumerate(tasks):
            row, col = divmod(i, columns)
            ax = fig.add_subplot(grid[row + 1, col])
            axes.append(ax)
            panel_style(ax, chr(98 + i))
            total = introduced[kind, version]
            ax.annotate(f"{SCENES[kind]} · {version} · {total} new", (0, 1), xycoords="axes fraction",
                        xytext=(12, 5), textcoords="offset points", va="bottom")
            ax.axvspan(0, INITIAL_COUNT, color="#E6EBEE", linewidth=0)
            for method in reversed(methods):
                hits = outcomes[kind, version, method]
                ax.step(range(len(hits)), [n / total for n in hits], where="post", color=LINES[method][1],
                        ls=LINES[method][2], linewidth=1.6 if method == "full" else 1.1)
            ax.set(xlim=(0, SUITE_SIZE), xticks=(0, 1000, 2000),
                   ylim=(-.035, 1.04), yticks=(0, .5, 1))
            if row == curve_rows - 1:
                ax.set_xlabel("Paid queries")
            if col == 0:
                ax.set_ylabel("New collision recall")
        fig.legend(handles=method_handles(methods), loc="lower center", bbox_to_anchor=(.5, .014),
                   frameon=False, ncols=2, columnspacing=1.6)
        save(fig, axes, "failure_response_evolution", DQN_CONTINUAL)


def capability_gate(policy, checks, initial):
    from experiments.qualification import COURSES

    labels = ("Normal\nfollowing", "Avoidable\nbraking", "Full\nstop",
              "Legal lane\nchange", "Rear\nconstraint")
    styles = {
        "Pass": ("o", "#30776D"), "Collision": ("X", "#A63A3A"),
        "Other failure": ("D", "#7958A0"),
    }
    if not checks:
        return
    rows = [("Original policy · h0", initial["courses"])]
    rows.extend((f"{SCENES[row['kind']]} · {row['version']}", row["courses"]) for row in checks)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.16, 1.38 + .25 * len(rows)))
        bottom = .58 / fig.get_figheight()
        top = 1 - .48 / fig.get_figheight()
        fig.subplots_adjust(left=.30, right=.98, top=top, bottom=bottom)
        used = set()
        for i, (_, courses) in enumerate(rows):
            for j, (course, observed) in enumerate(zip(COURSES, courses, strict=True)):
                if observed["course"] != course.name:
                    raise ValueError("Capability figure course order differs from the audit")
                status = "Pass" if observed["passed"] else ("Collision" if observed["collision"] else "Other failure")
                used.add(status)
                marker, color = styles[status]
                ax.plot(j, i, marker=marker, color=color, linestyle="none", markersize=6)
        ax.set(xlim=(-.5, len(labels) - .5), ylim=(len(rows) - .5, -.5),
               xticks=range(len(labels)), xticklabels=labels,
               yticks=range(len(rows)), yticklabels=[label for label, _ in rows])
        ax.xaxis.tick_top()
        ax.tick_params(axis="both", length=0, pad=6)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for i in range(len(rows) + 1):
            ax.axhline(i - .5, color="#DBE0E4", linewidth=.55, clip_on=False)
        handles = [Line2D([], [], marker=marker, color=color, linestyle="none", label=label)
                   for label, (marker, color) in styles.items() if label in used]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .025),
                   frameon=False, ncols=len(handles))
        save(fig, [ax], "repair_capability", policy)


def render(policy="ppo"):
    reports = output_root(policy) / "reports"
    audit = read_json(reports / "audit.json")
    expected = len(SCENES) * len(TESTED_VERSIONS) * len(methods_for(policy))
    if policy in ("ppo", DQN_CONTINUAL) and (audit["status"] != "passed" or audit["sequences"] != expected):
        raise ValueError("The complete fixed-suite audit must pass before plotting")
    with (reports / "trajectory_versions.csv").open(encoding="utf-8-sig", newline="") as stream:
        versions = list(csv.DictReader(stream))
    metrics = {}
    if (reports / "query_metrics.csv").exists():
        with (reports / "query_metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
            metrics = {(row["kind"], row["version"], row["method"]): int(row["new_in_first_100"])
                       for row in csv.DictReader(stream)}
    totals = {(row["kind"], row["version"]): int(row["total_collisions"]) for row in versions}
    curves = read_json(reports / "recall_curves.json.gz")
    outcomes = query_outcomes(curves, metrics, policy)
    if any(row["version"] != "h0" for row in versions):
        version_rates(versions, policy)
        collision_transitions(versions, policy)
    if policy != "ppo" and any(not row["qualified"] for row in audit["capability_checks"]):
        capability_gate(policy, audit["capability_checks"], audit["initial_capability"])
    if outcomes:
        recall(curves, totals, policy=policy)
        failure_response_evolution(outcomes, policy, versions=versions)
    if policy in ("ppo", DQN_CONTINUAL) and audit["status"] == "passed":
        dqn_query_savings(reports, policy)


if __name__ == "__main__":
    from experiments.paper_results import regression_policy

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=tuple(POLICIES), default="ppo")
    parser.add_argument("--versions-only", action="store_true",
                        help="Audit and plot complete native driving suites independently of test prioritization")
    args = parser.parse_args()
    args.policy = regression_policy(args.policy)
    if args.versions_only:
        from experiments.regression.fixed_report import complete_version_table
        versions = complete_version_table(args.policy)
        version_rates(versions, args.policy)
        collision_transitions(versions, args.policy)
    else:
        render(args.policy)
