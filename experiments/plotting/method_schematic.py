"""Render the four-stage historical response transfer method."""

from matplotlib.patches import FancyArrowPatch, Rectangle

from experiments.plotting.style import STYLE, plt
from experiments.protocol import ROOT

DESTINATION = ROOT / "figures/schematics/four_stage_method.png"


def stage(ax, xy, panel, name, lines, color, surface):
    x, y = xy
    ax.add_patch(Rectangle((x, y), 0.392, 0.36, facecolor=surface,
                           edgecolor="#CDD5DB", linewidth=0.65))
    ax.add_patch(Rectangle((x, y + 0.344), 0.392, 0.016, color=color, linewidth=0))
    ax.text(x + 0.024, y + 0.271, panel, color=color, fontweight="bold", va="center")
    ax.text(x + 0.065, y + 0.271, name, color="#23333D", fontweight="bold", va="center")
    ax.text(x + 0.025, y + 0.198, "\n".join(lines), color="#344954", va="top", linespacing=1.62)


def connect(ax, start, end, label, label_at, *, ha="center", color="#566974"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9,
                                linewidth=0.85, color=color, shrinkA=0, shrinkB=0))
    ax.text(*label_at, label, ha=ha, va="center", color=color, fontsize=8, linespacing=1.08)


def draw():
    fig = plt.figure(figsize=(7.16, 3.75))
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 1), ylim=(0, 1))
    ax.set_axis_off()
    cool, warm = "#246280", "#9B493C"
    stage(ax, (0.035, 0.585), "a", "Historical response priors", (
        "Joint collision and RSS response model",
        "Shared representation of the scenario",
        "Posterior for each historical ADS",
    ), cool, "#F2F7F9")
    stage(ax, (0.573, 0.585), "b", "Target risk adaptation", (
        "Adapt both target models to paid feedback",
        "Weight all branches by pre-query",
        "collision likelihood",
    ), warm, "#FBF5F2")
    stage(ax, (0.573, 0.055), "c", "Budgeted collision discovery", (
        "Start with common target queries",
        "Test the unqueried case with highest risk",
        "Record collision and RSS feedback",
    ), warm, "#FBF5F2")
    stage(ax, (0.035, 0.055), "d", "Cross-task knowledge", (
        "Archive paid responses by scenario",
        "Build a source for a later ADS",
        "Update shared model between tasks",
    ), cool, "#F2F7F9")
    connect(ax, (0.433, 0.765), (0.567, 0.765), "Source\nposteriors", (0.5, 0.825))
    connect(ax, (0.68, 0.577), (0.68, 0.424), "Predicted\nrisk", (0.667, 0.5), ha="right")
    connect(ax, (0.852, 0.424), (0.852, 0.577), "Paid\nfeedback", (0.868, 0.5),
            ha="left", color=warm)
    connect(ax, (0.567, 0.235), (0.433, 0.235), "Task\nrecords", (0.5, 0.294))
    connect(ax, (0.226, 0.424), (0.226, 0.577), "Next task", (0.24, 0.5), ha="left")
    return fig


def render():
    with plt.rc_context(STYLE):
        fig = draw()
        DESTINATION.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(DESTINATION, dpi=600)
        plt.close(fig)
    return DESTINATION


if __name__ == "__main__":
    print(render())
