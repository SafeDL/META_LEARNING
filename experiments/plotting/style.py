"""Shared typography and rendered quality checks for regression figures."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.storage import read_json, write_json

FIGURE_SCRIPTS = Path(os.environ.get("NATURE_FIGURE_SCRIPTS", Path.home() / ".codex/skills/nature-figure/scripts"))
STYLE = {
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "axes.linewidth": 0.7, "pdf.fonttype": 42,
    "svg.fonttype": "none", "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
}


def audit_figure(fig, axes, filename, *, report_root):
    spec = importlib.util.spec_from_file_location("regression_panel_alignment", FIGURE_SCRIPTS / "audit_panel_alignment.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory(prefix="regression-figure-qa-") as temporary:
        directory = Path(temporary)
        environment = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        alignment = directory / "alignment.json"
        module.require_matplotlib_panel_alignment(
            fig, axes=axes, json_out=alignment, tolerance_pt=1.5,
            gutter_tolerance_pt=1.5, require_panel_labels=len(axes) > 1, strict=True)
        pdf = directory / "figure.pdf"
        fig.savefig(pdf, dpi=600)
        fig.savefig(directory / "figure.svg")
        subprocess.run([sys.executable, "-B", str(FIGURE_SCRIPTS / "audit_pdf_text.py"), str(pdf),
                        "--min-pt", "5"], check=True, env=environment)
        collision = directory / "collision.json"
        subprocess.run([sys.executable, "-B", str(FIGURE_SCRIPTS / "audit_figure_collisions.py"), str(pdf),
                        "--json-out", str(collision), "--strict"], check=True, env=environment)
        write_json(report_root / f"{filename}.json", {
            "alignment": read_json(alignment), "collision_audit": read_json(collision),
            "export_contract": "PNG only; PDF and SVG are temporary QA files"})
