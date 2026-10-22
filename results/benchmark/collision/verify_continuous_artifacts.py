"""Review actual final continuous cost and figure exports after the full raw audit."""

from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from benchmark.audit_collision_results import compare, require
from benchmark.continuous_selection import METHODS
from benchmark.evaluate_continuous import validate_frozen
from benchmark.studies import CONTINUOUS_TASKS
from verify_cost_inventory import read, sha
from verify_finite_figures import NS, calibration, comments, distance, expected, pdf_curves, vertices


def figure(folder, stem, rows, curve):
    svg, png, pdf = [folder / (stem + suffix) for suffix in (".svg", ".png", ".pdf")]
    root = ET.parse(svg, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))).getroot()
    axes = root.find(".//s:g[@id='axes_1']", NS)
    xfit = calibration(axes.find("s:g[@id='matplotlib.axis_1']", NS), "x")
    yfit = calibration(axes.find("s:g[@id='matplotlib.axis_2']", NS), "y")
    lines = [g.find("s:path", NS) for g in axes if g.get("id", "").startswith("line2d_")]
    legend = axes.find("s:g[@id='legend_1']", NS)
    methods = sorted({row["method"] for row in rows})
    require(set(methods) == set(METHODS) and len(rows) == 25, "Continuous representative plot omitted runs")
    require(comments(legend) == methods and len(lines) == len(methods), "Continuous SVG legend differs")
    legend_lines = [g.find("s:path", NS) for g in legend if g.get("id", "").startswith("line2d_")]
    require(len(legend_lines) == len(lines), "Missing continuous legend line")
    svg_data, svg_error, endpoints = [], {}, {}
    for method, path, legend_path in zip(methods, lines, legend_lines):
        color = lambda p: re.search(r"stroke: (#[0-9a-f]+)", p.get("style")).group(1)
        require(color(path) == color(legend_path), "Continuous SVG method color changed")
        data = expected(rows, method, curve, {})
        projected = np.column_stack((np.polyval(xfit, data[:, 0]), np.polyval(yfit, data[:, 1])))
        actual = vertices(path)
        require(np.diff(actual[:, 0]).min() >= -1e-6, "Continuous plotted query order changed")
        require(np.max(np.abs(actual[[0, -1]] - projected[[0, -1]])) < 1e-5, "Continuous curve endpoints changed")
        error = max(distance(actual, projected), distance(projected, actual))
        require(error <= .25, "Continuous SVG differs from independently rebuilt curve")
        clip_id = path.get("clip-path")[5:-1]
        clip = root.find(f".//s:clipPath[@id='{clip_id}']/s:rect", NS)
        rect = np.asarray([float(clip.get(k)) for k in ("x", "y", "width", "height")])
        svg_data.append((actual, color(path), rect))
        svg_error[method], endpoints[method] = error, float(data[-1, 1])
    with Image.open(png) as image:
        require(image.size == (1152, 672), "Wrong continuous PNG size")
        image.verify()
    pages = PdfReader(pdf).pages
    require(len(pages) == 1 and abs(float(pages[0].mediabox.width) - 518.4) < .02
            and abs(float(pages[0].mediabox.height) - 302.4) < .02, "Wrong continuous standalone PDF")
    palette = [np.asarray([int(color[i:i+2], 16) / 255 for i in (1, 3, 5)]) for _, color, _ in svg_data]
    pdf_data = pdf_curves(pages[0], palette)
    require(len(pdf_data) == len(methods), "Missing continuous PDF data curve")
    require(all(method in pages[0].extract_text() for method in methods), "Missing PDF method label")
    pdf_error = {}
    for method, (sv, color, sr), (pv, rgb, pr), expected_rgb in zip(methods, svg_data, pdf_data, palette):
        require(np.max(np.abs(rgb - expected_rgb)) < 1e-7, "Continuous PDF method color changed")
        a, b = (sv - sr[:2]) / sr[2:], (pv - pr[:2]) / pr[2:]
        b[:, 1] = 1 - b[:, 1]
        a, b = a * sr[2:], b * sr[2:]
        error = max(distance(a, b), distance(b, a))
        require(error <= .25, "Continuous PDF curve differs from validated SVG")
        pdf_error[method] = error
    return {"stem": stem, "curve": curve, "methods": methods, "independent_endpoints": endpoints,
            "max_svg_deviation_pt": svg_error, "max_pdf_svg_deviation_pt": pdf_error,
            "artifacts": [{"path": str(p), "sha256": sha(p)} for p in (png, svg, pdf)]}


def main():
    output = Path(__file__).parent
    base = ROOT / "results/gcrt/collision/continuous"
    proof_path = output / "continuous_result_audit.json"
    proof = read(proof_path)
    require(proof["status"] == "complete" and proof["verified_runs"] == proof["expected_runs"] == 100
            and not proof["errors"] and not proof["pending_runs"] and proof["index_checked"]
            and proof["frozen_protocol_checked"], "Full100 continuous raw audit is required")
    require(proof["validator_sha256"] == sha(ROOT / "benchmark/audit_continuous_results.py"), "Continuous auditor changed")
    validate_frozen()
    index_path, report_path = base / "run_index.json", base / "report/report.json"
    index, report = read(index_path), read(report_path)
    require(index["status"] == report["status"] == "complete" and len(index["runs"]) == report["runs"] == 100,
            "Continuous native report/index incomplete")
    paid = {str(Path(row["path"]).resolve()): row for row in report["costs"]}
    require(len(paid) == len(report["costs"]) == 100, "Continuous cost rows duplicated or omitted")
    groups = defaultdict(list)
    for row in proof["runs"]:
        path = Path(row["path"])
        require(sha(path) == row["sha256"], "Audited final result changed")
        saved = read(path)
        compare({k: v for k, v in paid[str(path.resolve())].items() if k != "path"}, saved["cost"], "Continuous cost report")
        require(saved["cost"]["actual_physical_calls"] == 128 and saved["cost"]["execution_mode"] == "actual_on_demand",
                "Continuous actual call budget changed")
        groups[row["method"]].append({**saved["cost"], "selection_s": row["selection_s"],
                                     "adaptation_s": row["adaptation_s"],
                                     "child_model_construction_s": saved["result"].get("model_construction_s", 0)})
    require(set(groups) == set(METHODS) and all(len(rows) == 20 for rows in groups.values()), "Incomplete method costs")
    dev_path = output / "continuous_development_verification.json"
    dev = read(dev_path)
    require(dev["status"] == "verified_execution" and dev["actual_calls"] == 1280, "Development actual costs missing")
    development = []
    for row in dev["runs"]:
        require(sha(row["path"]) == row["sha256"], "Audited development result changed")
        development.append(read(row["path"])["cost"])
    require(len(development) == 40 and sum(r["actual_physical_calls"] for r in development) == 1280,
            "Development actual calls changed")
    flat = [r for rows in groups.values() for r in rows]
    clocks = ("physical_execution_s", "sut_initialization_s", "model_construction_s", "selector_process_wall_s")
    cost_result = {"status": "verified_continuous_cost_report", "observed_at_utc": datetime.now(timezone.utc).isoformat(),
                   "scope": "All100 final actual cost rows matched to raw-audited saved runs, plus40 separately charged actual development runs. Training/acquisition costs remain in the finite construction inventory.",
                   "validator_sha256": sha(__file__), "raw_audit_sha256": sha(proof_path),
                   "report_sha256": sha(report_path), "development_audit_sha256": sha(dev_path),
                   "additional_physical_calls": 0, "final_actual_calls": 12800, "development_actual_calls": 1280,
                   "final_totals": {k: math.fsum(r[k] for r in flat) for k in clocks},
                   "development_totals": {k: math.fsum(r[k] for r in development) for k in clocks},
                   "final_method_means": {m: {"runs": len(rows), **{
                       "mean_" + k: math.fsum(r[k] for r in rows) / len(rows)
                       for k in (*clocks, "selection_s", "adaptation_s", "child_model_construction_s")}}
                       for m, rows in groups.items()},
                   "accounting": ["Every initialization/repeated execution is charged in each actual-call budget.",
                                  "Actual selector process wall time includes physical-query waiting and child computation; do not add those components again.",
                                  "Parent model setup precedes the isolated selector; child model construction is separately recorded inside selector wall time.",
                                  "Concurrent worker duration sums are not pipeline elapsed time. Earlier unknown qualification duration remains unknown."]}
    (output / "continuous_cost_report_verification.json").write_text(json.dumps(cost_result, indent=2) + "\n", encoding="utf-8")
    figures = []
    for sut, kind in CONTINUOUS_TASKS:
        rows = [r for r in index["runs"] if r["target_id"] == sut and r["scenario"] == kind]
        figures += [figure(base / "report", f"{sut}_{kind}", rows, "discovery_curve"),
                    figure(base / "report", f"first_{sut}_{kind}", rows, "survival")]
        print("verified continuous exports", len(figures), "/8", flush=True)
    sheet = Image.new("RGB", (1152, 4 * 356), "white")
    draw, font = ImageDraw.Draw(sheet), ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 13)
    for i, row in enumerate(figures):
        x, y = i % 2 * 576, i // 2 * 356
        draw.text((x + 6, y + 3), row["stem"], fill="black", font=font)
        with Image.open(base / "report" / (row["stem"] + ".png")) as image:
            image.thumbnail((576, 336))
            sheet.paste(image.convert("RGB"), (x, y + 20))
    preview = output / "continuous_figures_contact_sheet.png"
    sheet.save(preview)
    result = {"status": "verified_numerical_exports_pending_visual_review", "observed_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "All8 actual continuous PNG/SVG/PDF sets; independent representative-task mean discoveries/censored survival, numeric ticks, legends, PDF vectors and image/page dimensions. Statistical inference and visual review remain separate.",
              "validator_sha256": sha(__file__), "shared_figure_checker_sha256": sha(output / "verify_finite_figures.py"),
              "index_sha256": sha(index_path), "report_sha256": sha(report_path),
              "additional_physical_calls": 0, "svg_geometry_tolerance_pt": .25,
              "figures": figures, "contact_sheet": str(preview), "visual_review_complete": False}
    (output / "continuous_figure_verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("verified12800 final/1280 development actual cost calls and8 numerical figure sets;visual review pending", flush=True)


if __name__ == "__main__":
    main()
