"""Check actual exported curves against independent lineage means and survival."""

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from benchmark.studies import REPRESENTATIVES
from verify_cost_inventory import read, require, sha

NS = {"s": "http://www.w3.org/2000/svg"}
NUMBER = r"[-+]?(?:\d*\.)?\d+(?:[eE][-+]?\d+)?"


def comments(node):
    return [x.text.strip() for x in node.iter() if x.tag is ET.Comment]


def calibration(axis, coordinate):
    points = []
    for tick in axis:
        use = tick.find(".//s:use", NS)
        labels = comments(tick)
        if use is not None and len(labels) == 1:
            try:
                value = float(labels[0].replace("\u2212", "-"))
            except ValueError:
                continue
            points.append((value, float(use.get(coordinate))))
    require(len(points) >= 2, "Numeric tick calibration missing")
    points = np.asarray(points)
    fit = np.polyfit(points[:, 0], points[:, 1], 1)
    require(np.max(np.abs(np.polyval(fit, points[:, 0]) - points[:, 1])) < 2e-6,
            "Nonlinear or inconsistent numeric tick positions")
    return fit


def vertices(path):
    data = path.get("d")
    require(not re.sub(r"[ML\s]", "", re.sub(NUMBER, "", data)), "Unexpected curve SVG commands")
    result = np.asarray([float(x) for x in re.findall(NUMBER, data)]).reshape(-1, 2)
    require(len(result) >= 2 and np.isfinite(result).all(), "Invalid SVG curve")
    return result


def distance(a, b):
    origin, vector = b[:-1], np.diff(b, axis=0)
    denominators = np.einsum("ij,ij->i", vector, vector)
    difference = a[:, None, :] - origin[None, :, :]
    projection = np.einsum("ijk,jk->ij", difference, vector) / np.where(denominators == 0, 1, denominators)
    closest = origin[None, :, :] + np.clip(projection, 0, 1)[:, :, None] * vector[None, :, :]
    return float(np.sqrt(np.square(a[:, None, :] - closest).sum(axis=2)).min(axis=1).max())


def pdf_curves(page, expected_colors):
    curves, points, color, rectangle = [], [], None, None
    for operands, operator in page.get_contents().operations:
        if operator == b"re" and float(operands[2]) > 200 and float(operands[3]) > 100:
            rectangle = np.asarray([float(v) for v in operands])
        elif operator == b"RG":
            color = np.asarray([float(v) for v in operands])
        elif operator == b"G":
            color = np.repeat(float(operands[0]), 3)
        elif operator == b"m":
            points = [[float(v) for v in operands]]
        elif operator == b"l":
            points.append([float(v) for v in operands])
        elif operator == b"S":
            path = np.asarray(points)
            if (color is not None and rectangle is not None and len(path) > 1 and np.ptp(path[:, 0]) > 200
                    and any(np.max(np.abs(color - expected)) < 1e-7 for expected in expected_colors)):
                curves.append((path, color.copy(), rectangle.copy()))
            points = []
    return curves


def expected(group, method, curve, labels):
    rows = [r for r in group if r["method"] == method]
    lineages = sorted({r["lineage"] for r in rows})
    if curve == "survival":
        times = sorted({0, 128, *(r["first_collision"]["time"] for r in rows)})
        values = [np.mean([np.mean([not r["first_collision"]["observed"] or r["first_collision"]["time"] > t
                                   for r in rows if r["lineage"] == name]) for name in lineages]) for t in times]
        return np.column_stack((np.repeat(times, 2)[1:], np.repeat(values, 2)[:-1]))
    def values(row):
        return row[curve] if curve in row else labels[str(Path(row["path"]).resolve())][curve]
    mean = np.mean([np.mean([values(r) for r in rows if r["lineage"] == name], axis=0)
                    for name in lineages], axis=0)
    require(len(mean) == 128, "Missing plotted discovery prefix")
    return np.column_stack((np.arange(1, 129), mean))


def main():
    output = Path(__file__).parent
    folder = ROOT / "results/gcrt/collision/report"
    index_path = ROOT / "results/gcrt/collision/run_index.json"
    receipt = read(output / "full_finite_completion_verification.json")
    require(sha(index_path) == receipt["run_index_sha256"], "Audited finite index changed")
    rows = read(index_path)["runs"]
    metrics_path = ROOT / "results/gcrt/collision/new_collision_metrics.json"
    metrics = read(metrics_path)
    require(metrics["status"] == "complete" and len(metrics["runs"]) == 10880, "Incomplete certificate curves")
    labels = {str(Path(r["path"]).resolve()): r for r in metrics["runs"]}
    plan = []
    for study in ("overall", "strict_family"):
        for condition in ("reuse", "new", "hard"):
            group = [r for r in rows if r["study"] == study and r["condition"] == condition]
            plan += [(f"{study}_{condition}", group, "discovery_curve"),
                     (f"first_discovery_{study}_{condition}", group, "survival")]
            if study == "overall":
                plan += [(f"version_new_{condition}", group, "version_new_curve"),
                         (f"history_specific_{condition}", group, "historical_set_specific_curve")]
    group = [r for r in rows if r["study"] == "mechanisms" or (
        r["study"] == "overall" and r["method"] == "history_response" and r["condition"] == "new"
        and r["target_id"] in REPRESENTATIVES)]
    plan.append(("mechanisms", group, "discovery_curve"))
    require(len(plan) == 19, "Wrong finite figure set")
    verified = []
    for stem, group, curve in plan:
        svg = folder / (stem + ".svg")
        root = ET.parse(svg, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))).getroot()
        axes = root.find(".//s:g[@id='axes_1']", NS)
        xfit = calibration(axes.find("s:g[@id='matplotlib.axis_1']", NS), "x")
        yfit = calibration(axes.find("s:g[@id='matplotlib.axis_2']", NS), "y")
        lines = [g.find("s:path", NS) for g in axes if g.get("id", "").startswith("line2d_")]
        legend = axes.find("s:g[@id='legend_1']", NS)
        methods = sorted({r["method"] for r in group})
        require(comments(legend) == methods and len(lines) == len(methods), "Curve/legend method mismatch")
        legend_lines = [g.find("s:path", NS) for g in legend if g.get("id", "").startswith("line2d_")]
        require(len(legend_lines) == len(lines), "Missing legend curve")
        deviations, endpoints, svg_paths = {}, {}, []
        for method, path, legend_path in zip(methods, lines, legend_lines):
            color = lambda p: re.search(r"stroke: (#[0-9a-f]+)", p.get("style")).group(1)
            require(color(path) == color(legend_path), "Legend color does not match plotted method")
            data = expected(group, method, curve, labels)
            projected = np.column_stack((np.polyval(xfit, data[:, 0]), np.polyval(yfit, data[:, 1])))
            actual = vertices(path)
            require(np.diff(actual[:, 0]).min() >= -1e-6, "Plotted query order changed")
            require(np.max(np.abs(actual[[0, -1]] - projected[[0, -1]])) < 1e-5, "Plotted curve endpoints changed")
            error = max(distance(actual, projected), distance(projected, actual))
            require(error <= .25, f"Rendered SVG curve deviates from independent data: {stem}/{method} ({error}pt)")
            deviations[method] = error
            endpoints[method] = float(data[-1, 1])
            clip_id = path.get("clip-path")[5:-1]
            clip = root.find(f".//s:clipPath[@id='{clip_id}']/s:rect", NS)
            rectangle = np.asarray([float(clip.get(k)) for k in ("x", "y", "width", "height")])
            svg_paths.append((actual, color(path), rectangle))
        png, pdf = folder / (stem + ".png"), folder / (stem + ".pdf")
        with Image.open(png) as image:
            require(image.size == (1152, 672), "Wrong PNG export size")
            image.verify()
        pages = PdfReader(pdf).pages
        require(len(pages) == 1 and abs(float(pages[0].mediabox.width) - 518.4) < .02
                and abs(float(pages[0].mediabox.height) - 302.4) < .02, "Wrong standalone PDF size/pages")
        palette = [np.asarray([int(color[i:i+2], 16) / 255 for i in (1, 3, 5)])
                   for _, color, _ in svg_paths]
        pdf_paths = pdf_curves(pages[0], palette)
        require(len(pdf_paths) == len(methods), "Missing actual PDF vector curve")
        require(all(method in pages[0].extract_text() for method in methods), "Missing PDF legend text")
        pdf_deviations = {}
        for method, (svg_points, hex_color, sr), (pdf_points, rgb, pr) in zip(methods, svg_paths, pdf_paths):
            expected_rgb = np.asarray([int(hex_color[i:i+2], 16) / 255 for i in (1, 3, 5)])
            require(np.max(np.abs(rgb - expected_rgb)) < 1e-7, "PDF method color changed")
            normalized_svg = (svg_points - sr[:2]) / sr[2:]
            normalized_pdf = (pdf_points - pr[:2]) / pr[2:]
            normalized_pdf[:, 1] = 1 - normalized_pdf[:, 1]
            a, b = normalized_svg * sr[2:], normalized_pdf * sr[2:]
            error = max(distance(a, b), distance(b, a))
            require(error <= .25, "PDF vector curve differs from validated SVG")
            pdf_deviations[method] = error
        verified.append({"stem": stem, "curve": curve, "methods": methods,
                         "max_svg_deviation_pt": deviations, "independent_endpoints": endpoints,
                         "max_pdf_svg_deviation_pt": pdf_deviations,
                         "artifacts": [{"path": str(p), "sha256": sha(p)} for p in (png, svg, pdf)]})
        print("verified actual figure exports", len(verified), "/19", stem, flush=True)
    sheet = Image.new("RGB", (1728, 7 * 356), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 13)
    for i, row in enumerate(verified):
        x, y = i % 3 * 576, i // 3 * 356
        draw.text((x + 6, y + 3), row["stem"], fill="black", font=font)
        with Image.open(folder / (row["stem"] + ".png")) as image:
            image.thumbnail((576, 336))
            sheet.paste(image.convert("RGB"), (x, y + 20))
    preview = output / "finite_figures_contact_sheet.png"
    sheet.save(preview)
    result = {"status": "verified_numerical_exports_pending_visual_review", "observed_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "All19 actual finite PNG/SVG/PDF exports; independent equal-lineage prefix means and censored survival from saved outcomes, tick calibration, method/color legend binding and vector curve geometry. Certification label correctness and statistical inference require their separate audits.",
              "validator_sha256": sha(__file__), "additional_physical_calls": 0,
              "index_sha256": sha(index_path), "new_collision_metrics_sha256": sha(metrics_path),
              "svg_geometry_tolerance_pt": .25, "figures": verified, "contact_sheet": str(preview),
              "visual_review_complete": False}
    (output / "finite_figure_verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
