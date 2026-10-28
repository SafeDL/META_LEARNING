"""Check native driving-quality fuzzy rules and every allowed paid safe trajectory."""

import ast
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import skfuzzy

from testing_methods.baselines.scenario_fuzz.driving_quality import (
    oversteer_level,
    understeer_level,
)

from .construction import load_archives
from .discovery_config import write_json
from .domain_risk import disclose_driving_quality
from .executions import ExecutionBank
from .protocol import ROOT, TEST_RECORDS_ROOT

SOURCE = (
    "https://raw.githubusercontent.com/AtongWang/ScenarioFuzz/"
    "49360794846786b133799a62827528bc4644bc62/src/driving_quality.py"
)


def fuzzy_rule_check():
    source = urllib.request.urlopen(SOURCE, timeout=30).read().decode("utf-8")
    tree = ast.parse(source)
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("get_oversteer_level", "get_understeer_level")
    ]
    namespace = {"np": np, "skfuzzy": skfuzzy}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])), SOURCE, "exec"
        ),
        namespace,
    )
    errors = [
        abs(oversteer_level(a, b, c) - namespace["get_oversteer_level"](a, b, c))
        for a in (5, 25, 45)
        for b in (0.05, 0.25, 0.45)
        for c in (5, 20, 40)
    ]
    errors += [
        abs(understeer_level(a) - namespace["get_understeer_level"](a)) for a in (0.1, 0.5, 0.9)
    ]
    if max(errors) > 1e-12:
        raise ValueError("Driving quality fuzzy rules differ from the author's implementation")
    return {
        "author_source": SOURCE,
        "same_input_cases": len(errors),
        "maximum_error": float(max(errors)),
        "documented_correction": "steering-difference filter receives steering angle instead of lateral speed",
    }


def audit_bank(task):
    kind, source, role = task
    bank = ExecutionBank(TEST_RECORDS_ROOT / "physical" / role / kind / source)
    scored, collision, short, minimum = 0, 0, 0, 0.0
    for scene in bank.candidates:
        execution = bank(scene)
        disclosed = disclose_driving_quality(execution).feedback
        if disclosed.collision:
            collision += 1
        elif disclosed.safety.driving_quality_score is None:
            short += 1
        else:
            scored += 1
            minimum = min(minimum, disclosed.safety.driving_quality_score)
    print("quality_audit", kind, source, "scored", scored, "short", short, flush=True)
    return {
        "scenario": kind,
        "source_id": source,
        "role": role,
        "scored": scored,
        "collisions_skipping_successor_score": collision,
        "short_safe_unscored": short,
        "minimum_score": minimum,
        "checked_executions": len(bank.candidates),
    }


def main():
    started = time.perf_counter()
    rules = fuzzy_rule_check()
    tasks = []
    for kind in ("cut_in", "cut_out"):
        history, development = load_archives(kind)
        tasks.extend(
            (kind, archive.source_id, archive.role) for archive in (*history, *development)
        )
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(audit_bank, tasks))
    report = {
        "status": "passed",
        "fuzzy_rules": rules,
        "banks": rows,
        "checked_executions": sum(r["checked_executions"] for r in rows),
        "short_safe_unscored": sum(r["short_safe_unscored"] for r in rows),
        "new_physical_calls": 0,
        "wall_s": time.perf_counter() - started,
    }
    write_json(
        ROOT
        / "experiment_outputs/benchmark/collision/discovery/domain_comparison/driving_quality_audit.json",
        report,
    )
    print({k: v for k, v in report.items() if k not in ("banks", "fuzzy_rules")}, flush=True)
    return report


if __name__ == "__main__":
    main()
