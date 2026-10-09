"""Evaluate EP discovery support and evidence scoring with cached controls."""
import multiprocessing as mp
import time
from zipfile import ZipFile

import numpy as np
from threadpoolctl import threadpool_limits
import torch

from methods.history_guided_testing.experiment import RemoteOracle
from methods.history_guided_testing.history import historical_risk, load_history, split_indices, subset
from methods.history_guided_testing.io import read_json, write_json
from methods.history_guided_testing.search import TestingSession
from research.risk_conditioned_response_testing.confirmation import CONFIRMATION

from .adaptive_margin_prior import AdaptiveMarginPrior
from .baselines import verify_baseline_inputs
from .censored_margin_response import historical_margin_prior
from .config import BUDGET, INITIAL_QUERIES, OUTPUT, POOL_SEEDS, SUT_IDS
from .decision_support import SUPPORT_THRESHOLD
from .evaluation import disclose
from .discovery_supported_coupling import run_supported_policy
from .feedback_audit import audit_adaptive, audit_support
from .frozen_coupling import FrozenReplay
from .ep_margin_response import ExpectationPropagatedMargin
from .event_report_response import EventReportResponse


RESULTS = OUTPUT / "feedback_correction"
SEED = 11
METHOD = "report_binary_prior"


def selector_worker(connection):
    try:
        torch.set_num_threads(1)
        with threadpool_limits(limits=1):
            while True:
                message, task = connection.recv()
                if message == "stop":
                    return
                reference = FrozenReplay(TestingSession(task["x"], SEED, mode="global_feedback", prior=task["risk_prior"]))
                alternative = (EventReportResponse(reference, task["x"][:, 4], task["margin_prior"])
                               if task["event_report"] else None)
                candidate = AdaptiveMarginPrior(reference, task["x"][:, 4], task["margin_prior"],
                                                model_class=ExpectationPropagatedMargin,
                                                evidence_score=task["evidence_score"], alternative_model=alternative)
                tie = np.random.default_rng(SEED).random(len(task["x"]))
                run = run_supported_policy(reference, candidate, RemoteOracle(connection), tie, BUDGET, INITIAL_QUERIES)
                run.update(candidate.diagnostics())
                connection.send(("result", run))
    except Exception as error:
        connection.send(("error", repr(error)))
        raise
    finally:
        connection.close()


def evaluate_stage(results, candidate_method, evidence_score, control_stage, control_method, *, event_report=False):
    verify_baseline_inputs(read_json(CONFIRMATION / "protocol.json"))
    results.mkdir(parents=True, exist_ok=True)
    write_json(results / "protocol.json", {
        "stage": candidate_method,
        "sut_ids": SUT_IDS, "pools": [0, 1], "seed": SEED, "budget": BUDGET,
        "new_curves": 12, "reused_controls": 24,
        "evidence_score": evidence_score,
        "secondary_event_report": event_report,
        "event_score_scope": "weights use pre-query Bernoulli log score; R still updates component posteriors; not a full joint-data Bayesian posterior",
        "event_score_bound": "on the actual shared queried sequence, mixture cumulative event log loss <= best component loss + log(2); no discovery-curve or calibration guarantee",
        "model": ("historical mixed EP plus normalized Gaussian event report; second component uses only collision signs"
                  if event_report else "two history/neutral mixed-observation EP models"),
        "decision": "candidate mean collision probability must improve; current reference slot can be replaced only if joint candidate-miss/reference-hit probability <=1/sqrt(200)",
        "joint_probability": "Gaussian report covariance includes independent observation noise; deterministic bivariate normal quadrature, then prior-model averaging",
        "scope": "working immediate binary loss, not strict propensity ranking or a true-world confidence statement",
        "threshold": float(1 - SUPPORT_THRESHOLD), "threshold_design": "locked budget-only 1/sqrt(B), no target-dependent threshold search",
        "protection": "unchanged actual discovery credit and independent paid-reference replay",
        "coverage": "fixed 105-position prefix guarantee removed when supported reference slots are skipped; actual count bound retained",
        "controls": ["frozen_reference", control_method],
        "control_stage": control_stage,
        "ep_inference": "parallel damped site refitting; exact safe Gaussian sites, inequality moment matching; tolerance 1e-7, maximum 200 iterations",
        "new_physical_calls": 0})
    snapshot = results / "source_inputs.zip"
    if not snapshot.exists():
        with ZipFile(snapshot, "w") as archive:
            for path in sorted(OUTPUT.parent.glob("*.py")):
                archive.write(path, path.name)
    with ZipFile(snapshot) as archive:
        if any(archive.read(n) != (OUTPUT.parent / n).read_bytes() for n in archive.namelist()):
            raise ValueError("Discovery support code changed after freezing")
    history = load_history()
    train, _ = split_indices(next(iter(history.values()))["x"])
    allowed = subset(history, train)
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=selector_worker, args=(child,))
    process.start()
    child.close()
    rows = []
    try:
        for pool in range(len(POOL_SEEDS)):
            with np.load(OUTPUT / f"pool_{pool}" / SUT_IDS[0] / "responses.npz") as bank:
                x = bank["x"].copy()
            risk_prior, margin = historical_risk(allowed, x), historical_margin_prior(x)
            for sut in SUT_IDS:
                with np.load(OUTPUT / f"pool_{pool}" / sut / "responses.npz") as bank:
                    baseline = read_json(OUTPUT / f"pool_{pool}" / sut / "baselines" / f"frozen_original_{SEED}.json")
                    controls = OUTPUT / "neutral_margin_prior_coupling" / f"pool_{pool}" / sut
                    for method in ("frozen_reference", control_method, candidate_method):
                        if method == candidate_method:
                            path = results / f"pool_{pool}" / sut / f"{candidate_method}_{SEED}.json"
                            if path.exists():
                                run = read_json(path)
                            else:
                                started = time.perf_counter()
                                run = disclose(parent, {"x": x, "risk_prior": risk_prior, "margin_prior": margin,
                                                        "evidence_score": evidence_score, "event_report": event_report}, bank["risk"], bank["collision"])
                                run.update(method=candidate_method, seed=SEED, selector_elapsed_s=time.perf_counter() - started)
                                write_json(path, run)
                            audit_adaptive(run, candidate_method, bank, baseline, evidence_score=evidence_score, event_report=event_report)
                            audit_support(run)
                            if any(len(updates) != BUDGET or any(u["relative_site_change"] >= 1e-7 for u in updates) for updates in run["component_inference_updates"]):
                                raise ValueError("EP inference records are missing or unconverged")
                            print("DISCOVERY SUPPORT", pool, sut, run["F200"], run["reference_observations_skipped"], flush=True)
                            write_json(results / "progress.json", {"status": "running", "latest": str(path.relative_to(results)), "expected_new_curves": 12})
                        else:
                            source = controls if method == "frozen_reference" else OUTPUT / control_stage / f"pool_{pool}" / sut
                            run = read_json(source / f"{method}_{SEED}.json")
                        rows.append({"pool": pool, "sut_id": sut, "method": method,
                                     "area_pct": 100 * run["area_200"], "recall_pct": 100 * run["recall_200"], "F200": run["F200"]})
    finally:
        if process.is_alive():
            parent.send(("stop", None))
        parent.close()
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"Discovery support selector exited {process.exitcode}")
    aggregate = {m: {k: float(np.mean([r[k] for r in rows if r["method"] == m])) for k in ("area_pct", "recall_pct", "F200")}
                 for m in ("frozen_reference", control_method, candidate_method)}
    changes = {m: {"area_pp": aggregate[candidate_method]["area_pct"] - aggregate[m]["area_pct"],
                   "recall_pp": aggregate[candidate_method]["recall_pct"] - aggregate[m]["recall_pct"]} for m in ("frozen_reference", control_method)}
    summary = {"rows": rows, "aggregate": aggregate, "candidate_changes": changes, "new_curves": 12,
               "reused_controls": 24, "joint_binary_support_verified": True, "full_reference_bounds_verified": True,
               "significant_advantage_established": False}
    completed = read_json(OUTPUT / "current_method_comparison.json")["aggregate"]
    reference_names = (
        "Task-mean Gaussian, unprotected", "Student-t, unprotected",
        "Student-t, protected", "EP event score", "Event signs, protected",
    )
    stronger = {name: completed[name] for name in reference_names}
    stronger_changes = {
        name: {key: aggregate[candidate_method][key] - value[key]
               for key in ("area_pct", "recall_pct", "F200")}
        for name, value in stronger.items()
    }
    summary["development_gate_passed"] = (all(v["area_pp"] > 0 and v["recall_pp"] > 0 for v in changes.values())
                                          and all(v["area_pct"] > 0 and v["recall_pct"] > 0
                                                  for v in stronger_changes.values()))
    summary["event_score_prefix_identity_verified"] = evidence_score == "event"
    write_json(results / "stronger_method_comparison.json", {"references": stronger,
               "candidate_changes": stronger_changes, "scope": "same first-seed development banks and allowed feedback",
               "significant_advantage_established": False})
    write_json(results / "summary.json", summary)
    lines = ["# 反馈解释与发现损失支持的实际对照", "", "新执行 12 条策略曲线，复用 24 条既有控制；同 SUT、场景与 200 次预算。",
             f"候选为 {candidate_method}，证据评分为 {evidence_score}，直接机制对照为 {control_method}。",
             "两组件接收同次执行的真实反馈；权重按查询前评分更新，第二组件的观测解释由协议锁定。",
             "不要求严格辨明潜在倾向排序。固定前缀至少 105 项的保证已移除，实际全程计数界保持不变。", "",
             "| 方法 | Area % | Recall % | 平均 F200 |", "|---|---:|---:|---:|"]
    for method, value in aggregate.items():
        lines.append(f"| {method} | {value['area_pct']:.3f} | {value['recall_pct']:.3f} | {value['F200']:.2f} |")
    lines += ["", "| 池 | SUT | 方法 | Area % | Recall % | F200 |", "|---|---|---|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['pool']} | {row['sut_id']} | {row['method']} | {row['area_pct']:.3f} | {row['recall_pct']:.3f} | {row['F200']} |")
    lines += ["", "查询前证据、联合概率、条件参照观测、实际反馈及原参照全程界核对通过。",
              "支持量来自 EP 工作后验，不是未知目标的频率保证；本轮不自动补种子或宣称显著优势。"]
    (results / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(results / "progress.json", {"status": "complete", "new_curves": 12, "reused_controls": 24,
                                         "new_disclosures": 12 * BUDGET, "new_physical_calls": 0})
    print({"aggregate": aggregate, "candidate_changes": changes}, flush=True)


def main():
    evaluate_stage(RESULTS, METHOD, "event", "event_evidence_prior_coupling", "event_evidence_prior", event_report=True)


if __name__ == "__main__":
    main()
