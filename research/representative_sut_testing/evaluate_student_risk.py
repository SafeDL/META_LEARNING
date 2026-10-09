"""Screen exact frozen-reference coupling, then expand only after joint gains."""
import multiprocessing as mp
import time
from zipfile import ZipFile

import numpy as np

from methods.history_guided_testing.history import historical_risk, load_history, split_indices, subset
from methods.history_guided_testing.io import read_json, write_json
from research.risk_conditioned_response_testing.confirmation import CONFIRMATION

from .baselines import verify_baseline_inputs
from .config import OUTPUT, POOL_SEEDS, SEEDS, SUT_IDS
from .evaluation import disclose, metrics
from .frozen_coupling import METHODS, selector_worker
from .risk_event_response import fit_event_readout
from .risk_task_prior import risk_templates
from .student_risk_prior import PRIOR_DEGREES_OF_FREEDOM


RESULTS = OUTPUT / "student_risk"


def summarize(seeds):
    rows = []
    for pool in range(len(POOL_SEEDS)):
        for sut in SUT_IDS:
            bank_folder = OUTPUT / f"pool_{pool}" / sut
            with np.load(bank_folder / "responses.npz") as bank:
                initial = {}
                values = {method: [] for method in METHODS}
                for seed in seeds:
                    baseline = read_json(bank_folder / "baselines" / f"frozen_original_{seed}.json")
                    for method in METHODS:
                        run = read_json(RESULTS / f"pool_{pool}" / sut / f"{method}_{seed}.json")
                        order = run["selected_indices"]
                        if len(order) != 200 or len(set(order)) != 200:
                            raise ValueError("Invalid frozen-coupling paid budget")
                        actual = metrics(bank["collision"], order)
                        if any(run[key] != actual[key] for key in ("area_200", "recall_200", "curve", "F200")):
                            raise ValueError("Frozen-coupling metric differs from paid events")
                        expected = [{"index": i, "risk": float(bank["risk"][i]), "collision": bool(bank["collision"][i])}
                                    for i in order]
                        if run["disclosures"] != expected:
                            raise ValueError("Frozen-coupling feedback differs from measured responses")
                        if run["event_residual_observations"] != 200 or len(run["event_residual_records"]) != 200:
                            raise ValueError("Missing paid event-residual feedback")
                        for observation, record in zip(expected, run["event_residual_records"]):
                            if any(record[key] != observation[key] for key in observation):
                                raise ValueError("Event-residual feedback differs from paid observations")
                        if len(run["risk_scale_records"]) != 200:
                            raise ValueError("Missing paid Student risk scale records")
                        for t, record in enumerate(run["risk_scale_records"], 1):
                            parameters = len(np.unique(bank["x"][order[:t], 4]))
                            df = PRIOR_DEGREES_OF_FREEDOM + t - parameters
                            multiplier = (PRIOR_DEGREES_OF_FREEDOM - 2 + record["residual_quadratic"]) / (df - 2)
                            if (record["index"] != order[t - 1] or record["observations"] != t
                                    or record["mean_parameters"] != parameters or record["effective_count"] != t - parameters
                                    or record["predictive_df"] != df or record["residual_quadratic"] < -1e-10
                                    or not np.isclose(record["covariance_multiplier"], multiplier, rtol=0, atol=1e-12)):
                                raise ValueError("Student risk scale does not match paid response evidence")
                        for check in run["regime_selection_checks"]:
                            if order[check["query"] - 1] != check["chosen_index"]:
                                raise ValueError("Regime-anchor decision differs from the actual query")
                            if (bank["x"][check["chosen_index"], 4] != check["family"]
                                    or bank["x"][check["reference_index"], 4] != check["family"]):
                                raise ValueError("Candidate changed the frozen reference functional regime")
                            if check["chosen_index"] != check["reference_index"]:
                                if check["chosen_probability"] <= check["reference_probability"] + 1e-12:
                                    raise ValueError("Regime correction has no declared strict probability advantage")
                        if seed in initial and order[:10] != initial[seed]:
                            raise ValueError("Frozen-coupling controls have different initialization")
                        initial[seed] = order[:10]
                        if method == "frozen_reference" and order != baseline["selected_indices"]:
                            raise ValueError("Native frozen reference differs from original saved trajectory")
                        prefix = run["reference_prefix_indices"]
                        if prefix != baseline["selected_indices"][:len(prefix)]:
                            raise ValueError("Coupled replay changes the original independent reference")
                        gain = np.asarray(run["curve"]) - baseline["curve"]
                        bounds = []
                        if len(run["coupling_checks"]) != 200:
                            raise ValueError("Missing reference coupling checks")
                        for t, check in enumerate(run["coupling_checks"], 1):
                            k = check["reference_prefix_length"]
                            if not 0 <= k <= t:
                                raise ValueError("Reference replay advanced past paid feedback")
                            extra = [i for i in order[:t] if i not in prefix[:k]]
                            negative = [i for i in extra if not bank["collision"][i]]
                            if (extra != check["extra_indices"] or negative != check["negative_extra_indices"]
                                    or len(negative) != check["negative_extra_count"]
                                    or check["discovery_gain_lower"] != -len(negative)):
                                raise ValueError("Reference accounting differs from actual paid observations")
                            lower = check["discovery_gain_lower"]
                            if gain[t - 1] < lower:
                                raise ValueError("Real discovery curve violates the full-reference bound")
                            found = run["curve"][t - 1]
                            found_before = run["curve"][t - 2] if t > 1 else 0
                            negative_before = run["coupling_checks"][t - 2]["negative_extra_count"] if t > 1 else 0
                            credit_rate = 1 / np.sqrt(200)
                            credit_before = 1 + credit_rate * found_before
                            allowed = negative_before + 1 <= credit_before + 1e-12
                            if (check["found"] != found or check["negative_extra_before"] != negative_before
                                    or not np.isclose(check["credit_rate"], credit_rate, rtol=0, atol=1e-12)
                                    or not np.isclose(check["loss_credit"], 1 + credit_rate * found, rtol=0, atol=1e-12)
                                    or not np.isclose(check["credit_before"], credit_before, rtol=0, atol=1e-12)
                                    or check["probe_allowed"] != allowed):
                                raise ValueError("Discovery credit differs from the paid prefix")
                            if method == "coupled_response":
                                if len(negative) > check["loss_credit"] + 1e-12:
                                    raise ValueError("Protected proposal exceeds earned discovery credit")
                                if check["chosen_index"] != check["reference_index"] and not allowed:
                                    raise ValueError("Protected alternative was queried without worst-case loss credit")
                                relative_lower = (baseline["curve"][t - 1] - 1) / (1 + credit_rate)
                                if found < relative_lower - 1e-12:
                                    raise ValueError("Discovery curve violates the earned-credit relative bound")
                            bounds.append(lower)
                        if (sum(bounds) != run["area_gain_lower"] or bounds[-1] != run["terminal_gain_lower"]
                                or gain.sum() < sum(bounds)):
                            raise ValueError("Area or terminal certificate is inconsistent")
                        values[method].append(run)
                for method, runs in values.items():
                    rows.append({"pool": pool, "sut_id": sut, "method": method, "runs": len(runs),
                                 "area_pct": 100 * np.mean([r["area_200"] for r in runs]),
                                 "recall_pct": 100 * np.mean([r["recall_200"] for r in runs]),
                                 "F200": np.mean([r["F200"] for r in runs]),
                                 "alternative_queries": np.mean([r["alternative_queries"] for r in runs]),
                                 "max_negative_extra_count": max(r["max_negative_extra_count"] for r in runs),
                                 "distinct_sequences": len({tuple(r["selected_indices"]) for r in runs})})
    aggregate = {method: {key: float(np.mean([r[key] for r in rows if r["method"] == method]))
                          for key in ("area_pct", "recall_pct", "F200", "alternative_queries")} for method in METHODS}
    full = aggregate["coupled_response"]
    changes = {method: {"area_pp": full["area_pct"] - aggregate[method]["area_pct"],
                        "recall_pp": full["recall_pct"] - aggregate[method]["recall_pct"]}
               for method in METHODS[:-1]}
    passed = all(c["area_pp"] > 0 and c["recall_pp"] > 0 for c in changes.values())
    return {"rows": rows, "aggregate": aggregate, "candidate_changes": changes, "gate_passed": passed,
            "seeds": seeds, "audited_runs": len(POOL_SEEDS) * len(SUT_IDS) * len(seeds) * len(METHODS),
            "reference_trajectory_matches_frozen_baseline": True, "full_curve_bounds_verified": True,
            "earned_credit_and_relative_bound_verified": True,
            "student_risk_scale_feedback_verified": True,
            "event_residual_feedback_verified": True,
            "significant_advantage_established": False}


def evaluate(connection, seeds, allowed, event_readout):
    for pool in range(len(POOL_SEEDS)):
        with np.load(OUTPUT / f"pool_{pool}" / SUT_IDS[0] / "responses.npz") as coordinates:
            x = coordinates["x"].copy()
        templates = risk_templates(allowed, x)
        for sut in SUT_IDS:
            with np.load(OUTPUT / f"pool_{pool}" / sut / "responses.npz") as bank:
                if not np.array_equal(x, bank["x"]):
                    raise ValueError("Cross-SUT pool coordinates differ")
                prior = historical_risk(allowed, bank["x"])
                for seed in seeds:
                    for method in METHODS:
                        path = RESULTS / f"pool_{pool}" / sut / f"{method}_{seed}.json"
                        if path.exists():
                            continue
                        started = time.perf_counter()
                        task = {"method": method, "seed": seed, "x": bank["x"], "prior": prior,
                                "event_readout": event_readout, "risk_templates": templates}
                        run = disclose(connection, task, bank["risk"], bank["collision"])
                        run.update(method=method, seed=seed, selector_elapsed_s=time.perf_counter() - started)
                        write_json(path, run)
                        write_json(RESULTS / "progress.json", {"status": "running", "latest": str(path.relative_to(RESULTS)),
                                                              "screening_runs": 36, "conditional_full_runs": 180})
                        print("FROZEN COUPLING", pool, sut, seed, method, run["F200"], flush=True)


def main():
    verify_baseline_inputs(read_json(CONFIRMATION / "protocol.json"))
    write_json(RESULTS / "protocol.json", {
        "stage": "existing six-SUT development; not blind confirmation",
        "sut_ids": SUT_IDS, "pools": list(range(len(POOL_SEEDS))), "methods": METHODS, "budget": 200,
        "initialization": "first ten actual queries of frozen_original; common across controls",
        "reference": "unchanged TestingSession(mode=global_feedback), same trained kernel and history split",
        "reference_feedback": "risk only; collision not passed to its state update",
        "candidate": "frozen risk GP plus historical risk-to-event mean and contextual probit residual GP",
        "risk_task_covariance": "original learned risk kernel plus empirical centered six-source response Gram matrix; mean/noise and source split unchanged",
        "task_mean": "generalized least-squares family intercept fitted only from paid target risk; trend uncertainty included; unobserved families retain original mean",
        "risk_scale_prior": "a~InverseGamma(nu/2,(nu-2)/2), E[a]=1; nu=5 fixed before screening, not tuned",
        "risk_scale_posterior": "df=nu+n-p, beta=(y-m-H*trend)^T*(K+noise*I)^(-1)*(y-m-H*trend); covariance multiplier=(nu-2+beta)/(df-2)",
        "risk_prediction": "Student-t marginal with exact scale marginalization and flat observed-family mean prior; same Gaussian GLS mean; signal and nugget share a scale",
        "risk_prior_degrees_of_freedom": PRIOR_DEGREES_OF_FREEDOM,
        "risk_scale_source": "https://proceedings.mlr.press/v33/shah14.html; observed-family trend integration is an explicit project adaptation",
        "selection": "same functional regime as current frozen-reference head; strict predicted event-probability improvement required",
        "response_functions": "event feedback changes only event residual; risk regression is protected from classification feedback",
        "candidate_history": "same original six sources and frozen training indices; no extra configurations",
        "event_readout": "per-functional-family isotonic mean fitted from actual source risk/event pairs",
        "event_inference": "probit Laplace approximation; exact constant risk-calibration segments and sixteen-node probability-space integration of linear segments",
        "event_residual_covariance": "equal normalized within-family intercept and learned local kernel; same unit marginal variance as local-only residual",
        "selector_payload": "x, seed, frozen reference prior, six historical risk templates and history-only event readout; no target identity/parameters/full response arrays",
        "loss_credit": "L_t <= 1 + F_actual(t)/sqrt(B); before an alternative reserve one worst-case noncollision",
        "credit_rate": float(1 / np.sqrt(200)),
        "relative_discovery_bound": "F_actual(t) >= (F_reference(t)-1)/(1+1/sqrt(B)) at every paid prefix",
        "area_bound": "raw cumulative gain >= -sum_t L_t; original at-most-one terminal-loss guarantee is replaced",
        "credit_design": "single rate determined only by locked B=200; no search over target-dependent loss thresholds",
        "screening_seeds": SEEDS[:1], "conditional_full_seeds": SEEDS,
        "promotion": "both Area and Recall improve against frozen reference and uncoupled proposal",
        "new_physical_calls": 0,
    })
    snapshot = RESULTS / "source_inputs.zip"
    if not snapshot.exists():
        with ZipFile(snapshot, "w") as archive:
            for path in sorted(OUTPUT.parent.glob("*.py")):
                archive.write(path, path.name)
    with ZipFile(snapshot) as archive:
        for name in archive.namelist():
            if archive.read(name) != (OUTPUT.parent / name).read_bytes():
                raise ValueError("Frozen-coupling source changed after freezing")
    original = load_history()
    training, _ = split_indices(next(iter(original.values()))["x"])
    allowed = subset(original, training)
    event_readout = fit_event_readout(RESULTS)
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=selector_worker, args=(child,))
    process.start()
    child.close()
    try:
        evaluate(parent, SEEDS[:1], allowed, event_readout)
        screening = summarize(SEEDS[:1])
        write_json(RESULTS / "screening.json", screening)
        if screening["gate_passed"]:
            evaluate(parent, SEEDS[1:], allowed, event_readout)
            final = summarize(SEEDS)
        else:
            final = screening
    finally:
        if process.is_alive():
            parent.send(("stop", None))
        parent.close()
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"Frozen-coupling selector exited {process.exitcode}")
    write_json(RESULTS / "summary.json", final)
    lines = ["# 冻结原方法参照上的受限纠正", "",
             f"六固定 SUT、两共享池、选择种子 {list(final['seeds'])}、200 次唯一查询和共同原参照前十次初始化。",
             "候选及去耦合对照使用同六来源、同历史训练划分、同核和实际风险／事件反馈；冻结参照保持原风险权限。", "",
             "风险 GP 只由实际风险更新；历史风险到事件的平均映射经目标场景事件残差修正，允许同风险有不同碰撞响应。",
             "事件残差采用 probit 拉普拉斯近似；风险映射的常量段精确积分，线性段在概率域作十六节点积分，不读取未查询标签。", "",
             "事件残差同时包含场景类内共享校准偏差和局部变化，两项归一化后保持与上一轮相同的初始边缘方差。", "",
             "候选风险 GP 加入同一六来源的经验响应差异协方差；初始风险均值、噪声、学习核与事件推断保持不变。", "",
             "每类已观测场景的整体风险偏移由目标风险反馈作广义最小二乘估计，预测方差包含均值估计不确定性。", "",
             "风险尺度不再固定：逆伽马尺度先验的自由度预先固定为 5，已查询风险的残差二次型更新尺度后验。",
             "积分使用 Student-t 预测分布；扣除已观测类型均值参数的自由度，均值保持原广义最小二乘形式。",
             "信号和数值噪声共享尺度；这是工作响应模型，不是未知目标上的频率校准保证。", "",
             "候选保留参照下一项的功能场景类型，只在同类中预测事件概率严格更高时改序；模型无区别时执行原参照。", "",
             "本轮只修改探测额度：L_t≤1+F_actual(t)/sqrt(200)，执行候选前预留一个最坏情况非碰撞。额度仅由固定预算和实际已发现数决定。",
             "全程界为 F_actual(t)≥(F_reference(t)−1)/(1+1/sqrt(200))；这是比原最多损失一个发现更弱的保证。", "",
             "| 方法 | Area % | Recall % | F200 | 平均偏离参照查询数 |", "|---|---:|---:|---:|---:|"]
    for method, value in final["aggregate"].items():
        lines.append(f"| {method} | {value['area_pct']:.3f} | {value['recall_pct']:.3f} | "
                     f"{value['F200']:.2f} | {value['alternative_queries']:.2f} |")
    lines += ["", "| 池 | SUT | 方法 | Area % | Recall % | F200 | 最大未接纳非碰撞数 | 不同序列数 |",
              "|---|---|---|---:|---:|---:|---:|---:|"]
    for row in final["rows"]:
        lines.append(f"| {row['pool']} | {row['sut_id']} | {row['method']} | {row['area_pct']:.3f} | "
                     f"{row['recall_pct']:.3f} | {row['F200']:.2f} | {row['max_negative_extra_count']} | {row['distinct_sequences']} |")
    lines += ["", f"{final['audited_runs']} 条曲线、{final['audited_runs'] * 200} 次披露核对通过。",
              "独立参照与既有冻结路径完全一致；回放只使用已测反馈。每个实际前缀的全程计数界、终点与累计面积界通过核对。"]
    for method, value in final["candidate_changes"].items():
        lines.append(f"相对 {method}：Area {value['area_pp']:+.3f}、Recall {value['recall_pp']:+.3f} 个百分点。")
    lines += ["", "单种子筛查未通过，不扩大重复。" if not screening["gate_passed"] else "筛查通过后补齐五种子；现有池仍是开发，显著性未确认。",
              "计数界、额度支出和相对发现数界均按实际前缀核对；额度规则不保证严格增益，新增物理仿真为 0。"]
    (RESULTS / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(RESULTS / "progress.json", {"status": "complete", "runs": final["audited_runs"],
                                         "disclosures": final["audited_runs"] * 200,
                                         "screening_gate_passed": screening["gate_passed"], "new_physical_calls": 0})
    print({"aggregate": final["aggregate"], "candidate_changes": final["candidate_changes"],
           "gate_passed": final["gate_passed"]}, flush=True)


if __name__ == "__main__":
    main()
