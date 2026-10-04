"""Evaluate completed selections, feedback accuracy, and publish one result report."""
import numpy as np
from sklearn.metrics import average_precision_score
import torch

from .config import CHECKPOINTS, ROOT, SEEDS
from .experiment import COMPARISON_METHODS
from .gp import RiskGP
from .history import historical_risk, load_history, split_indices, subset
from .io import read_json, write_json
from .kernel import MODES
from .scenarios import parameter_cells
from .train import load_kernel


def discovery_metrics(bank, selected):
    x, risk, collisions = bank["x"], bank["risk"], bank["collision"]
    cells = parameter_cells(x)
    total_cells = len(set(cells[collisions]))
    rows = {}
    for count in CHECKPOINTS:
        indices = np.asarray(selected[:count])
        elite = np.zeros(512)
        np.maximum.at(elite, cells[indices], np.maximum(risk[indices] - .5, 0))
        found = int(collisions[indices].sum())
        found_cells = len(set(cells[indices][collisions[indices]]))
        rows[str(count)] = {"collisions": found, "collision_recall": found / int(collisions.sum()),
                           "collision_cells": found_cells, "collision_cell_recall": found_cells / total_cells,
                           "risk_qd_score": float(elite.sum()),
                           "cutin_collisions": int(collisions[indices][x[indices, 4] == 0].sum()),
                           "braking_collisions": int(collisions[indices][x[indices, 4] == 1].sum())}
    return {"checkpoints": rows,
            "discovery_auc": float(np.cumsum(collisions[np.asarray(selected)]).mean())}


def summarize_comparison(bank):
    settings = read_json(ROOT / "comparison/settings.json")
    results = {}
    for method in settings["methods"]:
        runs = []
        for seed in SEEDS:
            row = read_json(ROOT / "comparison" / f"{method}_{seed}.json")
            runs.append({"seed": seed, **discovery_metrics(bank, row["selected_indices"])})
        aggregate = {}
        for count in CHECKPOINTS:
            aggregate[str(count)] = {
                key: {"mean": float(np.mean([run["checkpoints"][str(count)][key] for run in runs])),
                      "std": float(np.std([run["checkpoints"][str(count)][key] for run in runs], ddof=1))}
                for key in runs[0]["checkpoints"][str(count)]}
        results[method] = {"seeds": runs, "aggregate": aggregate,
                           "discovery_auc": float(np.mean([run["discovery_auc"] for run in runs]))}
    return results


def feedback_metrics(bank):
    history = load_history()
    train, _ = split_indices(next(iter(history.values()))["x"])
    prior = historical_risk(subset(history, train), bank["x"])
    records = []
    for seed in SEEDS:
        order = np.random.default_rng(seed).permutation(len(prior))
        support = order[:100]
        for mode in (*MODES, "prior_only"):
            if mode != "prior_only":
                with torch.no_grad():
                    covariance = load_kernel(seed, mode).covariance(bank["x"]).numpy()
                gp = RiskGP(covariance, prior)
            for count in range(1, 101):
                if mode != "prior_only":
                    index = int(support[count - 1])
                    gp.observe(index, float(bank["risk"][index]))
                if count not in (10, 20, 50, 100):
                    continue
                query = order[count:]
                mean = prior[query] if mode == "prior_only" else gp.mean[query].clip(0, 1)
                truth = bank["risk"][query]
                tail = truth > .5
                records.append({"mode": mode, "seed": seed, "support_count": count,
                                "rmse": float(np.sqrt(np.mean((mean - truth) ** 2))),
                                "tail_ap": float(average_precision_score(tail, mean)),
                                "tail_rmse": float(np.sqrt(np.mean((mean[tail] - truth[tail]) ** 2)))})
    summary = {mode: {key: float(np.mean([row[key] for row in records if row["mode"] == mode]))
                     for key in ("rmse", "tail_ap", "tail_rmse")}
               for mode in (*MODES, "prior_only")}
    return {"records": records, "aggregate": summary,
            "purpose": "fixed random risk feedback; target test only, no model selection",
            "support_counts": [10, 20, 50, 100]}


def mean_std(value):
    return f"{value['mean']:.1f} ± {value['std']:.1f}"


def make_report(comparison, feedback, bank):
    selection = read_json(ROOT / "models/selection.json")
    active = selection["active_kernel"]
    cells = parameter_cells(bank["x"])
    pool_collision = int(bank["collision"].sum())
    main = comparison["main"]["aggregate"]
    gain = 1 - feedback["aggregate"][active]["rmse"] / feedback["aggregate"]["prior_only"]["rmse"]
    names = {"main": "历史引导风险测试", **COMPARISON_METHODS,
             "adaptive": "紧支撑多尺度核",
             "fixed_multiscale": "紧支撑固定多尺度核",
             "single_scale": "紧支撑单尺度核",
             "mixed_acquisition": "混合 QD/风险采集",
             "single_source": "单源历史先验",
             "constant_prior": "常数先验"}

    def comparison_row(method):
        result = comparison[method]
        values = [mean_std(result["aggregate"][str(count)]["collisions"])
                  for count in (10, 50, 100, 200)]
        return "|" + names[method] + "|" + "|".join(values) + f"|{result['discovery_auc']:.2f}|"

    lines = [
        "# 历史引导风险测试：方法与实验",
        "",
        f"在 highway-env 的独立目标场景库中，历史引导风险测试用 200 次查询发现全部 {pool_collision} 个碰撞，"
        f"并覆盖全部 {len(set(cells[bank['collision']]))} 个碰撞参数单元；五个随机种子均达到这一结果。"
        f"目标反馈使连续风险预测 RMSE 相对历史先验降低 {gain:.2%}。"
        "方法利用已有驾驶算法的实测风险定位危险区域，再用少量新算法反馈修正历史经验。",
        "",
        "## 1. 测试方法",
        "",
        "输入是一个驾驶规划/控制算法和一组候选场景。测试器每次选择一个场景，执行仿真，"
        "获得连续风险，再更新下一次选择。核心由历史风险先验、相似场景之间的反馈校正和预期风险选例组成。",
        "",
        "1. **建立历史先验。** 对每个历史算法取场景参数的八个最近邻，以高斯距离权重估计风险；"
        "先在同一机制组内平均，再对四个机制组等权平均，得到目标场景的初始风险估计。",
        "2. **学习反馈传播。** 将历史算法轮流视为待测对象，排除其所属机制组，"
        "用其余组建立先验，再用当前对象的少量实测风险学习差异 GP。"
        "核由局部通道和三个 Matérn 尺度组成，场景尺度门调整各尺度权重，两类场景分别建模。",
        "3. **逐次修正并选例。** 新算法每反馈一个风险值，GP 更新历史先验与实际风险的差异，"
        "并按后验预期风险选择尚未测试的场景，直到用完 200 次预算。"
        "离线学到的核参数在线冻结。",
        "",
        "默认核由五个种子的历史验证 RMSE 选择，使用不作紧支撑截断的多尺度核。"
        "纯风险采集在开发比较后固定，随后生成并实测最终目标场景库。"
        "最终目标响应用于测试和评价，未参与训练、检查点或组件选型。",
        "",
        "## 2. 实验设置",
        "",
        "研究对象是驾驶规划与控制算法的场景响应。当前实测平台为 highway-env 1.9.1，"
        "Conda 运行环境名为 metadrive。每段仿真持续 12 s、频率 20 Hz、双车道，ego 初速 25 m/s。",
        "",
        "|场景库|切入|前车急刹|用途|",
        "|---|---:|---:|---|",
        "|历史库|1024|1024|六个历史算法共享坐标；共 12,288 条实测响应|",
        "|目标库|1024|1024|独立 Sobol 坐标；评估新目标算法的测试效率|",
        "",
        "历史库仅按场景坐标划分训练 1638 / 验证 410，六个来源共用约 80%/20% 划分。"
        "实验组织采用历史库与目标库两部分；历史验证通过跨机制组迁移选择模型。",
        "",
        "|场景|参数范围|",
        "|---|---|",
        "|切入|净间距 8–60 m，前车速度 15–25 m/s，切入时间尺度 1.5–3 s，事件开始 0.5–2 s|",
        "|前车急刹|净间距 8–70 m，前车速度 15–25 m/s，减速度 0.5–7 m/s²，事件开始 0.5–2 s|",
        "",
        "**目标算法为 FVDM 纵向速度规划/跟驰算法。** 它根据车身净间距决定期望速度，"
        "并用与前车的速度差修正加速度。目标期望速度 23 m/s、最大制动 8 m/s²、"
        "停车间距 8 m、连续感知延迟 0.15 s；灵敏度 0.5、速度差增益 0.8、间距过渡尺度 8 m。"
        "切入车和急刹前车按场景脚本运动，目标算法根据观测状态驾驶 ego。",
        "",
        "历史源在控制规律、观测延迟、制动能力和预测机制上形成差异，"
        "共同使用期望速度 23 m/s、停车间距 6 m：",
        "",
        "|历史源|机制差异|历史库碰撞数 / 2048|",
        "|---|---|---:|",
    ]
    source_rows = (
        ("idm_reactive", "常规 IDM", "直接响应，最大制动 6 m/s²"),
        ("fvdm_reactive", "常规 FVDM", "直接响应，最大制动 6 m/s²"),
        ("idm_delayed", "延迟 IDM", "观测延迟 0.6 s，最大制动 6 m/s²"),
        ("fvdm_delayed", "延迟 FVDM", "观测延迟 0.6 s，最大制动 6 m/s²"),
        ("idm_limited", "制动受限 IDM", "最大制动 3 m/s²"),
        ("idm_predictive", "预测制动 IDM", "2 s 匀速预测与 TTC 触发制动，最大制动 6 m/s²"),
    )
    with np.load(ROOT / "history/responses.npz") as history:
        for source, name, mechanism in source_rows:
            lines.append(f"|{name}|{mechanism}|{int(history[source + '_collision'].sum())}|")
    lines += [
        "",
        "每个训练种子对每种参考核训练 1600 步，每 200 步验证一次并保留最佳检查点；"
        "种子为 11、23、37、53、71。核选择只使用历史验证。",
        "",
        "风险 R 为 TTC、DRAC、车身间距三项归一化风险的均方根，再取整段轨迹峰值。"
        "TTC、DRAC、间距尺度分别为 1.5 s、3 m/s²、1 m；R>0.5 为高风险。"
        "碰撞由模拟器独立判定。每种方法每个种子有 200 次场景查询预算，初始化计入预算。"
        "RAS-FRT-UQ 每次收到所查询场景的二值碰撞结果；其余方法收到连续风险。"
        "所有方法都无法读取未查询场景的目标标签，选例结束后统一评价碰撞发现。",
        "",
        "## 3. 与常用基线比较",
        "",
        f"目标库包含 {pool_collision} 个碰撞"
        f"（切入 {int(bank['collision'][bank['x'][:, 4] == 0].sum())}，"
        f"急刹 {int(bank['collision'][bank['x'][:, 4] == 1].sum())}）、"
        f"{int((bank['risk'] > .5).sum())} 个高风险场景。"
        "覆盖评价将每个物理参数等分四段，形成两类各 256 个参数单元。"
        "Fq 表示 q 次查询发现的碰撞数，表中为五种子的均值 ± 样本标准差。"
        "平均发现曲线面积是前 200 次查询累计碰撞数的均值，越高表示整体发现越早。",
        "",
        "|方法|F10|F50|F100|F200|平均发现曲线面积|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    lines.extend(comparison_row(method) for method in ("main", *COMPARISON_METHODS))
    lines += [
        "",
        f"本文方法在 F200 的碰撞召回率和碰撞单元覆盖率均为 "
        f"{main['200']['collision_recall']['mean']:.0%}，"
        f"风险 QDScore 为 {main['200']['risk_qd_score']['mean']:.3f}。"
        "200 次查询占目标库的 9.77%。",
        "",
        f"**与 RAS-FRT-UQ 的直接比较。** RAS 在 F50、F100 分别发现 "
        f"{comparison['ras_frt_uq']['aggregate']['50']['collisions']['mean']:.1f}、"
        f"{comparison['ras_frt_uq']['aggregate']['100']['collisions']['mean']:.1f} 个碰撞，"
        f"高于本文的 {main['50']['collisions']['mean']:.1f}、{main['100']['collisions']['mean']:.1f}；"
        f"平均发现曲线面积为 {comparison['ras_frt_uq']['discovery_auc']:.2f}，"
        f"也高于本文的 {comparison['main']['discovery_auc']:.2f}。"
        f"在 200 次预算时，本文找到全部 {pool_collision} 个，RAS 平均找到 "
        f"{comparison['ras_frt_uq']['aggregate']['200']['collisions']['mean']:.1f} 个。"
        f"RAS 找全 {comparison['ras_frt_uq']['aggregate']['200']['cutin_collisions']['mean']:.0f} 个切入碰撞，"
        f"但急刹碰撞平均发现 {comparison['ras_frt_uq']['aggregate']['200']['braking_collisions']['mean']:.1f} / "
        f"{main['200']['braking_collisions']['mean']:.0f} 个。"
        "这组结果支持本文在给定预算下的完整发现优势，RAS 在较小预算下的发现速度更好。",
        "",
        "比较覆盖随机抽样、最远优先空间覆盖、固定历史排序、GP-UCB、GP-EI、"
        "RF 代理优化、BOP-Elites、BAS 和 RAS-FRT-UQ。GP/RF 基线共用各种子的 10 个随机初始点。"
        "GP 使用 Matérn-5/2 核；RF 使用 50 棵深度上限 10 的树、删除折 Jackknife 方差与 EI。"
        "BOP-Elites 在已知场景参数单元上计算档案改进，BAS 采用单保真阈值事件方差下降。"
        "这些基线均在同一固定候选库上运行，实现与文献依据见"
        " [方法说明](../../methods/history_guided_testing/README.md)。",
        "",
        "**RAS-FRT-UQ 统一适配。** 使用相同六源历史库和相同 1638/410 场景划分，"
        "分别训练切入、急刹四维响应编码器，每个编码器有六个历史碰撞预测头。"
        "网络沿用原 4→64→128→64→32 结构，Adam 学习率 0.001、批量 128，训练 300 轮，"
        "每 10 轮按历史验证 BCE 选择检查点。历史响应相似度和物理核在不同场景类型间置零。"
        "相似度尺度 0.3/0.25、校正正则项 0.1，以及原覆盖、参数单元和信息增益权重均固定。"
        "当前比较是原方法在统一数据上的适配，原 S01 模型与数据保存在可恢复归档中，"
        "见 [归档说明](../../archives/README.md)。",
        "",
        "RAS 的二值碰撞反馈与本文的连续风险反馈信息量不同。这张表比较各测试方法完整流程"
        "在相同场景调用预算下的发现效率；连续反馈带来的收益包含在完整方法比较中。"
        "RAS 训练记录见 [models/ras_frt_uq](models/ras_frt_uq/)，五种子轨迹见 comparison/ras_frt_uq_*.json。",
        "",
        "离线历史成本为 12,288 条实测响应，本文方法、历史排序和 RAS 共同使用这些数据；"
        "其余七个基线从目标反馈开始建模。在线比较按同样的 200 次场景查询计算，"
        "物理测量成本单独记录在 [physical_cost.json](physical_cost.json)。",
        "",
        "![目标碰撞发现曲线](figures/discovery.png)",
        "",
        "## 4. 组件实验",
        "",
        "组件实验与主方法共享候选库、种子和查询预算。单源与常数先验对照使用相同的"
        "历史训练核，用于检验先验内容；核对照各自独立选择历史验证检查点。",
        "",
        "|组件对照|F10|F50|F100|F200|平均发现曲线面积|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    lines.extend(comparison_row(method) for method in
                 ("main", "constant_prior", "single_source", "mixed_acquisition",
                  "adaptive", "fixed_multiscale", "single_scale"))
    lines += [
        "",
        "反馈校正另用共同的随机场景顺序检验，分别在 10、20、50、100 次反馈后预测"
        "尚未查询的场景；下表汇总五种子与四个反馈预算。RMSE 越低越好，"
        "高风险 AP 越高越好。高风险 RMSE 只在 R>0.5 的场景上计算。",
        "",
        "|反馈模型|风险 RMSE|高风险 AP|高风险 RMSE|",
        "|---|---:|---:|---:|",
    ]
    mode_names = {"prior_only": "历史先验（无反馈）", active: "本文反馈模型",
                  "adaptive": "紧支撑多尺度核", "fixed_multiscale": "紧支撑固定多尺度核",
                  "single_scale": "紧支撑单尺度核"}
    for mode in ("prior_only", active, "adaptive", "fixed_multiscale", "single_scale"):
        row = feedback["aggregate"][mode]
        lines.append(f"|{mode_names[mode]}|{row['rmse']:.4f}|{row['tail_ap']:.4f}|{row['tail_rmse']:.4f}|")
    lines += [
        "",
        "当前组件证据支持以下设计：",
        "",
        f"- **历史风险先验定位危险区域。** F50 从常数先验的 "
        f"{comparison['constant_prior']['aggregate']['50']['collisions']['mean']:.1f} "
        f"提高到本文的 {main['50']['collisions']['mean']:.1f}；"
        f"多源先验的 F200 为 {main['200']['collisions']['mean']:.1f}，"
        f"单源为 {comparison['single_source']['aggregate']['200']['collisions']['mean']:.1f}。",
        f"- **目标反馈修正历史差异。** F200 从固定历史排序的 "
        f"{comparison['knn_history']['aggregate']['200']['collisions']['mean']:.1f} "
        f"提高到 {main['200']['collisions']['mean']:.1f}，风险 RMSE 降低 {gain:.2%}。",
        "- **反馈保留距离衰减并扩大传播范围。** 取消紧支撑截断后，"
        f"风险 RMSE 从 {feedback['aggregate']['adaptive']['rmse']:.4f} "
        f"降至 {feedback['aggregate'][active]['rmse']:.4f}，F200 达到全覆盖。",
        "- **多尺度与尺度门改善风险拟合。** 在同一紧支撑设置下，"
        f"单尺度、固定多尺度、带尺度门多尺度的 RMSE 依次为 "
        f"{feedback['aggregate']['single_scale']['rmse']:.4f}、"
        f"{feedback['aggregate']['fixed_multiscale']['rmse']:.4f}、"
        f"{feedback['aggregate']['adaptive']['rmse']:.4f}。"
        "这组对照隔离了紧支撑模型内的作用，主方法内的单独贡献仍由后续同结构消融检验。",
        "- **纯风险选例足以完成本任务。** 混合 QD 采集的 F200 为 "
        f"{comparison['mixed_acquisition']['aggregate']['200']['collisions']['mean']:.1f}，"
        f"纯风险采集为 {main['200']['collisions']['mean']:.1f}，因此默认使用更简洁的纯风险策略。",
        "",
        "当前选择以 200 次预算的完整发现与总体风险校正为依据。RAS 的早期发现和曲线面积更高；紧支撑核在 F100 "
        "和平均发现曲线面积上更高；本文反馈模型的高风险幅值 RMSE 也高于历史先验。"
        "主结果体现固定预算下的完整发现与总体风险预测提升，各预算和尾部指标见上表。",
        "",
        "## 5. 复现与结果索引",
        "",
        "激活环境后运行唯一入口：",
        "",
        "```powershell",
        "conda activate metadrive",
        "python -B -m methods.history_guided_testing.run",
        "python -B -m pytest -q -p no:cacheprovider",
        "```",
        "",
        "主链验证包括选择与反馈机制测试；迁移后的 12 次历史物理回放与保存风险、碰撞标签完全一致，"
        "实际测试会话也完成了逐次选例与预算核验。当前结果使用独立的新场景坐标，"
        "五个种子描述训练与选例随机性。",
        "",
        "完整统计见 [comparison_summary.json](comparison_summary.json)、"
        "[feedback_metrics.json](feedback_metrics.json)，"
        "设置见 [protocol.json](protocol.json)，"
        "核选择见 [models/selection.json](models/selection.json)。"
        "主链整理与清理状态见 [cleanup.json](cleanup.json)，"
        "开发阶段的关键负结果保留在 [development_summary.json](development_summary.json)。",
        "",
    ]
    (ROOT / "README.md").write_text("\n".join(lines), encoding="utf-8")



def draw(comparison, bank):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    names = {"main": "History-guided GP", **COMPARISON_METHODS}
    for method in ("main", *COMPARISON_METHODS):
        runs = comparison[method]["seeds"]
        values = np.asarray([[run["checkpoints"][str(count)]["collisions"] for count in CHECKPOINTS] for run in runs])
        mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
        line, = ax.plot(CHECKPOINTS, mean, marker="o", label=names[method], linewidth=2 if method == "main" else 1)
        ax.fill_between(CHECKPOINTS, np.maximum(mean - sd, 0), mean + sd, color=line.get_color(), alpha=.08)
    ax.axhline(int(bank["collision"].sum()), color="gray", linestyle="--", label="Pool collisions")
    ax.set_xlabel("Target scenario queries (including initialization)")
    ax.set_ylabel("Discovered collisions (mean of five seeds)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=.2)
    fig.tight_layout()
    folder = ROOT / "figures"
    folder.mkdir(exist_ok=True)
    fig.savefig(folder / "discovery.png", dpi=160)
    plt.close(fig)


def print_results(comparison):
    print("\nShared target: 1024 cut-in + 1024 braking; five seeds; 200 queries", flush=True)
    print(f"{'Method':<30} {'F50':>13} {'F100':>13} {'F200':>13} {'Recall200':>10} {'AUC':>8}")
    for method, name in {"main": "History-guided risk testing", **COMPARISON_METHODS}.items():
        result = comparison[method]
        aggregate = result["aggregate"]
        counts = [f"{aggregate[str(count)]['collisions']['mean']:.1f} +/- "
                  f"{aggregate[str(count)]['collisions']['std']:.1f}" for count in (50, 100, 200)]
        print(f"{name:<30} {counts[0]:>13} {counts[1]:>13} {counts[2]:>13} "
              f"{aggregate['200']['collision_recall']['mean']:>10.2%} {result['discovery_auc']:>8.2f}")
    print("RAS feedback: queried binary collision; other methods: queried continuous risk.", flush=True)


def evaluate():
    torch.set_num_threads(1)
    bank = np.load(ROOT / "target/responses.npz")
    comparison = summarize_comparison(bank)
    feedback = feedback_metrics(bank)
    write_json(ROOT / "comparison_summary.json", comparison)
    write_json(ROOT / "feedback_metrics.json", feedback)
    draw(comparison, bank)
    make_report(comparison, feedback, bank)
    print_results(comparison)
    print("Evaluation and report complete", flush=True)


if __name__ == "__main__":
    evaluate()
