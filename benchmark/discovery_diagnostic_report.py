"""Consolidate completed development diagnostics without changing admission gates."""

import numpy as np

from .discovery import method_summary
from .discovery_config import OUTPUT, read_json, write_json


def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    original = read_json(OUTPUT / 'pilot/run_index.json')['runs']
    matched = read_json(OUTPUT / 'consistency/search/report.json')['runs']
    wide = read_json(OUTPUT / 'support/search/report.json')['runs']
    for path, expected in (('audit/report.json', 520),
                           ('consistency/search/audit/report.json', 80),
                           ('support/search/audit/report.json', 80)):
        audit = read_json(OUTPUT / path)
        if audit['status'] != 'passed' or audit['runs'] != expected:
            raise ValueError('All three discovery matrices must be independently audited')
    labels = {'fsbo': 'FSBO', 'pooled_ft': 'Pooled-FT',
              'source_local': 'Frozen encoder + local',
              'matched_source_local': 'Matched training + local',
              'wide_source_global': 'Wide supports + global',
              'wide_source_local': 'Wide supports + local'}
    all_rows = original + matched + wide
    selected = [r for r in all_rows if r['method'] in labels]
    summaries = method_summary(selected)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    curves = {}
    for method, label in labels.items():
        rows = [r for r in selected if r['method'] == method]
        if len(rows) != 40:
            raise ValueError('Each displayed method must retain its complete 40-run matrix')
        curves[method] = {}
        for axis, curve_name, metric in zip(axes, ('discovery_curve', 'recall_curve'),
                                             ('unique_collisions', 'collision_recall')):
            by_lineage = []
            for lineage in sorted({r['lineage'] for r in rows}):
                values = [r[curve_name] for r in rows
                          if r['lineage'] == lineage and r[curve_name][0] is not None]
                if values:
                    by_lineage.append(np.mean(values, axis=0))
            curve = np.mean(by_lineage, axis=0)
            np.testing.assert_allclose(curve[-1], summaries[method][metric]['mean'], atol=1e-12)
            curves[method][curve_name] = curve.tolist()
            axis.plot(np.arange(1, 129), curve, label=label)
    axes[0].set(xlabel='Paid queries (including 8 initial)', ylabel='Unique collision scenarios')
    axes[1].set(xlabel='Paid queries (including 8 initial)', ylabel='Recall over complete finite pool', ylim=(0, 1))
    for axis in axes:
        axis.legend(fontsize=7)
        axis.grid(alpha=0.2)
    destination = OUTPUT / 'diagnostic_summary'
    destination.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'svg', 'pdf'):
        figure.savefig(destination / f'curves.{extension}', dpi=180)
    plt.close(figure)
    write_json(destination / 'report.json', {
        'overall_objective_achieved': False, 'original_gate_status': 'not_passed',
        'total_search_runs': 680, 'total_equivalent_queries': 87040,
        'new_physical_calls': 0, 'summary': summaries, 'curves': curves,
        'displayed_methods': list(labels), 'displayed_runs': len(selected),
        'total_models_trained_in_diagnostics': 8, 'paired_prediction_episodes': 520})
    lines = ['# 开发诊断汇总', '',
             '原主矩阵与两项诊断共680项搜索、87,040次等效查询、新增物理调用为零。',
             '额外训练8个同规模历史响应模型，并完成360+160项配对预测验证；训练与验证开销独立保存。', '',
             '| 方法 | 平均去重碰撞数 | 平均召回率 | 发现 AUC |', '|---|---:|---:|---:|']
    for method, label in labels.items():
        stats = summaries[method]
        lines.append(f"| {label} | {stats['unique_collisions']['mean']:.3f} | {stats['collision_recall']['mean']:.2%} | {stats['discovery_auc']['mean']:.3f} |")
    lines.extend(['', '![数量与召回曲线](curves.png)', '',
                  '扩大元训练支持集改善了全局权重原型，但当前最好原型仍低于强基线。局部权重尚未显示必要性。',
                  '原5个百分点门槛在FSBO召回率95.4709%时不可达，原门槛保持未通过；未启动C/D/E。',
                  '下一轮的成功判据或实验条件仍待明确，整体研究目标未完成。', ''])
    (destination / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return summaries


def recovery_report():
    """Report all completed post-decision experiments, including failed variants."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    folders = ('recovery','safety_response','geometry_search')
    indices = {name:read_json(OUTPUT/name/'run_index.json') for name in folders}
    for name in folders:
        audit = read_json(OUTPUT/name/'audit/report.json')
        if audit['status'] != 'passed' or audit['runs'] != 80:
            raise ValueError('Post-decision summary requires all three completed raw-feedback audits')
    old = read_json(OUTPUT/'pilot/run_index.json')
    rows = [r for r in old['runs'] if r['method'] in ('fsbo','pooled_ft')]
    rows += [r for index in indices.values() for r in index['runs']]
    stats = method_summary(rows)
    labels = {'fsbo':'FSBO','pooled_ft':'Pooled-FT',
              'response_adaptive':'Collision fields, adapted',
              'response_fixed':'Collision fields, fixed',
              'safety_fixed':'Safety fields, fixed',
              'safety_adaptive':'Safety fields, adapted',
              'geometry_response':'Structured history fields',
              'kinematic_kernel':'Physical features control'}
    gate = read_json(OUTPUT/'geometry_search/gate.json')
    figure,axes = plt.subplots(1,3,figsize=(16,4.8),constrained_layout=True)
    curves = {}
    for method,label in labels.items():
        group = [r for r in rows if r['method'] == method]
        if len(group) != 40:
            raise ValueError('Post-decision plot requires every complete paired seed and task')
        curves[method] = {}
        for axis,curve_name,metric in zip(axes[:2],('discovery_curve','recall_curve'),
                                        ('unique_collisions','collision_recall')):
            means = []
            for lineage in sorted({r['lineage'] for r in group}):
                values = [r[curve_name] for r in group if r['lineage'] == lineage
                          and r[curve_name][0] is not None]
                if values:
                    means.append(np.mean(values,axis=0))
            curve = np.mean(means,axis=0)
            np.testing.assert_allclose(curve[-1],stats[method][metric]['mean'],atol=1e-12)
            curves[method][curve_name] = curve.tolist()
            axis.plot(np.arange(1,129),curve,label=label)
    axes[0].set(xlabel='Paid queries (8 initial included)',ylabel='Unique collision scenarios')
    axes[1].set(xlabel='Paid queries (8 initial included)',ylabel='Recall over the full finite pool',ylim=(0,1))
    for axis in axes[:2]:
        axis.legend(fontsize=6)
        axis.grid(alpha=.2)
    misses = [1-stats[m]['collision_recall']['mean'] for m in labels]
    axes[2].barh(list(labels.values()),misses)
    axes[2].invert_yaxis()
    threshold = .5*(1-gate['summary'][gate['strongest_baseline']]['collision_recall']['mean'])
    axes[2].axvline(threshold,color='black',linestyle='--',label='Half of strongest baseline misses')
    axes[2].xaxis.set_major_formatter(PercentFormatter(1))
    axes[2].set(xlabel='Missed fraction at 128 queries (lower is better)')
    axes[2].legend(fontsize=6)
    axes[2].grid(axis='x',alpha=.2)
    destination = OUTPUT/'post_decision_summary'
    destination.mkdir(parents=True,exist_ok=True)
    for extension in ('png','svg','pdf'):
        figure.savefig(destination/f'curves.{extension}',dpi=180)
    plt.close(figure)
    gates = {name:read_json(OUTPUT/name/'gate.json') for name in folders}
    result = {'original_gate_status':'not_passed','original_gate_changed':False,
              'remaining_miss_reduction_threshold':.5,'budget':128,
              'post_decision_runs':240,'post_decision_equivalent_queries':30720,
              'total_search_runs':920,'total_equivalent_queries':117760,
              'new_physical_calls':0,'summary':stats,'curves':curves,
              'gates':{name:{k:g[k] for k in ('status','remaining_miss_reduction','conditions')}
                       for name,g in gates.items()},
              'independent_target_evaluation_completed':False}
    write_json(destination/'report.json',result)
    lines = ['# 保留128次预算后的开发结果','',
             '用户确认保留128次预算，后续标准固定为相对最强基线减少至少50%的剩余漏检；碰撞数量不降低，至少两个谱系正收益。',
             '首轮5个百分点门槛及其失败记录保留。原FSBO召回率95.470932%，对应新门槛97.735466%；新增物理特征对照也参与其所在轮次的最强基线选择。',
             '三项后续实验共240运行、30,720次等效查询；累计920运行、117,760次等效查询，新增物理调用为零。全部后续运行已逐条核对原始反馈、账本、完整分母，并完成720次独立NumPy查询前概率与选点复核。','',
             '| 方法 | 平均去重碰撞数 | 平均召回率 | 发现AUC | 在线计算秒 |','|---|---:|---:|---:|---:|']
    for method,label in labels.items():
        stat = stats[method]
        online = stat['selection_s']['mean']+stat['adaptation_s']['mean']
        lines.append(f"| {label} | {stat['unique_collisions']['mean']:.3f} | {stat['collision_recall']['mean']:.5%} | {stat['discovery_auc']['mean']:.3f} | {online:.3f} |")
    lines += ['','在线计算仅指选例和目标适配；历史响应预测器构建、进程启动和物理建库成本在各运行cost中单独保存。实际运行并行执行，在线秒数不视为经过严格隔离的硬件速度比较。','',
              '![发现曲线和剩余漏检](curves.png)','',
              '| 后续阶段 | 门槛 | 漏检减少比例 |','|---|---|---:|']
    for name,value in gates.items():
        ratio = value['remaining_miss_reduction']
        lines.append(f"| {name} | {value['status']} | {ratio:.4%} |" if ratio is not None else
                     f"| {name} | {value['status']} | 不适用 |")
    lines += ['','各轮主方法均在运行前指定，辅助版本即使更好也不自动替代主方法。配对分层自助区间、完整谱系和逐项机制检查保留在各gate.json。',
              '连续安全信息与相对运动特征是不同的改变。历史预测Brier/MAE改善不等于碰撞发现机制成立；若物理特征对照解释收益，不认定历史响应迁移成立。',
              '这些仍是开发结果。C/D/E的准入和独立目标验证必须另行满足，不能将本报告写为正式独立验证成功。','']
    (destination/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    return result


def adaptive_report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    from .discovery_adaptive_target import REPORT, report as gate_report

    audit = read_json(REPORT/'audit/report.json')
    negative = read_json(REPORT/'audit/negative_controls.json')
    if audit['status'] != 'passed' or audit['runs'] != 200 or negative['status'] != 'passed':
        raise ValueError('Adaptive-target figures require the complete audit and corruption controls')
    result = gate_report()
    rows = read_json(REPORT/'run_index.json')['runs']
    rows += [r for r in read_json(OUTPUT/'pilot/run_index.json')['runs'] if r['method'] == 'fsbo']
    rows += [r for r in read_json(OUTPUT/'geometry_search/run_index.json')['runs'] if r['method'] == 'kinematic_kernel']
    stats = method_summary(rows)
    labels = {'fsbo':'FSBO','kinematic_kernel':'Physical features control',
              'hybrid_local':'Strong target + local sources','hybrid_global':'Strong target + global sources',
              'hybrid_shared':'Strong target + shared kernel','target_local':'Two targets, frozen physical GP',
              'gp_target_local':'Two targets, adapted physical GP'}
    figure,axes = plt.subplots(1,3,figsize=(16,4.8),constrained_layout=True)
    curves = {}
    for method,label in labels.items():
        group = [r for r in rows if r['method'] == method]
        if len(group) != 40:
            raise ValueError('Adaptive-target curves require all 40 paired runs per method')
        curves[method] = {}
        for axis,name,metric in zip(axes[:2],('discovery_curve','recall_curve'),
                                    ('unique_collisions','collision_recall')):
            means = []
            for lineage in sorted({r['lineage'] for r in group}):
                values = [r[name] for r in group if r['lineage'] == lineage and r[name][0] is not None]
                if values:
                    means.append(np.mean(values,axis=0))
            curve = np.mean(means,axis=0)
            np.testing.assert_allclose(curve[-1],stats[method][metric]['mean'],atol=1e-12)
            curves[method][name] = curve.tolist()
            axis.plot(np.arange(1,129),curve,label=label)
    axes[0].set(xlabel='Paid queries (8 initial included)',ylabel='Unique collision scenarios')
    axes[1].set(xlabel='Paid queries (8 initial included)',ylabel='Full-pool collision recall',ylim=(0,1))
    for axis in axes[:2]:
        axis.grid(alpha=.2)
        axis.legend(fontsize=6)
    axes[2].barh(list(labels.values()),[1-stats[m]['collision_recall']['mean'] for m in labels])
    axes[2].invert_yaxis()
    axes[2].axvline(1-result['required_recall'],color='black',linestyle='--',label='Required remaining misses')
    axes[2].xaxis.set_major_formatter(PercentFormatter(1))
    axes[2].set(xlabel='Missed fraction at 128 queries (lower is better)')
    axes[2].grid(axis='x',alpha=.2)
    axes[2].legend(fontsize=6)
    for ext in ('png','svg','pdf'):
        figure.savefig(REPORT/f'curves.{ext}',dpi=180)
    plt.close(figure)
    write_json(REPORT/'curves.json',{'summary':stats,'curves':curves,'audit':audit['status'],
               'total_search_runs':1120,'total_equivalent_queries':143360,'new_physical_calls':0})
    text = (REPORT/'report.md').read_text(encoding='utf-8')
    text += ('\n![发现曲线与剩余漏检](curves.png)\n\n'
             '独立审计通过：200运行、25,600条原始付费反馈及可靠性更新；'
             '143,360次来源分量概率重建；600次FSBO、120次从零GP数值预测检查；'
             '600次完整候选库贪心选点复核。概率、权重、分母和账本反馈四项人工损坏均被审计拒绝，原文件未修改。\n\n'
             '累计1,120项发现运行、143,360次等效查询，新增物理调用为零。'
             '该轮不支持以强化目标分量或一般融合解决历史来源收益问题；整体目标仍未完成。\n')
    (REPORT/'report.md').write_text(text,encoding='utf-8')
    return result


def joint_report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    from .discovery_joint_response import MODELS, REPORT, report as gate_report
    from .protocol import ROOT

    if not (REPORT/'audit/report.json').exists():
        raise ValueError('Joint-response figures require the complete 280-run audit first')
    audit = read_json(REPORT/'audit/report.json')
    negative = read_json(REPORT/'audit/negative_controls.json')
    if audit['status'] != 'passed' or audit['runs'] != 280 or negative['status'] != 'passed':
        raise ValueError('Joint-response figures require all 280 audited runs and corruption controls')
    result = gate_report()
    rows = read_json(REPORT/'run_index.json')['runs']
    rows += [r for r in read_json(OUTPUT/'pilot/run_index.json')['runs'] if r['method'] == 'fsbo']
    rows += [r for r in read_json(OUTPUT/'geometry_search/run_index.json')['runs'] if r['method'] == 'kinematic_kernel']
    stats = method_summary(rows)
    labels = {'fsbo':'FSBO','kinematic_kernel':'Physical features control',
              'joint_local':'Joint transfer, local','joint_global':'Joint transfer, global',
              'joint_shared':'Joint transfer, shared covariance','joint_mean':'Joint transfer, mean only',
              'joint_binary':'Joint transfer, binary feedback','joint_target':'Joint target only',
              'joint_uncorrelated':'Uncorrelated target only'}
    figure,axes = plt.subplots(2,2,figsize=(16,10),constrained_layout=True)
    curves,checkpoints = {},{}
    for method,label in labels.items():
        group = [r for r in rows if r['method'] == method]
        if len(group) != 40:
            raise ValueError('Joint figures require every paired seed and task for each method')
        curves[method] = {}
        for axis,name,metric in zip(axes[0],('discovery_curve','recall_curve'),
                                    ('unique_collisions','collision_recall')):
            means = []
            for lineage in sorted({r['lineage'] for r in group}):
                values = [r[name] for r in group if r['lineage'] == lineage and r[name][0] is not None]
                if values:
                    means.append(np.mean(values,axis=0))
            curve = np.mean(means,axis=0)
            np.testing.assert_allclose(curve[-1],stats[method][metric]['mean'],atol=1e-12)
            curves[method][name] = curve.tolist()
            axis.plot(np.arange(1,129),curve,label=label)
        checkpoints[method] = {str(b):{'unique_collisions':curves[method]['discovery_curve'][b-1],
                                      'collision_recall':curves[method]['recall_curve'][b-1]}
                               for b in (16,32,64,128)}
    axes[0,0].set(xlabel='Paid queries (8 initial included)',ylabel='Unique collision scenarios')
    axes[0,1].set(xlabel='Paid queries (8 initial included)',ylabel='Full-pool collision recall',ylim=(0,1))
    axes[0,1].yaxis.set_major_formatter(PercentFormatter(1))
    for axis in axes[0]:
        axis.grid(alpha=.2)
        axis.legend(fontsize=7)
    axes[1,0].barh(list(labels.values()),[1-stats[m]['collision_recall']['mean'] for m in labels])
    axes[1,0].invert_yaxis()
    axes[1,0].axvline(1-result['required_recall'],color='black',linestyle='--',label='Required remaining misses')
    axes[1,0].xaxis.set_major_formatter(PercentFormatter(1))
    axes[1,0].set(xlabel='Missed fraction at 128 paid queries (lower is better)')
    axes[1,0].grid(axis='x',alpha=.2)
    axes[1,0].legend(fontsize=7)
    lineages = sorted(stats['joint_local']['collision_recall']['by_lineage'])
    values = np.array([[stats[m]['collision_recall']['by_lineage'][lineage] for lineage in lineages]
                       for m in labels])
    heatmap = axes[1,1].imshow(values,vmin=0,vmax=1,cmap='Blues',aspect='auto')
    axes[1,1].set_xticks(range(len(lineages)),lineages)
    axes[1,1].set_yticks(range(len(labels)),labels.values())
    axes[1,1].set_title('Recall by lineage at 128 queries')
    for i in range(len(labels)):
        for j in range(len(lineages)):
            axes[1,1].text(j,i,f'{values[i,j]:.1%}',ha='center',va='center',
                           color='white' if values[i,j] > .6 else 'black',fontsize=8)
    figure.colorbar(heatmap,ax=axes[1,1],fraction=.04)
    for extension in ('png','svg','pdf'):
        figure.savefig(REPORT/f'curves.{extension}',dpi=180)
    plt.close(figure)
    construction = read_json(OUTPUT/'diagnosis/report.json')['banks']
    physical = {}
    for role in ('history','development'):
        banks = [b for b in construction if b['role'] == role and b['scenario'] in ('cut_in','cut_out')]
        physical[role] = {'banks':len(banks),'executions':sum(b['physical_bank_calls'] for b in banks),
                          'execution_s':sum(b['physical_bank_execution_s'] for b in banks),
                          'new_physical_calls_this_experiment':0}
    training = {str(path.relative_to(MODELS)):read_json(path)['wall_s'] for path in MODELS.glob('*/*.json')}
    priors = {kind:read_json(REPORT/'protocol.json')['prior_reports'][kind]['wall_s']
              for kind in ('cut_in','cut_out')}
    costs = {'model_training_wall_s_by_model':training,'source_posterior_construction_wall_s':priors,
             'existing_physical_pool_costs':physical,'search_by_method':{},
             'failed_smoke_and_reconstruction':read_json(REPORT/'failed_initialization_smoke/report.json')}
    for method in labels:
        if method.startswith('joint_'):
            group = [r for r in rows if r['method'] == method]
            payloads = [read_json(ROOT/r['path']) for r in group]
            costs['search_by_method'][method] = {
                'runs':40,'equivalent_queries':5120,'new_physical_calls':0,
                'mean_selection_s':stats[method]['selection_s']['mean'],
                'mean_adaptation_s':stats[method]['adaptation_s']['mean'],
                'sum_run_wall_s':sum(p['cost']['wall_s'] for p in payloads),
                'replay_equivalent_simulation_s':sum(record['feedback']['execution_s']
                    for p in payloads for record in p['query_ledger'])}
    write_json(REPORT/'curves.json',{'summary':stats,'curves':curves,'checkpoints':checkpoints,
               'complete_matrix_audited':True,'valid_total_search_runs':1400,
               'valid_total_equivalent_queries':179200,'new_physical_calls':0})
    write_json(REPORT/'costs.json',costs)
    text = (REPORT/'report.md').read_text(encoding='utf-8')
    text += ('\n![发现曲线、剩余漏检与谱系召回率](curves.png)\n\n'
             '独立复核通过全部280项运行、35,840条付费碰撞/间距反馈及可靠性更新，'
             '并在840个查询前状态以NumPy重建全库概率和实际选点。16个历史来源后验已独立检查。'
             '六项人工损坏均被拒绝，原记录保持不变。\n\n'
             '| 方法 | 查询预算 | 平均碰撞数 | 平均召回率 |\n|---|---:|---:|---:|\n')
    for method,label in labels.items():
        for budget,metrics in checkpoints[method].items():
            text += f"| {label} | {budget} | {metrics['unique_collisions']:.3f} | {metrics['collision_recall']:.5%} |\n"
    text += ('\n统计先在谱系内汇总再对谱系等权汇总；零碰撞任务保留在数量和成本中，召回不适用。'
             '数量与召回在同一候选库内成比例，不视为两份独立证据。配对分层自助区间见gate.json。\n\n'
             '成本见costs.json：已有历史建库、开发全库、四个模型训练、来源后验构建、在线适配和选例分别列出。'
             '共享GPU并行运行的耗时不用于严格硬件速度比较，重叠的运行时间不相加解释为总经过时间。'
             '全矩阵新增35,840次等效查询、零次物理调用；累计1,400项有效搜索、179,200次等效查询。'
             '失败冒烟及18次重建查询另列，原失败查询数为重建值9、可确认边界8至127。\n\n'
             '本报告为开发证据，独立目标和严格留家族尚待后续阶段，不能外推连续空间或真实道路的全部失效覆盖。\n')
    (REPORT/'report.md').write_text(text,encoding='utf-8')
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('scope',nargs='?',default='original',choices=('original','recovery','adaptive','joint'))
    {'original':report,'recovery':recovery_report,'adaptive':adaptive_report,'joint':joint_report}[parser.parse_args().scope]()
