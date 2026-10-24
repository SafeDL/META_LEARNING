"""Complete, audited curves and separate costs for the physical-mixture trial."""

import hashlib
import json

import numpy as np

from . import discovery_joint_response as joint
from .collision_metrics import paired_lineage_bootstrap
from .discovery import method_summary
from .discovery_config import OUTPUT, read_json, write_json
from .discovery_physical_mixture import MAIN, METHODS, REPORT, STRONG, checkpoint_path, gate, previous_rows, protocol
from .protocol import ROOT


def aggregate_curves(rows,methods):
    stats = method_summary(rows)
    result = {}
    for method in methods:
        group = [r for r in rows if r['method'] == method]
        if len(group) != 40:
            raise ValueError('Every displayed method needs all 40 paired task/seed runs')
        result[method] = {}
        for name,metric in (('discovery_curve','unique_collisions'),('recall_curve','collision_recall')):
            lineage_means = []
            for lineage in sorted({r['lineage'] for r in group}):
                values = [r[name] for r in group if r['lineage'] == lineage and r[name][0] is not None]
                if values:
                    lineage_means.append(np.mean(values,axis=0))
            if not lineage_means:
                if stats[method][metric]['mean'] is not None:
                    raise ValueError('Undefined recall differs from the summary')
                result[method][name] = [None]*128
                continue
            curve = np.mean(lineage_means,axis=0)
            np.testing.assert_allclose(curve[-1],stats[method][metric]['mean'],atol=1e-12)
            if np.any(np.diff(curve) < -1e-12):
                raise ValueError('An audited discovery curve decreased')
            result[method][name] = curve.tolist()
    return result


def recall_text(value):
    return f'{value:.6%}' if value is not None else '不适用'


def report():
    if not (REPORT/'audit/report.json').exists():
        raise ValueError('Finish the complete 320-run independent audit before the final report')
    audit = read_json(REPORT/'audit/report.json')
    negative = read_json(REPORT/'audit/negative_controls.json')
    index = read_json(REPORT/'run_index.json')
    if (audit['status'] != 'passed' or audit['runs'] != 320 or audit['queries'] != 40960
            or negative['status'] != 'passed' or len(negative['positive_controls']) != 8
            or len(negative['negative_controls']) != 8 or index['protocol'] != protocol()):
        raise ValueError('Final report needs the full frozen matrix and all corruption controls')
    result = gate(index['runs'])
    if result != read_json(REPORT/'gate.json'):
        raise ValueError('Final report admission differs from the frozen full-matrix result')
    rows = index['runs']+previous_rows()
    stats = result['summary']
    labels = {'physical_global':'Main: source posteriors + physical GP',
              'physical_local':'Local source weights','physical_mean':'Source means only',
              'physical_target':'Matched targets only','physical_adaptive':'Adaptive physical weight',
              'physical_margin':'Online clearance','physical_fsbo':'FSBO + physical GP (CUDA)',
              'fsbo_cuda':'FSBO (CUDA)','joint_global':'Previous joint transfer (global)',
              'fsbo':'FSBO (CPU)','pooled_ft':'Pooled-FT'}
    if result['strongest_baseline'] not in labels:
        labels[result['strongest_baseline']] = result['strongest_baseline']
    curves = aggregate_curves(rows,stats)
    checkpoints = {m:{str(b):{'unique_collisions':c['discovery_curve'][b-1],
                              'collision_recall':c['recall_curve'][b-1]} for b in (16,32,64,128)}
                   for m,c in curves.items()}
    budget_references,checkpoint_comparisons = {},[]
    for budget in (16,32,64,128):
        reference = max(STRONG,key=lambda m:(checkpoints[m][str(budget)]['collision_recall']
                        if checkpoints[m][str(budget)]['collision_recall'] is not None else -1,
                        checkpoints[m][str(budget)]['unique_collisions'],
                        float(np.mean(curves[m]['discovery_curve'][:budget])),m))
        budget_references[str(budget)] = reference
        if reference not in labels:
            labels[reference] = reference
        for metric,curve_name in (('unique_collisions','discovery_curve'),('collision_recall','recall_curve')):
            paired = [{**r,metric:r[curve_name][budget-1]} for r in rows
                      if r['method'] in (MAIN,reference) and r[curve_name][budget-1] is not None]
            if paired:
                checkpoint_comparisons.append({'budget':budget,
                    **paired_lineage_bootstrap(paired,MAIN,reference,metric)})

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    figure,axes = plt.subplots(1,2,figsize=(14,6.3))
    figure.subplots_adjust(left=.07,right=.98,top=.95,bottom=.29,wspace=.22)
    for method,label in labels.items():
        for axis,name in zip(axes,('discovery_curve','recall_curve')):
            if curves[method][name][0] is None:
                continue
            axis.plot(np.arange(1,129),curves[method][name],label=label,
                      linewidth=2.4 if method == MAIN else 1.3,
                      linestyle='--' if method == 'fsbo' else '-')
    axes[0].set(ylabel='Unique collision scenarios')
    axes[1].set(ylabel='Full-pool collision recall',ylim=(0,1))
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    for axis in axes:
        axis.set_xlabel('Paid queries (8 initial queries included)')
        axis.grid(alpha=.2)
    handles,names = axes[0].get_legend_handles_labels()
    figure.legend(handles,names,loc='lower center',bbox_to_anchor=(.5,.01),ncol=2,fontsize=8)
    for extension in ('png','svg','pdf'):
        figure.savefig(REPORT/f'curves.{extension}',dpi=180,bbox_inches='tight',pad_inches=.15)
    plt.close(figure)

    foundation = read_json(joint.REPORT/'costs.json')
    reused = {'path':str((joint.REPORT/'costs.json').relative_to(ROOT)),
              'existing_physical_pool_costs':foundation['existing_physical_pool_costs'],
              'source_posterior_construction_wall_s':foundation['source_posterior_construction_wall_s'],
              'correlated_model_training_wall_s':{k:v for k,v in foundation['model_training_wall_s_by_model'].items()
                                                   if k.startswith('correlated')},
              'fsbo_historical_training_elapsed_s':{kind:read_json(checkpoint_path(kind,'physical_fsbo').with_suffix('.json'))['elapsed_s']
                                                     for kind in ('cut_in','cut_out')}}
    costs = {'reused_foundation_costs':reused,
             'new_historical_training':False,'new_physical_calls':0,'by_method':{},'failed_attempts':[]}
    diagnosis_path = REPORT/'paid_prefix_diagnosis/report.json'
    if diagnosis_path.exists():
        diagnosis = read_json(diagnosis_path)
        costs['paid_prefix_prediction_diagnosis'] = {
            'path':str(diagnosis_path.relative_to(ROOT)),
            'prediction_reconstruction_wall_s':diagnosis['wall_s'],
            'paid_prefixes_reused':diagnosis['paid_prefixes'],
            'prediction_comparisons':diagnosis['prediction_comparisons'],
            'additional_selector_queries':diagnosis['additional_selector_queries'],
            'new_physical_calls':diagnosis['new_physical_calls']}
    influence_path = REPORT/'paid_prefix_diagnosis/collision_influence/report.json'
    if influence_path.exists():
        influence = read_json(influence_path)
        costs['paid_collision_influence_diagnosis'] = {
            'path':str(influence_path.relative_to(ROOT)),
            'prediction_reconstruction_wall_s':influence['wall_s'],
            'prediction_comparisons':influence['prediction_comparisons'],
            'additional_selector_queries':influence['additional_selector_queries'],
            'new_physical_calls':influence['new_physical_calls']}
    for method in METHODS:
        group = [r for r in index['runs'] if r['method'] == method]
        payloads = [read_json(ROOT/r['path']) for r in group]
        costs['by_method'][method] = {
            'runs':40,'equivalent_queries':5120,'new_physical_calls':0,
            'mean_selection_s':stats[method]['selection_s']['mean'],
            'mean_adaptation_s':stats[method]['adaptation_s']['mean'],
            'mean_learned_kernel_adaptation_s':float(np.mean([sum(q['adaptation_s'] if method == 'fsbo_cuda' else q['joint_adaptation_s']
                                                                  for q in p['result']['queries']) for p in payloads])),
            'mean_physical_adaptation_s':float(np.mean([sum(q.get('physical_adaptation_s',0) for q in p['result']['queries']) for p in payloads])),
            'sum_run_wall_s':sum(p['cost']['wall_s'] for p in payloads),
            'replay_equivalent_simulation_s':sum(record['feedback']['execution_s'] for p in payloads for record in p['query_ledger'])}
    for path in (REPORT/'failed_runs').rglob('*.jsonl'):
        attempts = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        costs['failed_attempts'].append({'path':str(path.relative_to(ROOT)),'attempts':len(attempts),
                                        'paid_attempts':sum(a['paid_attempts'] for a in attempts),
                                        'wall_s':sum(a['wall_s'] for a in attempts)})
    boundary = read_json(joint.REPORT/'curves.json')
    totals = {'prior_valid_search_runs':boundary['valid_total_search_runs'],'new_valid_search_runs':320,
              'valid_total_search_runs':boundary['valid_total_search_runs']+320,
              'valid_total_equivalent_queries':boundary['valid_total_equivalent_queries']+40960,
              'new_physical_calls':0,'failed_attempts_included_in_valid_totals':False}
    write_json(REPORT/'costs.json',costs)
    write_json(REPORT/'curves.json',{'curves':curves,'checkpoints':checkpoints,'summary':stats,
               'strongest_reference_by_budget':budget_references,'checkpoint_comparisons':checkpoint_comparisons,
               'complete_matrix_audited':True,'audit_sha256':hashlib.sha256((REPORT/'audit/report.json').read_bytes()).hexdigest(),
               **totals})
    lines = ['# 历史后验与独立物理GP：完整开发结果','',
             f"准入：{result['status']}；主方法：{MAIN}；最强参考：{result['strongest_baseline']}。",
             '全部320项完成并独立复核；每项128查询包含8次共同初始化。',
             '旧CPU与新CUDA的FSBO、同一物理GP的普通混合、同表示目标模型均参与参考选择。','',
             '| 方法 | 平均真实碰撞数 | 平均召回率 | 发现AUC |','|---|---:|---:|---:|']
    for method,s in stats.items():
        lines.append(f"| {method} | {s['unique_collisions']['mean']:.3f} | {recall_text(s['collision_recall']['mean'])} | {s['discovery_auc']['mean']:.3f} |")
    lines += ['','![数量与召回曲线](curves.png)','',
              '| 方法 | 查询预算 | 平均碰撞数 | 平均召回率 |','|---|---:|---:|---:|']
    for method,label in labels.items():
        for budget,m in checkpoints[method].items():
            lines.append(f"| {label} | {budget} | {m['unique_collisions']:.3f} | {recall_text(m['collision_recall'])} |")
    lines += ['','| 预算 | 该预算最强预登记参考 |','|---:|---|']
    for budget,reference in budget_references.items():
        lines.append(f'| {budget} | {reference} |')
    lines += ['','先在谱系内汇总再对谱系等权；零碰撞任务保留数量和成本，召回不适用。',
              '同库数量与召回不视为独立证据。配对分层区间、谱系差异和逐项门槛见gate.json。',
              '每个预算点均对预登记强集合中的最强者另算配对区间，见curves.json；低预算结果不替换128次主门槛。',
              '审计逐项核对完整真实碰撞分母、128条付费反馈与曲线，并独立重建960次选点。',
              '固定质量、局部权重、来源协方差、在线间距及同信息目标模型分别由对应消融解释。',
              '任何辅助版本更好也不改写原主方法结果。开发结论不代替独立目标、严格留家族与新场景验证。','',
              'costs.json分别记录继承的历史建库/基础训练、在线选例、两类适配和回放等效仿真成本。',
              '相同付费前缀诊断的预测重建计算单列，不计入在线选例耗时或有效搜索运行数。',
              '新实验不重新训练历史模型，新增物理调用为零；重叠运行时间不相加解释为总经过时间，硬件并行耗时不用于严格速度比较。',
              f"累计{totals['valid_total_search_runs']:,}项有效搜索、{totals['valid_total_equivalent_queries']:,}次等效查询，失败尝试另列。",'']
    (REPORT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    return result


if __name__ == '__main__':
    report()
