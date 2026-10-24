"""Diagnose historical representations and raw-feature predictions without final targets."""

import numpy as np
import torch

from methods.common.response import source_weights
from methods.gcrt.train import load_encoder

from .build_collision_models import model_path
from .construction import load_construction
from .discovery_config import OUTPUT, SCENARIOS, write_json


def diagnose_representation():
    torch.set_num_threads(1)
    reports = {}
    for kind in SCENARIOS:
        history, development = load_construction(kind)
        x = development[0].x
        if any(not np.array_equal(a.x, x) for a in development):
            raise ValueError('Development geometries must match before comparing source kernels')
        encoder, _ = load_encoder(model_path('history_response', 'all', kind))
        with torch.no_grad():
            encoded = encoder.encode(x, history).double()
            physical = encoder.physical_kernel(x).double().numpy()
            scale = encoder.log_response_scale.exp().double().square().item()
            response = scale * (encoded @ encoded.transpose(-1, -2)).numpy()
        pooled = np.einsum('s,sij->ij', source_weights(history), response)
        common = physical + pooled
        denominator = np.linalg.norm(common)
        reports[kind] = {
            'candidates': len(x), 'inputs': 'Development coordinates only; historical contexts; frozen encoder',
            'physical_frobenius_norm': float(np.linalg.norm(physical)),
            'response_frobenius_norm': float(np.linalg.norm(pooled)),
            'response_to_physical_norm_ratio': float(np.linalg.norm(pooled)/np.linalg.norm(physical)),
            'source_relative_kernel_change': {
                a.source_id: float(np.linalg.norm(response[i]-pooled)/denominator)
                for i, a in enumerate(history)},
            'same_candidate_source_cosine_similarity': {
                f'{left.source_id}/{right.source_id}': float((encoded[i]*encoded[j]).sum(-1).mean())
                for i, left in enumerate(history) for j, right in enumerate(history) if j > i}}
    output = {'status': 'complete', 'new_physical_calls': 0, 'scenarios': reports,
              'interpretation': 'Kernel geometry diagnosis only; does not establish cause or target discovery improvement'}
    write_json(OUTPUT / 'representation/report.json', output)
    lines = ['# 冻结编码器的来源关联差异', '',
             '仅使用开发场景坐标与历史上下文，不拟合新模型、不读取正式目标，也不使用开发碰撞标签。', '',
             '| 场景 | 响应核/物理核范数比 | 来源独立核相对共享核变化范围 |', '|---|---:|---:|']
    for kind, data in reports.items():
        differences = list(data['source_relative_kernel_change'].values())
        lines.append(f"| {kind} | {data['response_to_physical_norm_ratio']:.4f} | {min(differences):.4%}–{max(differences):.4%} |")
    lines.extend(['', '该诊断只量化现有表示中的来源差异；不能独自证明编码器塌缩或收益差距的因果来源。', ''])
    destination = OUTPUT / 'representation/report.md'
    destination.write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines), flush=True)
    return output


def diagnose_paid_representation():
    import copy
    import time

    from threadpoolctl import threadpool_limits

    from methods.common.classification import logistic_moments
    from methods.common.gp_classifier import ClassificationKernel, collision_probabilities, fit_target
    from methods.common.response import archive_fingerprints
    from methods.gcrt.joint_prediction import marginal_posterior
    from methods.gcrt.joint_response import JointResponseKernel, fit_joint, margin_values
    from methods.gcrt.joint_training import load_model

    from .discovery_config import development_tasks, read_json
    from .discovery_joint_response import REPORT, RUNS, model_path as joint_model_path, protocol
    from .executions import ExecutionBank
    from .protocol import DATA_ROOT
    from .scenarios import features
    from .schema import QueryFeedback

    frozen = read_json(REPORT/'protocol.json')
    if frozen != protocol():
        raise ValueError('Paid-prefix diagnosis requires the unchanged joint search protocol')
    destination = REPORT/'physical_feature_diagnosis'
    specification = read_json(destination/'protocol.json')
    if (specification['device'] != 'cuda' or not specification['no_hyperparameter_selection']
            or specification['additional_selector_queries'] != 0):
        raise ValueError('Use the predeclared prediction diagnosis without extra queries')
    torch.set_num_threads(1)
    torch.manual_seed(specification['seed'])
    banks,bases,truths = {},{},{}
    rows = []
    predictions = {}
    with threadpool_limits(1):
        for task in development_tasks():
            kind,target,seed = task['scenario'],task['target_id'],task['algorithm_seed']
            if kind not in bases:
                history,development = load_construction(kind)
                base,metadata = load_model(joint_model_path(kind))
                if (metadata['specification']['history_fingerprints'] != archive_fingerprints(history)
                        or metadata['specification']['development_fingerprints'] != archive_fingerprints(development)):
                    raise ValueError('Prediction diagnosis archives differ from the role-checked model')
                bases[kind] = base.to('cuda')
            key = kind,target
            if key not in banks:
                banks[key] = ExecutionBank(DATA_ROOT/'physical/development'/kind/target)
                truths[key] = [banks[key](s).feedback.to_dict() for s in banks[key].candidates]
            bank,feedback = banks[key],truths[key]
            saved = read_json(RUNS/'joint_local'/target/kind/f'{seed}.json')
            if saved['specification']['protocol'] != frozen or saved['result']['actual_queries'] != 128:
                raise ValueError('Prediction diagnosis requires a complete frozen main-method run')
            queries = saved['result']['queries'][:127]
            paid = [q['index'] for q in queries]
            if len(set(paid)) != 127 or any(QueryFeedback.from_dict(q['feedback']).to_dict() != feedback[q['index']] for q in queries):
                raise ValueError('Paid-prefix feedback differs from its original execution')
            labels = np.array([q['feedback']['collision'] for q in queries],dtype=bool)
            clearance = [np.inf if q['feedback']['safety']['min_clearance_m'] is None else
                         q['feedback']['safety']['min_clearance_m'] for q in queries]
            available = sorted(set(range(512))-set(paid))
            x = features(bank.candidates)
            truth = np.array([f['collision'] for f in feedback])
            for name,config in specification['models'].items():
                if name == 'deep_joint':
                    model = copy.deepcopy(bases[kind])
                elif name == 'physical_joint':
                    model = JointResponseKernel(x.shape[1])
                    model.embedding = torch.nn.Identity()
                    model.deep = False
                    with torch.no_grad():
                        model.mean.zero_()
                    model = model.to('cuda')
                else:
                    model = ClassificationKernel(x.shape[1]).to('cuda')
                started = time.perf_counter()
                if name == 'physical_binary':
                    fit_target(model,x[paid],labels,steps=config['steps'],learning_rate=config['learning_rate'])
                    with torch.no_grad():
                        probability = collision_probabilities(model,x[paid],labels,x).cpu().numpy()
                else:
                    fit_joint(model,x[paid],labels,clearance,steps=config['steps'],learning_rate=config['learning_rate'])
                    with torch.no_grad():
                        mean,variance,_ = marginal_posterior(model.joint_mean(512)[None],
                            model.joint_covariance(x)[None],paid,labels,margin_values(clearance))
                        probability = logistic_moments(mean[0,:512].cpu().numpy(),variance[0,:512].cpu().numpy())[0]
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-started
                p = np.clip(probability[available],1e-8,1-1e-8)
                y = truth[available]
                ranking = sorted(available,key=lambda i:(-probability[i],bank.candidates[i].scenario_id))
                positives = [{'index':int(i),'probability':float(probability[i]),'rank':ranking.index(i)+1}
                             for i in available if truth[i]]
                rows.append({**task,'model':name,'support_count':127,'evaluation_count':385,
                             'collision_nll':float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p))),
                             'brier':float(np.square(p-y).mean()),'remaining_collisions':positives,
                             'parameters':sum(a.numel() for a in model.parameters()),'wall_s':elapsed})
                predictions[f'{target}/{kind}/{seed}/{name}'] = probability
            print('paid_representation',len(rows)//3,'/ 40',target,kind,seed,flush=True)
    summary = {}
    for name in specification['models']:
        group = [r for r in rows if r['model'] == name]
        lineages = sorted({r['lineage'] for r in group})
        summary[name] = {metric:float(np.mean([np.mean([r[metric] for r in group if r['lineage'] == lineage])
                                               for lineage in lineages])) for metric in ('collision_nll','brier')}
    np.savez_compressed(destination/'probabilities.npz',**predictions)
    result = {'status':'complete','protocol':specification,'search_protocol':frozen,'summary':summary,
              'comparisons':120,'paid_prefixes':40,'rows':rows,'additional_selector_queries':0,
              'new_physical_calls':0,'interpretation':'Matched retrospective predictions on paid prefixes, not new online discoveries or independent validation'}
    write_json(destination/'report.json',result)
    return result


def report_paid_representation():
    from .discovery_config import read_json
    from .discovery_joint_response import REPORT

    destination = REPORT/'physical_feature_diagnosis'
    data = read_json(destination/'report.json')
    numerical = read_json(destination/'numerical_checks.json')
    if data['status'] != 'complete' or len(data['rows']) != 120 or numerical['status'] != 'passed':
        raise ValueError('Finish all matched paid-prefix predictions and numerical checks first')
    lines = ['# 相同付费前缀下的物理特征诊断','',
             '40个既定开发任务，各使用联合主方法的前127次已付费反馈。三种预测器共120项比较。',
             '完整标签只在评估端计算剩余385点的误差和真实碰撞排名；没有新增选例或物理调用。',
             '这不是新的128查询搜索结果。','',
             '| 预测器 | 谱系等权碰撞NLL | Brier |','|---|---:|---:|']
    for name,stats in data['summary'].items():
        lines.append(f"| {name} | {stats['collision_nll']:.6f} | {stats['brier']:.6f} |")
    lines += ['',
              '物理联合GP使用原始归一化场景特征，无历史拟合，6个标量参数；物理二值GP为原3参数分类核。',
              '深核联合GP使用冻结历史模型。联合模型使用相同的付费碰撞/间距；物理二值GP只用碰撞。',
              '配置在计算前登记，未按结果搜索参数。','',
              '| PPO切出漏检点 | 预测器 | 碰撞概率范围 | 剩余排名范围 |','|---|---|---:|---:|']
    for index in (90,386):
        for model in data['summary']:
            values = [f for row in data['rows'] if row['target_id'] == 'development_ppo'
                      and row['scenario'] == 'cut_out' and row['model'] == model
                      for f in row['remaining_collisions'] if f['index'] == index]
            lines.append(f"| {index} | {model} | {min(v['probability'] for v in values):.6f}–{max(v['probability'] for v in values):.6f} | {min(v['rank'] for v in values)}–{max(v['rank'] for v in values)} |")
    lines += ['','![相同反馈下的概率与排名](predictions.png)','',
              '深核的全剩余库平均NLL/Brier更好，物理核对两项PPO漏检的排名更靠前。',
              '这支持检查互补，不能支持整体替换历史模型，也不能把排名改善解释成已经发现。',
              '实际选例会改变后续反馈，在线收益仍需完整配对搜索验证。','',
              '两个物理模型在PPO切出种子11的全部512点预测经独立NumPy重建，',
              '最大绝对差均小于5.56e-16，见numerical_checks.json。PNG已目视检查，另有SVG/PDF。','',
              '本分析为开发后诊断；正式目标未参与。完整标签没有发送给选择器。','']
    (destination/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    return data['summary']


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('scope',nargs='?',default='frozen',choices=('frozen','paid','paid_report'))
    {'frozen':diagnose_representation,'paid':diagnose_paid_representation,
     'paid_report':report_paid_representation}[parser.parse_args().scope]()
