"""Reproduce the historical-frequency and latent-variance initialization check."""

from threadpoolctl import threadpool_limits
import torch

from methods.baselines.fsbo.method import load as load_fsbo
from methods.gcrt.joint_training import load_model
from methods.gcrt.phase_calibration import calibrated_phase_bases

from .construction import load_construction
from .discovery_config import OUTPUT, read_json, write_json
from .discovery_phase_response_diagnosis import recorded_observations
from .discovery_physical_mixture import MAIN, REPORT, checkpoint_path, protocol


def diagnose():
    if protocol() != read_json(REPORT/'protocol.json'):
        raise ValueError('The original online experiment must remain unchanged')
    data = read_json(REPORT/'phase_diagnosis/report.json')
    original = read_json(OUTPUT/'phase_response/prior_initialization_check.json')
    rows = []
    torch.set_num_threads(1)
    with threadpool_limits(1):
        for kind in ('cut_in', 'cut_out'):
            history, development = load_construction(kind)
            observations = recorded_observations(data['records'], kind, history)
            for initialization in ('joint', 'fsbo'):
                base = (load_model(checkpoint_path(kind, MAIN))[0] if initialization == 'joint' else
                        load_fsbo(checkpoint_path(kind, 'physical_fsbo'))[0]).to('cuda')
                _, checks = calibrated_phase_bases(base, history, observations, development[0].x)
                for row in checks:
                    old = next(r for r in original['rows'] if (r['scenario'], r['initialization'], r['phase'])
                               == (kind, initialization, row['phase']))
                    if any(abs(old[k]-row[k]) > 1e-9 for k in row) or row['root_error'] > 1e-10:
                        raise ValueError('Calibration helper differs from the independent initialization check')
                    rows.append({'scenario': kind, 'initialization': initialization, **row})
    result = {'status': 'passed', 'initializations_checked': len(rows), 'rows': rows,
        'target_labels_used': False, 'additional_selector_queries': 0, 'new_physical_calls': 0,
        'original_predictions_changed': False,
        'scope': 'initial target-component marginal probability only; no calibration guarantee for transferred or adapted predictions'}
    destination = OUTPUT/'phase_calibration'
    write_json(destination/'prior_check.json', result)
    lines = ['# 阶段GP初始化概率核查', '',
        '只使用历史阶段频率与公开候选特征，保持两个冻结模型的核不变。未读取目标碰撞标签，没有新增查询。',
        '对潜在均值直接取频率的logit，与对Gaussian潜变量积分后的边际概率并不相同。', '',
        '| 场景 | 初始化 | 阶段 | 历史风险集频率 | 原初始边际概率 | 匹配后的潜在均值 |', '|---|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['scenario']} | {r['initialization']} | {r['phase']} | {r['observed_history_hazard_frequency']:.6f} | {r['registered_marginal_probability_mean']:.6f} | {r['marginally_matched_latent_mean']:.6f} |")
    lines += ['', '匹配均值通过一维求根使初始独立目标分量的平均边际概率等于历史风险集频率。12项与先前独立计算一致。',
              '这只校正初始化的一项属性；来源协方差叠加、后验条件化与适配后不保证仍然校准。',
              '此前240项阶段预测保持原样。调整后的完整同信息对照须在单独协议中执行，不能把初始化核查当成发现收益。']
    (destination/'prior_check.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    diagnose()
