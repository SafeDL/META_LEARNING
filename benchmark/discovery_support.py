"""Historical-only support-size diagnosis; preserves the failed original gate."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits
import torch

from methods.common.response import archive_fingerprints
from methods.gcrt.config import CONFIG
from methods.gcrt.consistency import episode_loss, load_encoder, train, training_contract
from methods.gcrt.train import episode

from .construction import load_construction
from .discovery_config import OUTPUT, SCENARIOS, read_json, write_json
from .discovery_consistency import SEEDS, model_location
from .protocol import ROOT


WIDE_CONFIG = replace(CONFIG, support_counts=(8, 16, 32, 64, 128))
MODELS = ROOT / 'results/gcrt/collision/discovery/support/models'
REPORT = OUTPUT / 'support'


def model_location_wide(mode, kind):
    return MODELS / mode / f'{kind}.pt'


def train_job(mode, kind):
    torch.set_num_threads(1)
    with threadpool_limits(1):
        history, development = load_construction(kind)
        return train(history, development, model_location_wide(mode, kind), mode, WIDE_CONFIG)


def validate_all(protocol):
    torch.set_num_threads(1)
    rows = []
    with threadpool_limits(1), torch.no_grad():
        for kind in SCENARIOS:
            history, development = load_construction(kind)
            for mode in ('source_global', 'source_local'):
                for schedule in ('short', 'wide'):
                    path = (model_location(mode, kind) if schedule == 'short'
                            else model_location_wide(mode, kind))
                    encoder, metadata = load_encoder(path)
                    if (metadata['specification']['history_fingerprints'] != archive_fingerprints(history)
                            or metadata['specification']['development_fingerprints'] != archive_fingerprints(development)):
                        raise ValueError('Support-size comparison requires the same historical/development archives')
                    encoder = encoder.to('cuda' if torch.cuda.is_available() else 'cpu')
                    for target in development:
                        for seed in SEEDS:
                            contexts, indices = episode(history, target, np.random.default_rng(seed), 128, 32)
                            _, metrics = episode_loss(encoder, contexts, target, indices, 128, mode, WIDE_CONFIG)
                            rows.append({'scenario': kind, 'mode': mode, 'schedule': schedule,
                                         'development_id': target.source_id, 'family': target.family,
                                         'seed': seed, 'checkpoint_step': metadata['step'], **metrics})
                    print('support_validation', kind, mode, schedule, len(rows), '/ 160', flush=True)
    if len(rows) != 160:
        raise ValueError('Support comparison requires 160 complete paired validation episodes')
    summaries = []
    for kind in SCENARIOS:
        for mode in ('source_global', 'source_local'):
            for schedule in ('short', 'wide'):
                selected = [r for r in rows if (r['scenario'], r['mode'], r['schedule']) == (kind,mode,schedule)]
                summaries.append({'scenario': kind, 'mode': mode, 'schedule': schedule,
                                  **{key: float(np.mean([r[key] for r in selected]))
                                     for key in ('collision_nll', 'brier', 'auxiliary_mse')}})
    result = {'status': 'complete', 'protocol': protocol, 'rows': rows, 'summary': summaries,
              'new_physical_calls': 0, 'original_pilot_gate_changed': False,
              'discovery_improvement_proven': False}
    write_json(REPORT / 'report.json', result)
    lines = ['# 元训练支持集规模诊断', '',
             '保持网络、1500次优化更新、学习率、种子和历史档案不变。',
             '训练支持集由4/8/16扩展到8/16/32/64/128；每次更新的监督点数与计算量因此增加，成本单独记录。',
             '配对验证均使用128个支持点和32个查询点；4个开发家族等权，5个固定种子，完整160项。',
             '只回答预测与支持集规模问题，不改变原门槛，也不证明碰撞发现收益。', '',
             '| 场景 | 模式 | 训练支持集 | 碰撞 NLL | Brier |', '|---|---|---|---:|---:|']
    for r in summaries:
        lines.append(f"| {r['scenario']} | {r['mode']} | {r['schedule']} | {r['collision_nll']:.6f} | {r['brier']:.6f} |")
    lines.append('')
    (REPORT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return result


def run():
    if read_json(OUTPUT/'consistency/search/report.json')['status'] != 'complete':
        raise ValueError('Complete the controlled matched-training search before expanding support sizes')
    protocol = {'purpose': 'Development diagnosis of the planned final support-size range',
                'original_pilot_gate_changed': False, 'opens_phases_c_d_e': False,
                'reference_config': asdict(CONFIG), 'wide_config': asdict(WIDE_CONFIG),
                'scenarios': list(SCENARIOS), 'validation_support': 128, 'validation_queries': 32,
                'seeds': list(SEEDS), 'training_contract': training_contract(),
                'runner_source': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    protocol = json.loads(json.dumps(protocol))
    path = REPORT / 'protocol.json'
    if path.exists() and read_json(path) != protocol:
        raise ValueError('Support experiment configuration changed; preserve previous evidence')
    write_json(path, protocol)
    # Each large-support backward pass retains a longer graph; bound concurrent GPU learners.
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(train_job, mode, kind)
                   for kind in SCENARIOS for mode in ('source_global', 'source_local')]
        for future in as_completed(futures):
            result = future.result()
            print('support_training_complete', result['specification']['kind'],
                  result['specification']['mode'], result['best_score'], flush=True)
    return validate_all(protocol)


if __name__ == '__main__':
    run()
