"""Diagnose training/inference consistency without changing the failed pilot gate."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits
import torch

from methods.gcrt.config import CONFIG
from methods.gcrt.consistency import MODES, episode_loss, load_encoder, train, training_contract
from methods.gcrt.train import episode, load_encoder as load_reference

from .build_collision_models import model_path
from .construction import load_construction
from .discovery_config import OUTPUT, SCENARIOS, read_json, write_json
from .protocol import ROOT


MODELS = ROOT / 'results/gcrt/collision/discovery/consistency/models'
REPORT = OUTPUT / 'consistency'
SEEDS = (1104, 1201, 1207, 1213, 1223)


def model_location(mode, kind):
    if mode == 'shared_global':
        return model_path('history_response', 'all', kind)
    return MODELS / mode / f'{kind}.pt'


def prepare_protocol():
    gate = read_json(OUTPUT / 'pilot/gate.json')
    if gate['status'] != 'not_passed':
        raise ValueError('This diagnosis addresses the recorded failed development pilot')
    protocol = {'purpose': 'Development-only diagnosis of matched kernel/reliability training',
                'original_pilot_gate_changed': False, 'original_gate_status': 'not_passed',
                'opens_phases_c_d_e': False, 'network': 'Unchanged ResponseEncoder width=48 embedding=24',
                'config': asdict(CONFIG), 'trained_modes': ['source_global', 'source_local'],
                'reference_mode': 'shared_global (existing qualified historical checkpoint)',
                'scenarios': list(SCENARIOS), 'query_validation_seeds': list(SEEDS),
                'validation_support': max(CONFIG.support_counts), 'validation_queries': CONFIG.query_count,
                'training_contract': training_contract(),
                'runner_source': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    protocol = json.loads(json.dumps(protocol))
    path = REPORT / 'protocol.json'
    if path.exists() and read_json(path) != protocol:
        raise ValueError('Consistency diagnosis protocol changed; existing evidence must be preserved')
    write_json(path, protocol)
    return protocol


def train_job(mode, kind):
    torch.set_num_threads(1)
    with threadpool_limits(1):
        history, development = load_construction(kind)
        return train(history, development, model_location(mode, kind), mode)


def train_all():
    prepare_protocol()
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(train_job, mode, kind)
                   for kind in SCENARIOS for mode in ('source_global', 'source_local')]
        for future in as_completed(futures):
            result = future.result()
            print('training_complete', result['specification']['kind'],
                  result['specification']['mode'], 'best_score', result['best_score'], flush=True)
    return validate_all()


def validate_all():
    protocol = prepare_protocol()
    torch.set_num_threads(1)
    rows = []
    with threadpool_limits(1), torch.no_grad():
        for kind in SCENARIOS:
            history, development = load_construction(kind)
            for trained_mode in MODES:
                location = model_location(trained_mode, kind)
                encoder, metadata = (load_reference(location) if trained_mode == 'shared_global'
                                     else load_encoder(location))
                if set(metadata['history_ids']) != {a.source_id for a in history}:
                    raise ValueError('Consistency checkpoint uses a different allowed source set')
                encoder = encoder.to('cuda' if torch.cuda.is_available() else 'cpu')
                for inference_mode in MODES:
                    for target in development:
                        for seed in SEEDS:
                            contexts, indices = episode(history, target, np.random.default_rng(seed),
                                                         max(CONFIG.support_counts), CONFIG.query_count)
                            _, metrics = episode_loss(encoder, contexts, target, indices,
                                                       max(CONFIG.support_counts), inference_mode)
                            rows.append({'scenario': kind, 'training_mode': trained_mode,
                                         'inference_mode': inference_mode, 'development_id': target.source_id,
                                         'family': target.family, 'seed': seed,
                                         'checkpoint_step': metadata['step'], **metrics})
                print('prediction_validation', kind, trained_mode, len(rows), '/ 360', flush=True)
    if len(rows) != 360:
        raise ValueError('The full paired consistency matrix must contain 360 supervised validation episodes')
    summaries = []
    for kind in SCENARIOS:
        for trained_mode in MODES:
            for inference_mode in MODES:
                selected = [r for r in rows if r['scenario'] == kind and r['training_mode'] == trained_mode
                            and r['inference_mode'] == inference_mode]
                families = sorted({r['family'] for r in selected})
                summaries.append({'scenario': kind, 'training_mode': trained_mode,
                                  'inference_mode': inference_mode,
                                  **{metric: float(np.mean([np.mean([r[metric] for r in selected if r['family'] == f])
                                                           for f in families]))
                                     for metric in ('collision_nll', 'brier', 'auxiliary_mse')}})
    result = {'status': 'complete', 'protocol': protocol, 'rows': rows, 'summary': summaries,
              'new_physical_calls': 0, 'original_pilot_gate_changed': False,
              'discovery_improvement_proven': False,
              'interpretation': 'Paired supervised development prediction diagnosis; not an active-search success claim'}
    write_json(REPORT / 'report.json', result)
    lines = ['# 核与来源权重的训练/推断一致性诊断', '',
             '网络规模、历史/开发角色、1500次更新、学习率、训练支持集和辅助损失均与原训练保持一致。',
             '原共享核检查点作为对照；新增来源独立核的全局和局部权重匹配训练。',
             '比较360项配对监督验证：支持集16、查询集32，5个固定种子；其中1104曾用于开发检查点选择。',
             '该结果仅回答预测与训练一致性问题，不替代主动搜索实验，不改变原阶段B门槛，也不启动C/D/E。', '',
             '| 场景 | 训练模式 | 推断模式 | 碰撞 NLL | Brier |', '|---|---|---|---:|---:|']
    for row in summaries:
        lines.append(f"| {row['scenario']} | {row['training_mode']} | {row['inference_mode']} | {row['collision_nll']:.6f} | {row['brier']:.6f} |")
    lines.extend(['', 'NLL和Brier越小越好；全部推断组合、零碰撞开发对象与种子都保留。', ''])
    (REPORT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('train', 'validate'))
    stage = parser.parse_args().stage
    if stage == 'train':
        train_all()
    else:
        validate_all()


if __name__ == '__main__':
    main()
