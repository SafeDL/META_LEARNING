"""Raw-pool and independent prequential audits for development search diagnoses."""

import argparse
import hashlib

import numpy as np
import torch

from methods.gcrt.consistency import MODES, load_encoder, training_contract
from methods.gcrt.discovery import discovery_posterior

from .construction import load_construction
from .discovery import experiment_contract
from .discovery_audit import reconstruct_evidence
from .discovery_config import OUTPUT, development_tasks, read_json, write_json
from .discovery_consistency import model_location
from .discovery_support import model_location_wide
from .executions import ExecutionBank
from .initialization import initial_indices
from .protocol import DATA_ROOT, ROOT
from .scenarios import dedup_key, features
from .schema import QueryFeedback


def audit(study='matched'):
    torch.set_num_threads(1)
    settings = {'matched': ('consistency', 'matched', model_location, 'discovery_consistency_search.py'),
                'wide': ('support', 'wide', model_location_wide, 'discovery_support_search.py')}
    directory, prefix, checkpoint_location, runner = settings[study]
    path = OUTPUT / directory / 'search/report.json'
    report = read_json(path)
    expected_contract = {'pilot': experiment_contract(), 'training': training_contract(),
                         'search_runner': hashlib.sha256((ROOT/'benchmark'/runner).read_bytes()).hexdigest()}
    if report['contract'] != expected_contract or len(report['runs']) != 80:
        raise ValueError('Matched search source or complete run count differs')
    expected = {(t['target_id'], t['scenario'], t['algorithm_seed'], f'{prefix}_{mode}')
                for t in development_tasks() for mode in ('source_global','source_local')}
    actual = [(r['target_id'], r['scenario'], r['algorithm_seed'], r['method']) for r in report['runs']]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Diagnostic audit requires the exact registered development matrix')
    raw, histories = {}, {}
    checked = []
    for row in report['runs']:
        key = row['target_id'], row['scenario']
        if key not in raw:
            bank = ExecutionBank(DATA_ROOT/'physical/development'/row['scenario']/row['target_id'])
            raw[key] = bank.candidates, [bank(s).feedback.to_dict() for s in bank.candidates]
        candidates, feedbacks = raw[key]
        kind = row['scenario']
        if kind not in histories:
            histories[kind] = load_construction(kind)[0]
        mode = row['method'].removeprefix(f'{prefix}_')
        model_file = checkpoint_location(mode, kind)
        encoder, _ = load_encoder(model_file)
        payload = read_json(ROOT / row['path'])
        if (payload['specification']['checkpoint_sha256'] != hashlib.sha256(model_file.read_bytes()).hexdigest()
                or payload['specification']['contract'] != expected_contract):
            raise ValueError('Saved search differs from its matched checkpoint or source')
        queries, ledger = payload['result']['queries'], payload['query_ledger']
        selected = [q['index'] for q in queries]
        if (len(queries) != 128 or len(set(selected)) != 128 or len(ledger) != 128
                or selected[:8] != initial_indices(512,8,row['algorithm_seed'])
                or payload['result']['selected_indices'] != selected):
            raise ValueError('Matched search budget or common initialization differs')
        found, curve = set(), []
        failures = {dedup_key(s) for s,f in zip(candidates,feedbacks) if f['collision']}
        for n, (q, record) in enumerate(zip(queries,ledger),1):
            if (q['query'] != n or record['call'] != n or record['status'] != 'complete'
                    or record['index'] != q['index'] or record['feedback'] != q['feedback']
                    or record['purpose'] != ('initial' if n <= 8 else 'target')
                    or QueryFeedback.from_dict(q['feedback']).to_dict() != feedbacks[q['index']]):
                raise ValueError('Matched query differs from original feedback or paid ledger')
            if q['feedback']['collision']:
                found.add(dedup_key(candidates[q['index']]))
            curve.append(len(found))
        recalls = [n/len(failures) if failures else None for n in curve]
        metrics = payload['metrics']
        if (metrics['discovery_curve'] != curve or metrics['recall_curve'] != recalls
                or metrics['pool_unique_collisions'] != len(failures)
                or metrics['unique_collisions'] != curve[-1] or metrics['collision_recall'] != recalls[-1]
                or any(row[key] != value for key,value in metrics.items())):
            raise ValueError('Matched metrics differ from the complete original pool or run index')
        np.testing.assert_allclose(metrics['discovery_auc'], np.mean(curve))
        for b in (16,32,64,128):
            if metrics['checkpoints'][str(b)] != {'unique_collisions':curve[b-1], 'collision_recall':recalls[b-1]}:
                raise ValueError('Matched search checkpoint differs')
        independent, bandwidth = MODES[mode]
        model = discovery_posterior(encoder, features(candidates), histories[kind],
                                    independent=independent, bandwidth=bandwidth)
        trace = reconstruct_evidence(queries, model, features(candidates))
        write_json(path.parent/'audit/evidence'/mode/row['target_id']/kind/
                   f"{row['algorithm_seed']}.json", trace)
        if payload['cost']['new_physical_calls'] != 0:
            raise ValueError('Matched replay must not hide physical calls')
        checked.append(row['path'])
        if len(checked) % 8 == 0:
            print(study,'search_audit',len(checked),'/ 80',flush=True)
    result = {'status':'passed', 'runs':80, 'queries':10240, 'raw_banks':len(raw),
              'raw_pool_executions':len(raw)*512, 'source_evidence_reconstructions':80,
              'original_pilot_gate_changed':False, 'new_physical_calls':0,
              'checked_paths':checked, 'contract':expected_contract, 'study':study}
    write_json(path.parent/'audit/report.json',result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study', nargs='?', default='matched', choices=('matched','wide'))
    audit(parser.parse_args().study)
