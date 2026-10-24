"""Corrupt in-memory physical-mixture records and require audit rejection."""

import copy
import hashlib
from unittest.mock import patch

from . import discovery_physical_mixture_audit as audit
from .discovery_config import development_tasks, read_json, write_json
from .discovery_physical_mixture import MAIN, METHODS, REPORT, RUNS
from .executions import ExecutionBank
from .protocol import DATA_ROOT, ROOT


def check():
    task = development_tasks()[0]
    bank = ExecutionBank(DATA_ROOT/'physical/development'/task['scenario']/task['target_id'])
    feedbacks = [bank(s).feedback.to_dict() for s in bank.candidates]
    positives = []
    for method in METHODS:
        path = RUNS/method/task['target_id']/task['scenario']/f"{task['algorithm_seed']}.json"
        payload = read_json(path)
        row = {**task,'method':method,'path':str(path.relative_to(ROOT)),**payload['metrics']}
        positives.append(audit.check_run(row,bank.candidates,feedbacks))
    path = RUNS/MAIN/task['target_id']/task['scenario']/f"{task['algorithm_seed']}.json"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = read_json(path)
    row = {**task,'method':MAIN,'path':str(path.relative_to(ROOT)),**payload['metrics']}
    checks = []
    for name in ('paid_collision','paid_margin','pool_denominator','component_weight','physical_probability',
                 'joint_log_likelihood','outer_mass','ledger_purpose'):
        changed = copy.deepcopy(payload)
        query = changed['result']['queries'][0]
        if name == 'paid_collision':
            query['feedback']['collision'] = not query['feedback']['collision']
            changed['query_ledger'][0]['feedback'] = copy.deepcopy(query['feedback'])
        elif name == 'paid_margin':
            query['feedback']['safety']['min_clearance_m'] += 1
            changed['query_ledger'][0]['feedback'] = copy.deepcopy(query['feedback'])
        elif name == 'pool_denominator':
            changed['metrics']['pool_unique_collisions'] += 1
        elif name == 'component_weight':
            query['prequery_component_weights'][0] += .01
            query['prequery_component_weights'][-1] -= .01
        elif name == 'physical_probability':
            query['prequery_physical_probability'] += .01
        elif name == 'joint_log_likelihood':
            query['prequery_joint_log_likelihood'][0] += 1
        elif name == 'outer_mass':
            query['prequery_outer_weights'] = [.8,.2]
        else:
            changed['query_ledger'][8]['purpose'] = 'initial'

        def read_changed(requested):
            return changed if requested == path else read_json(requested)

        with patch.object(audit,'read_json',side_effect=read_changed):
            try:
                audit.check_run(row,bank.candidates,feedbacks)
            except (ValueError,AssertionError) as error:
                checks.append({'corruption':name,'rejected':True,'exception':type(error).__name__,'reason':str(error)})
            else:
                raise AssertionError(f'Physical-mixture audit accepted corruption: {name}')
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise AssertionError('Negative controls changed the original physical-mixture run')
    result = {'status':'passed','positive_controls':positives,'negative_controls':checks,
              'original_run_unchanged':True,'corruption_applied_in_memory_only':True,
              'new_physical_calls':0,'additional_selector_queries':0}
    write_json(REPORT/'audit/negative_controls.json',result)
    return result


if __name__ == '__main__':
    check()
