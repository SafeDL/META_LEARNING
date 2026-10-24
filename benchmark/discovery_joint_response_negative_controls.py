"""Check rejection of corrupted joint-search records without changing saved runs."""

import copy
import hashlib
from unittest.mock import patch

from . import discovery_joint_response_audit as audit
from .discovery_config import development_tasks, read_json, write_json
from .discovery_joint_response import REPORT, RUNS
from .executions import ExecutionBank
from .protocol import DATA_ROOT, ROOT


def check():
    task = development_tasks()[0]
    path = RUNS/'joint_local'/task['target_id']/task['scenario']/f"{task['algorithm_seed']}.json"
    original_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = read_json(path)
    row = {**task,'method':'joint_local','path':str(path.relative_to(ROOT)),**payload['metrics']}
    bank = ExecutionBank(DATA_ROOT/'physical/development'/task['scenario']/task['target_id'])
    feedbacks = [bank(s).feedback.to_dict() for s in bank.candidates]
    positive = audit.check_run(row,bank.candidates,feedbacks)
    checks = []
    for name in ('paid_margin','pool_denominator','component_weight','joint_log_likelihood',
                 'joint_covariance','ledger_purpose'):
        changed = copy.deepcopy(payload)
        query = changed['result']['queries'][0]
        if name == 'paid_margin':
            query['feedback']['safety']['min_clearance_m'] += 1
            changed['query_ledger'][0]['feedback'] = copy.deepcopy(query['feedback'])
        elif name == 'pool_denominator':
            changed['metrics']['pool_unique_collisions'] += 1
        elif name == 'component_weight':
            query['prequery_component_weights'][0] += .01
            query['prequery_component_weights'][-1] -= .01
        elif name == 'joint_log_likelihood':
            query['prequery_component_log_likelihood'][0] += 1
        elif name == 'joint_covariance':
            query['prequery_joint_covariances'][0][0][0] += .1
        else:
            changed['query_ledger'][8]['purpose'] = 'initial'

        def read_changed(requested):
            return changed if requested == path else read_json(requested)

        with patch.object(audit,'read_json',side_effect=read_changed):
            try:
                audit.check_run(row,bank.candidates,feedbacks)
            except (ValueError,AssertionError) as error:
                checks.append({'corruption':name,'rejected':True,'exception':type(error).__name__,
                               'reason':str(error)})
            else:
                raise AssertionError(f'Joint audit accepted corruption: {name}')
    if hashlib.sha256(path.read_bytes()).hexdigest() != original_digest:
        raise AssertionError('Negative-control checks changed the original run')
    result = {'status':'passed','valid_record_passed':positive,'negative_controls':checks,
              'original_run_unchanged':True,'original_run_sha256':original_digest,
              'corruption_applied_in_memory_only':True,'new_physical_calls':0,
              'additional_selector_queries':0}
    write_json(REPORT/'audit/negative_controls.json',result)
    return result


if __name__ == '__main__':
    check()
