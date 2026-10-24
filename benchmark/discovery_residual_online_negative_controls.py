"""Reject in-memory corruptions of verified actual residual-search runs."""

import copy
import hashlib

import numpy as np

from .discovery_config import read_json, write_json
from .discovery_physical_mixture_audit import raw_banks
from .discovery_residual_online import MAIN, REPORT
from .discovery_residual_online_audit import audit, check_run
from .protocol import ROOT


def negative_controls():
    positive = audit('smoke')
    rows = read_json(REPORT/'smoke.json')['runs']
    row = next(r for r in rows if r['method'] == MAIN)
    candidates, feedback = raw_banks([row])[(row['target_id'], row['scenario'])]
    original = read_json(ROOT/row['path'])
    identities = {r['path']: hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest() for r in rows}
    mutations = []

    def changed(name, edit):
        output = copy.deepcopy(original)
        edit(output)
        mutations.append((name, output))

    changed('collision_label', lambda o: o['result']['queries'][0]['feedback'].__setitem__('collision', not o['result']['queries'][0]['feedback']['collision']))
    changed('full_pool_denominator', lambda o: o['metrics'].__setitem__('pool_unique_collisions', o['metrics']['pool_unique_collisions']+1))
    changed('duplicate_query', lambda o: o['result']['queries'][1].__setitem__('index', o['result']['queries'][0]['index']))
    changed('wrong_scale', lambda o: o['result'].__setitem__('beta', o['result']['beta']+.1))
    changed('incorrect_prequery_source_score', lambda o: o['result']['queries'][0]['prequery_group_log_likelihood'].__setitem__(
        0, o['result']['queries'][0]['prequery_group_log_likelihood'][0]+.25))
    changed('historical_query_account', lambda o: o['query_ledger'][0].__setitem__('account', 'history_certification'))
    changed('budget_not_complete', lambda o: o['result'].__setitem__('actual_queries', 127))
    changed('prior_mass', lambda o: o['result']['group_prior_weights'].__setitem__(0, .9))

    def change_probability(output, physical):
        q = output['result']['queries'][8]
        p = np.array(q['prequery_component_probabilities'])
        position = -1 if physical else 0
        p[position] += .02 if p[position] < .97 else -.02
        weights = np.array(q['prequery_group_weights'])
        group = float(weights @ p[:-1])
        q.update(prequery_component_probabilities=p.tolist(), prequery_physical_probability=float(p[-1]),
                 prequery_group_probability=group, prequery_probability=.9*group+.1*p[-1])
        likelihood = np.log(np.maximum(p[:-1] if q['feedback']['collision'] else 1-p[:-1], 1e-15))
        after = weights*np.exp(likelihood)
        q.update(prequery_group_log_likelihood=likelihood.tolist(), group_weights_after_query=(after/after.sum()).tolist())

    changed('internally_consistent_wrong_learned_gp', lambda o: change_probability(o, False))
    changed('internally_consistent_wrong_physical_gp', lambda o: change_probability(o, True))
    rejected = []
    for name, output in mutations:
        try:
            check_run(row, candidates, feedback, output)
        except (ValueError, AssertionError) as error:
            rejected.append({'name': name, 'error': str(error)[:240]})
        else:
            raise AssertionError(f'Residual audit accepted corruption: {name}')
    if any(hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != identity for path, identity in identities.items()):
        raise ValueError('An in-memory corruption test modified a saved actual run')
    result = {'status': 'passed', 'positive_controls': positive['runs'], 'rejected_corruptions': len(rejected),
              'negative_controls': rejected, 'original_run_identities_unchanged': True, 'new_physical_calls': 0}
    write_json(REPORT/'audit/negative_controls.json', result)
    return result


if __name__ == '__main__':
    negative_controls()
