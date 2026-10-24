"""Original feedback audit with independent NumPy Laplace prediction checks."""

import argparse
import copy
import hashlib

import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits
import torch

from methods.baselines.fsbo.method import load as load_fsbo
from methods.common.classification import logistic_moments
from methods.common.gp_classifier import fit_target
from methods.gcrt.response_adaptation import ResponseKernel, response_coordinates

from .build_collision_models import model_path
from .construction import load_construction
from .discovery import method_summary
from .discovery_config import STRONG_BASELINES, read_json, write_json
from .discovery_recovery import REPORT, protocol, recovery_gate
from .executions import ExecutionBank
from .initialization import initial_indices
from .protocol import DATA_ROOT, ROOT
from .scenarios import dedup_key, features
from .schema import QueryFeedback


def numpy_probabilities(model, x, selected, labels):
    with torch.no_grad():
        prior = model.covariance(x[selected]).numpy()
        cross = model.covariance(x[selected],x).numpy()
        diagonal = model.covariance(x).diagonal().numpy()
        mean = float(model.mean)
    # Match deployment's feature batches: float32 linear layers can round
    # differently when support features are sliced from one large batch.
    prior = prior + 1e-5 * np.eye(len(selected))
    eye = np.eye(len(selected))
    labels = np.asarray(labels,dtype=float)
    f = np.full(len(selected),mean)
    for _ in range(10):
        probability = expit(f)
        weight = np.maximum(probability * (1 - probability),1e-9)
        root = np.sqrt(weight)
        factor = np.linalg.cholesky(eye + root[:,None] * prior * root[None,:])
        b = weight * (f - mean) + labels - probability
        solved = np.linalg.solve(factor.T,np.linalg.solve(factor,root * (prior @ b)))
        f = mean + prior @ (b - root * solved)
    probability = expit(f)
    root = np.sqrt(np.maximum(probability * (1 - probability),1e-9))
    factor = np.linalg.cholesky(eye + root[:,None] * prior * root[None,:])
    alpha = np.linalg.solve(prior,f - mean)
    posterior_mean = mean + cross.T @ alpha
    solved = np.linalg.solve(factor,root[:,None] * cross)
    variance = np.maximum(diagonal - np.square(solved).sum(0),1e-10)
    return logistic_moments(posterior_mean,variance)[0]


def audit(study='recovery'):
    torch.set_num_threads(1)
    directory, expected_protocol, coordinates = REPORT,protocol,response_coordinates
    if study == 'safety':
        from methods.gcrt.safety_response import safety_coordinates
        from .discovery_safety import REPORT as safety_directory, protocol as safety_protocol
        directory,expected_protocol,coordinates = safety_directory,safety_protocol,safety_coordinates
    elif study == 'geometry':
        from .discovery_geometry_search import REPORT as geometry_directory, protocol as geometry_protocol
        directory,expected_protocol = geometry_directory,geometry_protocol
    index = read_json(directory/'run_index.json')
    if index['protocol'] != expected_protocol():
        raise ValueError('Recovery audit requires the frozen source and configuration')
    baselines = [r for r in read_json(REPORT.parent/'pilot/run_index.json')['runs']
                 if r['method'] in STRONG_BASELINES]
    if study == 'recovery':
        if recovery_gate(index['runs'],baselines) != read_json(directory/'gate.json'):
            raise ValueError('Recovery gate differs from the complete registered matrix')
    else:
        expected = {(t['target_id'],t['scenario'],t['algorithm_seed'],method)
                    for t in index['protocol']['tasks'] for method in index['protocol']['methods']}
        actual = [(r['target_id'],r['scenario'],r['algorithm_seed'],r['method']) for r in index['runs']]
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError('Safety audit requires the complete 80-run matrix')
        control_directory = REPORT if study == 'safety' else REPORT.parent/'safety_response'
        control_method = 'response_fixed' if study == 'safety' else 'safety_fixed'
        control_rows = [r for r in read_json(control_directory/'run_index.json')['runs']
                        if r['method'] == control_method]
        stats = method_summary(index['runs'] + baselines + control_rows)
        gate = read_json(directory/'gate.json')
        if gate['summary'] != stats:
            raise ValueError('Safety gate summary differs from the registered result rows')
        main_method = 'safety_fixed' if study == 'safety' else 'geometry_response'
        main,ref,control = (stats[m] for m in (main_method,gate['strongest_baseline'],control_method))
        recall,reference = main['collision_recall']['mean'],ref['collision_recall']['mean']
        reduction = (recall-reference)/(1-reference) if reference < 1-1e-12 else None
        gains = {k:v-ref['collision_recall']['by_lineage'][k]
                 for k,v in main['collision_recall']['by_lineage'].items()}
        conditions = {'remaining_misses_reduced_by_half':reduction is not None and reduction >= .5-1e-12,
                      'collision_count_not_lower':main['unique_collisions']['mean'] >= ref['unique_collisions']['mean']-1e-12,
                      'two_positive_lineages':sum(v > 1e-12 for v in gains.values()) >= 2,
                      'safety_recall_exceeds_collision_only':recall > control['collision_recall']['mean']+1e-12,
                      'safety_count_not_lower':main['unique_collisions']['mean'] >= control['unique_collisions']['mean']-1e-12}
        if study == 'geometry':
            conditions['structured_recall_exceeds_raw'] = conditions.pop('safety_recall_exceeds_collision_only')
            conditions['structured_count_not_lower'] = conditions.pop('safety_count_not_lower')
        if (gate['remaining_miss_reduction'] != reduction or gate['lineage_recall_gains'] != gains
                or gate['conditions'] != conditions
                or gate['status'] != ('passed' if all(conditions.values()) else 'not_passed')):
            raise ValueError('Safety admission gate differs from independent metric recomputation')
    raw, public = {}, {}
    checks = []
    with threadpool_limits(1):
        for row in index['runs']:
            key = row['target_id'],row['scenario']
            if key not in raw:
                bank = ExecutionBank(DATA_ROOT/'physical/development'/row['scenario']/row['target_id'])
                raw[key] = bank.candidates,[bank(s).feedback.to_dict() for s in bank.candidates]
            candidates,feedbacks = raw[key]
            kind = row['scenario']
            checkpoint = model_path('fsbo','all',kind)
            public_key = (*key,row['method'])
            if public_key not in public:
                if study == 'geometry':
                    from methods.gcrt.response_geometry import kinematic_features, response_fields
                    responses = (response_fields(load_construction(kind)[0],candidates)
                                 if row['method'] == 'geometry_response' else
                                 kinematic_features(candidates)[:,features(candidates).shape[1]:])
                else:
                    responses = coordinates(load_construction(kind)[0],features(candidates))
                public[public_key] = responses
            responses = public[public_key]
            payload = read_json(ROOT/row['path'])
            specification = payload['specification']
            if (specification['protocol'] != expected_protocol() or specification['task'] !=
                    {k:row[k] for k in index['protocol']['tasks'][0]}
                    or specification['checkpoint_sha256'] != hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                    or specification['candidate_ids'] != [s.scenario_id for s in candidates]):
                raise ValueError('Recovery run differs from the frozen public inputs')
            np.testing.assert_allclose(responses,payload['history_response_coordinates'],atol=0,rtol=0)
            queries,ledger = payload['result']['queries'],payload['query_ledger']
            selected = [q['index'] for q in queries]
            if (len(queries) != 128 or len(set(selected)) != 128 or len(ledger) != 128
                    or selected[:8] != initial_indices(512,8,row['algorithm_seed'])
                    or payload['result']['selected_indices'] != selected):
                raise ValueError('Recovery budget or common initialization differs')
            failures = {dedup_key(s) for s,f in zip(candidates,feedbacks) if f['collision']}
            found,curve = set(),[]
            for n,(query,record) in enumerate(zip(queries,ledger),1):
                if (query['query'] != n or record['call'] != n or record['status'] != 'complete'
                        or record['index'] != query['index'] or record['feedback'] != query['feedback']
                        or record['purpose'] != ('initial' if n <= 8 else 'target')
                        or QueryFeedback.from_dict(query['feedback']).to_dict() != feedbacks[query['index']]):
                    raise ValueError('Recovery feedback differs from original execution or paid ledger')
                if not 0 <= query['prequery_probability'] <= 1:
                    raise ValueError('Recovery query probability is outside [0,1]')
                if query['feedback']['collision']:
                    found.add(dedup_key(candidates[query['index']]))
                curve.append(len(found))
            recall = [n/len(failures) if failures else None for n in curve]
            metrics = payload['metrics']
            if (metrics['discovery_curve'] != curve or metrics['recall_curve'] != recall
                    or metrics['pool_unique_collisions'] != len(failures)
                    or metrics['unique_collisions'] != curve[-1]
                    or metrics['collision_recall'] != recall[-1]
                    or any(row[k] != value for k,value in metrics.items())):
                raise ValueError('Recovery metrics differ from the exhaustive original candidate pool')
            for b in (16,32,64,128):
                if metrics['checkpoints'][str(b)] != {'unique_collisions':curve[b-1],
                                                       'collision_recall':recall[b-1]}:
                    raise ValueError('Recovery checkpoint differs from original labels')
            pretrained,_ = load_fsbo(checkpoint)
            base = ResponseKernel(pretrained,responses.shape[1],
                                  adaptive=row['method'] in ('response_adaptive','safety_adaptive'))
            x = np.column_stack((features(candidates),responses))
            numerical = []
            for query_number in (9,65,128):
                paid = selected[:query_number-1]
                labels = [feedbacks[i]['collision'] for i in paid]
                model = copy.deepcopy(base)
                fit_target(model,x[paid],labels,steps=10,learning_rate=.003)
                probability = numpy_probabilities(model,x,paid,labels)
                query = queries[query_number-1]
                np.testing.assert_allclose(probability[query['index']],query['prequery_probability'],atol=2e-9,rtol=2e-9)
                np.testing.assert_allclose(model.log_response_scale.exp().square().detach(),
                                           queries[query_number-2]['response_variances'],atol=1e-12)
                remaining = set(range(512)) - set(paid)
                maximum = max(probability[i] for i in remaining)
                chosen = min((i for i in remaining if probability[i] >= maximum - 1e-12),
                             key=lambda i:candidates[i].scenario_id)
                if chosen != query['index']:
                    raise ValueError('Recovery acquisition differs from independent prequery prediction')
                numerical.append({'query':query_number,'probability':float(probability[chosen])})
            checks.append({'path':row['path'],'numerical_prequery_checks':numerical})
            if len(checks) % 8 == 0:
                print(study,'audit',len(checks),'/ 80',flush=True)
    result = {'status':'passed','runs':80,'queries':10240,'raw_banks':len(raw),
              'raw_pool_executions':512*len(raw),'independent_numpy_prediction_checks':3*len(checks),
              'new_physical_calls':0,'original_gate_changed':False,'checks':checks}
    write_json(directory/'audit/report.json',result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('study',nargs='?',default='recovery',choices=('recovery','safety','geometry'))
    audit(parser.parse_args().study)
