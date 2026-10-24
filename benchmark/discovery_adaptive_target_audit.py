"""Raw-feedback audit and independent reconstruction of adaptive-target mixtures."""

import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib

import numpy as np
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
import torch

from methods.baselines.fsbo.method import load as load_fsbo
from methods.common.classification import logistic_moments, site_derivatives
from methods.common.gp_classifier import ClassificationKernel, fit_target
from methods.common.response import archive_fingerprints

from .build_collision_models import model_path
from .construction import load_construction
from .discovery_adaptive_target import REPORT, gate, protocol, source_prior
from .discovery_config import read_json, write_json
from .discovery_recovery_audit import numpy_probabilities
from .executions import ExecutionBank
from .initialization import initial_indices
from .protocol import DATA_ROOT, ROOT
from .scenarios import dedup_key, features
from .schema import QueryFeedback


def check_run(row,candidates,feedbacks):
    torch.set_num_threads(1)
    with threadpool_limits(1):
        payload = read_json(ROOT/row['path'])
        spec = payload['specification']
        bank = ExecutionBank(DATA_ROOT/'physical/development'/row['scenario']/row['target_id'])
        if (spec['protocol'] != protocol() or spec['candidate_ids'] != [s.scenario_id for s in candidates]
                or spec['task'] != {k:row[k] for k in protocol()['tasks'][0]}
                or spec['bank_contract'] != bank.manifest['contract']):
            raise ValueError('Adaptive-target audit differs from frozen public inputs')
        for name,digest in spec['checkpoints'].items():
            if hashlib.sha256(model_path(name,'all',row['scenario']).read_bytes()).hexdigest() != digest:
                raise ValueError('Adaptive-target checkpoint differs from the executed model')
        queries,ledger = payload['result']['queries'],payload['query_ledger']
        selected = [q['index'] for q in queries]
        if (len(queries) != 128 or len(set(selected)) != 128 or len(ledger) != 128
                or selected[:8] != initial_indices(512,8,row['algorithm_seed'])
                or payload['result']['selected_indices'] != selected):
            raise ValueError('Adaptive-target budget or common initialization differs')
        failures = {dedup_key(s) for s,f in zip(candidates,feedbacks) if f['collision']}
        found,curve = set(),[]
        for n,(q,record) in enumerate(zip(queries,ledger),1):
            if (q['query'] != n or record['call'] != n or record['status'] != 'complete'
                    or record['index'] != q['index'] or record['feedback'] != q['feedback']
                    or record['purpose'] != ('initial' if n <= 8 else 'target')
                    or QueryFeedback.from_dict(q['feedback']).to_dict() != feedbacks[q['index']]):
                raise ValueError('Adaptive-target query differs from raw feedback or paid ledger')
            if q['feedback']['collision']:
                found.add(dedup_key(candidates[q['index']]))
            curve.append(len(found))
        recall = [n/len(failures) if failures else None for n in curve]
        metrics = payload['metrics']
        if (metrics['discovery_curve'] != curve or metrics['recall_curve'] != recall
                or metrics['pool_unique_collisions'] != len(failures)
                or metrics['unique_collisions'] != curve[-1] or metrics['collision_recall'] != recall[-1]
                or any(row[k] != value for k,value in metrics.items())):
            raise ValueError('Adaptive-target metrics differ from the entire raw candidate pool')
        np.testing.assert_allclose(metrics['discovery_auc'],np.mean(curve),atol=1e-12)
        for b in (16,32,64,128):
            if metrics['checkpoints'][str(b)] != {'unique_collisions':curve[b-1],'collision_recall':recall[b-1]}:
                raise ValueError('Adaptive-target metric checkpoint differs')
        history,_ = load_construction(row['scenario'])
        if spec['history_fingerprints'] != archive_fingerprints(history):
            raise ValueError('Adaptive-target history differs from the executed construction')
        source = source_prior(row['scenario'],candidates,history,row['method'])
        raw_target = row['method'] == 'gp_target_local'
        if raw_target:
            names,prior = ['target_physical_gp','target_fsbo'],np.array([.5,.5])
        else:
            names = [*source.component_names,'target_fsbo']
            prior = np.r_[source.weights,.25] if len(source.weights) > 1 else np.array([1.,1.])
            prior /= prior.sum()
            mean,covariance = source.mean.copy(),source.covariance.copy()
        if payload['result']['component_names'] != names:
            raise ValueError('Adaptive-target component names differ from the public prior')
        np.testing.assert_allclose(payload['result']['component_prior_weights'],prior,atol=1e-14)
        x = features(candidates)
        evidence = np.zeros((len(prior),512))
        bandwidth = None if row['method'] == 'hybrid_global' else .3
        target_base,metadata = load_fsbo(model_path('fsbo','all',row['scenario']))
        if metadata['history_fingerprints'] != spec['history_fingerprints']:
            raise ValueError('Adaptive-target FSBO initialization differs from its permitted histories')
        numerical = []
        source_checks = 0
        for n,q in enumerate(queries,1):
            i,y = q['index'],q['feedback']['collision']
            scores = np.log(prior)+evidence[:,i]
            weight = np.exp(scores-logsumexp(scores))
            p = np.asarray(q['prequery_component_probabilities'])
            if not raw_target:
                expected = logistic_moments(mean[:,i],covariance[:,i,i])[0]
                np.testing.assert_allclose(p[:-1],expected,atol=2e-12,rtol=2e-12)
                source_checks += len(expected)
            np.testing.assert_allclose(q['prequery_component_weights'],weight,atol=2e-12)
            np.testing.assert_allclose(q['prequery_probability'],weight @ p,atol=2e-12)
            if n in (9,65,128):
                paid = selected[:n-1]
                labels = [feedbacks[j]['collision'] for j in paid]
                target = copy.deepcopy(target_base)
                fit_target(target,x[paid],labels,steps=10,learning_rate=.003)
                target_p = numpy_probabilities(target,x,paid,labels)
                np.testing.assert_allclose(p[-1],target_p[i],atol=2e-9,rtol=2e-9)
                if raw_target:
                    raw_model = ClassificationKernel(x.shape[1])
                    fit_target(raw_model,x[paid],labels,steps=10,learning_rate=.01)
                    source_p = numpy_probabilities(raw_model,x,paid,labels)[None]
                    np.testing.assert_allclose(p[0],source_p[0,i],atol=2e-9,rtol=2e-9)
                else:
                    source_p = logistic_moments(mean,covariance.diagonal(axis1=1,axis2=2))[0]
                components = np.vstack((source_p,target_p))
                logs = np.log(prior[:,None])+evidence
                weights = np.exp(logs-logsumexp(logs,axis=0))
                probability = (weights*components).sum(0)
                available = set(range(512))-set(paid)
                maximum = max(probability[j] for j in available)
                chosen = min((j for j in available if probability[j] >= maximum-1e-12),
                             key=lambda j:candidates[j].scenario_id)
                if chosen != i:
                    raise ValueError('Adaptive-target choice differs from independent full-pool prediction')
                numerical.append({'query':n,'mixture_probability':float(probability[i]),
                                  'target_fsbo_probability':float(target_p[i]),
                                  'raw_target_checked':raw_target})
            likelihood = np.log(np.maximum(p if y else 1-p,1e-15))
            np.testing.assert_allclose(q['prequery_component_log_likelihood'],likelihood,atol=2e-12)
            influence = (np.ones(512) if bandwidth is None else
                         np.exp(-np.square(x-x[i]).mean(1)/(2*bandwidth**2)))
            evidence += likelihood[:,None]*influence
            after = np.log(prior)+evidence[:,i]
            np.testing.assert_allclose(q['component_weights_after_query'],
                                       np.exp(after-logsumexp(after)),atol=2e-12)
            if not raw_target:
                _,gradient,curvature = site_derivatives(mean[:,i],covariance[:,i,i],y)
                cross = covariance[:,:,i].copy()
                mean += gradient[:,None]*cross
                covariance += curvature[:,None,None]*cross[:,:,None]*cross[:,None,:]
        if payload['cost']['new_physical_calls'] != 0:
            raise ValueError('Replay may not conceal physical calls')
        return {'path':row['path'],'numerical_prequery_checks':numerical,
                'source_component_probability_checks':source_checks}


def audit():
    index = read_json(REPORT/'run_index.json')
    if index['protocol'] != protocol() or gate(index['runs']) != read_json(REPORT/'gate.json'):
        raise ValueError('Adaptive-target frozen matrix or gate differs')
    raw = {}
    for row in index['runs']:
        key = row['target_id'],row['scenario']
        if key not in raw:
            bank = ExecutionBank(DATA_ROOT/'physical/development'/row['scenario']/row['target_id'])
            raw[key] = bank.candidates,[bank(s).feedback.to_dict() for s in bank.candidates]
    checks = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(check_run,row,*raw[(row['target_id'],row['scenario'])]) for row in index['runs']]
        for future in as_completed(futures):
            checks.append(future.result())
            if len(checks) % 8 == 0:
                print('adaptive_target_audit',len(checks),'/ 200',flush=True)
    checks.sort(key=lambda r:r['path'])
    result = {'status':'passed','runs':200,'queries':25600,'raw_banks':len(raw),
              'raw_pool_executions':512*len(raw),'reliability_updates_checked':25600,
              'source_component_probability_checks':sum(c['source_component_probability_checks'] for c in checks),
              'independent_fsbo_probability_checks':600,'independent_raw_gp_probability_checks':120,
              'independent_full_pool_acquisition_checks':600,'new_physical_calls':0,
              'previous_gates_changed':False,'checks':checks}
    write_json(REPORT/'audit/report.json',result)
    return result


if __name__ == '__main__':
    audit()
