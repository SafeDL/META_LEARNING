"""Original-feedback and independent numerical checks for physical mixtures."""

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
import torch

from methods.common.classification import logistic_moments
from methods.common.gp_classifier import ClassificationKernel, fit_target
from methods.common.response import archive_fingerprints
from methods.baselines.fsbo.method import load as load_fsbo
from methods.gcrt.joint_response import MARGIN_NOISE, fit_joint, margin_values
from methods.gcrt.joint_training import load_model

from . import discovery_joint_response as joint
from .construction import load_construction
from .discovery_config import development_tasks, read_json, write_json
from .discovery_joint_response_audit import predict_numpy, prepare_numpy, prior_numpy
from .discovery_physical_mixture import METHODS, REPORT, RUNS, checkpoint_path, gate, protocol
from .discovery_recovery_audit import numpy_probabilities
from .executions import ExecutionBank
from .initialization import initial_indices
from .protocol import DATA_ROOT, ROOT
from .scenarios import dedup_key, features
from .schema import QueryFeedback


def classifier_probabilities_numpy(model,x,paid,labels):
    with torch.no_grad():
        gram = model.covariance(x[paid]).cpu().numpy()
        cross = model.covariance(x[paid],x).cpu().numpy()
        diagonal = model.covariance(x).diagonal().cpu().numpy()
        center = float(model.mean)
    count = len(paid)
    mean,covariance = np.zeros((1,2*count)),np.zeros((1,2*count,2*count))
    mean[0,:count],covariance[0,:count,:count] = center,gram
    state = prepare_numpy(mean,covariance,range(count),np.asarray(labels,dtype=float),None)
    root,factor,alpha = (value[0] for value in state[-3:])
    predicted_mean = center+cross.T @ alpha
    solved = np.linalg.solve(factor,root[:,None]*cross)
    variance = np.maximum(diagonal-np.square(solved).sum(0),1e-10)
    return logistic_moments(predicted_mean,variance)[0]


def check_run(row,candidates,feedbacks):
    torch.set_num_threads(1)
    with threadpool_limits(1):
        output = read_json(ROOT/row['path'])
        spec = output['specification']
        task = {k:row[k] for k in spec['protocol']['tasks'][0]}
        if (spec['protocol'] != read_json(REPORT/'protocol.json') or spec['task'] != task
                or task not in spec['protocol']['tasks']
                or spec['method'] != row['method'] or spec['candidate_ids'] != [s.scenario_id for s in candidates]):
            raise ValueError('Physical-mixture task differs from its registered public inputs')
        checkpoint = checkpoint_path(row['scenario'],row['method'])
        if (spec['checkpoint_path'] != str(checkpoint.relative_to(ROOT))
                or spec['checkpoint_sha256'] != hashlib.sha256(checkpoint.read_bytes()).hexdigest()):
            raise ValueError('Physical-mixture checkpoint differs')
        queries,ledger = output['result']['queries'],output['query_ledger']
        selected = [q['index'] for q in queries]
        if (len(queries) != 128 or len(ledger) != 128 or len(set(selected)) != 128
                or selected[:8] != initial_indices(512,8,row['algorithm_seed'])
                or output['result']['selected_indices'] != selected):
            raise ValueError('Physical-mixture budget, initialization, or query uniqueness differs')
        failures = {dedup_key(s) for s,f in zip(candidates,feedbacks) if f['collision']}
        found,curve = set(),[]
        for n,(q,record) in enumerate(zip(queries,ledger),1):
            if (q['query'] != n or record['call'] != n or record['status'] != 'complete'
                    or record['index'] != q['index'] or record['feedback'] != q['feedback']
                    or record['purpose'] != ('initial' if n <= 8 else 'target')
                    or QueryFeedback.from_dict(q['feedback']).to_dict() != feedbacks[q['index']]):
                raise ValueError('Physical-mixture feedback differs from its original paid execution')
            if q['feedback']['collision']:
                found.add(dedup_key(candidates[q['index']]))
            curve.append(len(found))
        recall = [v/len(failures) if failures else None for v in curve]
        metrics = output['metrics']
        if (metrics['discovery_curve'] != curve or metrics['recall_curve'] != recall
                or metrics['unique_collisions'] != curve[-1] or metrics['collision_recall'] != recall[-1]
                or metrics['pool_unique_collisions'] != len(failures)
                or any(row[k] != value for k,value in metrics.items())):
            raise ValueError('Physical-mixture metrics differ from the full collision pool')
        for b in (16,32,64,128):
            if metrics['checkpoints'][str(b)] != {'unique_collisions':curve[b-1],'collision_recall':recall[b-1]}:
                raise ValueError('Physical-mixture checkpoint curve differs')
        np.testing.assert_allclose(metrics['discovery_auc'],np.mean(curve),atol=1e-12)
        history,development = load_construction(row['scenario'])
        if row['method'] in ('physical_fsbo','fsbo_cuda'):
            base,construction = load_fsbo(checkpoint)
        else:
            base,metadata = load_model(checkpoint)
            construction = metadata['specification']
        if (spec['history_fingerprints'] != archive_fingerprints(history)
                or construction['history_fingerprints'] != archive_fingerprints(history)
                or construction['development_fingerprints'] != archive_fingerprints(development)):
            raise ValueError('Physical-mixture role-checked construction archives differ')
        target = next(a for a in development if a.source_id == row['target_id'])
        if target.scenarios != candidates or spec['bank_contract'] != target.provenance['contract']:
            raise ValueError('Physical-mixture raw coordinates or execution contract differ from the development archive')
        if row['method'] == 'fsbo_cuda':
            x = features(candidates)
            base = base.to(spec['protocol']['device'])
            numerical = []
            for n in (9,65,128):
                paid = selected[:n-1]
                labels = np.array([feedbacks[i]['collision'] for i in paid],dtype=bool)
                model = copy.deepcopy(base)
                fit_target(model,x[paid],labels,steps=10,learning_rate=.003)
                probability = classifier_probabilities_numpy(model,x,paid,labels)
                index = selected[n-1]
                np.testing.assert_allclose(queries[n-1]['prequery_probability'],probability[index],atol=2e-7,rtol=2e-6)
                available = set(range(512))-set(paid)
                maximum = max(probability[i] for i in available)
                if probability[index] < maximum-2e-7:
                    raise ValueError('CUDA FSBO choice differs from its independently reconstructed probabilities')
                numerical.append({'query':n,'probability':float(probability[index]),'acquisition_gap':float(maximum-probability[index])})
            if output['cost']['new_physical_calls'] != 0:
                raise ValueError('CUDA FSBO replay conceals physical calls')
            return {'path':row['path'],'numerical_prefixes':numerical,'reliability_updates':0}
        sources = (None if row['method'] in ('physical_target','physical_fsbo') else
                   joint.read_prior(row['scenario'],candidates,history,spec['checkpoint_sha256']))
        prior = np.ones(1) if sources is None else np.r_[.75*sources['weights'],.25]
        names = ['target_deep'] if sources is None else [*sources['names'],'target_deep']
        if output['result']['component_names'] != [*names,'target_physical']:
            raise ValueError('Physical-mixture component names differ')
        np.testing.assert_allclose(output['result']['joint_component_prior_weights'],prior,atol=1e-14)
        outer_prior = np.array([.9,.1])
        np.testing.assert_allclose(output['result']['outer_prior_weights'],outer_prior,atol=1e-14)
        if output['result']['physical_mass_is_fixed'] != (row['method'] != 'physical_adaptive'):
            raise ValueError('Physical-mixture fixed mass flag differs')
        x = features(candidates)
        evidence = np.zeros((len(prior),512))
        outer_score = np.log(outer_prior)
        use_margin = row['method'] == 'physical_margin'
        prior_mode = 'joint_mean' if row['method'] == 'physical_mean' else 'joint_local'
        base = base.to(spec['protocol']['device'])
        numerical = []
        for n,q in enumerate(queries,1):
            i,y = q['index'],q['feedback']['collision']
            if row['method'] == 'physical_fsbo':
                if q['prequery_joint_means'] is not None or q['prequery_joint_covariances'] is not None:
                    raise ValueError('The original FSBO control has no clearance output channel')
                component_p = np.asarray(q['prequery_component_probabilities'][:-1])
                mean,cov = None,None
            else:
                mean,cov = np.asarray(q['prequery_joint_means']),np.asarray(q['prequery_joint_covariances'])
                component_p = logistic_moments(mean[:,0],cov[:,0,0])[0]
            score = np.log(prior)+evidence[:,i]
            weight = np.exp(score-logsumexp(score))
            outer = np.exp(outer_score-logsumexp(outer_score))
            p_raw = q['prequery_physical_probability']
            group = np.array([weight @ component_p,p_raw])
            np.testing.assert_allclose(q['prequery_joint_weights'],weight,atol=2e-12)
            np.testing.assert_allclose(q['prequery_outer_weights'],outer,atol=2e-12)
            np.testing.assert_allclose(q['prequery_joint_group_probability'],group[0],atol=2e-12)
            np.testing.assert_allclose(q['prequery_component_probabilities'],np.r_[component_p,p_raw],atol=2e-12)
            np.testing.assert_allclose(q['prequery_component_weights'],np.r_[outer[0]*weight,outer[1]],atol=2e-12)
            np.testing.assert_allclose(q['prequery_probability'],outer @ group,atol=2e-12)
            if n in (9,65,128):
                paid = selected[:n-1]
                labels = np.array([feedbacks[j]['collision'] for j in paid],dtype=bool)
                clearance = np.array([np.inf if feedbacks[j]['safety']['min_clearance_m'] is None else
                                      feedbacks[j]['safety']['min_clearance_m'] for j in paid])
                model = copy.deepcopy(base)
                if row['method'] == 'physical_fsbo':
                    fit_target(model,x[paid],labels,steps=10,learning_rate=.003)
                    expected_p = classifier_probabilities_numpy(model,x,paid,labels)[None]
                else:
                    fit_joint(model,x[paid],labels,clearance,steps=10,learning_rate=.003,use_margin=use_margin)
                    prior_mean,prior_cov = prior_numpy(model,x,sources,prior_mode)
                    state = prepare_numpy(prior_mean,prior_cov,paid,labels.astype(float),
                                           margin_values(clearance) if use_margin else None)
                    class_mean,class_variance = predict_numpy(prior_mean,prior_cov,state,range(512))
                    expected_p = logistic_moments(class_mean,class_variance)[0]
                    expected_mean,expected_cov = predict_numpy(prior_mean,prior_cov,state,[i,512+i],full_covariance=True)
                    np.testing.assert_allclose(mean,expected_mean,atol=2e-6,rtol=2e-6)
                    np.testing.assert_allclose(cov,expected_cov,atol=2e-7,rtol=2e-6)
                np.testing.assert_allclose(component_p,expected_p[:,i],atol=2e-7,rtol=2e-6)
                physical = ClassificationKernel(x.shape[1]).to(spec['protocol']['device'])
                fit_target(physical,x[paid],labels,steps=10,learning_rate=.01)
                raw_p = numpy_probabilities(physical.cpu(),x,paid,labels)
                np.testing.assert_allclose(p_raw,raw_p[i],atol=2e-9,rtol=2e-9)
                logs = np.log(prior[:,None])+evidence
                all_weights = np.exp(logs-logsumexp(logs,axis=0))
                full_probability = outer[0]*(all_weights*expected_p).sum(0)+outer[1]*raw_p
                available = set(range(512))-set(paid)
                maximum = max(full_probability[j] for j in available)
                if full_probability[i] < maximum-2e-7:
                    raise ValueError('Physical-mixture choice differs from independent full-pool probabilities')
                numerical.append({'query':n,'mixture_probability':float(full_probability[i]),
                                  'raw_probability':float(raw_p[i]),'acquisition_gap':float(maximum-full_probability[i])})
            density = np.zeros(len(prior))
            if row['method'] == 'physical_fsbo':
                conditional_p = component_p
            else:
                conditional_mean,conditional_var = mean[:,0].copy(),cov[:,0,0].copy()
            if use_margin:
                observed = q['feedback']['safety']['min_clearance_m']
                variance = cov[:,1,1]+MARGIN_NOISE**2
                residual = margin_values(np.inf if observed is None else observed)-mean[:,1]
                conditional_mean += cov[:,0,1]*residual/variance
                conditional_var -= cov[:,0,1]**2/variance
                density = -.5*(np.log(2*np.pi*variance)+residual**2/variance)
            if row['method'] != 'physical_fsbo':
                conditional_p = logistic_moments(conditional_mean,conditional_var)[0]
            likelihood = density+np.log(np.maximum(conditional_p if y else 1-conditional_p,1e-15))
            np.testing.assert_allclose(q['conditional_collision_probabilities'],conditional_p,atol=2e-12)
            np.testing.assert_allclose(q['prequery_joint_log_likelihood'],likelihood,atol=2e-10,rtol=2e-12)
            influence = (np.exp(-np.square(x-x[i]).mean(1)/(2*.3**2)) if row['method'] == 'physical_local' else np.ones(512))
            evidence += likelihood[:,None]*influence
            after = np.log(prior)+evidence[:,i]
            np.testing.assert_allclose(q['joint_weights_after_query'],np.exp(after-logsumexp(after)),atol=2e-11)
            outer_likelihood = np.log(np.maximum(group if y else 1-group,1e-15))
            np.testing.assert_allclose(q['prequery_outer_binary_log_likelihood'],outer_likelihood,atol=2e-12)
            if row['method'] == 'physical_adaptive':
                outer_score += outer_likelihood
            np.testing.assert_allclose(q['outer_weights_after_query'],np.exp(outer_score-logsumexp(outer_score)),atol=2e-12)
        if output['cost']['new_physical_calls'] != 0:
            raise ValueError('Physical-mixture replay conceals physical calls')
        return {'path':row['path'],'numerical_prefixes':numerical,'reliability_updates':128}


def raw_banks(rows):
    raw = {}
    for row in rows:
        key = row['target_id'],row['scenario']
        if key not in raw:
            bank = ExecutionBank(DATA_ROOT/'physical/development'/row['scenario']/row['target_id'])
            raw[key] = bank.candidates,[bank(s).feedback.to_dict() for s in bank.candidates]
    return raw


def audit_rows(rows,raw):
    sources = {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
               (str(Path(__file__).relative_to(ROOT)), 'benchmark/discovery_joint_response_audit.py',
                'benchmark/discovery_recovery_audit.py')}
    frozen_digest = hashlib.sha256((REPORT/'protocol.json').read_bytes()).hexdigest()
    feedback_digests = {key:hashlib.sha256(json.dumps(value[1],sort_keys=True).encode('utf-8')).hexdigest()
                        for key,value in raw.items()}
    checked = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = {}
        for row in rows:
            key = row['target_id'],row['scenario']
            identity = {'audit_sources':sources,'protocol_sha256':frozen_digest,
                        'run_sha256':hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest(),
                        'raw_feedback_sha256':feedback_digests[key]}
            path = REPORT/'audit/runs'/row['method']/row['target_id']/row['scenario']/f"{row['algorithm_seed']}.json"
            if path.exists():
                saved = read_json(path)
                if saved['status'] == 'passed' and saved['identity'] == identity:
                    checked.append(saved['result'])
                    continue
            futures[pool.submit(check_run,row,*raw[key])] = path,identity
        for future in as_completed(futures):
            path,identity = futures[future]
            result = future.result()
            write_json(path,{'status':'passed','identity':identity,'result':result})
            checked.append(result)
            if len(checked) % 8 == 0:
                print('physical_mixture_audit',len(checked),'/',len(rows),flush=True)
    checked.sort(key=lambda r:r['path'])
    return checked


def audit():
    index = read_json(REPORT/'run_index.json')
    if index['protocol'] != protocol() or gate(index['runs']) != read_json(REPORT/'gate.json'):
        raise ValueError('Physical-mixture matrix or admission differs from the frozen experiment')
    raw = raw_banks(index['runs'])
    checked = audit_rows(index['runs'],raw)
    result = {'status':'passed','runs':320,'queries':40960,'raw_banks':len(raw),
              'joint_and_outer_reliability_updates_checked':35840,'independent_joint_gp_prefixes':840,
              'independent_raw_gp_prefixes':840,'independent_fsbo_cuda_prefixes':120,'independent_full_pool_choices':960,
              'new_physical_calls':0,'previous_gates_changed':False,'checks':checked}
    write_json(REPORT/'audit/report.json',result)
    return result


def audit_available():
    if read_json(REPORT/'protocol.json') != protocol():
        raise ValueError('Available-run audit requires the unchanged frozen protocol')
    rows = []
    for method in METHODS:
        for task in development_tasks():
            path = RUNS/method/task['target_id']/task['scenario']/f"{task['algorithm_seed']}.json"
            if path.exists():
                saved = read_json(path)
                rows.append({**task,'method':method,'path':str(path.relative_to(ROOT)),**saved['metrics']})
    checked = audit_rows(rows,raw_banks(rows))
    result = {'status':'available_completed_runs_checked','runs':len(checked),'expected_runs':320,
              'entire_matrix_pass_claimed':False,'new_physical_calls':0}
    write_json(REPORT/'audit/available.json',result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('scope',nargs='?',default='complete',choices=('complete','available'))
    {'complete':audit,'available':audit_available}[parser.parse_args().scope]()
