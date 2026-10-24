"""Historical-only held-out prediction check for relative-motion response fields."""

from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from threadpoolctl import threadpool_limits

from methods.gcrt.response_geometry import kinematic_features, predict_fields

from .construction import load_construction
from .discovery_config import OUTPUT, SCENARIOS, write_json


def check(kind,source_id):
    with threadpool_limits(1):
        archive = next(a for a in load_construction(kind)[0] if a.source_id == source_id)
        folds = np.random.default_rng(1607).permutation(len(archive.scenarios)).reshape(4,-1)
        records = []
        for fold,query in enumerate(folds):
            support = np.concatenate([f for n,f in enumerate(folds) if n != fold])
            for geometry in (False,True):
                x = kinematic_features(archive.scenarios) if geometry else archive.x
                p,safety = predict_fields(archive,support,x[query],kinematic=geometry)
                records.append({'fold':fold,'kinematic':geometry,
                                'brier':float(np.square(p-archive.collision[query]).mean()),
                                'safety_mae':float(np.abs(safety-archive.auxiliary[query]).mean())})
        return {'scenario':kind,'source_id':source_id,'records':records}


def run():
    jobs = [(kind,a.source_id) for kind in SCENARIOS for a in load_construction(kind)[0]]
    rows = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(check,*job) for job in jobs]):
            rows.append(future.result())
            print('historical_geometry_cv',len(rows),'/ 16',flush=True)
    summary = {}
    for kind in SCENARIOS:
        summary[kind] = {str(geometry):{
            metric:float(np.mean([r[metric] for row in rows if row['scenario'] == kind
                                  for r in row['records'] if r['kinematic'] == geometry]))
            for metric in ('brier','safety_mae')} for geometry in (False,True)}
    result = {'status':'complete','historical_sources':16,'folds':4,'fold_seed':1607,
              'uses_development_labels':False,'uses_target_labels':False,
              'summary':summary,'runs':rows,'new_physical_calls':0}
    write_json(OUTPUT/'geometry_diagnosis/report.json',result)
    print(summary,flush=True)
    return result


if __name__ == '__main__':
    run()
