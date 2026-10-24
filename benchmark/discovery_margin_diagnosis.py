"""Read-only clearance/collision consistency diagnostics for historical archives."""

from importlib.metadata import version

import numpy as np

from .construction import load_construction
from .discovery_config import OUTPUT, write_json
from .scenarios import DOMAINS


def diagnose():
    records = []
    for kind in DOMAINS:
        history,development = load_construction(kind)
        for archive in (*history,*development):
            margin,collision = archive.safety[:,0],archive.collision
            safe = margin[~collision]
            finite_safe = safe[np.isfinite(safe)]
            records.append({'source_id':archive.source_id,'role':archive.role,'scenario':kind,
                            'samples':len(collision),'collision_count':int(collision.sum()),
                            'nonpositive_safe_clearance':int(np.sum(safe <= 0)),
                            'nonzero_collision_clearance':int(np.sum(margin[collision] > 1e-9)),
                            'nonfinite_clearance':int(np.sum(~np.isfinite(margin))),
                            'safe_clearance_quantiles':np.quantile(finite_safe,[0,.1,.5,.9,1]).tolist()
                            if len(finite_safe) else None})
    result = {'status':'complete','uses_final_targets':False,'archives':len(records),
              'responses':sum(r['samples'] for r in records),
              **{key:sum(r[key] for r in records) for key in
                 ('nonpositive_safe_clearance','nonzero_collision_clearance','nonfinite_clearance')},
              'highway_env_version':version('highway-env'),'records':records,'new_physical_calls':0,
              'interpretation':'saved geometric clearance and simulator collision flag are distinct observations'}
    write_json(OUTPUT/'margin_observation/report.json',result)
    print({k:v for k,v in result.items() if k != 'records'},flush=True)
    return result


if __name__ == '__main__':
    diagnose()
