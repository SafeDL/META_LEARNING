"""Scenario-only relative-motion coordinates for historical response prediction."""

import numpy as np
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from benchmark.scenarios import features
from .response_adaptation import SURROGATE_CONFIG


def kinematic_features(scenarios):
    raw = features(scenarios)
    rows = []
    for scenario in scenarios:
        p = scenario.parameters
        if scenario.kind not in ('cut_in','cut_out'):
            raise ValueError('Relative-motion response diagnosis is restricted to the two development scenarios')
        closing = (p['closing_speed_mps'] if scenario.kind == 'cut_in' else
                   p['ego_speed_mps'] - p['lead_speed_mps'])
        gap,t,duration = p['gap_m'],p['event_time_s'],p['lateral_duration_s']
        encounter = gap/max(closing,0.5)
        row = [(gap-closing*t)/50,(gap-closing*(t+duration/2))/50,
               (gap-closing*(t+duration))/50,
               max(closing,0)**2/(20*gap),
               (encounter-t)/10,(encounter-t-duration)/10]
        if scenario.kind == 'cut_out':
            obstacle_time = p['obstacle_distance_m']/p['ego_speed_mps']
            row.extend(((obstacle_time-t-duration/2)/10,
                        p['ego_speed_mps']**2/(20*p['obstacle_distance_m'])))
        rows.append(row)
    return np.column_stack((raw,np.clip(rows,-1,1)))


def predict_fields(archive, train_indices, query_features, *, kinematic):
    if archive.role != 'history':
        raise ValueError('Response field fitting requires historical archives')
    x = kinematic_features(archive.scenarios) if kinematic else archive.x
    y = archive.collision[train_indices]
    if len(np.unique(y)) == 1:
        probability = np.full(len(query_features),(y.sum()+.5)/(len(y)+1))
    else:
        classifier = HistGradientBoostingClassifier(**SURROGATE_CONFIG)
        classifier.fit(x[train_indices],y)
        probability = classifier.predict_proba(query_features)[:,1]
    safety = []
    for values in archive.auxiliary.T:
        regressor = HistGradientBoostingRegressor(**SURROGATE_CONFIG)
        regressor.fit(x[train_indices],values[train_indices])
        safety.append(regressor.predict(query_features))
    return probability,np.column_stack(safety)


def response_fields(history,candidates):
    if not history or any(a.role != 'history' for a in history):
        raise ValueError('Structured response coordinates use historical sources only')
    x = kinematic_features(candidates)
    collisions,safety = [],[]
    for archive in history:
        p,margin = predict_fields(archive,np.arange(len(archive.scenarios)),x,kinematic=True)
        collisions.append(logit(np.clip(p,.01,.99))/logit(.99))
        for values,prediction in zip(archive.auxiliary.T,margin.T):
            scale = max(3*values.std(),.1)
            safety.append(np.clip((prediction-values.mean())/scale,-1,1))
    return np.column_stack((*collisions,*safety))
