"""Continuous historical safety fields for response-similarity transfer."""

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from .response_adaptation import SURROGATE_CONFIG, response_coordinates


def safety_coordinates(history, x):
    collision = response_coordinates(history,x)
    columns = []
    for archive in history:
        for values in archive.auxiliary.T:
            model = HistGradientBoostingRegressor(**SURROGATE_CONFIG)
            model.fit(archive.x,values)
            center = values.mean()
            scale = max(3 * values.std(),0.1)
            columns.append(np.clip((model.predict(x) - center)/scale,-1,1))
    return np.column_stack((collision,*columns))
