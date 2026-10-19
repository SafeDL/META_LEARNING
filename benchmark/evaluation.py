"""Discovery counts and coverage shared by risk and collision evaluation."""

import numpy as np


def metrics(labels, selected):
    positives = int(np.count_nonzero(labels))
    found = int(np.count_nonzero(np.asarray(labels)[selected]))
    return {
        "F200": found,
        "recall_200": found / positives if positives else None,
    }
