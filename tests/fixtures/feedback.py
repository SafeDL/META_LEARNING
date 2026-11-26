"""Paid collision and RSS responses for selector tests."""

from experiments.rss import measurement_contract
from experiments.schema import QueryFeedback, SafetyMeasurements


class Oracle:
    def __init__(self, scenes, labels, margins):
        self.scenes, self.labels, self.margins = scenes, labels, margins
        self.queried = []

    def query(self, index, *, purpose):
        self.queried.append(index)
        return QueryFeedback(
            self.scenes[index].scenario_id,
            bool(self.labels[index]),
            SafetyMeasurements(1, 1, rss_margin=float(self.margins[index])),
            0,
            measurement_contract(),
        )
