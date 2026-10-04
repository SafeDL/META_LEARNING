"""Measured risk elites in the two families' 512 parameter cells."""
import numpy as np


class RiskArchive:
    def __init__(self, cell_count=512, risk_threshold=.5):
        self.threshold = risk_threshold
        self.elite_quality = np.zeros(cell_count)

    def observe(self, cell, risk):
        self.elite_quality[cell] = max(self.elite_quality[cell], risk - self.threshold)

    def metrics(self):
        return {"risk_occupied_cells": int((self.elite_quality > 0).sum()),
                "risk_qd_score": float(self.elite_quality.sum())}
