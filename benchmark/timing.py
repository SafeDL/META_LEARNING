"""Synchronized task and query computation, excluding response transport."""

import time

import torch

from .risk_channel import RiskOracle


def now():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


class QueryClock:
    def __init__(self):
        self.created = now()
        self.steps = []
        self.transport = []
        self.paused = 0.0

    def start(self):
        self.mark = now()
        self.initialization = self.mark - self.created

    def step(self):
        current = now()
        self.steps.append(current - self.mark - self.paused)
        self.mark, self.paused = current, 0.0

    def report(self):
        result = {
            "task_initialization_s": self.initialization,
            "query_compute_s": self.steps,
            "response_transport_s": self.transport,
        }
        if hasattr(self, "terminal"):
            result["final_decision_s"] = self.terminal
        return result

    def finish_decision(self):
        self.terminal = now() - self.mark - self.paused


class TimedRiskOracle(RiskOracle):
    def __init__(self, connection, clock):
        super().__init__(connection)
        self.clock = clock

    def query(self, index):
        started = now()
        observation = super().query(index)
        elapsed = now() - started
        self.clock.paused += elapsed
        self.clock.transport.append(elapsed)
        return observation
