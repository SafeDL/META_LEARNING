"""Paired-feedback transport used by frozen comparison baselines."""
from types import SimpleNamespace


class RemoteOracle:
    def __init__(self, connection):
        self.connection = connection
        self.count = 0

    def query(self, index):
        self.connection.send(("query", int(index)))
        risk, collision = self.connection.recv()
        self.count += 1
        return SimpleNamespace(
            risk=risk, valid_risk=risk is not None, collision=collision,
            scenario_id=f"target:{index}", query_number=self.count,
        )
