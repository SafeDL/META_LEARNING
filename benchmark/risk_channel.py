"""Disclose one requested risk per paid query to an isolated selector."""

from types import SimpleNamespace

from .config import BUDGET


class RiskOracle:
    def __init__(self, connection):
        self.connection = connection

    def query(self, index):
        self.connection.send(("query", int(index)))
        return SimpleNamespace(risk=self.connection.recv())


def disclose_risk(connection, task, risk, process, *, minimum_queries=BUDGET):
    if not 0 <= minimum_queries <= BUDGET:
        raise ValueError("Invalid minimum number of paid queries")
    connection.send(("task", task))
    selected, disclosures = [], []
    while True:
        while not connection.poll(1.0):
            if not process.is_alive():
                raise RuntimeError("Selector process exited before returning a result")
        message, value = connection.recv()
        if message == "query":
            index = int(value)
            if (
                index not in range(len(risk))
                or index in selected
                or len(selected) >= BUDGET
            ):
                raise ValueError("Invalid or repeated paid query")
            selected.append(index)
            disclosures.append(
                {"index": index, "risk": float(risk[index]), "collision": None}
            )
            connection.send(float(risk[index]))
        elif message == "result":
            if (
                value["selected_indices"] != selected
                or not minimum_queries <= len(selected) <= BUDGET
            ):
                raise ValueError(
                    "Selector returned an incomplete or different paid trajectory"
                )
            if len(selected) < BUDGET and (
                value.get("stop_reason") != "quality_retained"
                or not value.get("stopping")
                or not value["stopping"][-1]["stop"]
            ):
                raise ValueError("Early return needs an explicit stopping decision")
            for actual, reported in zip(disclosures, value["queries"]):
                if (
                    actual["index"] != reported["index"]
                    or actual["risk"] != reported["risk"]
                ):
                    raise ValueError("Returned risk differs from the disclosure")
            return {**value, "disclosures": disclosures}
        else:
            raise RuntimeError(value)
