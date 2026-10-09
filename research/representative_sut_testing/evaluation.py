"""Budgeted feedback disclosure and the fixed discovery metrics."""
import numpy as np

from .config import BUDGET, CHECKPOINTS


def metrics(collisions, selected):
    positives = int(np.count_nonzero(collisions))
    found = np.cumsum(np.asarray(collisions)[selected])
    upper = np.minimum(np.arange(1, BUDGET + 1), positives)
    return {
        "total_collisions": positives,
        "area_200": float(found.sum() / (BUDGET * positives)) if positives else None,
        "recall_200": float(found[-1] / positives) if positives else None,
        "F200": int(found[-1]), "missed": positives - int(found[-1]),
        "checkpoints": {str(t): int(found[t - 1]) for t in CHECKPOINTS},
        "terminal_ceiling_attainment": float(found[-1] / upper[-1]) if positives else None,
        "area_ceiling": float(upper.sum() / (BUDGET * positives)) if positives else None,
        "curve": found.astype(int).tolist(),
    }


def disclose(connection, task, risk, collision):
    connection.send(("task", task))
    selected, observations = [], []
    while True:
        message, value = connection.recv()
        if message == "query":
            index = int(value)
            if index not in range(len(risk)) or index in selected or len(selected) >= BUDGET:
                raise ValueError("Invalid, duplicate or over-budget disclosure")
            selected.append(index)
            pair = (float(risk[index]), bool(collision[index]))
            observations.append({"index": index, "risk": pair[0], "collision": pair[1]})
            connection.send(pair)
        elif message == "result":
            if value["selected_indices"] != selected or len(selected) != BUDGET:
                raise ValueError("Result differs from the actual disclosure order")
            for reported, actual in zip(value["queries"], observations):
                if any(reported[key] != actual[key] for key in actual):
                    raise ValueError("Selector observation record differs from disclosures")
            return {**value, **metrics(collision, selected), "disclosures": observations}
        else:
            raise RuntimeError(value)
