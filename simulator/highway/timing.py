"""One clock for native execution, policy training and controller prediction."""

PHYSICS_HZ = 20
DECISION_HZ = 5
PHYSICS_DT = 1 / PHYSICS_HZ
DECISION_DT = 1 / DECISION_HZ
PHYSICS_PER_DECISION = PHYSICS_HZ // DECISION_HZ


def timing_contract():
    return {"simulation_frequency": PHYSICS_HZ, "policy_frequency": DECISION_HZ}
