"""Register original fixed GP defaults without a development experiment."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

from testing_methods.history_transfer.adaptation import (
    FitSettings,
    OnlineSettings,
)
from testing_methods.history_transfer.inference import TARGET_PARAMETERS

from .storage import read_json

REGISTRY = "reports/method_settings.json"


def fit_settings(value):
    if (
        not isinstance(value, dict)
        or set(value) != {"steps", "learning_rate", "parameters"}
        or type(value["steps"]) is not int
        or not isinstance(value["parameters"], list)
        or not set(value["parameters"]).issubset(TARGET_PARAMETERS)
        or not isinstance(value["learning_rate"], (int, float))
        or isinstance(value["learning_rate"], bool)
    ):
        raise ValueError("Unknown or malformed registered GP fitting setting")
    return FitSettings(
        value["steps"],
        value["learning_rate"],
        tuple(value["parameters"]),
    )


@dataclass(frozen=True)
class MethodSettings:
    online: OnlineSettings
    pooled: FitSettings

    def to_dict(self):
        return {"online": self.online.to_dict(), "pooled": self.pooled.to_dict()}

    def prediction_kwargs(self, method):
        if method == "pooled_history":
            return {"pooled_fit_settings": self.pooled}
        if method in ("target_only", "full"):
            return {"online_settings": self.online}
        raise ValueError("Only the three registered GP prediction methods use these settings")

    def budget_kwargs(self, method):
        if method not in ("target_only", "full"):
            raise ValueError("Pooled GP is not a registered active-query method")
        return {"online_settings": self.online}


def native_settings():
    """Original six-scalar, ten-step fits, shared by the raw target branches."""
    learned = FitSettings(10, 0.003, TARGET_PARAMETERS)
    return MethodSettings(
        OnlineSettings(FitSettings(10, 0.01, TARGET_PARAMETERS), learned, -1.4),
        learned,
    )


def specification():
    from .method_contracts import method_contract

    return {
        "scope": "Fixed original GP defaults; no development data or hyperparameter selection",
        "settings": native_settings().to_dict(),
        "methods": {
            name: method_contract(name) for name in ("target_only", "pooled_history", "full")
        },
        "loader_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "formal_target_labels_read": False,
        "opens_formal_evaluation": False,
    }


def register_settings(root=None):
    from .history_transfer_experiment import OUTPUT, freeze

    return freeze(Path(OUTPUT if root is None else root) / REGISTRY, specification())


def load_settings(root):
    if read_json(Path(root) / REGISTRY) != specification():
        raise ValueError("Registered GP settings differ from the fixed defaults or implementation")
    return native_settings()


if __name__ == "__main__":
    register_settings()
