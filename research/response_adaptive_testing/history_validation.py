"""Train or reuse predictors with an entire historical mechanism group excluded."""
import numpy as np
import torch

from methods.history_guided_testing.config import GROUPS
from methods.history_guided_testing.history import split_indices
from methods.history_guided_testing.io import write_json
from research.history_response_testing.history_model import HistoryResponseModel, fit_history_model

from .config import OUTPUT


def predictions(bank, excluded):
    names = [
        name for group, members in GROUPS.items() if group != excluded
        for name in members
    ]
    path = OUTPUT / "models" / f"exclude_{excluded}.pt"
    train, validation = split_indices(bank["x"])
    if not path.exists() or not path.with_suffix(".json").exists():
        states, curves = {}, {}
        torch.manual_seed(11)
        for family in (0, 1):
            model, curve = fit_history_model(
                bank, names, train[bank["x"][train, 4] == family],
                validation[bank["x"][validation, 4] == family],
                np.random.default_rng(11))
            states[str(family)] = {
                key: value.cpu()
                for key, value in model.state_dict().items()
            }
            curves[str(family)] = curve
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"states": states, "source_names": names}, path)
        write_json(path.with_suffix(".json"), {
            "curves": curves,
            "excluded_group": excluded,
            "source_names": names
        })
    state = torch.load(path, weights_only=True)
    risk, collision = np.zeros((len(bank["x"]), len(names))), np.zeros(
        (len(bank["x"]), len(names)))
    for family in (0, 1):
        indices = np.flatnonzero(bank["x"][:, 4] == family)
        model = HistoryResponseModel(len(names)).cuda().eval()
        model.load_state_dict(state["states"][str(family)])
        with torch.no_grad():
            r, c = model(
                torch.as_tensor(bank["x"][indices, :4],
                                dtype=torch.float32,
                                device="cuda"))
            risk[indices], collision[indices] = r.cpu().numpy(), c.sigmoid(
            ).cpu().numpy()
    return names, risk, collision
