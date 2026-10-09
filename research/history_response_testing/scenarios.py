"""New coordinates and frozen baseline predictions, without target outcomes."""
import numpy as np
import torch
from scipy.stats import qmc

from methods.history_guided_testing.config import BOUNDS
from methods.history_guided_testing.scenarios import scene_library

POOL_SEEDS = tuple((1098001 + 191 * i, 1098002 + 191 * i) for i in range(20))


def confirmation_scenes(pool):
    scenes = scene_library("target")
    for family, seed in enumerate(POOL_SEEDS[pool]):
        points = qmc.Sobol(4, scramble=True, seed=seed).random_base2(10)
        bounds = np.asarray(BOUNDS[family])
        for index, point in enumerate(points):
            scene = scenes[family * 1024 + index]
            scene["numeric_input"] = [*point.tolist(), family]
            scene["active_parameters"] = dict(
                zip(scene["active_parameters"],
                    (bounds[:, 0] + point *
                     (bounds[:, 1] - bounds[:, 0])).tolist()))
            scene[
                "scenario_id"] = f"fresh_confirmation:{pool}:{family}:{index:04d}"
            scene["sampling_seed"] = seed
    return scenes


def ras_predictions(x, seed):
    from methods.ras_frt_uq.unified import MODEL_ROOT, ResponseEncoder, predict_response
    state = torch.load(MODEL_ROOT / f"seed_{seed}.pt",
                       map_location="cpu",
                       weights_only=True)
    result = np.zeros((len(x), 6))
    for family in (0, 1):
        indices = np.flatnonzero(x[:, 4] == family)
        model = ResponseEncoder(6).cuda().eval().requires_grad_(False)
        model.load_state_dict(state["encoders"][str(family)])
        result[indices] = predict_response(model, x[indices, :4])
    return result
