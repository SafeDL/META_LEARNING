"""Frozen RAS inference on the shared candidate pool."""
import numpy as np
import torch


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
