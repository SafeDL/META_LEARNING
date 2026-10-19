"""Frozen behavior predictions, without target outcomes."""
import torch


PREDICTION_BATCH = 65536


def response_tables(x, behaviors, models, return_logits=False):
    device = "cuda"
    x = torch.as_tensor(x, dtype=torch.float32, device=device)
    behaviors = torch.as_tensor(behaviors, dtype=torch.float32, device=device)
    risks = x.new_empty((len(behaviors), len(x)))
    collisions = torch.empty_like(risks)
    with torch.inference_mode():
        for family in (0, 1):
            indices = torch.nonzero(x[:, 4] == family, as_tuple=True)[0]
            risk_values = x.new_empty((len(behaviors) * len(indices), ))
            collision_values = torch.empty_like(risk_values)
            for start in range(0, len(risk_values), PREDICTION_BATCH):
                flat = torch.arange(start,
                                    min(start + PREDICTION_BATCH,
                                        len(risk_values)),
                                    device=device)
                inputs = torch.cat((x[indices[flat % len(indices)], :4],
                                    behaviors[flat // len(indices)]), 1)
                r, logits = models[family](inputs)
                risk_values[flat] = r
                collision_values[flat] = (logits if return_logits else
                                          logits.sigmoid())
            risks[:, indices] = risk_values.reshape(len(behaviors), -1)
            collisions[:,
                       indices] = collision_values.reshape(len(behaviors), -1)
    return risks, collisions
