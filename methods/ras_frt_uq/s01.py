"""The original four-input, binary-feedback RAS-FRT-UQ baseline."""
import torch

from methods.srd_tnp_bqd.common import REPO, read_json
from methods.srd_tnp_bqd.s01 import SOURCES
from .coverage_selector import similarities
from .fusion_selector import select_fusion_sequence
from .response_encoder import ResponseEncoder, predict_response
from .transfer_uncertainty import physical_kernel


MODEL_ROOT = REPO / "results/ras_frt_uq/model"


def select(x, cells, oracle, seed, budget=200):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(1)
    model = ResponseEncoder(len(SOURCES)).to(device).eval().requires_grad_(False)
    model.load_state_dict(torch.load(MODEL_ROOT / f"response_seed_{seed}.pt",
                                     map_location=device, weights_only=True))
    frozen = read_json(MODEL_ROOT / "frozen_training.json")
    settings = frozen["chosen_similarity"]
    responses = predict_response(model, x)
    prior = responses.mean(axis=1)
    similarity = similarities(x, responses, settings["sigma_x"], settings["sigma_r"])
    records = []

    class BinaryOracle:
        def query(self, index):
            observation = oracle.query(index)
            if observation.collision is None:
                raise ValueError("original RAS-FRT-UQ requires a measured collision outcome")
            records.append({"query_number": observation.query_number, "index": index,
                            "scenario_id": observation.scenario_id, "label": observation.collision,
                            "risk": observation.risk, "valid_risk": observation.valid_risk,
                            "cell": int(cells[index])})
            return observation.collision

    selected, _, scores = select_fusion_sequence(prior, similarity, physical_kernel(x), cells,
                                                BinaryOracle(), budget, settings["regularizer"])
    return {"selected_indices": selected, "queries": records, "final_risk_mean": scores.tolist(),
            "method": "RAS-FRT-UQ", "seed": seed, "feedback_used": "binary collision",
            "original_model_and_fusion_weights": True, "sources": list(SOURCES)}
