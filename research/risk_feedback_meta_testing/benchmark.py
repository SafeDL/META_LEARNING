"""Measure two validation queries before launching the fixed holdout policies."""
import time

import numpy as np
import torch

from methods.history_guided_testing.io import read_json, write_json
from research.behavior_response_testing.prediction import behavior_grid

from .config import COHORT, MODELS, RESULTS
from .model import FailureDecoder, FrozenResponseBackbone
from .planner import BudgetLookaheadSession


def main():
    torch.set_num_threads(1)
    protocol = read_json(RESULTS / "protocol.json")
    name = protocol["split"]["validation"][0]
    with np.load(COHORT / name / "pool_0" / "responses.npz") as bank:
        x, risk = bank["x"].copy(), bank["risk"].copy()
    seed = 11
    backbone = FrozenResponseBackbone(torch.load(
        MODELS / f"predictor_{seed}.pt", map_location="cuda",
        weights_only=True)).cuda()
    parameters = torch.load(RESULTS / "models" / f"decoder_meta_{seed}.pt",
                            map_location="cuda", weights_only=True)
    weight, bias = backbone.initial_head()
    decoder = FailureDecoder(weight, bias, parameters["center"],
                             parameters["scale"]).cuda()
    decoder.load_state_dict(parameters)
    predictions = backbone.tables(torch.as_tensor(x, dtype=torch.float32,
                                                   device="cuda"),
                                   torch.as_tensor(behavior_grid(), device="cuda"),
                                   decoder)
    discrepancy = read_json(COHORT / "protocol.json")["candidate"][
        "risk_discrepancy"][str(seed)]
    session = BudgetLookaheadSession(
        x, *predictions, discrepancy,
        {key: getattr(decoder, key).detach().cpu().tolist()
         for key in ("slope", "center", "scale")})
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    timings = []
    for _ in range(2):
        started = time.perf_counter()
        index = session.next_index()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        session.observe(float(risk[index]))
        timings.append({"query": session.count, "index": index,
                        "next_index_seconds": elapsed})
        print("LOOKAHEAD BENCHMARK", timings[-1], flush=True)
    prefix = read_json(COHORT / name / "pool_0" / "selection" /
                       "candidate_11.json")["selected_indices"]
    for index in prefix:
        if session.count >= 150:
            break
        if not session.remaining[index]:
            continue
        session.pending = index
        session.records.append({"index": index, "query_number": session.count + 1})
        session.observe(float(risk[index]))
    torch.cuda.synchronize()
    started = time.perf_counter()
    index = session.next_index()
    torch.cuda.synchronize()
    timings.append({"query": session.count + 1, "index": index,
                    "prefix_role": "timing-only frozen validation prefix",
                    "next_index_seconds": time.perf_counter() - started})
    print("LOOKAHEAD BENCHMARK", timings[-1], flush=True)
    result = {"role": "validation timing and implementation check, not performance evidence",
              "profile": name, "seed": seed, "scenes": len(x),
              "hypotheses": len(behavior_grid()), "queries": timings,
              "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3}
    write_json(RESULTS / "lookahead_benchmark.json", result)


if __name__ == "__main__":
    main()
