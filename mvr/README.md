# Transferable Scenario Mining

`mvr` mines safety-critical scenarios under a fixed simulator-episode budget. The active method uses a shared interaction encoder, MoE Outer policy, PEARL-inspired product-of-Gaussians context, and a shared Inner SAC controller.

## Method

Each episode follows `Outer interaction/x0 action → MetaDrive execution → Inner adversary rollout → trajectory/outcome token → posterior z → next action`. The Outer policy sees geometry-derived scene/candidate embeddings and `z`, never SUT or functional-family identifiers.

## Tasks and scope

`configs/geometry_catalog.json` defines four concrete geometries per family: three train and one disjoint test geometry. `configs/taskbook.json` combines them with the IDM profile split. Regimes R1–R4 separate seen/unseen SUT from seen/unseen geometry.

## Canonical pipeline

```text
inner_pretrain → posterior → inner_latent_calibration → outer
```

`training/pipeline.py` enforces this order and writes one checkpoint plus JSON summary per stage. `inner_latent_calibration` is required because the final Inner policy consumes posterior `z`.

## Evaluation and validation

Evaluation uses 20 total simulator episodes and K = 0/1/2/4 support shots; support episodes are charged to the same budget. G3 replays shared low-discrepancy configurations across IDM profiles to verify that their failure landscapes are non-degenerate.

Evaluation JSON records each concrete scenario's geometry hash, conflict-relative initial state, option, latent, Inner-policy hash, and episode seed so it can be reconstructed from the taskbook.

```powershell
python -m mvr.scripts.build_taskbook --output mvr/configs/taskbook.json
python -m mvr.scripts.validate_mvr --config mvr/configs/mvr.yaml --output results/validation/g3.json
python -m mvr.scripts.train_mvr --config mvr/configs/mvr.yaml --output results/mvr/run_001
python -m mvr.scripts.evaluate_mvr --config mvr/configs/mvr.yaml --checkpoint results/mvr/run_001/outer.pt --output results/mvr/run_001/evaluation.json
```

## Verification

```powershell
conda run -n metadrive python -m pytest mvr/tests -q
```
