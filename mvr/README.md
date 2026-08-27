# Transferable Scenario Mining

`mvr` mines safety-critical scenarios under a fixed simulator-episode budget. The active method uses a shared interaction encoder, MoE Outer policy, PEARL-inspired product-of-Gaussians context, and a shared Inner SAC controller.

## Method

Each episode follows `Outer interaction/x0 action → native navigation → semantic schedule → native IDM nominal action → 3-D interaction residual → hard action projection → event-time semantic failure → trajectory/outcome token → posterior z → next action`. The Outer policy sees geometry-derived scene/candidate embeddings and `z`, never SUT or functional-family identifiers.

The Inner SAC action is `interaction_residual_3d_v1 = [u_long, u_maneuver, u_lat]`. `u_maneuver` is a low-frequency, stateful conflict-timing reference; the four schedule values are included in the Inner observation. MetaDrive native IDM handles normal driving. The residual only makes an already lawful maneuver more or less challenging, while `TrafficActionShield` only performs a final physical projection and never lane-follows on behalf of the controller.

## Tasks and scope

`configs/geometry_catalog.json` defines five concrete geometries per family: three train, one validation, and one disjoint test geometry. `configs/taskbook.json` combines them with the IDM profile split. Regimes R1–R4 separate seen/unseen SUT from seen/unseen geometry.

Roundabout candidates are explicit entry-to-exit road contracts.  Spawn lane and destination drive MetaDrive native navigation; the adapter validates the resulting checkpoints without modifying navigation internals.

## Canonical pipeline

```text
inner_pretrain → posterior → inner_latent_calibration → outer
```

`training/pipeline.py` enforces this order and writes one checkpoint plus JSON summary per stage. `inner_latent_calibration` is required because the final Inner policy consumes posterior `z`.

## Evaluation and validation

Evaluation uses 20 total simulator episodes and K = 0/1/2/4 support shots; support episodes are charged to the same budget. G3 replays shared low-discrepancy configurations across IDM profiles to verify that their failure landscapes are non-degenerate.

Evaluation JSON records each concrete scenario's geometry hash, conflict-relative initial state, option, latent, Inner-policy hash, and episode seed so it can be reconstructed from the taskbook.

Checkpoints require `transferable_scenario_miner_v3`, `scenario_contract_v2`, `interaction_residual_3d_v1`, and `metadrive_native_idm_v1`; previous 2-D residual checkpoints are intentionally incompatible.

```powershell
conda run -n metadrive python -m mvr.scripts.build_taskbook --output mvr/configs/taskbook.json
conda run -n metadrive python -m mvr.scripts.train_mvr --config mvr/configs/mvr_stage1.yaml --output results/mvr/stage1_seed11 --stop-after inner_pretrain
conda run -n metadrive python -m mvr.scripts.validate_mvr --config mvr/configs/mvr_stage1.yaml --checkpoint results/mvr/stage1_seed11/inner_pretrain.pt --mode inner --output results/mvr/stage1_seed11/validation.json
```

After a checkpoint has been retrained against the current taskbook, create auditable high-frame-rate GIF replays for formal tasks. Each policy replay is written to its own GIF; each frame combines a blue-SUT tail-following 3D camera with a synchronized global top-down view where the adversary is red:

```powershell
conda run -n metadrive python -m mvr.scripts.visualize_stage1 --config mvr/configs/mvr_stage1.yaml --checkpoint results/mvr/stage1_seed11/inner_pretrain.pt --output results/mvr/stage1_seed11/visualization
```

## Verification

```powershell
conda run -n metadrive python -m pytest mvr/tests -q
```
