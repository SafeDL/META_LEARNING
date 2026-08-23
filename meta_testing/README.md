# Hierarchical Map-Aware Meta-Testing

`meta_testing` is the active MVR implementation. It mines safety-critical scenarios for held-out IDM controller profiles under a fixed simulator-episode budget.

## Method

Each episode follows `Outer scene action → MetaDrive execution → Inner adversary rollout → trajectory/outcome token → posterior z → next scene action`. The Inner controller receives only the explicit 10-D physical state. The Outer policy receives map embedding and posterior latent `z`; it never receives SUT identity.

## Tasks and scope

`configs/idm_taskbook.json` defines merge, cut-in, and roundabout tasks. The first four IDM profiles are meta-train, `idm_fast_small_gap` is validation, and `idm_late_response` is meta-test. Every family currently uses one fixed map, so this package supports map-conditioned testing, not unseen-map generalization.

## Canonical pipeline

```text
inner_pretrain → posterior → inner_latent_calibration → outer
```

`training/pipeline.py` enforces this order and writes one checkpoint plus JSON summary per stage. `inner_latent_calibration` is required because the final Inner policy consumes posterior `z`.

## Evaluation and validation

Evaluation uses 20 total simulator episodes and K = 0/1/2/4 support shots; support episodes are charged to the same budget. G3 replays shared low-discrepancy configurations across IDM profiles to verify that their failure landscapes are non-degenerate.

```powershell
python -m meta_testing.scripts.validate_mvr --config meta_testing/configs/mvr.yaml --output results/validation/g3.json
python -m meta_testing.scripts.train_mvr --config meta_testing/configs/mvr.yaml --output results/meta_testing/run_001
python -m meta_testing.scripts.evaluate_mvr --config meta_testing/configs/mvr.yaml --checkpoint results/meta_testing/run_001/outer.pt --output results/meta_testing/run_001/evaluation.json
```

## Verification

```powershell
conda run -n metadrive python -m pytest meta_testing/tests -q
```
