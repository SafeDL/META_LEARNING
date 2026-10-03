# Rare-event passing experiment

Reference setting: Yang et al., *Adaptive Testing Environment: the road to testing autonomous driving systems at scale* ([arXiv](https://arxiv.org/abs/2402.19275)).

## Outcome

The new three-vehicle passing benchmark fixes the dominant benchmark problem:
collisions are no longer common under the declared natural distribution. It
also makes importance sampling useful, but the current AdaTE adaptive behavior
does not yet outperform uniform target exploration.

## Scenario and controllers

- Layout: AV in the passing lane, BV cutting into the passing lane, and a slower
  LV ahead in the original lane.
- Initial grid: BV-AV gap `[10, 18, 26, 34, 42] m` and relative speed
  `[-8, -5, -2, 1] m/s`, with probability mass concentrated on the milder end.
- Natural BV residual policy: `[-6, 0, 2] m/s²` with probabilities
  `[0.01, 0.98, 0.01]`.
- Surrogates: normal IDM and limited/strong-braking FVDM profiles.
- Targets: reference IDM, calibrated IDM, and predictive emergency braking.

The paper uses a two-acceleration LV/BV action. A nine-action product was
tested here, but the scheduled BV did not react to the LV, so the LV action had
no causal effect on AV crashes. It was removed from the maintained experiment.

## Calibration

Each target received 512 independent natural draws. Collision counts were
`3`, `1`, and `0`; collision-rate estimates were `0.586%`, `0.195%`, and `0%`.
Critical-event rates were `0.586%`, `0.781%`, and `0.781%`. A complete
three-action stress sweep found `8`, `6`, and `5` collision-producing grid/action
combinations, so the zero natural count for the strongest target is a finite
sample outcome rather than an empty failure set.

This is substantially more suitable for rare-event evaluation than the old A1
targets, whose observed NDE collision rates were about 26%–51%.

## AdaTE screening

The retained single-target screening uses 180 source episodes, 400 target
episodes for Uniform-QP, 400 for AdaTE-QP, and 256 independent evaluation draws
per frozen strategy. Initial scenarios are proposed by surrogate criticality;
trajectory weights include both `p0/q0` and the action probability ratios.

| Strategy | Events | Estimate | RHW | ESS | Max weight |
| --- | ---: | ---: | ---: | ---: | ---: |
| NDE-phi-H | 2 | 0.00781 | 1.383 | 256.0 | 1.00 |
| Equal-Mixture-H | 175 | 0.00864 | 0.202 | 47.7 | 6.54 |
| Uniform-QP-H | 159 | 0.01035 | **0.168** | 48.9 | 6.23 |
| AdaTE-QP-H | 203 | 0.00956 | 0.233 | 52.0 | 6.36 |

Compared with the old cross-target archive, the passing AdaTE proposal no
longer loses to NDE: its RHW is 83% lower in this screening run and its weight
diagnostics are finite and moderate. However, AdaTE-QP is 39% worse than
Uniform-QP in RHW and 15% worse than the equal mixture. The final AdaTE
coefficient `[0, 0.325, 0.675]` also fails the reference-IDM sanity check because
the exactly matching normal-IDM surrogate receives zero weight. This prevents
a claim that adaptive exploration is superior.

## Interpretation

Three identified causes were fixed: excessive event prevalence, unpaired source
rollouts, and omission of the initial-state importance ratio. Joint source-Q
coverage improved and the proposal now finds hundreds of weighted events
without extreme weights. The remaining bottleneck is statistical consistency
between sparse source Q tables and the target Q learned on a differently
visited state-action subset. More target budget alone did not resolve that
identifiability problem.

## Risk Mining and DETOUR transfer

On the same passing layout, the independent discovery chain uses a uniform
240-scenario candidate library and evaluates only the three AV targets. At
B=50, mean critical-event recall is 21.48% for Random, 73.15% for DETOUR, and
100% for Risk Mining. At B=20, direct Mining-DETOUR fusion reaches 32.56% versus
31.52% for Risk Mining; by B=50 all Mining variants tie. The revised benchmark
therefore reveals strong Risk Mining transfer but only a small intermediate-budget
fusion benefit.
