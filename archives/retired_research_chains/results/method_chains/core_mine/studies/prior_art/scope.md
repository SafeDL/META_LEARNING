# Prior-art boundary for the source-safe regression task

The original [SPECTRE paper](https://link.springer.com/chapter/10.1007/978-3-030-88106-1_4)
and [author repository](https://github.com/ssbse2021/SPECTRE) establish that
prioritizing new ADS-version scenarios from old-version execution data is
already prior work. The repository was inspected at commit
`3ed99cee5aa7143fa529c69a9770a90318a97dba` (not copied into the method
implementation). Its `ScenarioSearching/Problems/problem.py` evaluates a
suite using collision occurrence, potential collision probability, demand,
diversity, and a weighted position-sensitive combination, then its
`IBEA_Selection.py` applies IBEA. The code reads a dataset-specific absolute
Windows path and uses 50,000-index bounds, so it cannot be run faithfully on
the present 320-scenario IDM pool by changing only a file path.

The [author dataset description](https://github.com/ssbse2021/SPECTRE/blob/main/scenarios/README.md)
specifies 19 properties and Apollo 5.0/LGSVL scenarios. The present study
does not contain equivalent old-run collision probability or ADS-demand
attributes. Substituting TTC for SPECTRE's collision probability would
collapse the comparison into our own proposed historical-margin signal;
calling such a proxy a faithful SPECTRE baseline would be misleading.

The possible narrower research gap is therefore **not** "reuse previous
results". It is whether *all-source-event-free* tests, where old collision
occurrence is constant, can be usefully prioritized by continuous safety
margin with only B=50 new executions. This task distinction still requires
a stronger empirical baseline with honestly mapped observable attributes,
fresh held-out validation, and demonstration beyond simple TTC ranking
before a publication-grade novelty claim is warranted.

Additional primary-source checks (2026-09-24):

- [Corso and Kochenderfer, AAAI 2021](https://ojs.aaai.org/index.php/AAAI/article/view/16876)
  already transfer safety-validation knowledge from previously tested
  systems to changed systems using source value functions and learned
  attention weights. Its sequential MDP/action-value setting differs from
  this fixed replay suite, but neither "transfer from old controllers" nor
  "learn weights over historical hypotheses" is standalone novelty for
  CoRe-Mine. Any new composition claim must beat its own mean-history and
  target-only ablations, not merely random search.
- [Uesato et al., ICLR 2019](https://openreview.net/pdf?id=B1xhQhRcK7)
  use a continuation strategy in simulated driving that learns failure
  modes in related, less robust agents. This is a close warning against
  claiming that cross-controller failure transfer itself is new. Their
  objective includes adversarial failure discovery and probability
  estimation, not prioritizing a fixed all-source-safe replay suite, so
  the tasks are not interchangeable.
- [DETOUR](https://www.sciencedirect.com/science/article/pii/S0167642326000560)
  clusters road-shape tests and favors unexecuted roads near *previously
  failing* roads. Its [author implementation](https://github.com/cetinkaya/detour)
  is for road-coordinate inputs and off-lane failure. Here all historical
  candidates passed, road geometry is fixed, and the endpoint is interaction
  collision/near miss. A direct DETOUR run would have neither matching
  features nor historical failing exemplars.
- [STRAP](https://arxiv.org/abs/2209.01546) prioritizes recorded driving
  segments by scene coverage and rarity after reduction. Its
  [author repository](https://github.com/ITSEG-MQ/STRAP) expects Apollo/SVL
  recordings and a richer scene schema; the present bank has neither those
  recordings nor the module-level features. A five-mode rarity heuristic
  could be tested but must be labeled an adapted proxy, not STRAP.
- [SDC-Prioritizer](https://arxiv.org/abs/2107.09614) uses static road
  features and simulation-time/diversity objectives. The present road is
  fixed, so its road-feature distinctions are absent. The frozen
  `RiskDiverse-Static` comparison is a deliberately labeled, available-
  feature heuristic, not a faithful published implementation.

This audit strengthens the case for reporting the available-input baselines
honestly, but it does not prove the all-source-safe problem or its simple
mode-quantile solution is unprecedented. The newer FVDM validation further
shows mode quantiles do not reliably beat raw historical-margin ranking
across controller families. The empirical contribution remains narrow.
