# AdaTE to highway-env deviations

This isolated replication keeps the paper's mechanism but does not claim to
reproduce its overtaking data, NADE distribution, or numerical table.  The
implementation was cross-checked against upstream commit
`39ef2b4c6593e4ef805b702a8dbbf436d3ae790a` of
`THU-AI-Testing/Adaptive-Testing-DenseRL`; no upstream source code or data is
vendored here (the upstream project is PolyForm Noncommercial 1.0.0).

- The original cross-target archive uses the repository's two-vehicle cut-in
  dynamics. The new rare-event experiment adds a three-vehicle passing layout:
  the AV passes while a BV cuts out from behind a slower LV. The paper controls
  both LV and BV accelerations; the final highway-env pilot controls only the
  BV residual `[-6, 0, 2] m/s²`. A nine-action LV/BV product was tested and
  removed because scheduled BV motion made the LV action causally irrelevant.
- The original archive declares φ as `[0.1, 0.8, 0.1]`. The rare-event passing
  experiment declares `[0.01, 0.98, 0.01]` and a non-uniform initial-state
  distribution. Neither distribution is estimated from naturalistic data, so
  reported quantities remain reference-distribution estimates rather than a
  real-world NDE crash rate.
- The finite-horizon implementation uses terminal crash reward one and
  `gamma=1`. The uploaded paper's convergence statement assumes `gamma<1`;
  that proof is not claimed for this configuration.
- The learner preserves every physical transition, starts target episodes by
  replaying a source-derived critical snapshot, gates updates only by the
  source-derived critical mask, and uses expected φ bootstrap. It neither uses
  `max_a Q` nor treats adaptive η as the bootstrap policy.
- Source Q_j values are estimated from source-only rollouts under a declared
  finite state encoder. Unlike the upstream complete tensor grid, a state that
  is not jointly observed for all source tables remains explicitly unknown and
  evaluation falls back to φ. `source_q_known_steps` records this coverage;
  unknown values are never silently replaced by zero.
- Source profiles now use paired background action plans for the same scenario.
  This prevents unrelated random action sequences from creating artificial
  disagreement and improved joint Q coverage in the passing screening study.
- The adaptation QP follows paper Eq. (11) over the deduplicated set of visited
  critical state-action rows. Upstream `comb_coef.py` additionally builds its
  equivalent quadratic form from maneuver-criticality tensors and updates at a
  fixed separation. This implementation records `qp_update_every` explicitly
  and does not label the two objective constructions identical.
- Evaluation constructs each surrogate policy from M_j=Q_j φ, mixes policies
  as ψα=Σα_jψ_j, and applies defensive mixtures. The passing experiment also
  proposes initial scenarios with q0 proportional to p0 times surrogate
  criticality, then records p0/q0 in the trajectory weight. Initial-state and
  action defense use separate declared strengths because repeated action ratios
  otherwise caused severe finite-sample weight collapse. Adaptation rollouts
  are not reused as evaluation samples.
- The formal protocol compares uniform/adaptive learning behavior and equal/QP
  coefficients at fixed target budget, then evaluates NDE-φ, each single
  surrogate, the equal mixture, and learned QP mixtures separately. CI, RHW,
  ESS, weight mean and maximum are reported rather than treating a single
  point estimate as a convergence proof.
