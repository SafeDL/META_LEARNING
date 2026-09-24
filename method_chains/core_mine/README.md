# CoRe-Mine

CoRe-Mine combines function-specific source hypotheses with a residual Gaussian
process and marginal coverage acquisition. It evaluates cached Highway-env
response banks under a fixed target query budget.

The frozen experiment found no reliable gain over the strongest matched
FPS-Marginal baseline, so the protocol did not authorize additional physical
simulation. This negative result is retained because it determines the current
research decision.

```powershell
conda run -n metadrive python -m pytest method_chains/core_mine/tests -q -p no:cacheprovider
conda run -n metadrive python -m method_chains.core_mine.experiment
```

Formal artifacts for the original candidate are under
`results/method_chains/core_mine/`. The `develop/` and `validate/` directories
are the declared data splits; `confirm/records.csv` records that physical
confirmation was not executed.

A later, separate corrected-geometry study tested a narrower historical
blind-spot problem at **B=50**. Its simple source-mean plus target-residual
variant has a conditional gain for VI/TTC, but the full compositional method
still has no demonstrated extra benefit. See
`docs/core_mine_source_safe_protocol.md` and
`results/method_chains/core_mine/studies/source_safe/research_decision.md`. A control
frequency audit shows most VI/TTC events persist from 5 to 20 Hz while many
MCTS-CV events disappear; it is a diagnostic, not a new matched B=50 study.
An independent one-seed 20 Hz confirmation physically executes every target
query and is recorded under `results/method_chains/core_mine/studies/source_safe/online20/`;
it supports target-feedback correction but not a strong gain over target-only
learning. The current work is not yet a validated software-version regression
study.
The old negative experiment above remains part of the research history.
The latest raw-static development control and its limits are summarized in
`results/method_chains/core_mine/studies/source_raw_static_development/research_decision.md`.

```powershell
conda run -n metadrive python -m method_chains.core_mine.source_safe_experiment --stage analyze
conda run -n metadrive python -m method_chains.core_mine.control_frequency_audit --stage all --workers 4
conda run -n metadrive python -m method_chains.core_mine.plot_source_safe_evidence
conda run -n metadrive python -m method_chains.core_mine.online20_confirmation --stage verify
```
