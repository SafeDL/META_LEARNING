# CoRe-Mine result index

All CoRe-Mine studies share this result root. The original cached experiment
and its frozen develop, validate, and confirm splits remain at the root.
Subsequent studies are grouped under `studies/` by their research question.
The result data are preserved; descriptive folder names and references were
updated where the old paths included stage-version suffixes.

## Current research evidence

- [Fresh-seed B=50 mode-shift confirmation](studies/mode_shift_fresh_confirmation/research_decision.md)
- [Target-TTC ablation](studies/mode_label_ablation/research_decision.md)
- [Raw historical-score static comparison](studies/source_raw_static_development/research_decision.md)
- [Fresh-seed comparison with static and UCB1 allocators](studies/mode_label_fresh_confirmation/research_decision.md)
- [Retrospective replay across paired IDM/FVDM controller revisions](studies/mode_label_revision_offline/research_decision.md)
- [Corrected evidence-gate development](studies/evidence_gate_development/research_decision.md)
- [Prior-art scope](studies/prior_art/scope.md)

The invalid first evidence-gate attempt was replaced by the corrected development run; its original files are stored under `archives/experimental_results/` outside the current result root.

The fresh-seed result supports a simulator-specific gain over static
rankings, but it does not show a reliable advantage over UCB1. The result
does not validate software-release-pair regression testing.
The retrospective paired-revision replay also does not establish a consistent
gain for mode-label feedback over within-mode quantiles or UCB1.
