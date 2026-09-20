# Paper-to-highway result alignment

| Paper result | Highway-env analogue | Status / boundary |
|---|---|---|
| Figure 6(a), four components | `paper_fig06a_component_efficiency.png` | Direct four-way mechanism analogue; three algorithm seeds, actual target calls. |
| Figure 6(b), 1k/2k/3k histories | `paper_fig06b_history_growth.png` | Adapted to 100/300/510 genuine source records; no duplicated history. |
| Figure 7, SEM across systems | `paper_fig07_sem_generalization.png` | Stronger LOSO target-SUT audit: each target is excluded from model fitting and early stopping. |
| Table 2, filter model quality | `paper_table2_filter_models.*` | RBF/MLP/RF execution-preknown baselines versus graph SEM on the same frozen audits. |
| Figure 10, efficiency by ADS | `paper_fig10_efficiency_systems.png` | Six highway controller profiles, three seeds, actual simulation time. |
| Figure 11, error types | `paper_fig11_event_types.png` | Only supported collision and near-miss events; no invented red-light/stuck oracle. |
| Figure 12, code coverage | `paper_fig12_behavioral_coverage.png` | Behavioral acceleration-TTC coverage only; explicitly not code coverage. |
| Figure 14, collided object types | `paper_fig14_mode_distribution.png` | Replaced by physical interaction modes because the harness has one NPC class. |
| Figure 15, 54 scenario patterns | `paper_fig15_interaction_atlas.png` | Representative real trajectories; clusters are not claimed as bugs. |
| Table 3, cost/errors | `paper_table3_highway.*` | Per-SUT wall time and events with mean±SD; CARLA scene-construction costs are not claimed. |

Figures 8, 9 and 13 depend on multi-road urban maps, multiple object classes, or invalid route placement. They are not numerically reproduced in the straight two-lane harness. The existing seed-graph, mutation and trajectory figures cover the supported mechanism without manufacturing unsupported variables.
