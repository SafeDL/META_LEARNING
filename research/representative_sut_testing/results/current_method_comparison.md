# Current completed method comparison

B=200; 6 fixed SUTs, 2 shared pools, selection seed 11. Development means only.

| Method | Area % | Recall % | Mean F200 |
|---|---:|---:|---:|
| Frozen original | 52.837 | 76.885 | 118.92 |
| Task-mean Gaussian, unprotected | 52.987 | 77.592 | 118.67 |
| Student-t, unprotected | 53.025 | 77.473 | 118.50 |
| Student-t, protected | 53.048 | 77.157 | 119.25 |
| EP event score | 53.051 | 77.067 | 119.75 |
| Event signs, protected | 53.088 | 77.011 | 119.50 |
| Safe risk order, protected | 53.075 | 76.945 | 119.42 |
| Adaptive evidence scope | 53.070 | 77.011 | 119.50 |

| SUT | Frozen Area % | Protected Student Area % | Frozen Recall % | Protected Student Recall % |
|---|---:|---:|---:|---:|
| idm_ref | 12.461 | 12.265 | 25.000 | 24.620 |
| fvdm_target | 67.768 | 67.498 | 98.936 | 97.836 |
| mobil_ref_v2 | 78.967 | 79.009 | 100.000 | 100.000 |
| vi_ttc_ref_audit_v4 | 26.917 | 27.682 | 53.904 | 55.001 |
| mcts_cv_ref_audit_v4 | 53.958 | 54.786 | 83.468 | 85.484 |
| ppo_ref_v2 | 76.950 | 77.050 | 100.000 | 100.000 |
