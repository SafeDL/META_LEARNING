# FBRT scenario source mapping

All simulated parameters are in SI units. Every continuous interval in
`scenario_catalogue_v2.yaml` is a project `RESEARCH_RANGE`; no interval is
presented as a national standard limit or a fitted natural-driving percentile.
The source identifiers retain the distinction between national standards,
Shenzhen's local standard, Euro NCAP protocol versions, and research papers.

| ID | Evidence used | Source and location | Scope limit |
|---|---|---|---|
| S01 | Cut-in structure; 8–60 m and 1.5–3.0 s are project ranges | DB4403/T 359.1-2023 C.4.3.3.2; Euro NCAP Safe Driving—Vehicle Assistance v1.2 (July 2026), 2.1.4; Feng et al. (2022); Li et al. (2024); Karunakaran et al. (2024) | Protocol speed/TTC anchors are not reproduced; the time scale is the installed controller gain scale, not measured lane-change duration |
| S02 | Lead cuts out to reveal a static target; TTC at cut-out start | DB4403/T 359.1-2023 C.4.3.3.5; Euro NCAP Safe Driving—Vehicle Assistance v1.2, 2.1.5 | State observation only; not a perception occlusion test; project ranges are adapted |
| S03 | Lead brakes to a stop | DB4403/T 359.1-2023 C.4.3.3.6.2; Euro NCAP Safe Driving—Vehicle Assistance v1.2, CCRb; AEB Car-to-Car v4.3.1 (February 2024), 8.2.2.3 | Existing legacy contract retains its instantaneous-brake behavior; the catalogue range is a project range |
| S04 | Stop, hold, restart phases | DB4403/T 359.1-2023 C.4.3.3.4.2 | The shared scripted 2 s hold is an adaptation; it is not an exact replay of every state-dependent clause |
| S05 | Ego considers a lane change with a closing rear vehicle | Euro NCAP Lane Support Systems v4.3 (December 2023), 7.4 | The ego chooses whether to change lanes; this is not the protocol's prescribed trajectory |
| S06 | Same-lane slower/moving lead | Euro NCAP Safe Driving—Vehicle Assistance v1.2, CCRm; DB4403/T 359.1-2023 C.4.3.3.8 | The selected 20–27 m/s lead range follows the available policy action contract and is a project range |
| S07 | Static lead structure | Euro NCAP Safe Driving—Vehicle Assistance v1.2, CCRs; AEB Car-to-Car v4.3.1 | Candidate only; not selected by the current recipe |
| S08 | Cut-in followed by braking as a composed interaction | DB4403/T 359.1-2023 C.4.3.3.2 and C.4.3.3.6; Li et al. (2024); Karunakaran et al. (2024) | This combined sequence is a project design, not a claimed standard test case; braking starts after measured merge completion |
| S09 | Cut-out and adjacent traffic after road release | DB4403/T 359.1-2023 C.4.3.3.3 | Candidate only; the adjacent-rear range is a project extension |
| S10 | Bounded target-lane gap with a front and rear vehicle | DB4403/T 359.1-2023 C.4.3.3.2; Euro NCAP Lane Support Systems v4.3, 7.4 | Project combination; candidate only |
| S11 | Static vehicle partially occupying the lane | DB4403/T 359.1-2023 C.4.3.3.1.1 | Candidate only; polygon geometry is required before execution |
| S12 | Successive cut-ins and target switching | DB4403/T 359.1-2023 C.4.3.3.2; Li et al. (2024); Karunakaran et al. (2024) | Project composition; candidate only |
| S13 | Curved road radius range | DB4403/T 359.1-2023 C.4.3.3.9.1 | Deferred; current runner is straight-road only |
| S14 | Same-lane powered two-wheeler target structure | DB4403/T 359.1-2023 C.4.3.3.7; Euro NCAP Safe Driving—Vehicle Assistance v1.2 | Deferred; small-target geometry and policy capability are not implemented |

## Source register

- **S1 — GB/T 41798-2022**, *Intelligent and connected vehicles—Test methods and requirements for automated driving functions*. National field-test framework; the plan records the official listing as current, but does not claim its complete parameter tables were obtained. [Official listing](https://openstd.samr.gov.cn/bzgk/std/newGbInfo?hcno=C3FD7FF23C6D06A9F7459DCD73E68905)
- **S2 — GB/T 47025-2026**, *Intelligent and connected vehicles—Simulation test methods and requirements for automated driving functions*. National simulation-test reference; the plan records official metadata only. [Official listing](https://openstd.samr.gov.cn/bzgk/std/newGbInfo?hcno=F1D96EE9F6E84D109F1AC57BDF7A1412)
- **S3 — DB4403/T 359.1-2023**, Shenzhen local standard, Part 1: highways and expressways. [Official publication](https://amr.sz.gov.cn/gkmlpt/content/10/10800/post_10800873.html) · [Official PDF](https://amr.sz.gov.cn/attachment/1/1342/1342711/10800873.pdf)
- **S4 — Euro NCAP Safe Driving—Vehicle Assistance Test & Assessment Protocol v1.2, July 2026.** [Protocol PDF](https://cdn.euroncap.com/cars/assets/Euro_NCAP_Protocol_Safe_Driving_Vehicle_Assistance_v1_2_b6fb486fa6.pdf)
- **S5 — Euro NCAP AEB Car-to-Car Test Protocol v4.3.1, February 2024.** Fixed historical version used as a reproducibility anchor. [Protocol PDF](https://cdn.euroncap.com/cars/assets/euro_ncap_aeb_c2c_test_protocol_v431_532926aad1.pdf)
- **S6 — Euro NCAP Lane Support Systems Test Protocol v4.3, December 2023.** [Protocol PDF](https://cdn.euroncap.com/cars/assets/euro_ncap_lss_test_protocol_v43_f2ddd5f6d6.pdf)
- **L1 — Feng et al.**, “Testing Scenario Library Generation for Connected and Automated Vehicles: An Adaptive Framework,” *IEEE T-ITS*, 2022. [DOI record and preprint](https://arxiv.org/abs/2003.03712)
- **L2 — Li et al.**, “面向自动驾驶仿真测试的高覆盖切入场景库生成方法,” *中国公路学报*, 2024. The plan uses the accessible abstract for parameterization structure and does not infer unavailable numerical tables. [Journal page](https://zgglxb.chd.edu.cn/CN/10.19721/j.cnki.1001-7372.2024.07.019)
- **L3 — Karunakaran et al.**, “Generating Edge Cases for Testing Autonomous Vehicles Using Real-World Data,” *Sensors*, 24(1):108, 2024. [Publisher full text](https://www.mdpi.com/1424-8220/24/1/108)
- **R1/R2 — Failure-Based Testing and Search for Boundary.** The implementation uses the failure-region memory idea and does not claim to reproduce a complete external testing system. [ICST 2026 keynote](https://conf.researchr.org/details/icst-2026/icst-2026-keynote/3/Failure-Based-Testing) · [Paper](https://arxiv.org/abs/2007.15231)

The active five-card recipe is frozen by static action capability before any
method comparison. The PPO speed set is read from the installed
`highway-env` package; no zero-speed action is available, so S03/S04 are not
used in the shared bank. S1/S2 establish national-standard context only; local
scenario clauses and Euro NCAP protocol anchors remain separately identified.
