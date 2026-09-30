# 2026 novelty decision for source-safe ADS regression prioritization

This is a bounded primary-source audit, not a claim that every paper in
the field has been found. It tests the specific thesis now supported by
the experiments: from a fixed suite that all historical controllers
passed, spend B=50 new-controller executions using historical continuous
safety margins plus online mode-level calibration. The comparison is
between **problem assumptions, information available at decision time,
and measured benefit**, not method names alone.

| Prior primary work | What it already establishes | Meaningful difference here | Consequence for our claim |
| --- | --- | --- | --- |
| [SPECTRE, SSBSE 2021](https://link.springer.com/chapter/10.1007/978-3-030-88106-1_4), [author code](https://github.com/simplexity-lab/SPECTRE) | ADS newer-version scenario selection/prioritization from previous-version execution data; four objectives include collision information, collision probability, demand, and diversity; 60,000-scenario study. | Our candidate pool is deliberately all-source-event-free, uses old TTC-derived margins, and charges each new target query sequentially. SPECTRE's published inputs are not all available in our two-vehicle bank. | **The regression-testing motivation and reuse of history are not novel.** All-old-safe is a potentially distinct *input regime*, not proof of a novel algorithm. An honest available-input comparator or dataset with SPECTRE attributes is still needed. |
| [Corso and Kochenderfer, AAAI 2021](https://ojs.aaai.org/index.php/AAAI/article/view/16876) | Transfers prior system safety-validation knowledge via action values and learned attention in sequential MDP validation. | Our fixed scenario replay suite is a different decision problem from adversarial state/action validation. | Cross-system transfer or source weighting alone cannot be a CoRe novelty claim. The complete CoRe composition also loses to simple baselines in our frozen studies. |
| [SDC-Prioritizer, ACM TOSEM, DOI 10.1145/3533818](https://www.christianbirchler.org/publications/) | Black-box regression-test ordering in self-driving simulators using road geometry, diversity, and test cost, without needing prior executions. | Our road geometry is constant, while interaction parameters and old continuous responses vary; a faithful road-feature implementation would have no discriminating road feature here. | Generic simulation-test prioritization or diversity is not novel. A fixed-road available-feature adaptation must be clearly labelled as such. |
| [STRaP, ESEC/FSE 2022](https://tianyi-zhang.github.io/files/fse2022-ADS-test-reduction.pdf) | Reduces and prioritizes Apollo driving-recording segments by scene coverage and rarity across three maps and regression mutants. | Our unit is a parameterized interaction episode rather than a recorded segment; old safety margin and charged target feedback are available, but Apollo scene-schema fields are absent. | Regression replay and scenario diversity are prior art; our simple synthetic benchmark is less realistic than STRaP's Apollo evaluation. |
| [DETOUR, Science of Computer Programming 2026](https://www.sciencedirect.com/science/article/pii/S0167642326000560) | Clusters roads and prioritizes unexecuted roads near previously failing executed roads for lane departure. | All our historical candidates passed; our road is fixed and endpoint is ego interaction collision/near miss. | DETOUR's failure-neighbor premise is unavailable here, but this does not itself make our method original. |
| [Coverage-Guided Road Selection, SANER 2026](https://conf.researchr.org/details/saner-2026/saner-2026-papers/31/Coverage-Guided-Road-Selection-and-Prioritization-for-Efficient-Testing-in-Autonomous) | Road-section clustering and prioritization using geometry, behavior, difficulty, and historical failures. | Our road shape is fixed and historical failures absent. | The paper further narrows any claim based only on coverage or historical information. |
| [Wuersching et al., ICST 2023](https://portal.fis.tum.de/en/publications/severity-aware-prioritization-of-system-level-regression-tests-in/) | Industry automotive regression prioritization; simple history/cost heuristics were more cost-effective than search/ML with sparse history. | System-level automotive tests are not the same as interaction scenarios in a closed-loop ADS simulator. | Our observation that simple ModeShift can beat a larger GP/CoRe stack is not, by itself, surprising algorithmic novelty. |

## Evidence already in this repository

The mixed-lane B=50 fresh-seed study confirms a **conditional empirical
effect**: ModeShift averages 22.0 collision-bearing physical cells and
41.625 ego collisions per 50 charged queries, versus 20.125 cells and
37.0 collisions for the strongest static mode-quantile comparator.
However, its advantage over that static method on the FVDM target is
only 0.25 cell per seed. Retrospective same-family IDM and FVDM
replays favor static mode quantiles on new-failure yield. The original
CoRe composition, local GP, and marginal objective do not pass their
frozen component gates. A development-only binary event-rate switch
also fails to preserve ModeShift's VI/TTC collision-cell advantage.
More directly, a 400-episode ablation on reused seeds shows that
removing the **target** TTC term from ModeShift leaves collision-cell
yield essentially unchanged (22.125 versus 22.000 cells@50), with
47–50 shared selected cases per unit. Continuous historical margin
ranking remains in both methods. Thus the positive evidence currently
supports online mode allocation from target event labels, not a novel
continuous target residual model.
The endpoint counts physical parameter regions, not distinct bugs.

Therefore the defensible statement is **not** "we invented ADS regression
prioritization," "we invented cross-controller transfer," or "the full
CoRe method is superior." The narrow candidate contribution is an
empirically characterized *all-history-passing, low-budget interaction
regression regime*, in which continuous old safety margin and coarse
online target calibration can help for some controller changes but not
others. Whether that is publication-level novelty remains unproven.

## Evidence required before an algorithmic novelty claim

1. A method must outperform the already strong static mode-quantile and
   ModeShift baselines under a **fresh, frozen** mixed-regime validation,
   not just be a new combination of existing components.
2. The comparison must include a faithful prior method when its required
   inputs exist, or explicitly report why the current data cannot supply
   them and provide an honestly named available-input adaptation. Calling
   a TTC-substituted SPECTRE objective "SPECTRE" would be misleading.
3. At least one real software-release pair or a clearly independent
   controller/scene architecture is needed before broad ADS regression
   effectiveness claims. Current targets are synthetic algorithms or
   parameter changes in one Highway-env road grammar.
4. Collision count, collision/near-miss count, multiple fixed cell
   resolutions, and continuous coverage must all be visible. A method
   that gains one grid's cells by losing many actual collisions has
   already failed in development.

No manuscript should assert that the above requirements have already
been met. A negative empirical paper about when simple historical
ranking beats sophisticated transfer might still be valuable, but it
would be a different contribution from the original CoRe algorithm.

## Reference metadata check

SPECTRE: Lu, Zhang, Yue, Ali, SSBSE 2021, pp. 41–55, DOI
`10.1007/978-3-030-88106-1_4` (Springer). Corso and Kochenderfer:
AAAI 2021, 35(8), pp. 7125–7132, DOI `10.1609/aaai.v35i8.16876`
(AAAI). STRaP: Deng, Zheng, Zhang, Lou, Zhang, ESEC/FSE 2022,
pp. 82–93, DOI `10.1145/3540250.3549152` (author paper and
Macquarie institutional record). SDC-Prioritizer: Birchler et al.,
DOI `10.1145/3533818`; the author publication list gives ACM TOSEM
32(2), Article 28, 2023, while one institutional repository exports
conflicting volume/year fields, so those fields should be rechecked
against the publisher before making a BibTeX entry. Wuersching et al.:
ICST 2023, pp. 398–409, DOI `10.1109/ICST57152.2023.00044`
(TUM institutional record).
