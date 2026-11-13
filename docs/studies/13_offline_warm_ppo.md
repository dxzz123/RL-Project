# Study 13: Offline-warm PPO against five matched baselines

- **Date:** 12 November 2026
- **Status:** current `model_1112` scorecard
- **Source commit:** `574ef42b1c3d5a4ce1dc9998a5670e89eb5fc452`

## Question

Can PPO close the gap to three heuristic controllers, Rollout ADP, and NN ADP when its pricing and dispatch networks are initialized from the same sealed offline trajectory bank used by NN ADP, then jointly trained on fresh on-policy experience?

## Methods

| Method | Decision rule |
|---|---|
| Service-first heuristic | Greedily serve profitable requests; recharge below a tuned energy threshold |
| Constant-inventory heuristic | Current profit plus one constant shadow value for stored battery energy |
| Time-varying-inventory heuristic | Current profit plus a larger early-day and smaller late-day battery shadow value |
| Rollout ADP | Exact current profit plus Monte Carlo continuation value for candidate fleet plans |
| NN ADP | Neural continuation value fitted from the sealed offline trajectories, used to rank candidate plans |
| Offline-warm PPO | Offline initialization of pricing/dispatch actors and critics, followed by joint on-policy PPO |

### Offline-warm PPO

The dispatch actor is centralized and produces a categorical distribution for each MCV. The fleet policy factorizes over vehicles, while the PPO clipped objective uses the joint fleet-action probability ratio. Separate pre-choice and post-choice critics estimate future operating value before and after customer choices. Their estimates are combined through interleaved generalized advantage estimation.

The warm start uses the same sealed 768-day mixed trajectory bank as NN ADP:

- the pricing actor clones the logged price vectors;
- the dispatch actor clones the logged fleet operations;
- both critics regress to complete discounted final-profit returns;
- 96 additional offline days select the warm-start checkpoint;
- all four networks then receive 512 fresh on-policy training days;
- realized business profit jointly optimizes pricing and dispatch.

This definition supersedes the older inventory-imitation PPO warm start for the current benchmark.

## Protocol

- Scenario: 25 zones and 10 MCVs under the fitted historical market model.
- Learned-policy seeds: 7, 19, and 37.
- Test regimes: baseline, 40% higher demand, and a spatiotemporal demand shift.
- Test days: 128 per regime, shared across all methods.
- NN ADP offline training: 768 trajectory days.
- PPO warm start: the same 768 training days plus 96 offline validation days.
- PPO on-policy phase: 512 days per seed.
- Test execution: frozen policies; no refitting or test-time tuning.
- Uncertainty: paired-day bootstrap conditional on the fixed trained policies.
- Guardrails: no invalid actions and no private future-demand queries.

## Results

The table reports mean discounted profit per test day. Pairwise bootstrap intervals are reported below.

| Method | Baseline | 40% higher demand | Spatiotemporal shift |
|---|---:|---:|---:|
| Service-first heuristic | 474.21 | 302.55 | 435.62 |
| Constant-inventory heuristic | 458.88 | 316.29 | 420.36 |
| **Time-varying-inventory heuristic** | **503.48** | 355.29 | **461.72** |
| Rollout ADP | 494.35 | 352.62 | 451.54 |
| **NN ADP (offline trajectories)** | 500.44 | **361.30** | 459.10 |
| Offline-warm PPO | 342.19 | 180.28 | 296.07 |

### Paired findings

- NN ADP exceeds Rollout ADP by 6.09 [1.87, 10.17] in the baseline, 8.68 [4.49, 13.01] under higher demand, and 7.56 [3.21, 11.82] after the spatiotemporal shift.
- NN ADP and time-varying inventory are not clearly separated in the baseline or spatiotemporal regime. Under higher demand, NN ADP leads by 6.01 [2.11, 9.97].
- Offline-warm PPO trails the regime winner by 161.28 [155.23, 167.00] in the baseline, 181.01 [174.92, 187.33] under higher demand, and 165.64 [159.76, 171.53] after the shift.

The matched warm start therefore does not eliminate PPO's performance gap in this experiment. This is a result about the implemented architecture, training budget, and scenario—not a general claim that PPO cannot work for joint pricing and dispatch.

## Artifacts

- [Benchmark figure](../assets/offline_warm_ppo_benchmark.png)
- [Machine-readable scorecard](../results/offline_warm_ppo_benchmark_summary.json)

The JSON contains the full precision means, bootstrap intervals, rankings, pairwise comparisons, seeds, and training budgets.

## Limitations

- The bootstrap conditions on three fixed learned-policy seeds and does not capture retraining uncertainty.
- The offline trajectories and all test days are synthetic.
- The comparison is not equal-compute: Rollout ADP simulates futures online, while NN ADP and PPO use network inference.
- Only two specified demand shifts are tested; the results do not establish robustness to arbitrary model error or real-world transfer.
