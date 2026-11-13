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
- Test regimes: demand increases of 0%, 20%, 40%, 60%, and 80%.
- Test days: the same 128 frozen days at every demand level, shared across all methods.
- NN ADP offline training: 768 trajectory days.
- PPO warm start: the same 768 training days plus 96 offline validation days.
- PPO on-policy phase: 512 days per seed.
- Test execution: frozen policies; no refitting or test-time tuning.
- Guardrails: no invalid actions and no private future-demand queries.

## Results

The table reports mean discounted profit per test day. The line chart intentionally compares the five non-PPO methods, while this complete table retains Offline-warm PPO.

| Method | Baseline | 20% higher demand | 40% higher demand | 60% higher demand | 80% higher demand |
|---|---:|---:|---:|---:|---:|
| Service-first heuristic | 470.02 | 404.90 | 302.55 | 175.42 | 43.10 |
| Constant-inventory heuristic | 457.04 | 400.83 | 316.29 | 216.69 | 100.82 |
| **Time-varying-inventory heuristic** | 502.34 | **445.16** | 355.29 | 265.91 | 163.81 |
| Rollout ADP | 491.88 | 438.43 | 352.62 | 255.71 | 147.44 |
| **NN ADP (offline trajectories)** | **502.62** | 444.44 | **361.30** | **267.74** | **164.99** |
| Offline-warm PPO | 345.64 | 282.99 | 180.28 | 80.42 | -38.82 |

### Demand-sweep findings

- NN ADP has the highest mean profit at 0%, +40%, +60%, and +80% demand.
- Time-varying inventory is highest at +20% demand and remains close to NN ADP throughout the sweep.
- Offline-warm PPO remains below all five alternatives and falls below zero at +80% demand.

The matched warm start therefore does not eliminate PPO's performance gap in this experiment. This is a result about the implemented architecture, training budget, and scenario—not a general claim that PPO cannot work for joint pricing and dispatch.

## Artifacts

- [Five-method demand-sweep figure](../assets/demand_sweep_0_80_five_methods.png)
- [Complete six-method result table](../results/demand_sweep_0_80_six_methods.csv)

The CSV contains full-precision means for all six methods and five demand levels.

## Limitations

- Learned-method means average three fixed policy seeds and do not capture retraining uncertainty.
- The offline trajectories and all test days are synthetic.
- The comparison is not equal-compute: Rollout ADP simulates futures online, while NN ADP and PPO use network inference.
- Five specified demand levels are tested; the results do not establish robustness to arbitrary model error or real-world transfer.
