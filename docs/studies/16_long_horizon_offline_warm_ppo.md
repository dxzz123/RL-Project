# Study 16: Long-horizon Offline-warm PPO

- **Date:** 17 November 2026
- **Status:** current `model_1112` PPO scorecard
- **Training source commit:** `574ef42b1c3d5a4ce1dc9998a5670e89eb5fc452`

## Question

Does Offline-warm PPO continue improving when the pure on-policy phase is extended from 32 updates to 2,048 updates, without retaining a decaying behavior-cloning constraint on either actor?

## Method

Each policy seed starts from the same sealed offline warm-start checkpoint used in Study 13:

- 768 offline trajectory days train the pricing actor, dispatch actor, pre-choice critic, and post-choice critic;
- 96 separate offline days monitor and select the warm-start checkpoint;
- online training then uses pure PPO: no pricing behavior-cloning loss, no dispatch behavior-cloning loss, no potential shaping, and no counterfactual auxiliary loss;
- each PPO update collects 16 fresh complete 12-hour on-policy days;
- 2,048 updates therefore collect 32,768 on-policy days per seed;
- pricing and dispatch advantages are computed with the same interleaved GAE implementation;
- seeds 7, 19, and 37 are trained independently;
- the final update-2,048 checkpoint is tested, rather than selecting a checkpoint from the test results.

The actor learning rate is 0.0001, critic learning rate is 0.001, PPO clip ratio is 0.2, GAE lambda is 1.0, dispatch entropy coefficient is 0.005, target KL is 0.03, and each update uses six optimization epochs with minibatches of 192 transitions.

## Training-length comparison

All rows use the same offline warm start, PPO hyperparameters, three policy seeds, and 128 baseline test days. Values are mean discounted profit per day.

| Pure-PPO updates | On-policy days per seed | Mean discounted profit/day |
|---:|---:|---:|
| 32 | 512 | 345.64 |
| 128 | 2,048 | 376.32 |
| 512 | 8,192 | 417.28 |
| **2,048** | **32,768** | **465.86** |

The update-2,048 policy improves on update 512 by 48.57 per day (11.64%) and is higher on 127 of the 128 paired baseline test days.

### Seed-level baseline results

| Policy seed | 512 updates | 2,048 updates | Difference |
|---:|---:|---:|---:|
| 7 | 415.88 | 464.93 | +49.05 |
| 19 | 418.07 | 467.27 | +49.20 |
| 37 | 417.91 | 465.37 | +47.46 |

The validation curves continue rising well beyond 512 updates. Their highest monitoring values occur at updates 1,984, 1,936, and 2,000 for seeds 7, 19, and 37. All three final update-2,048 values are slightly below their own monitoring peaks, indicating that the curve is approaching a plateau.

## Updated demand benchmark

Every demand level uses the same 128 frozen test days. The PPO row averages policy seeds 7, 19, and 37. The other five rows reuse their previously frozen Study 13 results.

| Method | Baseline | 20% higher demand | 40% higher demand | 60% higher demand | 80% higher demand |
|---|---:|---:|---:|---:|---:|
| Service-first heuristic | 470.02 | 404.90 | 302.55 | 175.42 | 43.10 |
| Constant-inventory heuristic | 457.04 | 400.83 | 316.29 | 216.69 | 100.82 |
| **Time-varying-inventory heuristic** | 502.34 | **445.16** | 355.29 | 265.91 | 163.81 |
| Rollout ADP | 491.88 | 438.43 | 352.62 | 255.71 | 147.44 |
| **NN ADP (offline trajectories)** | **502.62** | 444.44 | **361.30** | **267.74** | **164.99** |
| Offline-warm PPO, 2,048 updates | 465.86 | 398.63 | 301.62 | 203.51 | 87.70 |

The longer PPO phase raises the five demand-level values by 120.22, 115.64, 121.34, 123.09, and 126.52 relative to Study 13's 32-update PPO. Profit remains positive at +80% demand. NN ADP remains the strongest method at every demand level; the long-horizon PPO remains below the strongest non-PPO method at each level.

## Evaluation checks

- 15 policy-condition evaluations: five demand levels by three policy seeds.
- 128 complete test days in every evaluation, for 1,920 policy episodes.
- No invalid actions.
- No private future-demand model queries.
- Checkpoint hashes match the saved training metadata.
- The same action-randomness seed and stochastic evaluation convention are used as in Study 13.

## Artifacts

- [Complete six-method result table](../results/demand_sweep_0_80_six_methods.csv)
- [Compact PPO demand-sweep summary](../results/ppo_2048_demand_sweep_summary.json)
- [Five non-PPO method chart](../assets/demand_sweep_0_80_five_methods.png)

The chart intentionally remains a five-method chart, matching the existing README convention. The full table contains the updated PPO results.

## Limitations

- The 32-, 128-, 512-, and 2,048-update policies were successively inspected on the same 128-day bank. The training-length curve is therefore exploratory and the update count has not been confirmed on a new untouched test bank.
- The final update-2,048 checkpoint is reported consistently, but validation suggests that per-seed early stopping near updates 1,936–2,000 could perform differently.
- More training uses substantially more simulator interaction; this is not an equal-compute comparison.
- Learned-method means average three fixed seeds and do not quantify uncertainty from collecting another offline dataset or retraining with new seeds.
- All trajectories and test days are synthetic; the results do not establish real-world transfer.
