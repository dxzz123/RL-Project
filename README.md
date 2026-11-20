# Mobile charging: pricing and fleet dispatch

This branch studies how a mobile-charging operator should set prices and dispatch a fleet of mobile charging vehicles (MCVs). Each simulated hour follows the same sequence: post a price, observe customer choices, dispatch every MCV, and record operating profit. One episode is a 12-hour day; the reported score is total discounted daily profit in dollars.

`model_1112` now uses one six-method benchmark as its current scorecard. Older README tables described earlier scenarios and PPO variants and are not carried forward here.

## Current demand benchmark
### The six methods

1. **Service-first heuristic** greedily assigns vehicles to currently profitable requests and recharges only when energy is low.
2. **Constant-inventory heuristic** maximizes current profit plus one constant shadow value for stored battery energy.
3. **Time-varying-inventory heuristic** uses a larger battery shadow value early in the day and a smaller value later.
4. **Rollout ADP** scores feasible fleet plans by exact current profit plus Monte Carlo continuation value under the fitted historical market model.
5. **NN ADP** trains a neural continuation-value estimator on a sealed mixed bank of 768 offline trajectory days, then uses it to rank feasible fleet plans.
6. **Warm-started PPO** uses the same sealed 768-day offline bank to initialize the pricing actor/critic and dispatch actor/critic. It then runs 2,048 PPO updates with 16 fresh on-policy days per update (32,768 days per seed). The online phase uses pure PPO with no pricing- or dispatch-behavior-cloning constraint. Realized final profit jointly updates pricing and dispatch through interleaved GAE with separate pre-choice and post-choice critics.


Every demand level uses the same 128 frozen test days. Learned methods average policy seeds 7, 19, and 37 within each test day. Cells report mean discounted profit per day with a 95% percentile-bootstrap confidence interval in brackets. Each interval uses 100,000 resamples of the 128 day-level values.

| Method | Baseline | 20% higher demand | 40% higher demand | 60% higher demand | 80% higher demand |
|---|---:|---:|---:|---:|---:|
| Service-first heuristic | 470.02 [463.34, 476.59] | 404.90 [397.72, 412.10] | 302.55 [292.44, 312.64] | 175.42 [163.46, 187.41] | 43.10 [32.84, 53.35] |
| Constant-inventory heuristic | 457.04 [450.00, 463.99] | 400.83 [393.79, 408.00] | 316.29 [306.59, 325.86] | 216.69 [204.85, 228.43] | 100.82 [90.76, 110.91] |
| **Time-varying-inventory heuristic** | 502.34 [495.27, 509.32] | **445.16 [438.49, 451.89]** | 355.29 [346.06, 364.46] | 265.91 [254.75, 277.04] | 163.81 [154.33, 173.34] |
| Rollout ADP | 491.88 [485.54, 498.06] | 438.43 [432.23, 444.69] | 352.62 [343.65, 361.64] | 255.71 [244.52, 266.78] | 147.44 [138.06, 156.86] |
| **NN ADP** | **502.62 [495.64, 509.51]** | 444.44 [438.08, 450.82] | **361.30 [351.95, 370.68]** | **267.74 [257.00, 278.39]** | **164.99 [155.09, 174.97]** |
| Warm-started PPO | 465.86 [459.25, 472.39] | 398.63 [391.84, 405.42] | 301.62 [292.51, 310.73] | 203.51 [192.12, 214.84] | 87.70 [77.41, 97.97] |

The confidence intervals quantify variation across the frozen test days conditional on the fitted model, offline data, and trained policies. They do not include uncertainty from collecting another offline dataset or retraining the methods. Bold cells identify the largest sample mean, not necessarily a statistically significant advantage. The line chart compares the six methods.

![Six-method demand sweep](docs/assets/Six-Method_Performance_across_Demand_Levels.png)


### Paired ranking summary

Because all methods use the same test days, the leading methods are compared with paired bootstrap intervals. A paired interval containing zero does not establish a difference between the two methods.

| Demand increase | Highest sample mean | Difference from runner-up [95% paired CI] | Interpretation |
|---:|---|---:|---|
| 0% | NN ADP | +0.28 [-3.09, 3.70] vs. time-varying inventory | No clear difference |
| 20% | Time-varying inventory | +0.72 [-2.52, 3.89] vs. NN ADP | No clear difference |
| 40% | NN ADP | +6.01 [2.12, 9.94] vs. time-varying inventory | NN ADP has a clear advantage |
| 60% | NN ADP | +1.82 [-2.52, 6.19] vs. time-varying inventory | No clear difference |
| 80% | NN ADP | +1.18 [-3.88, 6.05] vs. time-varying inventory | No clear difference |

### Main conclusions

- NN ADP and time-varying inventory form the leading group across the demand sweep, with NN ADP showing a statistically distinguishable advantage only at the \(+40\%\) demand level.
- NN ADP remains above Rollout ADP at every demand level, with paired intervals excluding zero throughout the sweep.
- Warm-started PPO remains profitable through +80% demand after 2,048 updates, but its gap from the leading group widens as demand increases.
- Inventory-aware dispatch is substantially more robust to demand growth than service-first dispatch, which deteriorates most sharply under high demand.

The updated long-training protocol, learning-curve comparison, limitations, and artifact links are in [Study 16](docs/studies/16_long_horizon_offline_warm_ppo.md). The original 512-day benchmark remains documented in [Study 13](docs/studies/13_offline_warm_ppo.md). Machine-readable artifacts: [mean table](docs/results/demand_sweep_0_80_six_methods.csv) and [confidence intervals](docs/results/demand_sweep_0_80_six_methods_ci.csv).

### Fleet-size sensitivity

Study 14 evaluates all six methods under baseline demand on one shared 128-day test bank. NN ADP and Warm-started PPO are retrained for fleet sizes 5 and 15 because their network dimensions depend on the number of vehicles. Every PPO seed uses the matched 768-day offline warm start followed by 2,048 PPO updates of 16 fresh on-policy days each (32,768 days per seed).

| Method | Fleet size = 5 | Fleet size = 10 | Fleet size = 15 |
|---|---:|---:|---:|
| Service-first heuristic | 347.28 | 470.02 | 562.37 |
| Constant-inventory heuristic | 386.90 | 457.04 | 540.32 |
| Time-varying-inventory heuristic | 431.97 | 502.34 | 567.24 |
| Rollout ADP | 423.20 | 491.88 | **573.41** |
| NN ADP | **436.09** | **502.62** | 555.62 |
| Warm-started PPO (2,048 updates) | 399.98 | 465.86 | 516.54 |

The exact fleet layouts, training protocol, validation counts, and limitations are in [Study 14](docs/studies/14_fleet_size_sensitivity.md).

### Warm-started PPO ablation at fleet size 10

| Ablation | Offline days | PPO updates | On-policy PPO days | Mean discounted profit/day |
|---|---:|---:|---:|---:|
| Warm-start only | 768 | 0 | 0 | 274.11 |
| PPO only | 0 | 2,048 | 32,768 | 439.03 |
| Warm-started PPO | 768 | 2,048 | 32,768 | **465.86** |

At the matched 2,048-update budget, Warm-started PPO improves on Warm-start only by 191.75 per day and on PPO only by 26.83 per day. PPO from random initialization becomes profitable with this much simulator experience, but the warm start retains a measurable advantage. Full protocol and limitations: [Study 15](docs/studies/15_offline_warm_ppo_ablation.md).

## Quick start

Python 3.10 or later is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[train]'

python examples/manual_walkthrough.py
```

Run the tracked NN ADP studies with:

```bash
python experiments/run_logged_nn_adp_study.py --help
python experiments/run_candidate_adp_study.py --help
```

## Documentation

| Guide | Contents |
|---|---|
| [MODEL.md](docs/MODEL.md) | Customer choice, pricing, fleet state, batteries, and profit |
| [METHODS.md](docs/METHODS.md) | How the controllers decide and what information they use |
| [Study 16](docs/studies/16_long_horizon_offline_warm_ppo.md) | Current six-method scorecard with 2,048-update Warm-started PPO |
| [Study 15](docs/studies/15_offline_warm_ppo_ablation.md) | Warm-start-only, PPO-only, and Warm-started PPO at 2,048 updates |
| [Study 14](docs/studies/14_fleet_size_sensitivity.md) | Fleet-size sensitivity with 2,048-update Warm-started PPO |
| [Study 13](docs/studies/13_offline_warm_ppo.md) | Original 512-day Offline-warm PPO benchmark |
| [RESULTS.md](docs/RESULTS.md) | Current scorecard followed by clearly labeled historical studies |
| [RUNNING.md](docs/RUNNING.md) | Commands for the maintained experiment workflows |

## Code map

| Component | Main files |
|---|---|
| Heuristic policies | `src/charging_env/benchmarks/policies.py`, `src/charging_env/benchmarks/scalable.py` |
| Rollout ADP | `src/charging_env/rl/rollout_adp.py` |
| NN ADP | `src/charging_env/rl/candidate_value.py`, `src/charging_env/rl/candidate_training.py` |
| PPO actors, critics, and trainer | `src/charging_env/rl/paper_ppo.py`, `src/charging_env/rl/paper_train.py` |
| NN ADP experiment runners | `experiments/run_logged_nn_adp_study.py`, `experiments/run_candidate_adp_study.py` |

## Evidence boundaries

The histories and test days are synthetic. Controllers are frozen during testing, invalid actions are rejected, and learned controllers cannot query future private demand. Results compare the implemented controllers under this protocol; they are not evidence that one algorithm family dominates in every environment or at equal compute.
