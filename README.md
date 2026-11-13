# Mobile charging: pricing and fleet dispatch

This branch studies how a mobile-charging operator should set prices and dispatch a fleet of mobile charging vehicles (MCVs). Each simulated hour follows the same sequence: post a price, observe customer choices, dispatch every MCV, and record operating profit. One episode is a 12-hour day; the reported score is total discounted daily profit in dollars.

`model_1112` now uses one six-method benchmark as its current scorecard. Older README tables described earlier scenarios and PPO variants and are not carried forward here.

## Current 0%–80% demand benchmark

The benchmark is anchored to commit `574ef42b1c3d5a4ce1dc9998a5670e89eb5fc452`. Every demand level uses the same 128 frozen test days. Learned methods average policy seeds 7, 19, and 37.

| Method | 0% | +20% | +40% | +60% | +80% |
|---|---:|---:|---:|---:|---:|
| Service-first heuristic | 470.02 | 404.90 | 302.55 | 175.42 | 43.10 |
| Constant-inventory heuristic | 457.04 | 400.83 | 316.29 | 216.69 | 100.82 |
| **Time-varying-inventory heuristic** | 502.34 | **445.16** | 355.29 | 265.91 | 163.81 |
| Rollout ADP | 491.88 | 438.43 | 352.62 | 255.71 | 147.44 |
| **NN ADP (offline trajectories)** | **502.62** | 444.44 | **361.30** | **267.74** | **164.99** |
| Offline-warm PPO | 345.64 | 282.99 | 180.28 | 80.42 | -38.82 |

Values are mean discounted profit per test day. The line chart compares the five non-PPO methods; Offline-warm PPO remains in the complete result table.

![Five-method demand sweep](docs/assets/demand_sweep_0_80_five_methods.png)

### The six methods

1. **Service-first heuristic** greedily assigns vehicles to currently profitable requests and recharges only when energy is low.
2. **Constant-inventory heuristic** maximizes current profit plus one constant shadow value for stored battery energy.
3. **Time-varying-inventory heuristic** uses a larger battery shadow value early in the day and a smaller value later.
4. **Rollout ADP** scores feasible fleet plans by exact current profit plus Monte Carlo continuation value under the fitted historical market model.
5. **NN ADP** trains a neural continuation-value estimator on a sealed mixed bank of 768 offline trajectory days, then uses it to rank feasible fleet plans.
6. **Offline-warm PPO** uses the same sealed 768-day offline bank to initialize the pricing actor/critic and dispatch actor/critic. It then runs 512 fresh on-policy PPO training days. Realized final profit jointly updates pricing and dispatch through interleaved GAE with separate pre-choice and post-choice critics.

### Main findings

- NN ADP has the highest mean profit at 0%, +40%, +60%, and +80% demand.
- Time-varying inventory is highest at +20% demand and stays close to NN ADP across the sweep.
- Offline-warm PPO remains below all five alternatives and falls below zero at +80% demand.

The full protocol, limitations, and artifact links are in [Study 13](docs/studies/13_offline_warm_ppo.md). The machine-readable table is [demand_sweep_0_80_six_methods.csv](docs/results/demand_sweep_0_80_six_methods.csv).

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
| [Study 13](docs/studies/13_offline_warm_ppo.md) | Current six-method protocol and results |
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
