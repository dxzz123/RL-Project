# 10. Rollout ADP at fixed prices, and extra configurations

**Date:** 2026-10-25 · **Runners:** `experiments/run_general_adp_study.py`, `experiments/run_adp_transfer.py` · **Saved results:** `outputs/general_adp_study_20261025/`, `outputs/adp_transfer_20261025/`

**Question.** Table ADP only works for two vehicles. Can a planner that simulates futures at decision time handle any fleet size, and how does it compare with PPO and the rules?

**Answer.** Yes, it runs on both cities with no training. It beats saved PPO in both, by 180 per day in the large city. At a fixed price it is roughly tied with the bounded inventory rule: slightly ahead in the small city, slightly behind in the large one. It stays below Table ADP and the exhaustive inventory rule in the small city.

## Setup

- **Controller:** [rollout ADP](../METHODS.md#rollout-adp) with up to five candidate plans, eight imagined futures per plan, and service-first as the base policy.
- **Seeds:** planning seeds 7, 19 and 37. These are three repeated runs of an online planner, not training runs.
- **Fixed price:** rollout ADP was tried at $0.60 and $0.80 on 32 validation days per city. It chose $0.60 for the small city and $0.80 for the large one.
- **Rules:** service-first and the bounded inventory rule were retuned over 70 settings, including price.
- **Saved controls, not retrained:** warm-started Joint PPO from studies 08 and 09, and Table ADP in the small city.
- **Test:** 256 days per city shared by all controllers, 5,376 final policy episodes. Plus 256 episodes of the historical exhaustive inventory rule ($0.60, battery values 0.30/0.15) on the small-city days.

## Results

| Controller | Small city | Large city |
|---|---:|---:|
| Table ADP | **136.10** | not applicable |
| Exhaustive inventory rule (reference) | 131.41 | not feasible |
| **Rollout ADP** | 127.78 | 910.21 |
| Joint PPO (saved) | 126.53 | 729.89 |
| Inventory rule (bounded, retuned) | 126.35 | **913.65** |
| Service-first (retuned) | 123.20 | 896.22 |
| Current profit only | 105.64 | 816.39 |

"Current profit only" uses the bounded search with zero battery value, which ignores the future entirely. Its candidate search differs from rollout ADP's, so the gap is not a pure measure of what simulation adds.

Rollout ADP minus each controller, with paired 95% intervals:

| Comparison | Small city | Large city |
|---|---|---|
| − Joint PPO | +1.25 [0.18, 2.30] | +180.32 [175.44, 185.30] |
| − bounded inventory | +1.43 [0.62, 2.29] | −3.45 [−5.15, −1.70] |
| − service-first | +4.58 [2.63, 6.41] | +13.99 [12.42, 15.65] |
| − exhaustive inventory | −3.63 [−4.64, −2.61] | — |

Per planning seed:

| Seed | Small, rollout | Large, rollout | Small, PPO | Large, PPO |
|---|---:|---:|---:|---:|
| 7 | 128.04 | 910.60 | 126.71 | 730.71 |
| 19 | 127.56 | 909.66 | 126.41 | 718.81 |
| 37 | 127.74 | 910.37 | 126.47 | 740.14 |

Large-city operations, fleet kWh per day:

| Controller | Mobile delivered | Recharge bought | Unmet mobile | Battery left above reserve |
|---|---:|---:|---:|---:|
| Rollout ADP | 379.34 | 187.19 | 12.10 | 34.27 |
| Joint PPO | 474.45 | 194.37 | 397.42 | 72.17 |
| Inventory rule (bounded) | 368.47 | 206.47 | 21.31 | 46.75 |
| Service-first | 390.25 | 214.88 | 2.68 | 189.40 |

PPO attracts far more mobile demand than it serves, leaving 397 kWh unmet per day. Leftover battery alone says little: service-first ends with the most and still profits well.

## Why rollout ADP can lose to a simpler rule

- **Its future is service-first.** Every imagined future follows service-first, not an optimal policy. Even when the inventory-rule plan is chosen for this hour, its consequences are judged as if service-first took over afterward.
- **Few plans.** At most five complete plans are compared. A good plan that needs several vehicles to change together, or a partial recharge, can be missing.
- **Sampling noise.** Eight futures per plan leave noise in every comparison.

In the small city, rollout ADP delivers more mobile energy than the exhaustive rule (146.88 against 137.69 kWh) but buys more recharge (102.64 against 77.38 kWh). Its extra 4.93 of mobile revenue is outweighed by 3.93 more electricity cost, 3.49 more travel cost and 1.49 less station revenue.

**Speed.** Mean large-city dispatch time: 433.16 ms for rollout ADP, 20.99 ms for the bounded inventory rule, 0.52 ms for PPO.

## Extra configurations (no retuning)

The same code, without changes or retraining, was run on two more scenarios. These scenarios add unequal starting batteries, three constrained stations, shifting demand patterns and noisier station prices (normal shocks, 15% standard deviation). The 36-zone case also pays 0.08 per kWh left at the end of the day. All controllers use a fixed price of $0.80 and their earlier settings. Test set: 32 days each.

| Configuration | Rollout ADP | Inventory rule (bounded) | Service-first | Rollout − inventory |
|---|---:|---:|---:|---|
| 9 zones, 5 vehicles, 18 one-hour periods | 72.21 | **75.83** | 62.81 | −3.62 [−6.85, −0.39] |
| 36 zones, 12 vehicles, 8 half-hour periods | 104.64 | **105.19** | 93.92 | −0.55 [−2.13, 1.02] |

These intervals use a normal approximation over 32 days. Each planner was given the correct model for its scenario. These runs check that the code works across configurations; they don't test robustness to a wrong model.

## Correctness checks

All final episodes, 6,016 including the reference and extra configurations, have zero invalid actions. 274 tests passed at the time. The audits:

- reconstruct each planner's private random stream and check every candidate choice and query count;
- reconcile profit ledgers and recompute the intervals;
- verify source and checkpoint hashes.

Before the final runs, an independent replay through the real simulator checked 6,912 future transitions and 51 decisions, with a maximum score difference of 2.27e-13.

## Reproduce

Show the saved scorecards:

```bash
python experiments/run_general_adp_study.py --phase show --study-dir outputs/general_adp_study_20261025
python experiments/run_adp_transfer.py --show --output-dir outputs/adp_transfer_20261025
```

Run the full fixed-price comparison in a new directory. `--control-root` must contain the saved PPO and Table ADP controls (`outputs/recharge_study_20261024/`, `outputs/scale_25_study_20261024/`, `outputs/paper_ppo_study/`):

```bash
S=outputs/general_adp_repeat
python experiments/run_general_adp_study.py --phase init     --study-dir $S --control-root .
python experiments/run_general_adp_study.py --phase rules    --study-dir $S
python experiments/run_general_adp_study.py --phase validate --study-dir $S
python experiments/run_general_adp_study.py --phase select   --study-dir $S
python experiments/run_general_adp_study.py --phase evaluate --study-dir $S --workers 3
python experiments/run_general_adp_study.py --phase show     --study-dir $S
```

The exhaustive-inventory reference has its own runner, saved inside the result bundle rather than in Git:

```bash
mkdir -p $S/reference_inventory
cp outputs/general_adp_study_20261025/reference_inventory/run_reference.py $S/reference_inventory/
python $S/reference_inventory/run_reference.py
```

Run the extra configurations:

```bash
python experiments/run_adp_transfer.py --workers 2 --output-dir outputs/adp_transfer
python experiments/run_adp_transfer.py --show --output-dir outputs/adp_transfer
```

These seeds reproduce the recorded experiment. Once a test bank has guided a change, a new performance claim needs new test days.
