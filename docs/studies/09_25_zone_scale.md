# 09. Scaling to a 25-zone, 10-vehicle city

**Date:** 2026-10-24 · **Runner:** `experiments/run_scale_study.py` · **Saved results:** `outputs/scale_25_study_20261024/`

**Question.** Does warm-started Joint PPO keep up with the rules when the city and fleet get five times larger?

**Answer.** No. PPO improves a lot over its imitation start (+102.80 per day) but all three seeds lose to both rules. It trails the inventory rule by 135.12 per day and service-first by 164.68.

## The large city

`configs/synthetic_25_zones.json`:

- **Layout.** A 5 × 5 grid of zones over 10 km × 10 km, 2.5 km apart, with 10 vehicles and 5 stations.
- **Vehicles.** Each starts with 55 kWh in an 80 kWh battery and keeps a 5 kWh reserve.
- **Stations.** Each can deliver 100 kWh per hour to customers and exchange 100 kWh per hour with vehicles.
- **Demand.** Two busy areas. Residential zones peak early and late; commercial zones peak at midday. 1,032.5 expected requests per day before customers choose.
- **Same as the small city:** 12 hours, `b = 3`, request sizes uniform on 5–15 kWh, Manhattan travel at 20 km/h.

This is a synthetic scaling test. It uses no real road network or demand data.

## What had to change for 10 vehicles

- **Table ADP** stores a value for every combination of two vehicles' states, so it cannot run here.
- **The exhaustive inventory rule** tries every joint action, which is infeasible for 10 vehicles. It is replaced by the **bounded** version: start from two complete plans, improve one vehicle at a time, for up to two passes ([METHODS.md](../METHODS.md#the-rules)).
- **PPO** gets fresh networks sized for this scenario: 532,697 parameters across the four networks.

## Setup

- **PPO (seeds 7, 19, 37).** 256 days of imitating the inventory rule (100 supervised passes), then 500 PPO updates of 16 new days each. The teacher uses $0.60 and battery values 0.30/0.15, and follows PPO's action mask.
- **Rules,** retuned on 32 validation days:
  - Bounded inventory rule: price fixed at $0.60, 16 battery-value schedules. Chose 0.40 early and 0.10 late.
  - Service-first: 7 prices × 3 recharge thresholds. Chose $0.80 with threshold 20%.
- **Test:** 256 shared days. The final update-500 models are used even if an earlier checkpoint scored higher in monitoring.

## Results

| Controller | Profit | Mobile delivered | Recharge bought | Unmet mobile | Battery left above reserve |
|---|---:|---:|---:|---:|---:|
| Imitation only | 625.56 | 509.64 | 268.43 | 588.89 | 44.18 |
| Imitation + 500 PPO updates | 728.36 | 475.04 | 196.08 | 396.95 | 72.64 |
| Inventory rule (bounded) | 863.49 | 760.77 | 475.52 | 306.59 | 20.81 |
| **Service-first** | **893.04** | 387.40 | 210.17 | 3.73 | 188.66 |

Energy columns are fleet kWh per day. "Battery left" subtracts the fleet's 50 kWh reserve.

| Seed | Imitation only | Final PPO | PPO − inventory rule |
|---:|---:|---:|---|
| 7 | 626.34 | 733.30 | −130.19 [−134.47, −126.01] |
| 19 | 621.53 | 714.60 | −148.89 [−154.40, −143.59] |
| 37 | 628.81 | 737.19 | −126.29 [−132.02, −120.68] |

- PPO − imitation: +102.80 [98.92, 106.67], or 16.4%.
- PPO − inventory rule: −135.12 [−138.99, −131.40].
- PPO − service-first: −164.68 [−169.67, −159.45].

Monitoring profit was still moving late in training, so 500 updates don't establish convergence.

## What the large city reveals

**The problem isn't only buying too little energy.** Compared with the inventory rule, PPO issues about as many service commands per day (63.45 against 66.11) but delivers less per command (7.49 against 11.51 kWh). It ends with more battery left and more unmet demand, even though fewer customers choose its mobile service. The profit gap is mostly 160.42 less mobile revenue and 30.70 more unmet-demand cost.

**Where PPO's gain over imitation came from.** Less travel and fewer unmet penalties, not more energy delivered. Travel energy fell from 170.46 to 112.48 kWh per day, and mobile-selected energy fell from 1,098.54 to 871.99 kWh.

**Leftover battery is not the failure measure.** Service-first prices at $0.80, attracts only 391 kWh of mobile demand, serves about 99% of it, and earns the most while leaving the most energy unused.

**Plausible bottlenecks, none isolated yet:**

- ten separately sampled vehicle actions competing for the same requests;
- imitation errors leading into states the teacher never visited;
- KL-based early stopping limiting optimization: seeds completed 2,046, 2,307 and 2,377 of a possible 3,000 steps.

## Correctness checks

All 2,048 final episodes have zero invalid actions. 252 tests passed at the time. The audit checks:

- profit ledgers and seed separation;
- source snapshots, model budgets and checkpoint replay;
- the paired intervals.

Mean dispatch time: 0.33 ms for PPO, 31.50 ms for the bounded inventory rule, 0.17 ms for service-first.

## Reproduce

Show the saved scorecard:

```bash
python experiments/run_scale_study.py --phase show --study-dir outputs/scale_25_study_20261024
```

Run it in a new directory (needs the `[train]` dependencies):

```bash
S=outputs/scale_25
python experiments/run_scale_study.py --phase init   --study-dir $S
python experiments/run_scale_study.py --phase select --study-dir $S
for seed in 7 19 37; do
  python experiments/run_scale_study.py --phase train --seed "$seed" --study-dir $S
done
python experiments/run_scale_study.py --phase evaluate --study-dir $S
python experiments/run_scale_study.py --phase show     --study-dir $S
```

A short standalone check that large-city PPO training runs:

```bash
python -m charging_env.rl.paper_train --scenario configs/synthetic_25_zones.json \
  --warmstart-dispatch-solver coordinate --warmstart-coordinate-sweeps 2 \
  --warmstart-episodes 8 --warmstart-epochs 2 \
  --updates 2 --episodes-per-update 4 --seed 7 \
  --reward-scale 20 --normalize-advantages --gae-lambda 1 \
  --eval-episodes 4 --eval-interval 1 --output-dir outputs/scale_smoke
```
