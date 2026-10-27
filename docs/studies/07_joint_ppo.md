# 07. Joint PPO against Table ADP, earlier PPO and three rules

**Date:** 2026-10-21 · **Runner:** `experiments/run_paper_ppo_study.py` (results viewer: `summarize_paper_ppo_study.py`) · **Saved results:** `outputs/paper_ppo_study/`

**Question.** Does Joint PPO, the implementation of the two-stage algorithm in [MDP_RL_2.pdf](../MDP_RL_2.pdf), improve on the earlier PPO? And where does it stand against Table ADP and the rules?

**Answer.** Joint PPO beats earlier PPO by 5.77 per day, and all three seeds improve. It still trails the time-dependent inventory rule by 10.22 and Table ADP by 14.57. Both PPO versions almost never recharge.

## What was compared

All runs use the small city (`configs/utility_b3.json`). The six methods:

| Method | Origin |
|---|---|
| Joint PPO | Newly trained, seeds 7, 19, 37, 500 updates of 16 days each |
| Earlier PPO | Saved models from [study 05](05_ppo_improvements.md), not retrained |
| Table ADP | Saved models from [study 06](06_table_adp.md), not retrained |
| Service-first | Tuned on validation: chose $0.80, recharge below 20% usable battery |
| Constant inventory rule | Tuned on validation: chose $0.60, $0.25 per stored kWh |
| Time-dependent inventory rule | Tuned on validation: chose $0.60, $0.30 per kWh early, $0.15 late |

The three rule families were tuned over 82 settings on 128 validation days. Validation picked the time-dependent rule as the main heuristic baseline. Both PPO methods were run stochastically from their final update-500 checkpoints, decided before testing.

**Tests:**

- base: 512 days;
- high demand (arrival rates +50%): 256 days;
- low starting battery (35 kWh instead of 55): 256 days.

Every setting stayed frozen for the stress tests. Each column has its own days, so compare within a column.

## Results

| Controller | Base | High demand | Low battery |
|---|---:|---:|---:|
| **Table ADP** | **137.76** | 140.56 | **126.67** |
| Time-dependent inventory rule | 133.41 | 133.63 | 121.75 |
| Constant inventory rule | 129.02 | 129.77 | 115.90 |
| Service-first | 123.25 | **144.48** | 113.56 |
| Joint PPO | 123.19 | 131.93 | 106.90 |
| Earlier PPO | 117.42 | 127.09 | 105.70 |

Base-case paired differences:

- Joint PPO − earlier PPO: **+5.77** [4.95, 6.55], or 4.91%.
- Table ADP − Joint PPO: +14.57 [13.54, 15.64]. Every Table ADP seed beats its matching Joint PPO seed.
- Joint PPO − time-dependent inventory rule: −10.22 [−11.35, −9.12]. Joint PPO is roughly tied with service-first.

| Seed | Earlier PPO | Joint PPO | Table ADP | Joint − earlier PPO |
|---:|---:|---:|---:|---|
| 7 | 115.55 | 124.82 | 137.95 | +9.27 [8.05, 10.56] |
| 19 | 116.44 | 121.40 | 137.57 | +4.96 [3.69, 6.19] |
| 37 | 120.28 | 123.36 | 137.76 | +3.07 [1.71, 4.43] |

Joint PPO also beats earlier PPO for every seed under high demand. Under low battery, seed 19 drops by 0.86 with an interval including zero, so the average gain of 1.20 there is not a gain for every seed. Service-first wins under high demand, so no controller wins everywhere.

## Operating behavior behind the gap

Fleet totals per base day:

| Controller | Mobile delivered (kWh) | Recharge bought (kWh) | Battery left at end (kWh) |
|---|---:|---:|---:|
| Earlier PPO | 63.27 | 0.21 | 31.07 |
| Joint PPO | 69.57 | 0.17 | 28.67 |
| Time-dependent inventory rule | 140.34 | 81.17 | 16.89 |
| Table ADP | 159.06 | 108.78 | 17.58 |

Both PPO versions recharge almost nothing. Joint PPO uses the starting battery better and travels less than earlier PPO (7.76 against 11.63 kWh of travel energy). Its gain decomposes into 2.60 less travel cost, 2.42 more mobile revenue and 1.14 less unmet-demand cost, partly offset elsewhere. It improves day-to-day operation but leaves replenishment unsolved. [Study 08](08_recharge_warm_start.md) investigates why.

## How the two PPO versions differ

Joint PPO changes several things at once, so this study measures the package, not any single change.

| Aspect | Earlier PPO | Joint PPO |
|---|---|---|
| Dispatch update | Clip each vehicle's probability ratio separately, then average | Multiply the vehicles' ratios and clip the fleet ratio once |
| Pricing input | Includes current requests | Hides current requests; uses fleet state, arrival rates, time and prices |
| Dispatch input | Each vehicle's own view, including peers | The full post-choice state; one output head per vehicle |
| Dispatch actions | Five modes, a target, and a continuous energy fraction | Four operations and a target; recharge and export use the full feasible amount |
| Advantage trace | 1 within an hour; λ = 1.0 between hours in the saved models | λ = 0.95 at both pricing and dispatch steps |
| Training signal | Normalized advantages, period weights, inventory reward shaping | Raw advantages, uniform weights, original rewards |
| Exploration bonus | Pricing and decaying dispatch entropy | Dispatch entropy only, summed over vehicles |
| Networks | Tanh layers with LayerNorm and a SiLU final dispatch layer; vehicle ID input | SiLU with LayerNorm throughout, four separate optimizers |

## Information and computation are not equal

| Per trained model | Table ADP | Earlier PPO | Joint PPO |
|---|---|---|---|
| Learning budget | 124,416 fleet-state queries | 8,000 days (96,000 hours) | 8,000 days (96,000 hours) |
| Extra work | Evaluates every candidate action at every queried state | Reuses each batch in PPO updates | Reuses each batch for 6 epochs |
| What is saved | 16,848 table values | Four networks | Four networks |
| Decision time | Joint action search + table lookup | One network pass | One network pass |

Table ADP's 124,416 queries reuse 96 distinct demand draws across the fleet grid, so they are not 124,416 independent days. Across its three models it evaluated about 53.7 million candidate outcomes. PPO only gets rewards for actions it actually tried. Table ADP also solves a narrower problem, because its price is fixed. The result ranks these implementations under these conditions; it does not establish that ADP beats PPO in general.

## Correctness checks

The test phase ran 12,288 policy episodes over 1,024 distinct days, with zero invalid actions. Rule validation and testing used separate days.

- **Audit.** Rechecks episode profits, intervals, reward components, budgets, and source and checkpoint fingerprints.
- **Replay.** After installation, 18 recorded episodes across all nine learned models were replayed exactly.

## Reproduce

Show the saved scorecard and figure:

```bash
python experiments/summarize_paper_ppo_study.py --show
open docs/assets/algorithm_comparison.png      # macOS
```

![Six-method comparison, Joint PPO gains, stress tests and monitoring curves](../assets/algorithm_comparison.png)

Run the study in a new directory. It needs the saved controls from study 06 in `outputs/fitted_value_study/fitted/seed_*/` and `outputs/fitted_value_study/ppo/seed_*/`:

```bash
S=outputs/paper_ppo_repeat
python experiments/run_paper_ppo_study.py --phase init   --study-dir $S --control-root outputs/fitted_value_study
python experiments/run_paper_ppo_study.py --phase select --study-dir $S
for seed in 7 19 37; do
  python experiments/run_paper_ppo_study.py --phase train --seed "$seed" --study-dir $S
done
python experiments/run_paper_ppo_study.py --phase evaluate --study-dir $S
python experiments/summarize_paper_ppo_study.py --study-dir $S          # audit, writes analysis.json
python experiments/summarize_paper_ppo_study.py --study-dir $S --show
python experiments/plot_paper_ppo_study.py --study-dir $S               # optional, needs matplotlib
```

The runner fixes its day banks in code. A newly designed study must change and record them before `init`.
