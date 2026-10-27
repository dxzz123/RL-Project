# 08. Why PPO stopped recharging, and a warm-start repair

**Date:** 2026-10-24 · **Runner:** `experiments/run_recharge_study.py` · **Saved results:** `outputs/recharge_study_20261024/`

**Question.** Joint PPO bought almost no recharge energy (0.17 kWh per day against 108.78 for Table ADP in [study 07](07_joint_ppo.md)). Why, and can training be changed so it learns to replenish?

**Answer.** Recharging only pays when later prices and later dispatch both make use of the energy. Early random policies waste it, so PPO learns to avoid recharging and then never sees the sequences that would reverse that. Starting PPO from imitations of the inventory rule removes the collapse. The repaired PPO earns +5.33 per day more, with every seed better, but still trails the inventory rule by 5.50.

## Diagnosis

The analysis used 96 new development days for each of the three saved Joint PPO models.

1. **Recharging was available.** It was physically feasible in 98.64% of vehicle decisions. Its learned probability fell from 23.50% at initialization to 0.0329% after training. Even with less than 20 kWh in the battery, the trained probability was only 0.0861%. No mask errors were found.
2. **The useful experience vanished early.** For seed 7, validation recharge fell from 122.61 kWh at the start to 62.23, 29.55 and 15.15 kWh at updates 25, 50 and 75. During those early updates the dispatch critic explained almost none of the variation in returns.
3. **The payoff depends on what happens next.** In 30 low-battery states where PPO waited, one wait was replaced by a feasible recharge, and the rest of the day was simulated 12 times. Recharging cost $6.80 more immediately. The total effect on the rest of the day:

| Policy used after the single recharge | Recharge − wait | 95% interval over states |
|---|---:|---|
| Initial (random) PPO | −$7.13 | [−10.19, −4.07] |
| Trained PPO | +$0.28 | [−1.18, +1.73] |
| Table ADP | +$2.42 | [1.33, 3.50] |
| Fixed $0.60 pricing with PPO dispatch | +$2.20 | [0.31, 4.09] |

These states were chosen to diagnose the mechanism, so they don't measure the value of recharging in general. Late in the day, recharging can still be a bad decision, because leftover energy is worth nothing.

**It is not simply a critic that undervalues batteries.** The final critics predict their own policy's returns well: explained variance is about 0.91 on new rollouts. A critic estimates what *its own* policy will do next, not what a better controller could do with the same battery.

## The repair

An optional **warm start**, enabled with `--warmstart-episodes` (code: `rl/paper_warmstart.py`):

1. Run the time-dependent inventory rule for 512 days. It is restricted to Joint PPO's action mask and exchange amounts, uses $0.60, and values stored energy at $0.30 early, $0.15 late and zero in the last hour.
2. Fit the actors to the rule's actions (cross-entropy), and the critics to the rule's returns (mean-squared error), for 100 passes.
3. Discard the demonstrations, then train with ordinary on-policy PPO for 500 updates of 16 days.

The warm-started runs also changed four training settings:

- rewards divided by 20 for learning (reported profits stay in dollars);
- normalized advantages;
- λ = 1, so critic targets are complete-day returns;
- a smaller actor learning rate.

There is no extra reward for recharging and no change to the simulator. Because the settings changed together, this study measures the combined repair, not any single change.

The teacher uses the known simulator to compare actions, so the repaired PPO is model-informed. A separate imitation-only checkpoint shows how much comes from imitation and how much from PPO afterward.

## Results

Test set: 512 new small-city days, shared by every controller. Learned rows average seeds 7, 19 and 37.

| Controller | Profit | Recharge (kWh/day) | Mobile delivered (kWh/day) |
|---|---:|---:|---:|
| Original Joint PPO | 123.86 | 0.21 | 69.25 |
| Imitation only | 127.17 | 78.44 | 132.43 |
| **Imitation + 500 PPO updates** | **129.20** | **90.27** | **141.72** |
| Time-dependent inventory rule | 134.70 | 80.79 | 140.26 |
| Table ADP | 138.58 | 109.86 | 159.77 |

| Seed | Original PPO | Repaired PPO | Gain |
|---:|---:|---:|---|
| 7 | 125.09 | 128.37 | +3.28 [1.90, 4.64] |
| 19 | 122.27 | 129.93 | +7.67 [6.24, 9.06] |
| 37 | 124.24 | 129.29 | +5.05 [3.68, 6.37] |

- Repaired PPO − original PPO: **+5.33** [4.22, 6.42], or 4.31%.
- Repaired PPO − imitation only: +2.03 [1.36, 2.71]. Every seed's lower bound is positive, so PPO training adds value beyond imitation.
- Repaired PPO − inventory rule: −5.50 [−6.25, −4.76]. The repair closes about half of the original gap of 10.84.
- Repaired PPO − Table ADP: −9.38 [−9.95, −8.83].

**The remaining gap looks different from the original failure.** Repaired PPO delivers about as much mobile energy as the rule. It buys 9.48 kWh more recharge and ends the day with 25.02 kWh in the batteries, against 15.93 for the rule. Its accounting shows 2.05 more unmet-demand cost, 1.61 more electricity cost and 1.55 less station revenue, partly offset by more mobile revenue. Timing, routing and customer allocation still need work. Simply recharging less would not necessarily recover the gap.

## Correctness checks

All 6,656 test episodes have zero invalid actions. Checkpoint reloads reproduce evaluations exactly. 231 tests passed at the time. The intervals use 5,000 bootstrap resamples over test days.

## Reproduce

Show the saved scorecard:

```bash
python experiments/run_recharge_study.py --phase show --study-dir outputs/recharge_study_20261024
```

Run the study again in a new directory. It needs the saved original controls in `outputs/paper_ppo_study/`:

```bash
S=outputs/recharge_repeat
python experiments/run_recharge_study.py --phase init --study-dir $S --control-root outputs/paper_ppo_study
for seed in 7 19 37; do
  python experiments/run_recharge_study.py --phase train --seed "$seed" --study-dir $S
done
python experiments/run_recharge_study.py --phase evaluate --study-dir $S
python experiments/run_recharge_study.py --phase show     --study-dir $S
```

Train one warm-started model directly. The command is also in [RUNNING.md](../RUNNING.md#joint-ppo).

```bash
python -m charging_env.rl.paper_train --scenario configs/utility_b3.json --seed 7 \
  --warmstart-episodes 512 --warmstart-epochs 100 \
  --reward-scale 20 --normalize-advantages --gae-lambda 1 \
  --actor-learning-rate 0.0001 --dispatch-entropy-coef 0.005 --target-kl 0.03 \
  --updates 500 --episodes-per-update 16 --output-dir outputs/joint_ppo_warmstart
```

The saved directory also keeps the development diagnostics, including unsuccessful attempts, under `development/` and `development_summary.json`.
