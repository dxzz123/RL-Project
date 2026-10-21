# Historical fitted-value study — 2026-10-20

Paths under `outputs/` refer to local experiment records and are not included in Git. Use the saved study files or reproduce the experiment to inspect them; [TRAINING.md](../TRAINING.md) explains the setup and current workflow.

This report preserves the first five-method comparison of approximate DP with earlier PPO and three rules. The current six-method comparison also includes Joint PPO and uses different test scenarios; read [COMPARISON.md](../COMPARISON.md) for that result. [FITTED_VALUE.md](../methods/FITTED_VALUE.md) explains the method itself.

## What was compared

The study used the `b=3` scenario and three fitted-value training seeds: 7, 19, and 37. Each fitted model used a 10-kWh battery grid, eight draws per period, a fixed mobile price of 0.60 dollars/kWh, and half/full exchange actions. The controls were the final 500-update earlier-PPO checkpoints from the [targeted training study](B3_TRAINING_IMPROVEMENTS.md), reused without further training.

Three heuristic families were tuned on 128 separate validation scenarios: service-first, constant inventory value, and time-dependent inventory value. The compact grid contained 82 candidates. The time-dependent rule was strongest on validation and was fixed as the primary heuristic comparison before testing. All learned models and selected rule settings stayed fixed during the base and stress tests.

## Recorded results

Each entry is mean discounted original business profit per episode, in dollars. Learned-method rows average three separately executed models. Controllers within a column faced the same scenarios.

| Method | Base: 512 scenarios | High demand: 256 | Low inventory: 256 |
|---|---:|---:|---:|
| Approximate DP / fitted value | **137.76** | 139.48 | **127.15** |
| Earlier SA-PPO/MAPPO | 117.46 | 127.25 | 106.93 |
| Service-first | 122.89 | **145.50** | 115.02 |
| Constant inventory value | 129.46 | 130.73 | 117.11 |
| Time-dependent inventory value | 133.39 | 132.84 | 122.94 |

On the base test, approximate DP gained **4.37 dollars** over the validation-selected time-dependent rule, with paired 95% bootstrap interval **[3.64, 5.11]**. Its gain over earlier PPO was **20.31**, with interval **[19.28, 21.42]**. Each fitted-value seed had a positive paired gain interval against both controls.

| Fitted-value seed | Base profit | Gain over time-dependent rule | Paired 95% gain interval |
|---|---:|---:|---|
| 7 | 137.710 | +4.318 | [3.550, 5.118] |
| 19 | 137.797 | +4.405 | [3.603, 5.198] |
| 37 | 137.788 | +4.396 | [3.588, 5.196] |

The intervals use 5,000 paired scenario resamples with these trained models fixed. They do not quantify arbitrary retraining uncertainty. The higher-demand case raises arrival means by 50%; the low-inventory case starts each vehicle at 35 kWh. Service-first wins under higher demand, so the base advantage is not universal across operating conditions.

Approximate DP used model queries and exact current-action calculations, whereas PPO learned from episode rollouts. This was a comparison of complete controllers, not equal model access or equal computation. Three fitted-value runs made 373,248 state/draw queries and evaluated 53,714,325 candidate outcomes. Rule selection used 10,496 policy episodes; final testing used 9,216 policy episodes over 1,024 distinct scenarios, with zero invalid actions. The automated suite at that time passed 174 tests; later additions are recorded in [VALIDATION.md](../reference/VALIDATION.md).

## Reproduce this study or build its control directory

The runner requires the exact earlier-PPO control configuration. A newly trained model using default PPO arguments is not interchangeable with those controls. First obtain `checkpoint.pt` for seeds 7, 19, and 37 from `outputs/b3_improved_study/training/`, or reproduce that study using [its archived instructions](B3_TRAINING_IMPROVEMENTS.md).

From the project root, after installing the `[train]` dependencies, run:

```bash
source .venv/bin/activate
python experiments/run_fitted_value_study.py --phase init \
  --study-dir outputs/fitted_value_repeat \
  --ppo-root outputs/b3_improved_study/training
python experiments/run_fitted_value_study.py --phase select \
  --study-dir outputs/fitted_value_repeat
for seed in 7 19 37; do
  python -m charging_env.rl.fitted_train \
    --scenario configs/utility_b3.json \
    --seed "$seed" --grid-step 10 --samples 8 \
    --output-dir "outputs/fitted_value_repeat/fitted/seed_$seed"
done
python experiments/run_fitted_value_study.py --phase evaluate \
  --study-dir outputs/fitted_value_repeat
python experiments/summarize_fitted_value_study.py \
  --study-dir outputs/fitted_value_repeat
```

Use a fresh study directory and run each training command once, so the fixed checkpoint paths do not receive automatic numeric suffixes. The resulting directory contains `fitted/seed_{7,19,37}/checkpoint.npz` and `ppo/seed_{7,19,37}/checkpoint.pt`. It can serve as `--control-root outputs/fitted_value_repeat` for the [six-method study](../TRAINING.md#reproduce-the-six-method-study).

To generate the historical study's plot after summarizing:

```bash
python -m pip install matplotlib
python experiments/plot_fitted_value_study.py \
  --study-dir outputs/fitted_value_repeat
```

These commands reproduce fixed seed banks. If their results inform a new algorithm design, use a separately specified untouched test bank for new performance claims.

## Saved evidence

The original study remains in `outputs/fitted_value_study/`:

- Protocol and exact settings (`outputs/fitted_value_study/protocol.json`)
- Selected heuristic parameters (`outputs/fitted_value_study/rule_selection/selection.json`)
- Independent numerical audit and uncertainty intervals (`outputs/fitted_value_study/analysis.json`)
- Test summary (`outputs/fitted_value_study/benchmarks/summary.json`)
- Comparison figure (`outputs/fitted_value_study/fitted_value_comparison.png`)
- Executed training source (`outputs/fitted_value_study/provenance/executed_training_source/`)
