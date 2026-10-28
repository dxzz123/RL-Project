# Experiment scripts

These scripts run the full studies: fixed seeds, rule tuning on validation days, training, and one final test on held-out days. To run a single controller, use the commands in [docs/RUNNING.md](../docs/RUNNING.md) instead. Each study's report in [docs/studies/](../docs/studies/README.md) shows how to run its script.

All runners need the `[train]` dependencies, even ones that don't train PPO, because they share PPO comparison helpers.

## Current studies

| Script | Study | What it does |
|---|---|---|
| `run_history_adp_study.py` | [12 Historical model](../docs/studies/12_historical_model.md) | Fits demand and choice from quote histories; trains PPO and pricing in the fitted simulator; tests frozen controllers under private demand shifts |
| `history_training_checks.py` | helper | Checks saved CEM search arithmetic, PPO training budgets, checkpoints and evaluation ledgers without rerunning episodes |
| `run_priced_adp_study.py` | [11 Learned pricing](../docs/studies/11_learned_pricing.md) | Fits price rules for rollout ADP and the inventory rule, selects on validation, tests both cities |
| `run_general_adp_study.py` | [10 Rollout ADP](../docs/studies/10_rollout_adp.md) | Compares fixed-price rollout ADP with saved PPO, Table ADP and retuned rules |
| `run_adp_transfer.py` | [10 Rollout ADP](../docs/studies/10_rollout_adp.md#extra-configurations-no-retuning) | Runs the same controllers on a 9-zone and a 36-zone configuration |
| `run_scale_study.py` | [09 25-zone city](../docs/studies/09_25_zone_scale.md) | Trains warm-started PPO for the large city and compares it with two rules |
| `run_recharge_study.py` | [08 Recharge warm start](../docs/studies/08_recharge_warm_start.md) | Trains warm-started PPO for the small city |
| `run_paper_ppo_study.py` | [07 Joint PPO](../docs/studies/07_joint_ppo.md) | Trains Joint PPO and compares six methods |
| `summarize_paper_ppo_study.py` | 07 | Audits study 07 and writes `analysis.json`; `--show` prints the saved scorecard |
| `plot_paper_ppo_study.py` | 07 | Draws the study 07 figure (needs Matplotlib) |
| `run_fitted_value_study.py` | [06 Table ADP](../docs/studies/06_table_adp.md) | Study 06 runner; also provides helpers that studies 07–10 import |
| `summarize_fitted_value_study.py` | 06 | Audits study 06; also used by the study 07 summarizer |
| `plot_fitted_value_study.py` | 06 | Draws the study 06 figure |
| `source_provenance.py` | helper | Compares saved source snapshots with the current code |

Most runners work in phases on one study directory: `--phase init`, then middle phases (tune, train, validate, select, depending on the study), then `evaluate` and `show`. `show` only reads saved results. `run_adp_transfer.py` uses `--show` instead of phases.

## Older studies

`archive/` holds the scripts for studies 01–05 (Oct 17–18). They still run. Their recorded source hashes no longer match the current code, so exact reproduction of those studies uses the source snapshot saved with each study's results.

| Scripts in `archive/` | Study |
|---|---|
| `run_convergence.py`, `evaluate_convergence.py`, `summarize_stability.py`, `plot_convergence.py` | [01 Training curve](../docs/studies/01_ppo_training_curve.md) |
| `run_policy_benchmarks.py`, `plot_policy_benchmarks.py` | [02 First benchmarks](../docs/studies/02_first_benchmarks.md) |
| `run_activation_pilot.py` | [03 SiLU activation](../docs/studies/03_silu_activation.md) |
| `run_b3_training.py`, `run_b3_benchmarks.py`, `summarize_b3_study.py`, `plot_b3_study.py` | [04 SiLU with b = 3](../docs/studies/04_silu_b3.md) |
| `run_b3_improved_training.py`, `run_b3_improved_benchmarks.py`, `summarize_b3_improved_study.py`, `plot_b3_improved_study.py`, `diagnose_b3_gap.py` | [05 PPO improvements](../docs/studies/05_ppo_improvements.md) |
| `check_choice_intercept.py` | Customer-benefit check in [OUTSIDE_OPTION.md](../docs/reference/OUTSIDE_OPTION.md) |

## Reproducing frozen studies

Studies 09–11 froze a hash of every file in `src/` and of these scripts:

- `run_priced_adp_study.py`
- `run_general_adp_study.py`
- `run_adp_transfer.py`
- `run_scale_study.py`
- `run_recharge_study.py`
- `run_fitted_value_study.py`

The recorded source snapshots preserve those studies. Their runners check the source inventory as well as file hashes, so adding a module can also change the match. Use the saved snapshot for exact historical reproduction. For new behavior, start a new study directory with the new implementation; never edit runtime source during a frozen run.
