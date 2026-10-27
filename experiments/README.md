# Experiments

To run a controller, start with [TRAINING.md](../docs/TRAINING.md). Its `python -m charging_env.rl...` commands run one method. The scripts here coordinate larger studies with fixed seeds, benchmark selection, and independent tests. You do not need to read every script to use the project.

## Current comparison

The newest study learns pricing separately for rollout ADP and inventory dispatch, then compares both with their fixed-price versions on the small and 25-zone networks. The preceding rollout/PPO comparison and original two-vehicle ADP study remain available. [COMPARISON.md](../docs/COMPARISON.md) explains the results and their limits.

| Script | Use it to |
|---|---|
| [run_priced_adp_study.py](run_priced_adp_study.py) | Learn pricing, select policies, and compare rollout with inventory; `--phase show` reads saved results |
| [run_general_adp_study.py](run_general_adp_study.py) | Compare rollout ADP, saved PPO and selected rules on fresh independent scenarios; `--phase show` reads saved results |
| [run_adp_transfer.py](run_adp_transfer.py) | Check rollout ADP under two additional known configurations; `--show` reads saved results |
| [run_scale_study.py](run_scale_study.py) | Train and compare PPO with two rules on 25 synthetic zones and 10 mobile vehicles; no old checkpoints required |
| [run_recharge_study.py](run_recharge_study.py) | Reproduce the recharge repair study or display its saved scorecard with `--phase show` |
| [run_paper_ppo_study.py](run_paper_ppo_study.py) | Initialize the study, select rules on validation scenarios, train Joint PPO, and evaluate all six methods |
| [summarize_paper_ppo_study.py](summarize_paper_ppo_study.py) | Display saved results with `--show`, or audit a completed study |
| [plot_paper_ppo_study.py](plot_paper_ppo_study.py) | Plot saved results and training history; requires Matplotlib |

The newest saved-results command needs `outputs/priced_adp_study_20261025/`; fixed-price rollout uses `outputs/general_adp_study_20261025/`, and the original six-method scorecard needs `outputs/paper_ppo_study/`. These directories are local and ignored by Git, so a fresh GitHub checkout does not contain them. The [comparison report](../docs/COMPARISON.md) and [historical six-method figure](../docs/assets/algorithm_comparison.png) are included in the repository and can be read immediately.

Follow [the learned-pricing workflow](../docs/TRAINING.md#learn-prices-for-rollout-adp), [the fixed-price rollout workflow](../docs/TRAINING.md#compare-rollout-adp-with-ppo-and-rules), or [the original six-method workflow](../docs/TRAINING.md#reproduce-the-six-method-study). Each lists the required saved bundles. The learned-pricing study trains its own price policies; the older comparisons reuse specified learned controls. Standalone evaluations automatically number occupied output directories, while a new comparison study requires a new study directory.

<details>
<summary>Supporting modules and earlier studies — optional reference</summary>

You can skip this section on a first reading. Earlier studies explain how the current setup developed; their results use different scenario sets and are not additional rows in the current comparison.

The original six-method runner imports helpers from `run_fitted_value_study.py`. Its summarizer uses `summarize_fitted_value_study.py` and `source_provenance.py` to verify saved statistics and source records. These supporting files remain necessary for that workflow.

| Study or task | Scripts | Guide |
|---|---|---|
| Approximate DP versus earlier PPO and three rules | `run_fitted_value_study.py`, `summarize_fitted_value_study.py`, `plot_fitted_value_study.py` | [Fitted-value study](../docs/archive/FITTED_VALUE_STUDY.md) |
| Targeted earlier-PPO changes under `b=3` | `run_b3_improved_training.py`, `run_b3_improved_benchmarks.py`, `summarize_b3_improved_study.py`, `plot_b3_improved_study.py`, `diagnose_b3_gap.py` | [Training improvements](../docs/archive/B3_TRAINING_IMPROVEMENTS.md) |
| First SiLU plus `b=3` study | `run_b3_training.py`, `run_b3_benchmarks.py`, `summarize_b3_study.py`, `plot_b3_study.py` | [SiLU + b=3 study](../docs/archive/SILU_B3_STUDY.md) |
| Customer-choice benefit check | `check_choice_intercept.py` | [Outside option](../docs/reference/OUTSIDE_OPTION.md) |
| Activation pilot under `b=0` | `run_activation_pilot.py` | [Activation study](../docs/archive/ACTIVATION_EXPERIMENT.md) |
| Broad benchmark study under `b=0` | `run_policy_benchmarks.py`, `plot_policy_benchmarks.py` | [Earlier benchmarks](../docs/archive/POLICY_BENCHMARKS.md) |
| Longer-training study under `b=0` | `run_convergence.py`, `evaluate_convergence.py`, `summarize_stability.py`, `plot_convergence.py` | [Training analysis](../docs/archive/TRAINING_ANALYSIS.md) |

Older script paths are retained because scripts share helpers and recorded experiment protocols refer to exact source locations. Keeping those paths stable preserves imports and reproducibility. Before rerunning an earlier study, read its report and `--help`; pass explicit paths to saved controls with `--ppo-root` for the fitted-value runner or `--control-root` for the current comparison.

</details>
