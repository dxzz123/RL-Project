# Experiment entry points

Start with [the training workflow](../docs/TRAINING.md) for commands or [the algorithm comparison](../docs/COMPARISON.md) for results. Most users only need the three current scripts below. Standalone learner commands live in `charging_env.rl`, not in this directory.

## Current six-method comparison

| Script | Purpose |
|---|---|
| [run_paper_ppo_study.py](run_paper_ppo_study.py) | `init`, `select`, `train`, and `evaluate` phases for manuscript PPO, frozen earlier PPO, approximate DP, and three rules |
| [summarize_paper_ppo_study.py](summarize_paper_ppo_study.py) | Audit a completed study; with `--show`, simply display its existing results |
| [plot_paper_ppo_study.py](plot_paper_ppo_study.py) | Plot the saved comparison and monitoring history; requires Matplotlib |

To see results already saved in this project:

```bash
python3 experiments/summarize_paper_ppo_study.py --show
```

Run it from the project root. The complete reproduction sequence is in [TRAINING.md](../docs/TRAINING.md#reproduce-the-six-method-study). It explicitly passes `--control-root outputs/fitted_value_study`, so it does not depend on the author's machine-specific default control path. The control directory must contain the saved models; initializing the comparison does not train missing controls.

## Earlier studies and supporting tools

These scripts are retained for reproduction and shared imports. Their results belong to their own scenario sets and are not additional rows in the current scorecard.

| Study or task | Scripts | Explanation |
|---|---|---|
| Approximate DP versus earlier PPO and three rules | `run_fitted_value_study.py`, `summarize_fitted_value_study.py`, `plot_fitted_value_study.py` | [Fitted-value guide](../docs/FITTED_VALUE.md) |
| Targeted earlier-PPO changes under `b=3` | `run_b3_improved_training.py`, `run_b3_improved_benchmarks.py`, `summarize_b3_improved_study.py`, `plot_b3_improved_study.py` | [Targeted-training report](../docs/archive/B3_TRAINING_IMPROVEMENTS.md) |
| Diagnose the remaining earlier-PPO gap | `diagnose_b3_gap.py` | Same report; a diagnostic, not a new main benchmark |
| First SiLU plus `b=3` study | `run_b3_training.py`, `run_b3_benchmarks.py`, `summarize_b3_study.py`, `plot_b3_study.py` | [SiLU + b=3 report](../docs/archive/SILU_B3_STUDY.md) |
| Customer-choice benefit check | `check_choice_intercept.py` | [Outside option](../docs/OUTSIDE_OPTION.md) |
| Activation pilot under `b=0` | `run_activation_pilot.py` | [Activation report](../docs/archive/ACTIVATION_EXPERIMENT.md) |
| First broad benchmark study under `b=0` | `run_policy_benchmarks.py`, `plot_policy_benchmarks.py` | [Historical benchmark report](../docs/archive/POLICY_BENCHMARKS.md) |
| First longer-training study under `b=0` | `run_convergence.py`, `evaluate_convergence.py`, `summarize_stability.py`, `plot_convergence.py` | [Training analysis](../docs/archive/TRAINING_ANALYSIS.md) |

All script names and imports remain valid. For a moved project, pass explicit paths to saved controls: the older fitted-value runner accepts `--ppo-root`, and the current runner accepts `--control-root`. Historical scripts also have study-specific defaults; read their report and `--help` before reproducing one.

Standalone training CLIs automatically number occupied output directories. Study runners instead require a new study root and preserve fixed internal paths across phases. That difference keeps the checkpoints, source records, rule selection, and test evidence linked to one protocol.
