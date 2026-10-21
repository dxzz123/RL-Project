# Run training and understand the results

Use this page for the practical workflow: choose a controller, run it, and compare its profit with other methods. [MODEL.md](MODEL.md) explains the environment. [COMPARISON.md](COMPARISON.md) explains why the algorithms differ and what the completed experiments show.

All commands below run from the project root, the folder containing `pyproject.toml`. The examples explicitly use `configs/utility_b3.json` so they share the same customer-choice model. Omitting `--scenario` uses the original `b=0` toy scenario; changing `b` changes customer behavior, not just a training setting.

## Set up once

Activate the existing virtual environment and install the training dependencies:

```bash
source .venv/bin/activate
python -m pip install -e '.[train]'
```

If the folder is missing, first run `python3 -m venv .venv`. Python 3.10 or later is required. `-e` means source edits take effect without reinstalling; `[train]` adds PyTorch for the two PPO methods. The simulator and approximate DP need only `python -m pip install -e .`. Optional tested direct dependency versions are recorded in `requirements-tested.txt` and `requirements-training-tested.txt`; these are not complete cross-platform lock files.

In VS Code, choose `.venv/bin/python` with **Python: Select Interpreter**. The Run and Debug menu contains clearly named entries for each learner, the simulator walkthroughs, and the saved comparison.

## View the results already available

The [comparison report](COMPARISON.md) and [figure](assets/algorithm_comparison.png) are included in the repository and can be read without installing anything. If you also have the saved study in `outputs/paper_ppo_study/`, these commands display its scorecard and figure:

```bash
python3 experiments/summarize_paper_ppo_study.py --show
open docs/assets/algorithm_comparison.png
```

`--show` reads `outputs/paper_ppo_study/analysis.json` and prints the six-method scorecard. It does not run policies or rewrite results. The figure shows base profits, the joint-PPO improvement over earlier PPO, stress cases, and monitoring curves. `open` is the macOS command; elsewhere, open the image in your file browser.

The [comparison guide](COMPARISON.md) explains each row, why approximate DP is not a PPO variant, and what the uncertainty intervals mean. In this local project the completed study is present. A Git checkout alone does not contain ignored `outputs/`; copy the study artifacts or follow the reproduction workflow below.

## Choose the method you want to run

| Your purpose | Command module | Detailed guide |
|---|---|---|
| Learn a continuation-value table and search fleet actions using the known model | `charging_env.rl.fitted_train` | [Approximate DP / fitted value](methods/FITTED_VALUE.md) |
| Train pricing and fleet dispatch with joint PPO | `charging_env.rl.paper_train` | [Joint PPO](methods/JOINT_PPO.md) |
| Study or continue the earlier SA-PPO/MAPPO implementation | `charging_env.rl.train` | [Earlier PPO/MAPPO](methods/LEGACY_PPO.md) |

A **policy** is the controller's decision rule. A **checkpoint** saves a learned policy and the scenario/settings needed to load it. The two PPO trainers save `.pt` files, but their checkpoints are different formats and must be loaded with the matching trainer. Approximate DP saves `.npz` arrays and metadata.

### Approximate DP: fit a value table

```bash
python -m charging_env.rl.fitted_train \
  --scenario configs/utility_b3.json \
  --seed 7 --grid-step 10 --samples 8 \
  --output-dir outputs/fitted_value
```

This method learns dispatch continuation values for exactly two vehicles. It uses a fixed mobile price of 0.60 dollars/kWh under this scenario, then searches joint dispatch actions with that value. `--grid-step` sets battery-grid spacing in kWh; `--samples` sets the number of demand/price draws used at each time/state query. Smaller grid spacing or more samples increases training work. There is no PPO update count: fitting works backward once through the finite horizon. See [FITTED_VALUE.md](methods/FITTED_VALUE.md) for the equations and saved-policy example.

### Joint PPO: check a short training run

```bash
python -m charging_env.rl.paper_train \
  --scenario configs/utility_b3.json \
  --updates 10 --episodes-per-update 4 --seed 7 \
  --output-dir outputs/joint_ppo
```

An **episode** runs all 12 physical periods. An **update** collects fresh episodes and then adjusts the networks. This command collects 40 training episodes, or 480 physical periods, plus evaluation episodes. It checks that training and saving work; it is too small to establish reliable performance. The dispatch critic uses dispatch advantage A₂ by default. The separate A₁ option is retained for controlled experiments.

For a longer standalone run:

```bash
python -m charging_env.rl.paper_train \
  --scenario configs/utility_b3.json \
  --updates 500 --episodes-per-update 16 --seed 7 \
  --eval-episodes 64 --eval-interval 25 \
  --output-dir outputs/joint_ppo_500
```

This collects 8,000 training episodes and 96,000 physical periods. It uses the current trainer defaults for other settings. The dedicated study below additionally fixes separate seed banks, control checkpoints, and the complete comparison protocol. A standalone command is not a reproduction of that full study.

### Earlier PPO: keep it as a comparison method

```bash
python -m charging_env.rl.train \
  --scenario configs/utility_b3.json --dispatch-final-activation silu \
  --updates 10 --episodes-per-update 4 --seed 7 \
  --output-dir outputs/earlier_ppo
```

This selects `b=3` and the SiLU final dispatch activation. It does not reproduce every optional change used in the saved earlier-PPO control models. Their exact settings are stored with the checkpoints and described in [the historical targeted-training report](archive/B3_TRAINING_IMPROVEMENTS.md). The [earlier PPO reference](methods/LEGACY_PPO.md) documents all flags, network structure, and the older learning equations.

## Find the files from a run

Each standalone training CLI prints the directory it actually uses. If the requested directory exists, it chooses a numbered sibling such as `outputs/joint_ppo_2`. Repeating the same command therefore preserves earlier checkpoints without requiring you to invent a name. Direct Python training calls and the dedicated study runners have stricter unused-directory checks.

| What you need | Approximate DP | Joint PPO | Earlier PPO |
|---|---|---|---|
| Learned controller | `checkpoint.npz` | `checkpoint.pt` | `checkpoint.pt` |
| Run overview | `summary.json` | `summary.json` | `summary.json` |
| Progress | `training_metrics.csv` | `training_metrics.csv` | `training_metrics.csv` |
| Validation over learning | No PPO-style monitoring curve | `validation_history.json` | `validation_metrics.csv` |
| Validation-selected snapshot | Not used | `best_checkpoint.pt` | `best_checkpoint.pt` |
| Exact environment | `scenario.json` and checkpoint | `scenario.json` and checkpoint | `scenario.json` and checkpoint |

Read the method guides for the remaining metadata files. `checkpoint.pt` is the final state; `best_checkpoint.pt` is the best monitored snapshot according to that run's validation rule. The completed six-method comparison uses the final 500-update PPO snapshots, fixed in advance.

A learning loss is an optimization diagnostic, not profit. Even a lower critic error or a flat training curve does not show that a controller beats a benchmark. Use the original business-profit metric on held-out scenarios.

## Continue or evaluate a PPO model

Use the same module that created the checkpoint. For joint PPO:

```bash
python -m charging_env.rl.paper_train \
  --resume outputs/joint_ppo/checkpoint.pt --updates 10 \
  --output-dir outputs/joint_ppo_continued
```

`--updates 10` adds ten updates to the saved counters. Resume restores the scenario, parameters, optimizer state, and random-generator state. Use the actual printed directory if the original run received a suffix. Do not supply a replacement scenario when resuming.

To evaluate one of the completed joint-PPO models without changing it:

```bash
python -m charging_env.rl.paper_train \
  --evaluate-only outputs/paper_ppo_study/training/seed_7/checkpoint.pt \
  --eval-episodes 128 --eval-seed-base 193000001 \
  --eval-policy stochastic
```

This prints a fresh evaluation; it does not reproduce the recorded scorecard. Repeating the same seed bank repeats those scenarios. To evaluate earlier PPO, use `charging_env.rl.train` and one of its checkpoints; to evaluate approximate DP, use the short Python example in [FITTED_VALUE.md](methods/FITTED_VALUE.md). Each method guide explains its evaluation options.

## Reproduce the six-method study

The current comparison trains three joint-PPO models and evaluates them alongside three frozen approximate-DP models, three frozen earlier-PPO models, and three tuned heuristic families. The learned-method mean rows average the three model results; they do not combine the models into an ensemble.

The commands below require the existing control files in `outputs/fitted_value_study/fitted/seed_{7,19,37}/` and `outputs/fitted_value_study/ppo/seed_{7,19,37}/`. These are present in this local project. If starting without those artifacts, the [fitted-value guide](methods/FITTED_VALUE.md) and [archive index](archive/README.md) explain the earlier studies that produced them. This workflow deliberately reuses fixed controls, rather than silently retraining different ones.

<details>
<summary>Full reproduction procedure — expand after preparing the control models</summary>

Choose an unused study directory and keep it unchanged through all phases:

```bash
python experiments/run_paper_ppo_study.py --phase init \
  --study-dir outputs/paper_ppo_repeat \
  --control-root outputs/fitted_value_study
python experiments/run_paper_ppo_study.py --phase select \
  --study-dir outputs/paper_ppo_repeat
for seed in 7 19 37; do
  python experiments/run_paper_ppo_study.py --phase train --seed "$seed" \
    --study-dir outputs/paper_ppo_repeat
done
python experiments/run_paper_ppo_study.py --phase evaluate \
  --study-dir outputs/paper_ppo_repeat
python experiments/summarize_paper_ppo_study.py \
  --study-dir outputs/paper_ppo_repeat
python experiments/summarize_paper_ppo_study.py \
  --study-dir outputs/paper_ppo_repeat --show
```

`init` records the settings, copies the controls, and freezes source hashes. `select` tunes the three rules on validation scenarios. `train` fits one new PPO seed at a time. `evaluate` runs all fixed policies on the same independent test scenarios. The summary step audits the saved evidence and writes `analysis.json`; `--show` then displays it. Do not edit frozen source files midway through a study. To change the implementation, start a new study directory.

Plotting is optional and needs Matplotlib:

```bash
python -m pip install matplotlib
python experiments/plot_paper_ppo_study.py \
  --study-dir outputs/paper_ppo_repeat
```

The recorded study used 512 base scenarios and 256 scenarios for each of two stress cases. No test result selected the models or heuristic settings. These banks reproduce that experiment; if you use their results to guide a new algorithm, reserve a new untouched bank before claiming new independent performance. The current runner fixes its seed banks in code, so a newly designed study must explicitly change and document them before initialization.

</details>

## Judge a result

Compare **discounted business profit per episode** on the same scenario seeds. Keep customer model, physical settings, checkpoint choice, and execution mode explicit. Training randomness, validation for selection, and final testing have different roles; using one set for all three produces an optimistic comparison.

Read profit alongside service, recharge, travel, and unmet-demand measures. Then check whether improvements hold across training seeds and changed operating conditions. [COMPARISON.md](COMPARISON.md) applies that reasoning to the saved results; [VALIDATION.md](reference/VALIDATION.md) separates implementation correctness from performance evidence.
