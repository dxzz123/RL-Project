# Mobile charging: model, controllers, and experiments

This project studies an operator that owns fixed charging stations and mobile charging vehicles. Each period, it sets mobile prices, customers choose where to charge, and the mobile fleet travels, serves customers, recharges, or exports energy. The simulator accounts for battery energy and business profit.

Three learned controllers use this same simulator:

| Method | What it learns | Guide |
|---|---|---|
| **Approximate dynamic programming (ADP)**, called **fitted-value dispatch** in the code | The future value of fleet locations and batteries; uses that value to search dispatch actions with a fixed mobile price | [FITTED_VALUE.md](docs/FITTED_VALUE.md) |
| **Manuscript joint PPO** | Pricing and dispatch policies, using the two-stage algorithm in the latest manuscript | [PAPER_PPO.md](docs/PAPER_PPO.md) |
| **Earlier PPO/MAPPO** | Pricing and a shared vehicle policy, with different action representations and learning rules | [LEGACY_PPO.md](docs/LEGACY_PPO.md) |

**Approximate DP currently earns the highest base-case test profit.** Manuscript PPO improves on earlier PPO, but still trails approximate DP and the strongest inventory rule. This is a result for the saved models in this example city: approximate DP has access to the one-period model and searches candidate actions, while PPO learns from episodes. [COMPARISON.md](docs/COMPARISON.md) gives the shared test results, assumptions, uncertainty, and stress cases.

## Read in this order

1. [Model](docs/MODEL.md): what happens in one period, why customers choose a provider, and how energy and profit are calculated.
2. [Training and results](docs/TRAINING.md): install, view the completed comparison, train a controller, and find its saved files.
3. [Algorithm comparison](docs/COMPARISON.md): understand the differences between approximate DP and both PPO implementations, then interpret their performance.
4. Open the method guide above when you need equations or implementation details.

For a specific question, use [ENVIRONMENT.md](docs/ENVIRONMENT.md) for the Python API, [NOTATION.md](docs/NOTATION.md) for symbols and manuscript mappings, [OUTSIDE_OPTION.md](docs/OUTSIDE_OPTION.md) for the customer benefit `b`, or [ARCHITECTURE.md](docs/ARCHITECTURE.md) to find the relevant code. [VALIDATION.md](docs/VALIDATION.md) records checks. Earlier experiment reports are in [docs/archive](docs/archive/README.md).

## Run the environment

Open a terminal in the project folder containing `pyproject.toml`. Use the existing virtual environment:

```bash
source .venv/bin/activate
python -m pip install -e '.[train]'
python examples/manual_walkthrough.py
```

If `.venv` does not exist, create it once with `python3 -m venv .venv` before activation. Python 3.10 or later is required. The `[train]` extra installs PyTorch for PPO; `python -m pip install -e .` is sufficient for the simulator and approximate DP. The manual example follows scripted actions and saves `outputs/manual_episode.csv`; it is an explanation of the environment, not a performance benchmark.

## See the completed comparison

From the project root:

```bash
python3 experiments/summarize_paper_ppo_study.py --show
open outputs/paper_ppo_study/paper_ppo_comparison.png
```

The first command prints the saved six-method comparison without training or installing PyTorch. The second opens its figure on macOS; on other systems, open that PNG normally. These files are included in this local project's `outputs/` folder. That folder is ignored by Git, so a source-only checkout will need the saved study copied over or the experiment reproduced using [TRAINING.md](docs/TRAINING.md#reproduce-the-six-method-study).

## Train a controller

[TRAINING.md](docs/TRAINING.md) gives short commands for all three methods, with the same explicit `b=3` customer model. For example, a short manuscript-PPO execution check is:

```bash
python -m charging_env.rl.paper_train \
  --scenario configs/utility_b3.json \
  --updates 10 --episodes-per-update 4 --seed 7 \
  --output-dir outputs/paper_joint_ppo
```

You can repeat a standalone training command without renaming its output: the CLI automatically chooses `_2`, `_3`, and so on when needed, and prints the actual directory. A short run verifies execution; use independent comparisons to judge policy quality.

## Check the project

```bash
python -m unittest discover -s tests -v
python examples/check_gym.py
```

The full suite covers simulator accounting, customer choice, all three learners, checkpoint continuation, and experiment protocols. [VALIDATION.md](docs/VALIDATION.md) explains what those checks establish.

The city and demand parameters are illustrative, not fitted to real operations. The physical-model source is [MDP_RL.pdf](docs/MDP_RL.pdf); the later PPO specification is [MDP_RL_2.pdf](docs/MDP_RL_2.pdf). Their roles and implementation assumptions are explained in [NOTATION.md](docs/NOTATION.md).
