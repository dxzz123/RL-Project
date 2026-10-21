# Mobile charging: pricing and fleet dispatch

This project studies how an operator can earn profit from fixed charging stations and mobile charging vehicles. The example city has four locations, two stations, and two mobile vehicles. An episode covers twelve one-hour periods.

Each period follows the same sequence:

1. Set a mobile charging price at each location.
2. Customers choose mobile charging, an accessible station, or leave without charging.
3. Dispatch the mobile vehicles to serve customers, recharge, export energy, move, or wait.
4. Calculate energy use and business profit, then begin the next period.

For example, fully serving a 10-kWh mobile request at 0.60 dollars/kWh earns 6 dollars in revenue. Profit also accounts for energy purchases, travel, operating costs, and unmet-service penalties. Recharging costs money now but can make later service possible; the controllers must balance those consequences.

**Approximate dynamic programming (ADP)** learns the future value of the fleet and searches dispatch actions using the known model. **Proximal Policy Optimization (PPO)** learns policies for pricing and dispatch from simulated experience. Joint PPO and the earlier PPO implementation are both retained for comparison.

## Start here

| Your question | Read |
|---|---|
| What does the charging system do? | [Model walkthrough](docs/MODEL.md) |
| Which controller performs better, and why? | [Current comparison](docs/COMPARISON.md) |
| How do I train a controller or compare results? | [Training workflow](docs/TRAINING.md) |

In the current base-case tests, ADP earns the most profit. Joint PPO improves on earlier PPO but still trails the strongest inventory rule. These results describe a small simulated city with assumed demand patterns; the comparison explains the limits of that conclusion.

## Run a walkthrough

Use Python 3.10 or later. In the project folder, create an environment with `python3 -m venv .venv` if one does not already exist, then run:

```bash
source .venv/bin/activate
python -m pip install -e .
python examples/manual_walkthrough.py
```

The walkthrough prints customer choices, battery levels, energy flows, and profit, and saves `outputs/manual_episode.csv`. It uses scripted actions to explain the simulator. PPO's additional dependencies and training commands are in the training workflow above.

## Where things live

| Folder | Purpose |
|---|---|
| `src/charging_env/` | Simulator, controllers, and benchmark rules |
| `configs/` | City and customer-model settings |
| `examples/` | Small runnable demonstrations |
| `docs/` | Three main guides, followed by optional method and reference material |
| `experiments/` | Scripts for reproducible training and performance studies |
| `tests/` | Checks for physical accounting, learning, and saved results |

The [documentation index](docs/README.md) points to a specific explanation when you need more detail. Previous studies are kept in `docs/archive/`; they are optional background. Generated results live in `outputs/`, which is not included in Git.
