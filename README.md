# Mobile charging: pricing and fleet dispatch

An operator runs fixed charging stations and a fleet of mobile charging vehicles. Each hour it decides **what price to charge for mobile service** and **where to send each vehicle**. This project simulates that business and compares ways to make those two decisions: simple rules, approximate dynamic programming (ADP), and reinforcement learning (PPO).

Every simulated hour has the same four steps:

1. **Price.** Post a mobile charging price at each zone.
2. **Customers choose.** Each arriving electric-vehicle driver picks mobile charging, a nearby station, or leaves.
3. **Dispatch.** Each mobile vehicle serves customers, recharges or sells energy at a station, moves, or waits.
4. **Profit.** Count revenue minus electricity, travel, operating and unmet-demand costs, then start the next hour.

One episode is a 12-hour day. The score is total discounted profit for the day, in dollars.

## What we have found so far

All results come from simulated days the controllers never saw during training or tuning. [RESULTS.md](docs/RESULTS.md) has the details and the uncertainty intervals.

| Question | Answer so far |
|---|---|
| Best controller on the 25-zone, 10-vehicle city? | **Rollout ADP with learned prices**: 941.91 per day, against 924.80 for the inventory rule with its own learned prices and 914.50 for the fixed-price inventory rule. All three seeds beat both rules. |
| Best on the 4-zone, 2-vehicle city? | Methods that exploit how small this case is. On the same test days, **Table ADP** earns 136.10 and the exhaustive inventory rule 131.41, against 127.78 for rollout ADP. Learned prices lift rollout ADP only slightly. |
| Does learned pricing help? | A lot in the large city: +30.39 per day for rollout ADP, +10.30 for the inventory rule. Very little in the small city: +0.70 and +2.08. |
| How does PPO do? | It learned to almost never recharge. Warm-starting it from the inventory rule fixed that (+5.33 per day in the small city). PPO still trails the strongest rule in both cities: by 5.50 per day in the small city and by more than 130 per day in the large one. |

These are results inside one simulator with assumed demand. They show which implemented controller earns more under that model, not how a real operator would fare.

## Quick start

You need Python 3.10 or later. From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[train]'        # use -e . if you don't need PPO

python examples/manual_walkthrough.py      # one scripted day, printed step by step
```

Run the current best controller for a few simulated days (no training needed):

```bash
python -m charging_env.rl.rollout_adp --scenario configs/utility_b3.json \
  --seed 7 --price-index 4 --episodes 4 --output-dir outputs/rollout_adp
```

[RUNNING.md](docs/RUNNING.md) has commands for every controller and for viewing saved results.

## Read in this order

| Step | Guide | You will learn |
|---|---|---|
| 1 | [MODEL.md](docs/MODEL.md) | How customers, vehicles, batteries and profit work |
| 2 | [METHODS.md](docs/METHODS.md) | How each controller decides, side by side |
| 3 | [RESULTS.md](docs/RESULTS.md) | What the experiments show, and what they don't |
| 4 | [RUNNING.md](docs/RUNNING.md) | How to run, train, evaluate and reproduce |

Everything else is optional depth: method deep dives in [docs/methods/](docs/methods/), lookups in [docs/reference/](docs/reference/), and one report per experiment in [docs/studies/](docs/studies/README.md).

## Method names and where they live in code

The code kept its original file names, so a few names differ from the docs:

| Name in the docs | What it is | Code |
|---|---|---|
| Rollout ADP | Simulates a few futures for each candidate fleet plan at decision time; any fleet size | `rl/rollout_adp.py` |
| Learned pricing | A 12-coefficient price rule fitted by direct search, usable with rollout ADP or the inventory rule | `rl/adaptive_pricing.py`, `rl/priced_rollout.py` |
| Table ADP | Learns a value table offline; exactly two vehicles | `rl/fitted_value.py`, `rl/fitted_train.py` |
| Joint PPO | Learns pricing and dispatch networks; optional inventory-rule warm start | `rl/paper_ppo.py`, `rl/paper_train.py`, `rl/paper_warmstart.py` |
| Earlier PPO | The first SA-PPO–MAPPO implementation, kept as a control | `rl/trainer.py`, `rl/networks.py`, `rl/returns.py`, `rl/train.py` |
| Service-first, inventory rule | Benchmark rules | `benchmarks/policies.py`, `benchmarks/scalable.py`, `benchmarks/compact.py` |

All paths are under `src/charging_env/`.

## Folder layout

| Folder | Contents |
|---|---|
| `src/charging_env/` | Simulator (`simulator.py`, `config.py`, `choice.py`), controllers (`rl/`), rules and evaluation (`benchmarks/`) |
| `configs/` | Scenario files; see [configs/README.md](configs/README.md) |
| `examples/` | Three small scripts that walk through the simulator |
| `experiments/` | Scripts that run the full studies; older ones are in `experiments/archive/` |
| `tests/` | 309 automated checks (`python -m unittest discover -s tests`) |
| `docs/` | The guides above |
| `outputs/` | Results written by runs. Not in Git, so a fresh clone won't have the saved studies |
