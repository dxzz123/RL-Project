# Mobile charging: pricing and fleet dispatch

An operator runs fixed charging stations and a fleet of mobile charging vehicles. Each hour it decides **what price to charge for mobile service** and **where to send each vehicle**. This project simulates that business and compares ways to make those two decisions: simple rules, approximate dynamic programming (ADP), and reinforcement learning (PPO).

Every simulated hour has the same four steps:

1. **Price.** Post a mobile charging price at each zone.
2. **Customers choose.** Each arriving electric-vehicle driver picks mobile charging, a nearby station, or leaves.
3. **Dispatch.** Each mobile vehicle serves customers, recharges or sells energy at a station, moves, or waits.
4. **Profit.** Count revenue minus electricity, travel, operating and unmet-demand costs, then start the next hour.

One episode is a 12-hour day. The score is total discounted profit for the day, in dollars.

## What we have found so far

The latest comparison estimates demand and customer choice from past quote records instead of giving controllers the true simulator parameters. It uses the 25-zone, ten-vehicle city, policy seeds 7, 19 and 37, and 128 new test days per condition. PPO is retrained for 500 updates per seed in the same estimated model.

| Mean daily discounted profit | Original pattern | 40% more arrivals | Shifted places and times |
|---|---:|---:|---:|
| **Rollout ADP with learned prices** | **940.46** | **1,081.91** | **932.46** |
| PPO | 718.00 | 771.83 | 717.58 |
| Inventory with learned prices | 925.53 | 1,038.58 | 917.91 |
| Fixed-price inventory | 915.67 | 1,060.72 | 908.82 |

ADP beats both inventory variants and PPO for every policy seed in all three conditions, with positive paired 95% intervals. It is also slower: about 453 ms per dispatch on the original-pattern tests, versus 9 ms for PPO and 29 ms for learned-price inventory. Learned pricing is not always better: fixed-price inventory wins over learned-price inventory when arrivals rise by 40%.

The quote histories and test days are synthetic, and the assumed distribution families remain known. The model stays frozen during testing, so these results cover two specified demand shifts, not arbitrary real-world changes. [Study 12](docs/studies/12_historical_model.md) explains the data, information limits and uncertainty.

Earlier studies include the four-zone case, where Table ADP and exhaustive inventory search are especially strong, and known-model experiments with learned prices. [RESULTS.md](docs/RESULTS.md) separates that historical evidence from the current comparison.

## Quick start

You need Python 3.10 or later. From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[train]'        # use -e . if you don't need PPO

python examples/manual_walkthrough.py      # one scripted day, printed step by step
```

Try fixed-price rollout ADP for a few simulated days using a known simulator configuration (no training needed):

```bash
python -m charging_env.rl.rollout_adp --scenario configs/utility_b3.json \
  --seed 7 --price-index 4 --episodes 4 --output-dir outputs/rollout_adp
```

For the current comparison, use the [historical-model workflow](docs/RUNNING.md#historical-model-comparison): it estimates demand and choice from quote records before training and private testing. [RUNNING.md](docs/RUNNING.md) also has commands for every controller and for viewing saved results.

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
| Historical market model | Estimates arrivals, request sizes and customer choice from quote records; supplies forecasts and simulated futures | `rl/history_model.py`, `rl/history_control.py` |
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
| `tests/` | Automated correctness checks (`python -m unittest discover -s tests`) |
| `docs/` | The guides above |
| `outputs/` | Results written by runs. Not in Git, so a fresh clone won't have the saved studies |
