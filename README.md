# Mobile charging: pricing, customer choice, and vehicle dispatch

This project simulates an operator that owns fixed charging stations (FCSs) and mobile charging vehicles (MCVs). In each period, the operator sets mobile prices, customers choose a charging option, and the vehicles then decide where to travel and what to do. The simulator calculates the energy delivered, battery changes, and profit. The training code learns **policies**, meaning decision rules for prices and vehicle actions. It uses **reinforcement learning**: trying actions in the simulator and adjusting the rules using their rewards, which here come from the operator's profit.

The city and numerical settings are examples chosen for simulation, rather than estimates fitted to real-world data. Earlier PPO policies earned positive profit but remained below a strong inventory rule, which considers the value of battery energy for later use. The project now also includes fitted-value dispatch, a model-based learner that estimates that future value from simulator queries, and the latest manuscript's interleaved joint PPO. The new PPO changes fleet probability ratios, two-stage credit assignment, and the information available before pricing. Each method has its own guide and saved independent comparisons.

## Where to start reading

Start with the question you want to answer. You do not need to read the experiment history before understanding the model or running the environment.

| Your question | Read this |
|---|---|
| What happens in one period, and why? | [MODEL.md](docs/MODEL.md): the model explained in operational order, with worked examples |
| What does a symbol mean, and how does it match the PDF? | [NOTATION.md](docs/NOTATION.md): symbols, units, paper/code translations, and manuscript ambiguities |
| How do I run the simulator and read what it returns? | [ENVIRONMENT.md](docs/ENVIRONMENT.md): a complete example, configuration, allowed actions, and outputs |
| How does the new manuscript's joint PPO work? | [PAPER_PPO.md](docs/PAPER_PPO.md): interleaved advantages, joint fleet clipping, commands, and independent comparisons |
| How does the new learned-value dispatcher work? | [FITTED_VALUE.md](docs/FITTED_VALUE.md): intuition, value learning, commands, and the focused comparison |
| How do the pricing and vehicle decision rules learn together? | [TRAINING.md](docs/TRAINING.md): explains PPO and MAPPO, the learning cycle, commands, saved models, and settings |
| What does the charging benefit `b` change? | [OUTSIDE_OPTION.md](docs/OUTSIDE_OPTION.md): customer participation and the `b=0/2/3/4` scenarios |
| What has been checked? | [VALIDATION.md](docs/VALIDATION.md): correctness checks, execution checks, and the limits of performance evidence |

[MODEL.md](docs/MODEL.md) is written to be understandable without prior knowledge of PPO or MAPPO. The notation guide is a reference to consult when needed, not a prerequisite.

## Run the environment

Open **charging_env 2** as the project folder in VS Code, then open a terminal there. Python 3.10 or later is required. The `.venv` folder holds the project's Python packages separately from other projects. If it already exists, use it; otherwise create it once:

```bash
python3 -m venv .venv
```

Activate it and install the environment:

```bash
source .venv/bin/activate
python -m pip install -e .
python examples/manual_walkthrough.py
```

Here `.venv` is the Python package environment. The **simulation environment** is the `ChargingSimulator` or `ChargingEnv` object that receives actions and returns the next situation and profit. These two uses of “environment” refer to different things. The `-e` installation option means saved source changes are picked up without reinstalling the project.

The manual example demonstrates service, recharge, export, repositioning, and waiting over 12 periods, and writes a table to `outputs/manual_episode.csv`. It follows a fixed script to demonstrate the operations; it does not search for the most profitable decisions. To follow one period directly in Python, use the complete example in [ENVIRONMENT.md](docs/ENVIRONMENT.md#2-run-one-period-with-the-direct-api).

In VS Code, select `.venv/bin/python` using **Python: Select Interpreter** so the editor uses this project's Python installation. A **breakpoint** pauses the program at a chosen line, allowing you to inspect its variables. Place one in `examples/manual_walkthrough.py`, select **Manual environment walkthrough** in Run and Debug, and press F5. Inspect `obs` for the available information, `info["reward_terms"]` for revenue and costs, and `info["energy_ledger"]` for the energy accounting.

## New manuscript PPO: training and performance

This version uses one joint fleet probability ratio, applies GAE at both decision stages, and restricts pricing to background demand rates. The earlier PPO remains available as a separate method. Start a short execution check with:

```bash
source .venv/bin/activate
python -m charging_env.rl.paper_train \
  --scenario configs/utility_b3.json --updates 10 --episodes-per-update 4 \
  --seed 7 --output-dir outputs/paper_joint_ppo
```

The completed performance experiment trained seeds 7, 19, and 37 for 500 updates each. On 512 fresh shared base-test scenarios, mean discounted profit was:

| Method | Profit |
|---|---:|
| New joint PPO | 123.19 |
| Earlier PPO/MAPPO | 117.42 |
| Fitted-value dispatch | **137.76** |
| Service first | 123.25 |
| Constant inventory rule | 129.02 |
| Time-dependent inventory rule | 133.41 |

All three new PPO seeds beat their earlier counterparts. The average gain is **5.77 dollars, or 4.91%**, with a paired 95% interval **[4.95, 6.55]**. The new PPO is better than earlier PPO in this base experiment, while inventory rules and fitted value remain stronger. Under low initial inventory, improvement is not consistent across seeds.

To see the saved comparison immediately, with no further training:

```bash
python3 experiments/summarize_paper_ppo_study.py \
  --study-dir outputs/paper_ppo_study --show
open outputs/paper_ppo_study/paper_ppo_comparison.png
```

[PAPER_PPO.md](docs/PAPER_PPO.md) explains the exact manuscript details, full training configuration, all stress results, checkpoints, and complete reproduction commands. Repeated standalone training commands automatically choose numbered output directories.

## Train the learned-value dispatcher

This controller learns when battery energy and vehicle locations are valuable for later service. It fixes the mobile price at 0.60 and searches coordinated dispatch actions using that learned continuation value. The current table implementation supports two vehicles. It needs NumPy but no neural-network library.

```bash
source .venv/bin/activate
python -m charging_env.rl.fitted_train \
  --scenario configs/utility_b3.json --seed 7 --grid-step 10 --samples 8 \
  --output-dir outputs/fitted_value
```

Repeated commands automatically choose a fresh numbered directory and print its path. Each run saves `checkpoint.npz`, the exact scenario and configuration, the query budget, and its training records. Select **Train fitted-value dispatch (b=3)** in VS Code to debug the same command. [FITTED_VALUE.md](docs/FITTED_VALUE.md) explains the method and how to reproduce the comparison with PPO and three important heuristic families.

## Train earlier PPO for comparison

Training additionally requires PyTorch, the library used for the neural networks. With the same virtual environment active, install the optional training packages:

```bash
python -m pip install -e '.[train]'
```

Then run a short check using the default training settings:

```bash
python -m charging_env.rl.train \
  --updates 10 --episodes-per-update 4 --seed 7 \
  --output-dir outputs/quickstart
```

This uses the default `b=0` customer model and the original network settings. An **episode** runs the simulator through all its periods. An **update** collects episodes and then adjusts the networks. Ten updates with four episodes each therefore collect 40 episodes, or 480 physical periods in the default 12-period scenario. This checks that training, saving, and loading work; it does not establish reliable profit improvement.

The parameter `b` controls how attractive charging is relative to leaving to the outside option. **SiLU** is an alternative calculation inside the dispatch network, called an activation function. For a new run using the later `b=3` customer model and SiLU at the final dispatch activation, use:

```bash
python -m charging_env.rl.train \
  --scenario configs/utility_b3.json --dispatch-final-activation silu \
  --updates 10 --episodes-per-update 4 --seed 7 \
  --output-dir outputs/silu_b3_quickstart
```

This command selects two settings; it does not reproduce every change used in the latest study. A **checkpoint** is a saved snapshot of the model and training state. [TRAINING.md](docs/TRAINING.md) explains all optional settings, how to resume or evaluate a checkpoint, and why the final snapshot may differ from the one selected using validation scores. The earlier targeted PPO experiment can be reproduced from [B3_TRAINING_IMPROVEMENTS.md](docs/B3_TRAINING_IMPROVEMENTS.md). The latest manuscript's separate PPO implementation is explained in [PAPER_PPO.md](docs/PAPER_PPO.md).

<!-- You can repeat either training command without changing its output name. The CLI uses the requested directory if it is available; otherwise it creates a numbered sibling, such as `outputs/quickstart_2`, then `outputs/quickstart_3`, and prints the absolute path it uses. Keeping each run preserves its checkpoints and settings so you can compare results or recover an earlier model. Use `--resume` to continue from a checkpoint; the CLI also chooses an available destination for that continuation. Direct Python calls and dedicated experiment runners still require an unused output directory, and evaluation-only mode refuses to overwrite an existing `evaluation.json`. -->

## The two-stage sequence to keep in mind

```text
Potential requests + current fleet + station prices
                 |
                 v
        Choose mobile prices
                 |
                 v
 Customers choose mobile / an FCS / outside
                 |
                 v
         Dispatch all vehicles
                 |
                 v
 Service + energy changes + profit + next period
```

Pricing does not advance physical time or earn a separate reward. Dispatch produces the period profit and advances time. A 12-period episode therefore contains 24 stage transitions. **Discounting** gives later profit less weight when calculating total return. It applies once per physical period, rather than once for each decision stage.

Customers choose mobile service at their origin, rather than booking a particular vehicle. After choice, any vehicle sent to that node can serve the node's remaining mobile requests. A customer receives their full requested energy or remains unserved; the request is not retried or carried into a later period. [MODEL.md](docs/MODEL.md) explains the limits imposed by travel time, battery energy, and shared station capacity.

The manuscript uses pricing stage 1 and dispatch stage 2. The API uses `stage=0` for pricing, `stage=1` for dispatch, and `stage=2` for termination. Mathematical periods start at 1, while Python `period` starts at 0. These are index translations, not additional decision stages.

## Earlier fitted-value comparison (2026-10-20)

On 512 fresh `b=3` base-test scenarios, the focused comparison gives the following mean discounted profits in dollars per episode:

| Method | Profit |
|---|---:|
| Fitted-value dispatch, mean of seeds 7, 19, and 37 | **137.76** |
| SA-PPO/MAPPO, the three saved final models | 117.46 |
| Service-first rule | 122.89 |
| Constant inventory-value rule | 129.46 |
| Time-dependent inventory-value rule | 133.39 |

All three fitted-value seeds beat the strongest validation-selected heuristic. Their average gain is **4.37 dollars, or 3.28%**, with a paired 95% interval of **[3.64, 5.11]**. The rules were tuned on 128 separate validation scenarios; the test set was reserved from training and selection. The new comparison runs these five methods; older benchmark code and records remain available for reproducing previous studies.

The result has a clear limit: with 50% higher demand, service-first earns **145.50**, above fitted value's **139.48**. Fitted value uses a known simulator model, fixed mobile price 0.60, and two vehicles, so this comparison is not an equal-information training contest with PPO. The [fitted-value guide](docs/FITTED_VALUE.md#6-independently-generated-results) gives all seed results, uncertainty, operating behavior, stress tests, checkpoints, and a reproducible protocol.

## Earlier PPO experiments

The earlier targeted PPO study ran three fresh training runs using seeds 7, 19, and 37, with 500 updates each, SiLU, `b=3`, and targeted training changes. A **seed** sets a repeatable sequence of random draws. On 512 new independent base scenarios, average discounted profit was **116.72**, compared with **115.93** for the previous policies evaluated again on those same scenarios and **125.51** for the inventory rule with parameters retuned on separate validation scenarios. Seeds 19 and 37 improved, but seed 7 declined. Average profit also improved over the previous policies in both stress tests—tests with changed operating conditions—while remaining below the rule.

The new policies still barely recharge and make ineffective service assignments. These particular models show a small average gain, but the result does not establish consistent gains from repeating training or an advantage over the rule. See the [targeted PPO report](docs/B3_TRAINING_IMPROVEMENTS.md) for uncertainty estimates using shared test scenarios, explanations of vehicle behavior, training costs, checkpoints, and plots.

The earlier reports answer different questions and should be read in their original customer model:

| Report | Customer model | Question answered |
|---|---|---|
| [TRAINING_ANALYSIS.md](docs/TRAINING_ANALYSIS.md) | `b=0` | Why did the initial short training run lose profit, and what happened with longer training? |
| [POLICY_BENCHMARKS.md](docs/POLICY_BENCHMARKS.md) | `b=0` | How did the policies compare with practical rules? |
| [ACTIVATION_EXPERIMENT.md](docs/ACTIVATION_EXPERIMENT.md) | `b=0` | Did replacing the final dispatch Tanh activation with SiLU help in a bounded pilot? |
| [SILU_B3_STUDY.md](docs/SILU_B3_STUDY.md) | `b=3` | How did three 500-update policies compare with rules retuned for the changed market? |
| [B3_TRAINING_IMPROVEMENTS.md](docs/B3_TRAINING_IMPROVEMENTS.md) | `b=3` | Did targeted changes close the inventory-rule gap on another fresh test set? |

Changing `b` changes how many customers choose to charge. A profit difference between `b=0` and `b=3` therefore cannot be attributed only to a better controller. Each report identifies its training scenarios, validation scenarios used to choose settings or checkpoints, and independent test scenarios used to measure the final result. Links point to the recorded evidence under `outputs/`.

## Check the installation and implementation

```bash
python -m unittest discover -s tests -v
python examples/fcs_discharge_walkthrough.py
python examples/check_gym.py
```

The recorded complete test suite has 211 passing tests. These check energy and profit accounting, customer choices, Python inputs and outputs, both PPO implementations, fitted-value interpolation and Bellman updates, compact benchmarks, source protocols, and checkpoints. A successful run ends with `OK`; passing tests does not establish that a policy earns high profit. The controlled export example has a hand calculation in [ENVIRONMENT.md](docs/ENVIRONMENT.md#5-check-a-transition-by-hand). Gymnasium may report warnings because some observations have no fixed upper bound and energy actions are expressed in kWh; [VALIDATION.md](docs/VALIDATION.md) explains these choices.

## Project layout and sources

| Location | Purpose |
|---|---|
| `src/charging_env/config.py` | Scenario parameters and validation |
| `src/charging_env/simulator.py` | Requests, choices, dispatch, physical changes, and profit |
| `src/charging_env/gym_env.py` | Adapter that exposes the simulator through Gymnasium's standard interface |
| `src/charging_env/rl/` | Networks, future-return estimates, recorded episodes, learning updates, saved training state, and training commands |
| `src/charging_env/benchmarks/` | Alternative decision rules and tools for comparing policies |
| `configs/` | Baseline and participation scenarios |
| `examples/` | Short executable demonstrations |
| `experiments/` | Reproducible training and benchmark studies |
| `tests/` | Simulator, interface, and training checks |
| `docs/` | Explanations, notation, experiment reports, and source PDFs |
| `outputs/` | Saved models, configurations, metrics, and study evidence |

[MDP_RL.pdf](docs/MDP_RL.pdf) is an unchanged copy of the attachment used for the current model. [proposal.pdf](docs/proposal.pdf) is an older source retained for historical reference. The model guide identifies simulator assumptions that fill gaps in the manuscript and later optional extensions. The manuscript's illustrative experiment table is not a repository result.
