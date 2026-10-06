# Mobile charging simulation environment

This project implements EV arrivals, price-dependent charging choices, mobile vehicle actions, battery accounting, station capacity, and operator profit. It contains no learning algorithm or trained policy.

The proposal leaves several modeling choices unspecified. They are made explicit in [docs/MODEL.md](docs/MODEL.md), and can be changed in `Scenario`. The default city and parameter values are synthetic examples, not calibrated data.

## What “environment” means here

There are two different environments:

| Environment | Purpose |
|---|---|
| Python virtual environment, `.venv` | Keeps this project's installed Python packages separate from other projects. |
| Simulation environment, `ChargingSimulator` / `ChargingEnv` | Receives prices and dispatch actions, then returns observations, profit, and the next state. A future algorithm interacts with this object. |

An algorithm is a separate component that decides which actions to send. For now, the walkthrough sends hand-written actions so that you can inspect the model before adding an algorithm.

## First run in VS Code on macOS

1. Choose **Terminal → New Terminal**. 
2. Check that you have Python 3.10 or later:

   ```bash
   python3 --version
   ```

   If Python is missing or older than 3.10, install a current Python 3 release from [python.org](https://www.python.org/downloads/), then open a fresh terminal.

3. Create and activate the Python virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   The terminal prompt will normally begin with `(.venv)`. Run the activation command again when opening a new terminal for this project.

4. Install this project and its dependencies:

   ```bash
   python -m pip install -e .
   ```

   `-e` means an editable installation: changes to Python files under `src/` are used without reinstalling the project. The first installation requires internet access to download dependencies.

5. Press **Cmd+Shift+P**, choose **Python: Select Interpreter**, and select the Python interpreter inside `.venv`. If it is not listed, use **Enter interpreter path…** and select `.venv/bin/python`.
6. Run the manual walkthrough:

   ```bash
   python examples/manual_walkthrough.py
   ```

7. Run the simulator checks:

    ```bash
    python -m unittest discover -s tests -v
    ```

    A successful test run ends with `OK`. The tests check model behavior and accounting; they do not establish that any strategy is good.

If `ModuleNotFoundError: charging_env` appears, activate `.venv` and rerun `python -m pip install -e .` from the project folder. If the terminal and debugger behave differently, check that both use `.venv/bin/python`.

## Inspect the model in the debugger

Open `examples/manual_walkthrough.py` and click in the margin beside a line to set a breakpoint. Choose **Run and Debug**, select **Manual environment walkthrough**, and press **F5**. Use **Step Over** to move through the code; inspect `obs`, `reward`, and `info` in the Variables panel.

The most useful values to inspect are:

- `obs["stage"]`: `0` means pricing, `1` means dispatch, and `2` means terminal.
- `obs["energy_kwh"]` and `obs["locations"]`: fleet batteries and positions.
- `obs["mobile_demand"]` and `obs["fixed_demand"]`: choices realized after pricing.
- `info["reward_terms"]`: service revenue, grid revenue, costs, and terminal inventory value.
- `info["energy_ledger"]`: the quantities used to check battery conservation.

## One physical period has two decisions

The default episode lasts 12 one-hour periods. Each period has two calls:

1. Quote one charging price per MCV. Users then choose a specific MCV, an FCS, or the outside option. This call returns zero reward and does not advance physical time.
2. Send one dispatch action per MCV. Travel and operation occur, station limits are applied, and the period's profit is returned. This call advances physical time by one period.

This means a complete default episode has 24 action calls. Customer choices are fixed after the first call: dispatch cannot change them, and customers do not retry another provider if their chosen service is unavailable.

The example city is a 4 km square. FCSs remain in zones 0 and 3, while the MCVs start in zones 1 and 2:

| Zone | Coordinates (km) | Fixed station | Initial MCV |
|---|---|---|---|
| 0 | `(0, 0)` | FCS 0 | — |
| 1 | `(4, 0)` | — | MCV 0 |
| 2 | `(0, 4)` | — | MCV 1 |
| 3 | `(4, 4)` | FCS 1 | — |

These are initial positions. MCVs can subsequently visit FCSs to recharge or discharge. `DISCHARGE` exports energy from an MCV battery through the FCS connection to the station-side grid, earning revenue at the period's grid sell price. FCSs have no modeled storage battery; this export does not replenish an FCS battery.

For example, the core API can be used directly:

```python
from charging_env import ChargingSimulator, DispatchAction, Mode

sim = ChargingSimulator()
obs, info = sim.reset(seed=7)

# One price for each of the two default MCVs, in dollars per delivered kWh.
obs, reward, terminated, truncated, info = sim.quote_prices([0.40, 0.45])

# An illustrative manual action: both vehicles wait at their current locations.
actions = [
    DispatchAction(Mode.WAIT, int(zone))
    for zone in obs["locations"]
]
obs, reward, terminated, truncated, info = sim.dispatch(actions)
print(reward)
print(info["reward_terms"])
```

Waiting is useful for understanding the transition and costs; it is not an optimization benchmark.

The longer walkthrough cycles through a fixed sequence of serve, serve, recharge, discharge, reposition, and wait commands. Service targets are the vehicles' initial zones 1 and 2; recharge and discharge targets are the FCS zones 0 and 3. Repositioning returns the vehicles to their service zones. Some `SERVE` commands can be infeasible because nobody selected that vehicle at its target zone; the simulator then executes `WAIT` and reports it. The script writes a detailed accounting log to `outputs/manual_episode.csv`. A negative total profit from this fixed script is an inspection result, not evidence about a trained policy or the research question.

For a controlled example of MCVs supplying energy through an FCS, run:

```bash
python examples/fcs_discharge_walkthrough.py
```

This example sets the first period's EV requests to zero, quotes prices, and sends MCV 0 from zone 1 to FCS 0 in zone 0 and MCV 1 from zone 2 to FCS 1 in zone 3. Each requests a 10 kWh export. Each vehicle travels 4 km, taking 0.2 hours and consuming 1.4 stored kWh; export then consumes `10 / 0.95` stored kWh. Its battery ends at approximately 43.073684 kWh. Total export revenue is $2.00 at $0.10/kWh, travel costs $2.00, and operating costs $1.00, giving period profit of −$1.00. The script deliberately demonstrates travel and energy accounting with a manual action; it does not claim that this action is profitable. The zero-request override applies only to the first period.

## Files you will use

| File or folder | Purpose |
|---|---|
| `src/charging_env/config.py` | All physical, demand, pricing, and accounting settings in `Scenario`. |
| `src/charging_env/simulator.py` | State transitions, user choice, feasibility, reward, and observations. |
| `src/charging_env/gym_env.py` | Gymnasium adapter around the same simulator. |
| `configs/toy_city.json` | Editable example scenario. |
| `examples/manual_walkthrough.py` | Runs hand-written actions and explains the outputs. |
| `examples/fcs_discharge_walkthrough.py` | Shows vehicles traveling from separate initial locations to export energy through FCSs. |
| `examples/check_gym.py` | Checks the Gymnasium interface. |
| `tests/` | Checks demand, dynamics, capacity, and accounting. |
| `docs/MODEL.md` | Exact mathematical model and current assumptions. |
| `.vscode/launch.json` | Debugger configuration. |

You can load a JSON scenario from Python:

```python
from charging_env import ChargingSimulator, Scenario

scenario = Scenario.from_json("configs/toy_city.json")
sim = ChargingSimulator(scenario)
obs, info = sim.reset(seed=7)
```

Every time-dependent array must have `horizon` entries. Zone, station, and vehicle arrays must have matching sizes. Validation rejects inconsistent settings. Recreate the simulator after changing a scenario.

For a first edit, change one value in `configs/toy_city.json`, such as `unmet_penalty_per_kwh`, save the file, and rerun the walkthrough. Inspect its accounting log to see the effect. Keep the number of periods, vehicles, stations, and zones unchanged until you are comfortable with the matching array sizes.

## Interface for future algorithms

The Gymnasium adapter exposes `reset(seed=...)` and `step(action)`, with the standard observation, reward, termination, truncation, and information outputs. Run:

```bash
python examples/check_gym.py
```

The adapter preserves the same alternating pricing and dispatch stages. Read `docs/MODEL.md` and `gym_env.py` before wiring in a trainer: the active action fields depend on the stage.

Every Gymnasium action is a dictionary containing `prices`, `modes`, `targets`, and `energy_kwh`. Pricing uses only `prices`; dispatch uses the other three fields. All fields must still match the action space's shapes and types. The checker may recommend normalized `Box` action bounds; the current bounds deliberately use physical prices and kWh. This recommendation does not mean the API check failed. Normalization can be added when choosing a learner.

**Discounting must follow physical periods.** Pricing transitions have continuation discount `1.0`; dispatch transitions have `discount_per_period`, or `0.0` at the terminal transition. These are supplied in `info["discount"]`. A trainer that applies its usual fixed gamma after every `step()` will discount twice per physical period. Most generic trainers do not automatically use this information field, so the future algorithm must deliberately implement the correct timing or use a suitable wrapper.

The simulator also provides a local observation for each MCV after pricing. These are partial views of one shared system. The vehicles share demand and FCS resources; they are not independent environments. A multi-agent training interface and its policies remain future work.

## Recommended procedure before training

1. Run the walkthrough and tests with the default scenario.
2. Read the event sequence, demand choice, energy ledger, and reward in `docs/MODEL.md`.
3. Agree on the current simplifying assumptions and change them if they do not match your intended research problem.
4. Inspect several short seeded episodes and check that costs and battery changes have the intended meaning.
5. Only then choose an algorithm and its observation, action, and discount interfaces.

Useful official references: [Gymnasium environment API](https://gymnasium.farama.org/api/env/) and [VS Code Python environments](https://code.visualstudio.com/docs/python/environments).
