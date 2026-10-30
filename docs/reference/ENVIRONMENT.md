# Using and inspecting the environment

Use this guide to run the simulator, change its settings, or understand what it returns. A **transition** is one step from the current situation to the next after an action. An **observation** is the information returned for making a decision. [MODEL.md](../MODEL.md) explains the physical rules; this page explains the Python calls that apply them. Those calls and their input/output fields form the **application programming interface (API)**. The names in backticks are the actual code names you will see in Python.

## 1. Choose the interface that fits the task

`ChargingSimulator` is the direct interface. Call `reset` to start an episode, quote prices, dispatch vehicles, then repeat until the episode ends. This makes it easy to inspect one physical calculation or write an alternative decision rule for comparison, called a benchmark.

`ChargingEnv` provides the same simulator through **Gymnasium**, a standard Python interface used by reinforcement-learning tools. Its `step` method handles one decision stage at a time. A learner using it must still handle the pricing/dispatch stages, lists of allowed actions called **masks**, once-per-period discounting, and changing numbers of customer requests.

Both interfaces use the same physical model. Every controller in this project calls `ChargingSimulator` directly. [RUNNING.md](../RUNNING.md) shows their commands.

This page describes the **raw observation API**, which includes more information than every policy is permitted to use. The joint PPO hides current requests and waiting estimates from its pricing actor and critic; its dispatch networks use the post-choice information. The earlier PPO can use current requests for pricing, while the completed fitted-value study fixes the mobile price. See [RESULTS.md](../RESULTS.md) for the comparison.

## 2. Run one period with the direct API

Run Python from the project root with the project installed in the active virtual environment:

```python
from charging_env import ChargingSimulator, DispatchAction, Mode

sim = ChargingSimulator()
obs, info = sim.reset(seed=7)
assert obs["stage"] == 0  # Ready to price, in Python period 0.

# One actual dollar/kWh price for every node, taken from this period's menu.
obs, reward, terminated, truncated, choice_info = sim.quote_prices(
    obs["mobile_price_menu"][:, 1]
)
assert obs["stage"] == 1  # Choices are now observed; ready to dispatch.
assert reward == 0.0

# One vehicle action each. WAIT must target that vehicle's current node.
actions = [DispatchAction(Mode.WAIT, int(z)) for z in obs["locations"]]
obs, reward, terminated, truncated, info = sim.dispatch(actions)
print("Period profit:", reward)
print("Profit components:", info["reward_terms"])
print("Energy balance:", info["energy_ledger"]["balance_residual_kwh"])
```

The pricing call draws each customer's provider choice and returns the information available after those choices. Dispatch calculates the period's profit and moves time forward. The default episode has 12 periods, so it has 12 pricing calls and 12 dispatch calls. The example's `seed=7` makes the random draws repeatable. Its WAIT actions demonstrate the interface; they are not chosen to maximize profit.

There are two ways to submit a price, and they take different inputs:

| Call | Input for each node | Example |
|---|---|---|
| `quote_prices(...)` | Actual dollars per delivered kWh, from `obs["mobile_price_menu"]` | `0.4073` |
| `quote_price_levels(...)` | Integer position in that node's menu row | `1`, the medium 1.4x premium |

Alternate one pricing call and one dispatch call. Calling pricing twice or dispatching before customers choose causes a stage error. After the episode ends, call `reset` to start another one.

## 3. Read the stage and period together

The mathematical model uses periods `t=1,...,T`, but Python starts counting periods from zero. At mathematical period `t`, the observation's `period[0]` is `t-1`. After the last dispatch it is `T`.

| Observation `stage` | Meaning | Valid next operation |
|---|---|---|
| `0` | Pre-choice pricing state | Quote prices |
| `1` | Post-choice dispatch state | Dispatch every vehicle |
| `2` | Terminal state | Reset |

In the mathematical model, pricing is Stage 1 and dispatch is Stage 2. These correspond to API stages `0` and `1`; API stage `2` means termination. [NOTATION.md](NOTATION.md) provides the complete translation.

A transition returns five values, `(obs, reward, terminated, truncated, info)`:

- `obs` is the information available for the next decision.
- `reward` is zero after pricing and the period's operator profit after dispatch.
- `terminated` says whether the episode has finished all its periods.
- `truncated` would mark an episode stopped early by a time limit; it is always false here.
- `info` contains extra details, such as revenue, costs, energy accounting, and the discount to use for learning.

After pricing, `info["discount"]` is `1.0`. After dispatch, it is `Scenario.discount_per_period` if another period remains, or zero if the episode has ended. This **continuation discount** tells the learner how much weight to give later profit.

A learner must read and use those discounts itself. Gymnasium passes along `info` but does not apply the values inside it automatically. Discounting after both pricing and dispatch would incorrectly count two periods of time for one physical period.

## 4. Choose and configure a scenario

A `Scenario` contains the city's settings: vehicle and station limits, average demand over time, prices, and the customer-choice model. For example:

```python
from charging_env import ChargingSimulator, Scenario

scenario = Scenario.from_json("configs/utility_b3.json")
sim = ChargingSimulator(scenario)
obs, info = sim.reset(seed=7)
```

The baseline `Scenario()` and `configs/toy_city.json` use common charging benefit `b=0`, called the charging-utility intercept in the code. The `utility_b2.json`, `utility_b3.json`, and `utility_b4.json` scenarios raise this benefit, making customers more likely to charge rather than leave. Compare decision rules within the same customer model, and select benchmark settings for that model. [OUTSIDE_OPTION.md](OUTSIDE_OPTION.md) explains `b`.

The main demand and price settings are listed below. In a table shape such as `[horizon, n_zones]`, the first number is the number of rows and the second is the number of columns. Here, each row represents a period and each column represents a zone or station.

| Setting | Meaning |
|---|---|
| `arrival_rates` | Optional `[horizon, n_zones]` table of average arrival counts; actual counts are randomly drawn from a Poisson distribution |
| `mean_requests_by_zone`, `arrival_profile` | Zone baseline demand and a time-dependent multiplier; multiply them when no full arrival table is supplied |
| `request_min_kwh`, `request_max_kwh` | Smallest and largest energy amount a customer may request |
| `fcs_charge_price_profile` | Optional `[horizon, n_fcs]` table of baseline customer charging prices |
| `fcs_discharge_price_profile` | Optional `[horizon, n_fcs]` table of baseline prices paid for MCV energy exported through stations |
| `fcs_pricing_mode` | `"base_price"` to draw station base prices by period type, or `"legacy"` for the profile/shock scheme |
| `period_types`, `first_period_hour_of_day` | Explicit off-peak/normal/peak label per period, or the clock hour the day starts at so LADWP windows supply them |
| `fcs_base_price_mean`, `fcs_base_price_halfwidth` | Uniform band of the station base price for each of the three period types |
| `fcs_service_fee_rate` | Fee `alpha` an EV pays over the base price; the fleet still exchanges energy at base |
| `mcv_pricing_mode`, `mcv_premium_levels`, `mcv_premium_reference` | Premium menu over the nearest station's price, its multipliers, and whether the anchor includes the fee |
| `fcs_retail_per_kwh` | Legacy-mode baseline customer price for each station, kept constant over periods |
| `grid_sell_per_kwh` | Default baseline export price for each period, used at every station |
| `grid_buy_per_kwh` | Price the operator pays for electricity from the external grid in each period |
| `price_noise_distribution` | How random price changes are drawn: `"none"`, `"uniform"`, or `"normal"` |
| `price_noise_scale` | Size of the random percentage change: the half-width of the uniform range or standard deviation of the normal distribution |

The configured average demand and baseline prices stay fixed for a scenario. Actual arrivals, customer energy needs, random price changes, and customer choices can vary between episodes. `max_requests_per_zone` is an older field kept for compatibility; it does not cap arrivals or set a maximum request-list length. Loading a scenario checks that setting names, list/table sizes, and physical values are valid.

### Control the first period for inspection

For a hand calculation, you can replace the first period's sampled requests:

```python
obs, info = sim.reset(
    seed=7,
    options={
        "requests": [2, 0, 0, 0],
        "request_energy_kwh": [6.0, 12.0],
        "accessible_fcs": [[True, False], [True, True]],
    },
)
```

The count list gives the number of customers in each origin zone. The request list then groups customers by their origin zone, with all node-0 customers first. Both customers in this example start at node 0. Accessibility columns follow the station list, rather than the list of city nodes: with the default `fcs_zones=(0,3)`, column 0 refers to node 0 and column 1 to node 3. This station-list position is called a **station slot**.

The individual energy list and station-accessibility table are optional, but require explicit customer counts in `requests`. With counts alone, every customer requests `request_kwh` (10 kWh by default). Supplied individual energies must stay within the configured bounds. The accessibility table replaces the usual distance-based station check, although each station must still have capacity for the smallest request. These manually supplied inputs affect only the first period. Later periods return to random generation, and customers still choose their providers randomly according to the model.

## 5. Check a transition by hand

The controlled example `examples/fcs_discharge_walkthrough.py` disables price noise and uses zero first-period requests. Each default MCV travels 4 km to an FCS and exports 10 grid-side kWh.

For each MCV, the calculation is:

```text
Travel time:       4 km / 20 km per hour = 0.2 hour
Travel energy:     4 km * 0.35 kWh per km = 1.4 stored kWh
Export drain:      10 grid-side kWh / 0.95 = 10.526316 stored kWh
Battery afterward: 55 - 1.4 - 10.526316 = 43.073684 stored kWh
```

Across both vehicles, export earns 2 dollars, travel costs 2 dollars, and operation costs 1 dollar. Period profit is therefore -1 dollar. This example checks units, losses, and accounting; its action sequence is not intended to maximize profit. FCSs have no storage batteries, so the exported energy passes through their grid connections.

The manual example, `examples/manual_walkthrough.py`, demonstrates service, recharge, discharge, repositioning, and waiting over an episode. It writes `outputs/manual_episode.csv`. Use a debugger breakpoint to compare its `reward_terms` and `energy_ledger` with [MODEL.md](../MODEL.md).

## 6. Understand observations without confusing their units

An **array** is a list or table of numbers. Its **shape** describes how many values it holds along each dimension. The table below abbreviates sizes as `I = n_zones`, `M = n_mcv`, `F = n_fcs`, `L = n_price_levels`, and `N` for the number of customers who actually arrived this period. Thus `[M, I]` means one row per vehicle and one column per zone. The fields `period` and `outside_requests` each hold a one-element array, while `stage` is a single integer.

| Fields | Shape | Meaning |
|---|---|---|
| `locations`, `energy_kwh` | `[M]` each | Node positions and stored MCV battery kWh |
| `arrival_rates` | `[I]` | Configured average arrival count in each zone this period |
| `requests_by_zone` | `[I]` | Number of customers who actually arrived in each zone |
| `potential_energy_kwh` | `[I]` | Total energy requested in each origin zone before provider choice |
| `request_details` | `[N, 3+F]` | Individual requests, as described below |
| `mobile_prices` | `[I]` | Posted prices per delivered kWh; before choice, valid placeholders |
| `mobile_demand`, `mobile_requested_kwh` | `[I]` each | Post-choice mobile request counts and energy by origin |
| `fixed_demand`, `fixed_requested_kwh` | `[F]` each | Post-choice FCS counts and energy by station slot |
| `outside_requests` | `[1]` | Count of customers choosing outside |
| `mobile_wait_hours`, `mobile_available` | `[I]` each | Delay estimate and whether mobile service can be offered |
| `eligible_mcvs` | `[M, I]` | Which vehicles can reach each node, keep the battery reserve, and deliver at least the minimum request |
| `fcs_wait_hours`, `fcs_retail_prices`, `fcs_discharge_prices` | `[F]` each | Station queue estimates and customer/exchange prices |
| `fcs_base_prices` | `[F]` | This period's drawn base energy price at each station |
| `mobile_price_menu` | `[I, L]` | The admissible mobile prices at each node, in dollars per kWh |
| `grid_buy_price` | `[1]` | External wholesale dollars per grid-side kWh |
| `pricing_mask` | `[I, L]` | Which price choices are available at each node during pricing |
| `action_mask` | `[M, 5, I]` | Which operation/destination combinations each vehicle may choose during dispatch |

A `request_details` row is `[origin, energy_kwh, choice, accessibility...]`. Values use `float64`, a number format that can store decimal values with high precision. The `choice` code is `-1` before choice, `0` for mobile, `1..F` for station slots `0..F-1`, and `F+1` for leaving. Each customer keeps the same row during the current period. With no customers, the table has zero rows and shape `[0, 3+F]`.

The number of customer rows can change each period. Gymnasium represents this as a variable-length sequence, `Sequence(Box(...), stack=True)`. The environment neither fills the list with dummy customers (**padding**) nor drops customers to shorten it (**truncation**). The PPO dispatch encoders convert each request into numeric features and combine them into a fixed-length summary, called **pooling**. It includes list position because request allocation depends on order. A PPO library expecting a fixed-length input needs extra input-processing code to use this request list.

### What a vehicle observes

Call `local_observation(m)` after pricing to get the input for vehicle `m`. It contains that vehicle's location, battery, allowed-action mask, and `agent_id`, plus the other vehicles' locations and batteries and shared customer, demand, price, and waiting information. Arrays describing the current vehicle have one vehicle row; arrays describing peers contain the other vehicles.

The earlier PPO uses this per-vehicle API for its observation with peer information, $`\zeta_{m,t}`$. It does not reveal the actions other vehicles are about to take. Without the optional role feature, the input-processing code uses `agent_id` only to match availability information to the current vehicle and its peers. The optional feature also supplies explicit vehicle identity as a **one-hot vector**: a list with a 1 at this vehicle's position and zeros elsewhere. The dispatch **critic**, a network that predicts future team return during training, sees the full shared observation. [EARLIER_PPO.md](../methods/EARLIER_PPO.md) explains these actors, critics, and optional role features. The joint PPO instead supplies the global post-choice observation to its dispatch network, which emits a separate action distribution for each vehicle. Fitted-value dispatch compares joint fleet actions directly.

## 7. Submit valid actions

Simulator mode codes are `WAIT=0`, `SERVE=1`, `REPOSITION=2`, `RECHARGE=3`, and `DISCHARGE=4`. These codes belong to the physical API. The joint PPO has its own four-operation representation, which its adapter converts to these five modes. A target is a city node ID, even for recharge/export. In the default city, targeting the second FCS means node `3`, not station slot `1`.

For the direct API, use `DispatchAction(mode, target_zone, energy_kwh=0.0)`. The energy request is used for RECHARGE and DISCHARGE. The default `0.0` requests **zero energy exchange**, even when the mode is RECHARGE or DISCHARGE; supply a positive amount to exchange energy. Travel, battery bounds, efficiency, power, and shared station budgets determine the final allocated amount. SERVE derives its delivery from the selected requests and does not use this field.

The mask checks each vehicle's allowed actions before vehicles compete for customer requests and station capacity. Choosing an allowed action does not guarantee that resources remain for that vehicle. An action with the correct input format but an impossible operation/destination becomes WAIT and is recorded in `invalid_actions`. Incorrectly formatted lists, invalid mode codes, or negative energy amounts instead cause an **exception**, meaning Python stops that call and reports an input error.

A node without mobile service has an all-zero pricing-mask row: no price choice is active there. The policy supplies a valid placeholder price index such as `0` and excludes that node from its probability calculation. There is no valid probability distribution to calculate from an empty list of choices. Pricing masks are entirely zero outside pricing; dispatch masks are entirely zero outside dispatch.

### Gymnasium's action dictionary

A Python **dictionary** stores named fields and their values. `ChargingEnv.step` always expects the four fields below, even though each stage reads only the relevant ones. A field's **dtype** specifies its number format, such as whole-number integers or decimal-valued `float64`:

| Field | Shape and dtype | Meaning |
|---|---|---|
| `price_levels` | `[I]`, integer | Menu-row indices, not dollar prices |
| `modes` | `[M]`, integer | Vehicle mode codes |
| `targets` | `[M]`, integer | Target city-node IDs |
| `energy_kwh` | `[M]`, float64 | Requested grid-side exchange energy |

A minimal two-call example is:

```python
import numpy as np
from charging_env.gym_env import ChargingEnv

env = ChargingEnv()
obs, info = env.reset(seed=7)
c = env.simulator.config

action = {
    "price_levels": np.full(c.n_zones, 1, dtype=np.int64),
    "modes": np.zeros(c.n_mcv, dtype=np.int64),
    "targets": obs["locations"].copy(),
    "energy_kwh": np.zeros(c.n_mcv, dtype=np.float64),
}
obs, _, _, _, _ = env.step(action)  # Price-list index 2, then customer choice.
action["targets"] = obs["locations"].copy()
obs, profit, terminated, truncated, info = env.step(action)  # All vehicles WAIT.
```

All four fields must have the declared sizes, number formats, and allowed values, even when the current stage ignores some fields. Together these rules define the **action space**. The Gymnasium checker may warn that some observations have no fixed upper bound, because random arrival counts and normally distributed price changes can be arbitrarily large in the model. It may also warn that energy actions use physical kWh rather than scaled fractions. These warnings describe the chosen interface; [VALIDATION.md](VALIDATION.md) explains their limits.

## 8. Inspect the period that just finished

After dispatch, the returned `obs` usually contains the **next** period's requests. The returned `info` describes the **completed** period. Use `info["completed_period"]`, `info["request_origins"]`, `info["request_energy_kwh"]`, and `info["request_choices"]` when matching outcomes to requests.

Important outcomes include:

| `info` key | Meaning |
|---|---|
| `served_mobile[m, i]` | Number of mobile requests served by a vehicle at an origin |
| `delivered_mobile_kwh[m, i]` | Their delivered energy |
| `served_fixed[f]` | Number served by each FCS slot |
| `served_request_mask[r]` | Whether the whole request was served |
| `assigned_mcv[r]` | Serving vehicle ID; `-1` if there is no mobile assignment |
| `unmet_mobile_kwh`, `unmet_fixed_kwh` | Selected operator demand that was not fulfilled |
| `reward_terms` | Each revenue and cost component used in period profit |
| `energy_ledger` | Battery energy before and after, travel, grid exchange, customer delivery, and any difference left when checking that the energy accounting balances |

The period summary `info["metrics"]` counts the shares of **all arriving customers** who leave or receive service. The remaining share chose the operator but stayed unserved. Both reported shares are zero when no customers arrive. MCV active fraction is the fraction of vehicles that actually execute an operation other than WAIT.

**Utilization** compares energy delivered with available service capacity. Mobile service utilization divides delivered mobile kWh by the fleet's nominal capacity for a full period; it does not subtract travel time from that denominator. Station utilization divides station delivery by its customer-service energy limit and is reported as zero when that limit is zero.

These measurements describe operations. To judge a learned policy, compare independent profit, service, and risk results against benchmarks; see the reading guide in [README.md](../../README.md).
