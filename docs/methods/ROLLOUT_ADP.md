# Rollout ADP, with optional learned pricing

Rollout ADP asks: **if the fleet takes this action now, how much profit could it earn during the rest of the day?** It answers by simulating several possible futures, then chooses the current action with the best estimated total profit.

This is an online, model-based controller. It estimates a continuation value when a decision is needed; it does not fit a value table or network beforehand. It supports different fleet sizes, zone layouts, station configurations, and horizons within the project's `Scenario` model and computation limits.

The planning model can be supplied directly or [estimated from historical quotes](#learning-the-model-from-quote-histories). These are different information assumptions: knowing the simulator's generating parameters is easier than estimating demand and customer choice from observations.

The basic version uses one fixed mobile price all day. An optional [learned pricing policy](#learning-prices-as-well-as-planning-dispatch) chooses prices by zone and hour; dispatch is the same in both versions. [METHODS.md](../METHODS.md#rollout-adp) has the short version. Results are in [studies 10 and 11](../studies/README.md); commands are in [RUNNING.md](../RUNNING.md#rollout-adp). The study runner for the fixed-price version is named `run_general_adp_study.py`, so older text calls this "general ADP".

## Why consider future profit?

Consider two feasible fleet actions. The following numbers illustrate the calculation; they are not experimental results.

| Current action | Profit now | Estimated discounted profit from next period onward | Total score, with discount 0.99 |
|---|---:|---:|---:|
| Serve immediately and leave little usable battery | 12.00 | 10.00 | 21.90 |
| Serve less now and recharge one vehicle | 8.00 | 20.00 | 27.80 |

The second action wins because $`8+0.99\times20=27.80`$. Its future benefit compensates for the smaller immediate profit. Near closing time, that same recharge may be unprofitable because there is too little time left to use the energy.

Locations matter too. Two equally charged fleets can have different prospects if one is near likely demand and the other is far away. Rollout carries both battery levels and vehicle locations through every imagined future period.

This follows the dynamic-programming principle of comparing immediate reward plus the value of the resulting state. [MIT's dynamic programming notes](https://underactuated.csail.mit.edu/dp.html) explain that principle. Here, simulation supplies an approximate future value under a specified **base policy**.

## What happens at one dispatch decision?

Dispatch occurs after the current requests and customer choices are observed. The controller uses that actual observation for the current period; it samples only the future.

1. **Construct a few complete fleet plans.** Each plan specifies one action for every vehicle.
2. **Calculate each plan's current outcome.** Use the exact joint dispatch calculation to obtain current business profit, next locations, and next battery levels.
3. **Simulate the remaining periods.** Starting from each resulting fleet state, run several independent future trajectories. Each trajectory follows the service-first base policy until closing time.
4. **Average the future returns.** Add the current profit and select the plan with the highest estimated total.
5. **Execute only the current plan.** At the next real period, observe what actually happened and repeat the calculation.

The imagined future controller is always the base policy. It does not recursively run rollout inside rollout. The real controller nevertheless replans every period, so it is free to depart from those imagined actions when new information arrives.

### Which plans are compared?

The defaults generate at most five plans before removing exact duplicates:

| Candidate | How it is constructed |
|---|---|
| Service-first, threshold 0.2 | Serve profitable available requests first; otherwise consider recharging a vehicle below 20% usable battery |
| Inventory rule | Coordinate search using current profit plus a battery-inventory adjustment of 0.30 dollars/kWh in the first half of the horizon and 0.15 later |
| Current-profit rule | The same coordinate search with no inventory adjustment |
| Service-first, threshold 0.4 | The service-first rule with a higher recharge threshold |
| Service-first, threshold 0.6 | The service-first rule with a still higher recharge threshold |

Usable battery fraction is $`(e-e_{\min})/(e_{\max}-e_{\min})`$. The service-first threshold applies only when that vehicle has no profitable service assignment; it does not force a low-battery vehicle to abandon a profitable request. The rule skips recharging in the final period.

The inventory adjustment is used **only to propose a candidate**. Rollout scores that candidate using simulated business profit, without adding the adjustment again. The inventory proposal's adjustment is zero in the final period, while the scenario's actual terminal inventory payment remains part of profit.

The base service-first plan is always included. Identical action lists are evaluated once, and exact score ties prefer the earlier candidate. This inclusion gives rollout a direct comparison with its base action; finite sampling still means it can choose an action whose true expected return is worse.

### How does the coordinate search work?

The inventory and current-profit proposals use bounded coordinate search. Starting from all-wait and service-first plans, it tries changing one vehicle's action while holding the others fixed. Every trial is evaluated as a complete fleet action. It accepts improvements and repeats for up to two sweeps by default.

Recharge and export choices in this search use configurable fractions of the vehicle's individually feasible maximum; the default is `1.0`. Shared station capacity can reduce the amount actually delivered. Service-first proposals separately account for remaining station capacity and may request smaller quantities.

This avoids enumerating every possible combination of vehicle actions. It can miss an improvement that requires several vehicles to change actions together. The final rollout choice is therefore the best among the proposed plans, not an exhaustive optimum over all feasible fleet actions.

## The continuation-value calculation

Let physical periods be $`t=1,\ldots,T`$; Python uses `period = t - 1`. Write $`\widetilde S_t`$ for the current dispatch observation and $`\mathbf a`$ for a complete fleet action. The exact current transition gives

```math
x_{t+1}=f(\widetilde S_t,\mathbf a),
```

where $`x_{t+1}`$ contains the ordered vehicle locations and battery energies. Vehicle order matters when allocating whole customer requests. Station exchange uses a separate rule: if combined recharge or export requests exceed a station's corresponding capacity, the simulator scales those requests proportionally.

The base policy is $`\mu`$, which is service-first with recharge threshold 0.2 by default. For each of $`K`$ sampled futures, apply $`\mu`$ from the next period until the horizon. Average those discounted future profits:

```math
\widehat V_{t+1}^{\mu}(x_{t+1})
=\frac{1}{K}\sum_{k=1}^{K}
\sum_{j=t+1}^{T}\gamma^{j-t-1}R_j^{(k,\mu)}.
```

The score used to compare current plans is

```math
\widehat Q_t(\widetilde S_t,\mathbf a)
=R_t^d(\widetilde S_t,\mathbf a)
+\gamma\widehat V_{t+1}^{\mu}
\bigl(f(\widetilde S_t,\mathbf a)\bigr).
```

Here $`R_t^d`$ is the original current-period business profit, $`R_j^{(k,\mu)}`$ is the profit in simulated future period $`j`$, and $`\gamma`$ is the scenario's discount per physical period. The default is $`K=8`$ trajectories per candidate. Discounting advances once per physical period.

The future term estimates the return of the base policy, not the optimal future return. There is no fitted regressor, learned-value bootstrap, or additional reward shaping in this calculation. At the final period, the future sum is empty, so the controller simply selects the candidate with the largest exact current profit.

### Why share the simulated futures?

Every candidate receives the **same future-period/draw seed table** at the current decision. This is called common random numbers. It lets candidates face the same underlying sampled arrivals and price shocks, making their differences easier to compare. Their fleet states and subsequent decisions can still differ.

Planning uses its own random generator, separate from the live environment. It is initialized from the planning seed and episode identifier. During evaluation, the environment seed serves only as that identifier: the planner receives neither the live generator nor the actual future draws. A new shared seed table is generated at each decision, and every seed is recorded in the planning trace.

## Physical constraints and model assumptions

Current outcomes and future simulations use the same full-fleet physics: travel time and energy, battery reserve and capacity, service and charging rates, efficiencies, whole-request allocation, and shared station limits. Profit includes service and export revenue, energy procurement, travel and operating costs, and unmet-demand penalties. Any configured terminal inventory payment is included once in the final period.

The simulated continuation is sufficient for the current model because requests do not wait across periods, trips finish within a period, and future demand and station-price shocks follow the scenario's prescribed distributions. Time-varying arrival patterns are allowed; the distribution need not be identical at every time. The planner does need a model from which it can sample those future conditions.

Changing zone count, vehicle count, station locations, battery initial states, horizon, or valid price levels does not require retraining the fixed-price rollout controller. Supply the new valid `Scenario` and check its settings and performance. A saved pricing checkpoint includes its training scenario. The supplied workflow fits a new checkpoint for a changed scenario; this study does not establish transfer of learned prices without retraining. Physical rules remain those implemented by the simulator, including Manhattan travel distances and at most one fixed station per zone. Longer episodes do not automatically create overnight operation or connected days.

If real demand, customer choice, or travel conditions differ from the planning model, the forecasts can be wrong. Persistent hidden demand regimes, waiting queues, or trips spanning several periods would require changes to the simulator and continuation state. Support for different scenario dimensions does not imply universal optimality or robustness to an incorrect model.

## Learning the model from quote histories

The historical-model workflow first estimates how customers arrive and choose, then gives that estimated model to the controllers. The simulator that generates the historical data and final test days is kept separate. Its true demand and preference parameters are never supplied to a controller.

The input is a **complete quote history**, including customers who chose no service. Each record contains the offered payments, requested energy, available alternatives, waiting and travel attributes, and the customer's choice. Daily counts include zero-arrival hours and zones. The supplied experiment generates these records synthetically using randomized legal price offers and a simple service-first logging rule. It does not contain real-city data. Sales records alone would miss customers who declined and would not satisfy this data assumption.

Three parts of the model are fitted:

| Part | What is estimated from history |
|---|---|
| Arrivals | A Poisson mean for each hour and zone, with smoothing toward that zone's daily average |
| Requested energy | The bounds of a uniform request-size distribution |
| Customer choice | A common service intercept and sensitivity to payment, mobile waiting, station waiting and station travel |

The Poisson, uniform and multinomial-logit families are assumptions; the fitting process estimates their parameters. Physical specifications, network geometry and tariff processes remain known. Separate historical days check demand predictions and choice likelihood. They do not choose policy settings.

PPO and the pricing policies are then trained in the **same fitted simulator**. The inventory rule's settings are selected there too. A separate set of fitted-model validation days selects policies before any private test days are opened. PPO is retrained for this model; earlier checkpoints trained with the true generating parameters are not reused.

During a private test, the forecast model and selected policies stay frozen. Before pricing, every controller receives fitted arrival forecasts, fleet state and observed electricity prices; latent current requests remain hidden. At dispatch, it receives the actual current requests and choices, because customers have now responded. Rollout samples imagined futures only from the fitted model. The private simulator alone determines what actually happens and the profit earned.

This tests both estimation error and demand shifts: the experiment includes the original demand pattern, higher arrival volume, and a changed spatial and hourly pattern. There is no online model update or adjustment to each test regime. The service-first continuation policy and restricted candidate plans remain the same approximations described above. Learning the market model does not make those dispatch approximations optimal.

See [RUNNING.md](../RUNNING.md#historical-model-comparison) for the complete workflow and [study 12](../studies/12_historical_model.md) for its protocol and evidence. This workflow still assumes complete quote visibility, known physical rules and the chosen distribution families; it does not establish reliability under arbitrary real-world changes.

## How it differs from the other controllers

| Controller | Preparation | Decision-time calculation |
|---|---|---|
| Inventory rule | Specify or validate a battery value | Current profit plus a simple inventory adjustment |
| Table ADP | Fit a continuation-value table for the two-vehicle scenario | Current profit plus an interpolated table value; exhaustive search on its finite action grid |
| Rollout ADP | Configure the model, fixed price, base policy, and sample budget | Current exact profit plus Monte Carlo future profit under the base policy |
| Rollout ADP with learned pricing | Fit a small pricing policy from complete simulated returns | Learned prices followed by rollout dispatch; imagined future periods use those same learned prices |
| PPO | Train policy and critic networks on trajectories | Networks produce pricing and dispatch action probabilities |

Table ADP learns offline and reuses its table; see [TABLE_ADP.md](TABLE_ADP.md). Rollout computes a fresh estimate for the states it is comparing and stores no trained value model. PPO learns a direct policy and is usually much cheaper at deployment because it does not simulate future trajectories for each decision.

Including an inventory-rule plan among the candidates does not mean rollout compares two complete policies. After that current inventory action, its imagined future still follows service-first. An action that would work well with repeated inventory-based decisions can therefore receive an inaccurate ranking for that purpose. The continuation estimates the base policy's value, not the best achievable value.

Likewise, the candidate set contains only a few complete fleet plans. It may omit a useful combination of actions, or a partial recharge available to the original exhaustive solver. These approximation choices explain why a flexible rollout implementation need not outperform a specialized rule or the original small-state ADP table. Their contribution to a particular profit gap would require a separate controlled experiment.

A strong simulator is an important resource for rollout. Its access to candidate transitions and future model queries differs from PPO's network inference, so comparisons should report decision time as well as profit. Neither approach is guaranteed to beat the other on a new scenario.

## Learning prices as well as planning dispatch

Fixed pricing leaves a useful decision unused. For example, a fleet with little available energy may benefit from a higher mobile price that reduces the amount of demand it attracts. A well-positioned fleet with spare energy may profit from a lower price. The effect on fixed-station demand matters too, because the operator earns revenue from both services.

The new pricing policy maps the current permitted information to one legal price at each zone. It shares twelve coefficients across all zones. Its inputs summarize time, forecast demand, available service capacity, fleet battery, travel time, competing station prices, replacement-energy cost and expected station load. Two interaction terms let battery and capacity have different effects near closing time. The features are physically normalized and bounded; the linear price target is rounded to the nearest configured price level.

Pricing uses the same information boundary as Joint PPO. It sees arrival forecasts, fleet locations and batteries, current station prices and the grid purchase price. Forecasts are prescribed in the known-model examples and estimated from history in the historical-model workflow. Pricing cannot see realized current requests, individual request sizes, customer choices or the congestion those requests reveal. Availability is recomputed from fleet physics rather than accepted from an input mask.

### How the price coefficients are learned

The learner uses **cross-entropy policy search**:

1. Propose several coefficient vectors around the current search mean.
2. Run complete simulated episodes for each proposal. All proposals in a generation face the same environment and planning seeds.
3. Rank them by original discounted business profit, including fixed-station revenue and every operating cost.
4. Move the search distribution toward the better proposals, keep the winning policy, and repeat on fresh training scenarios.

The first phase learns prices with inexpensive service-first dispatch. This is a computationally cheaper initial policy. A second phase refines the coefficients using the intended dispatch controller: rollout ADP or the inventory rule. Inventory receives its own pricing refinement so that the comparison can distinguish better pricing from better dispatch. Separate validation scenarios choose among the fixed-price fallback, the initial learned policy and the refined policy. Final test scenarios are used only after those choices are frozen.

This learns a pricing policy directly from rewards. It does not fit a dispatch value table or use PPO gradients. Training scores are noisy optimization evidence; selecting a large training score does not establish a performance gain. The independent comparison reports what the selected policies actually earn.

### Keeping the imagined future consistent

The same frozen pricing actor is called at every future period inside a rollout, before simulated customers choose a service. Future dispatch still follows service-first. Thus each continuation estimate describes service-first dispatch under the learned pricing rule.

Changing only the live price callback would be inconsistent: it would value future fleet states assuming the old fixed prices. The new implementation changes both live and imagined pricing. A constant-price actor reproduces the original rollout outcomes with the same seeds, providing a direct check of the integration.

The learned coefficients are saved in a small `checkpoint.npz`. Planning remains online after that checkpoint is loaded. [RUNNING.md](../RUNNING.md) gives the training, evaluation and comparison commands; [RESULTS.md](../RESULTS.md) separates gains from pricing and dispatch.

## Run and inspect the fixed-price controller

From the project root, evaluate the four-zone `b=3` scenario at a fixed mobile price of 0.80 dollars/kWh:

```bash
source .venv/bin/activate
python -m charging_env.rl.rollout_adp \
  --scenario configs/utility_b3.json \
  --seed 7 --price-index 5 \
  --rollout-samples 8 --episodes 16 \
  --eval-seed-base 801000001 \
  --output-dir outputs/rollout_adp
```

Use `--scenario configs/synthetic_25_zones.json` for the 25-zone, ten-vehicle case. Both supplied files now use the three premium levels, so index 2 is the 1.7x premium over the nearest station's EV price. For another scenario, use an index from its own menu. Omitting `--price-index` selects the middle index, not a validation-selected price.

This command is a **known-model evaluation**: the same scenario defines planning and the live simulator. It does not estimate a model from historical data or test a hidden demand shift. Use the historical-model study runner for that comparison.

Here `--seed` controls planning randomness; there is no training seed or learned checkpoint. The run uses NumPy and the base environment dependencies. PyTorch is required separately for PPO comparisons. If the output directory is occupied, the command creates a numbered sibling and prints its actual path.

| Output | What it contains |
|---|---|
| `summary.json` | Mean discounted profit, uncertainty interval, unmet demand, ending battery, invalid actions, and timing |
| `evaluation.json` | Complete metrics for every test episode |
| `planning_trace.json` | Candidate scores, selected plans, per-draw returns, future seeds, and query counts at every decision |
| `protocol.json` | Configuration, scenario, evaluation seeds, and source hashes recorded before evaluation |
| `provenance.json` | Algorithm settings and cumulative planning statistics after evaluation |

For a comparison with frozen PPO, Table ADP where applicable, and validation-tuned rules, use [run_general_adp_study.py](../../experiments/run_general_adp_study.py) and the workflow in [RUNNING.md](../RUNNING.md). The separate [run_adp_transfer.py](../../experiments/run_adp_transfer.py) checks different fleet dimensions, operating periods, moving demand patterns, and constrained stations using fixed settings. Those are known-model configuration changes: each policy receives the correct scenario model. They do not test robustness to model error, and existing PPO checkpoints cannot be reused across incompatible input dimensions.

## Computation and performance limits

With $`C`$ distinct candidate plans, $`K`$ rollout samples, and $`T-t`$ future periods, one decision uses

```math
C K (T-t)
```

future period transitions. Only $`K(T-t)`$ distinct future draw seeds are needed because candidates share draws. The coordinate searches that construct proposals require additional current-period outcome calculations; these are counted separately.

With the default five candidates, eight samples, and twelve-period horizon, the first decision needs at most 440 future transitions. Later decisions are cheaper, and duplicate plans reduce the count. The implementation limits candidate plans to 16, samples to 256, and future transitions per decision to 50,000; a configuration can be rejected when its worst-case budget exceeds a limit.

More samples reduce sampling noise but increase runtime. They do not fix a weak base policy, a missing candidate action, or an inaccurate model. Including the base plan does not guarantee improvement under finite sampling.

Judge performance on independent scenarios after settings are fixed. Compare policies on the same scenario seeds, repeat planning seeds, and inspect paired profit differences alongside unmet demand, charging, final battery, invalid actions, and runtime. Raw ending battery alone is not a success measure: it may be useful insurance, unused inventory, or economically valuable terminal stock depending on the scenario. See [RESULTS.md](../RESULTS.md) for the measured comparisons.

| Implementation | Responsibility |
|---|---|
| [rollout_adp.py](../../src/charging_env/rl/rollout_adp.py) | Candidate comparison, future rollouts, local randomness, trace, and evaluation command |
| [adaptive_pricing.py](../../src/charging_env/rl/adaptive_pricing.py) | Restricted pricing features, shared price coefficients, policy search, and checkpoints |
| [priced_rollout.py](../../src/charging_env/rl/priced_rollout.py) | Apply the learned prices in live operation and simulated futures; evaluate a saved pricing policy |
| [history_model.py](../../src/charging_env/rl/history_model.py) | Fit and save arrival, request-size and customer-choice models from complete quote histories |
| [history_control.py](../../src/charging_env/rl/history_control.py) | Collect observable histories and separate fitted-model controllers from private test environments |
| [scalable.py](../../src/charging_env/benchmarks/scalable.py) | Bounded coordinate search for inventory and current-profit proposals |
| [policies.py](../../src/charging_env/benchmarks/policies.py) | Service-first rule and exact joint current-period outcomes |
| [fitted_value.py](../../src/charging_env/rl/fitted_value.py) | Shared helper for sampling a future dispatch observation from a specified fleet state |
