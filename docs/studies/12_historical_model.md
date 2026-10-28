# 12. Learning the market model from historical quotes

**Date:** 2026-10-27 · **Runner:** `experiments/run_history_adp_study.py` · **Saved results:** `outputs/history_adp_study_20261027/`

**Question.** Can rollout ADP remain useful when it has to estimate demand and customer choice from past observations, and what happens when test demand changes?

This study uses the 25-zone, ten-vehicle city. The evaluator knows the true environment, but controllers receive only a model fitted from historical quote records. PPO is retrained in that same fitted simulator. The inventory rule is tuned there and receives its own learned prices. This tests the complete controllers with estimated market parameters; it is not an equal-computation comparison.

```mermaid
flowchart LR
    A["Past quote records"] --> B["Fit demand and customer choice"]
    B --> C["Train prices and PPO; tune inventory"]
    C --> D["Freeze model and controllers"]
    D --> E["Compare on private test days"]
    F["Original or shifted demand"] --> E
```

The rollout calculation itself is unchanged: it still compares current profit plus Monte Carlo continuation profit under service-first dispatch. What changes is the model supplying those imagined customers. The history fit learns a market model, not a dispatch value network; the separate price search learns a pricing policy from simulated rewards.

## What comes from data?

The history contains 120 fitting days and 32 separate diagnostic days. A logging policy offers independently randomized legal mobile prices and dispatches with a simple service-first rule. It records every quote opportunity, including customers who leave. These histories are synthetic; no real operator dataset is included.

| Quantity | How it is obtained |
|---|---|
| Arrival rate for each hour and zone | Average observed count, including zeros, with one pseudoday of shrinkage toward that zone's all-hour average |
| Request energy range | Observed minimum and maximum, expanded slightly to account for finite sampling |
| Customer service benefit and price sensitivity | Bounded, ridge-penalized likelihood fit to the chosen alternative and all offered alternatives; ridge coefficient 0.000001 |
| Mobile waiting, station waiting and station travel sensitivity | Fitted in the same customer-choice model |
| Network, fleet, batteries, operating costs and tariff process | Known operational inputs |

The assumed families are Poisson arrivals, uniformly distributed request energy, and multinomial-logit choice with outside utility zero. Fitting estimates their parameters; it does not discover the distribution families. The choice model uses five coefficients: a common service intercept and penalties for total payment, mobile wait, fixed-station wait, and station travel. Physical battery and travel calculations are unchanged.

Fit data contain **123,880 requests**. The held-out histories contain **32,963 requests**. Their prediction checks are:

| Check on held-out history | Result |
|---|---:|
| Predicted / observed requests per day | 1,032.33 / 1,030.09 |
| Hour-zone mean count RMSE | 0.368 |
| Same RMSE for a fitted zone-only, time-constant baseline | 0.998 |
| Choice mean negative log likelihood, nats/request | 1.106 |
| Predicted / observed outside-choice share | 41.36% / 41.77% |
| Observed request energies inside fitted range | 100% |

The choice optimizer converged in six iterations and its observed information matrix had full rank. These checks describe prediction on the original historical regime. They do not establish accuracy after a demand shift, and aggregate choice shares alone do not prove calibration for every customer group.

## Keeping the test environment private

At pricing time, each controller sees fleet state, current public prices and the **fitted** arrival forecast. It cannot see the current realized requests. At dispatch, it also sees the actual requests and customer choices that have just occurred. It still receives the fitted forecast, even under a demand shift.

The rollout planner generates every imagined future from the fitted simulator. It never receives the private test scenario or its random generator. Current candidate outcomes use known physics and the observed current requests, which is information available when dispatch occurs. PPO and inventory receive the same forecast replacement. This closes the earlier path through which a raw observation could reveal the true arrival-rate table.

The fitted model remains frozen for all test conditions. There is no online refitting, checkpoint selection using test profit, or separate tuning for a shift.

## Training and selection

All policies share one historical fit. Seeds 7, 19 and 37 repeat policy optimization and, for ADP, planning randomness; they are not three independently collected histories.

| Stage | Budget |
|---|---|
| Fixed-price inventory tuning | 49 price/battery-value settings, each on 32 fitted-model days |
| Initial pricing fit, per seed | 12 search rounds × 24 candidates × 16 days, with service-first dispatch |
| Pricing refinement, per seed and dispatcher | 4 rounds × 8 candidates × 4 days, separately for rollout and inventory |
| PPO warm start, per seed | 256 inventory-teacher days and 100 imitation epochs |
| PPO training, per seed | 500 updates × 16 episodes = 8,000 days |
| PPO monitoring | 32 fitted-model days at initialization, after imitation and every 100 updates |
| Final policy selection | 16 further fitted-model days, shared by all candidates |

PPO uses hidden width 64, six epochs per update, minibatches of 192, actor/critic learning rates 0.0001/0.001, reward scale 20, normalized advantages, GAE coefficient 1, dispatch entropy coefficient 0.005 and target KL 0.03. Its teacher uses the established bounded two-sweep inventory recipe: price 0.60 USD/kWh and early/late stored-energy values 0.30/0.15 USD/kWh, with zero terminal value. The benchmark is retuned separately; its selected fixed setting is 0.80 USD/kWh and early/late values 0.20/0.10 USD/kWh. PPO can change both pricing and dispatch during its 500 updates. This comparison evaluates these specified training recipes, rather than an exhaustive PPO hyperparameter search. The configuration and all training metrics are saved.

For rollout and inventory, each seed's pricing selection compares the inventory-selected fixed price, the initial learned rule and its dispatcher-specific refinement. ADP does not receive a separate seven-price tuning sweep in this study. PPO selection compares final, monitoring-best and imitation-only checkpoints, in that tie order, using stochastic actions. Monitoring-best is allowed to be the untrained initial checkpoint if it has the highest monitoring profit; the selected update and warm-start status are recorded.

The resulting selections are:

| Seed | ADP pricing | Inventory pricing | PPO checkpoint |
|---|---|---|---|
| 7 | Refined | Initial learned | Final, update 500 |
| 19 | Initial learned | Refined | Final, update 500 |
| 37 | Initial learned | Initial learned | Final, update 500 |

These choices use the same 16 fitted-model validation days. All selected PPO checkpoints completed the inventory warm start before reinforcement learning.

Across three seeds, pricing search evaluates 14,592 candidate episodes and PPO collects 24,000 training episodes, plus 768 teacher episodes. Pricing candidates in a round share days, so candidate episodes are not independent demand samples. Every ADP decision uses eight imagined futures per candidate plan. All methods retain the original business-profit objective.

## Independent tests

There are 128 fresh paired days in each condition. Each condition runs three selected ADP policies, three PPO policies, three selected learned-price inventory policies, and one fixed-price inventory rule: **3,840 policy episodes** in total.

| Condition | Change in the private environment |
|---|---|
| Original pattern | Independent days from the historical generating regime |
| Higher volume | Every arrival rate multiplied by 1.4 |
| Shifted places and times | Demand locations rotated 180 degrees and the hourly pattern delayed by three hours, circularly; expected daily request total unchanged |

Customer-choice parameters remain unchanged in these tests. The controllers are not told which condition is running. Separate conditions have separate day banks; within a condition, every controller faces the same seed bank. Shared randomness reduces comparison noise, although different prices can produce different customer choices.

## Results

Rollout ADP earns more than both inventory variants and the implemented PPO recipe in all three conditions. The table reports mean discounted profit in dollars per day. Each learned-method row averages the three separately optimized policies.

| Controller | Original pattern | 40% more arrivals | Shifted places and times |
|---|---:|---:|---:|
| **Rollout ADP, learned prices** | **940.46** | **1,081.91** | **932.46** |
| Joint PPO | 718.00 | 771.83 | 717.58 |
| Inventory, selected learned prices | 925.53 | 1,038.58 | 917.91 |
| Inventory, fixed price | 915.67 | 1,060.72 | 908.82 |

For each comparison below, the interval comes from 5,000 paired bootstrap resamples of the 128 test days. Policies are first averaged within each day, then days are resampled. These intervals are conditional on this historical fit and these trained policies; they do not include uncertainty from new histories or retraining. They are individual 95% intervals, without a multiple-comparison adjustment.

| ADP minus comparator | Original pattern | 40% more arrivals | Shifted places and times |
|---|---:|---:|---:|
| PPO | +222.46 [216.16, 228.75] | +310.08 [302.76, 317.59] | +214.88 [209.12, 221.14] |
| Learned-price inventory | +14.93 [12.50, 17.38] | +43.33 [39.23, 47.50] | +14.55 [12.30, 16.99] |
| Fixed-price inventory | +24.79 [20.97, 28.61] | +21.18 [16.25, 26.14] | +23.64 [19.76, 27.48] |

The advantage is not carried by one training seed. For example, comparing each ADP seed with the corresponding learned-price inventory seed gives:

| Policy seed | Original pattern | 40% more arrivals | Shifted places and times |
|---|---:|---:|---:|
| 7 | +10.10 [6.15, 13.94] | +45.28 [39.29, 51.02] | +12.70 [9.30, 16.17] |
| 19 | +15.31 [11.41, 19.10] | +27.70 [22.14, 33.22] | +14.22 [10.33, 18.20] |
| 37 | +19.40 [15.69, 23.11] | +57.00 [52.11, 62.02] | +16.74 [13.60, 19.94] |

All 27 individual comparisons—three seeds, three conditions, and three comparators—have positive lower interval bounds. All 3,840 final episodes have zero invalid actions. ADP's mean gain over the better inventory variant in each condition is 14.93, 21.18 and 14.55 dollars per day, respectively. Naming that better variant describes the results; no controller is switched or retuned using test scores.

Learned pricing does not help every controller under every shift. With 40% more arrivals, fixed-price inventory earns 22.14 more than learned-price inventory. ADP remains ahead of both. Higher absolute profit in that condition also reflects a larger market; compare controllers within a column, not the absolute profit levels across columns.

These results show that rollout can remain effective with an estimated market model and these two specified demand shifts. They do not establish robustness to arbitrary distribution changes, misspecified choice families, or unobserved quote traffic. Nor do they establish that PPO in general is inferior: this is one specified 500-update training recipe. The earlier known-model study used different test banks, so its score difference from this study is not an estimate of the cost of learning the model.

## Computation

Decision times below are means in milliseconds on the original-pattern tests, including the fitted-model observation and consistency checks. Three evaluation workers ran concurrently, so these are measurements of this run, not isolated hardware benchmarks.

| Controller | Pricing decision | Dispatch decision |
|---|---:|---:|
| Rollout ADP | 2.26 | 453.18 |
| Joint PPO | 5.70 | 8.70 |
| Inventory, selected learned prices | 2.25 | 29.26 |
| Inventory, fixed price | 2.23 | 26.26 |

The successful pipeline took about 96 minutes with three workers: 44.1 minutes for policy training, 5.4 for rule tuning, 5.7 for policy selection and 40.5 for private evaluation, plus initialization, history collection and fitting. Audits and the earlier serialization-failure attempt are additional. ADP uses much more computation while operating; the profit comparison does not equalize training or decision-time budgets.

## What the controllers do on the original-pattern tests

The following are per-day averages on the 128 original-pattern test scenarios. Learned-method rows average seeds 7, 19 and 37. Energy columns sum over the whole fleet.

| Controller | Mobile energy delivered (kWh) | Grid energy bought for fleet recharge (kWh) | Unmet mobile demand (kWh) | Final stored battery energy (kWh) |
|---|---:|---:|---:|---:|
| Rollout ADP | 601.9 | 403.9 | 37.1 | 86.8 |
| Joint PPO | 449.5 | 135.9 | 414.2 | 86.3 |
| Inventory, selected learned prices | 463.0 | 177.4 | 50.2 | 82.2 |
| Inventory, fixed price | 370.8 | 205.6 | 21.3 | 97.5 |

ADP attracts and serves more mobile business, with an average active-zone price of 0.757 USD/kWh versus 0.786 for learned-price inventory. Compared with that inventory controller, ADP adds 73.26 in discounted mobile revenue but loses 21.15 in fixed-station revenue. Discharge revenue adds 3.73; additional procurement and travel cost 30.57 and 16.24; reduced unmet-demand cost saves 5.91. Together these explain the 14.93 profit difference, up to rounding. These are differences between complete pricing-and-dispatch controllers, so this table does not isolate dispatch at a common price.

PPO attracts about 863.7 kWh of mobile demand per day and delivers 449.5; ADP attracts 639.1 and delivers 601.9. PPO's final fleet battery total is almost the same as ADP's, so total closing inventory alone cannot explain the gap. Possible contributors include prices that attract demand the fleet cannot serve and the timing and location of energy use. This study does not isolate their causal effects or compare alternative PPO training recipes. Unmet mobile demand counts customers who chose mobile service and were not served; a lower number can also result from attracting fewer mobile customers.

## Using an existing historical dataset

`HistoricalLog` is the data interface, independent of the synthetic collector. Convert records into the following arrays, using `D` days, `T` periods, `Z` zones, `F` stations and `N` quote opportunities:

| Field | Shape and meaning |
|---|---|
| `counts` | `[D,T,Z]` integer arrival counts, with zeros retained |
| `request_day`, `request_period`, `request_zone` | `[N]` zero-based indices; day indexes the first axis of `counts` |
| `request_energy_kwh` | `[N]` requested energy |
| `alternative_payment` | `[N,F+2]` total dollars for the request, not price per kWh |
| `mobile_wait_hours` | `[N]` mobile waiting time |
| `fixed_wait_hours`, `fixed_travel_km` | `[N,F]` station attributes |
| `available`, `chosen` | `[N,F+2]` Boolean availability and `[N]` chosen column |
| `day_ids` | `[D]` strictly increasing, nonnegative integer chronological identifiers |

Alternative columns are mobile, each station in configured order, then outside. The outside option is always available and has zero payment. Counts must agree with the request rows. `public_config` contains operational fields defined in `history_model.PUBLIC_FIELDS`; `public_model_config(scenario)` extracts them and excludes demand and customer-preference parameters.

After constructing and saving a valid `HistoricalLog`, fitting requires only the log:

```python
from charging_env.rl.history_model import HistoricalLog, fit_history_model

log = HistoricalLog.load("historical_quotes.npz")
model, report = fit_history_model(log)
model.save_checkpoint("estimated_market.npz")
model.scenario.to_json("estimated_scenario.json")
```

Use `model.scenario` to construct or train a controller. When evaluating against a different simulator, use `evaluate_history_policy(private_scenario, model.scenario, seeds, controller_factory)`. The factory receives the fitted scenario only. Its returned controller must expose `scenario`, `prices` and `dispatch`. For deployment, apply the fitted forecast projections to the live operational observations rather than copying a private simulator forecast.

Sales-only records do not satisfy this interface: they omit declined offers and hide some arrivals. Missing alternatives, endogenous prices, incomplete quote traffic or unseen travel conditions require a revised observation and estimation model. Real-data validation would also need chronological future periods, several historical fitting windows, and shifts in choice, prices, travel and fleet availability. This study covers estimation error and two specified demand shifts within the current simulator.

An earlier preparation run completed all 500 PPO updates for seeds 7, 19 and 37, then failed during the final restricted checkpoint reload: fitted `Scenario` metadata contained NumPy `float64` scalars that PyTorch's `weights_only=True` loader rejected. The metadata now uses native Python `float` values. The study restarted from history collection with unchanged hyperparameters and the same preparation banks, without importing old checkpoints or results. No policy-selection validation or private test had run. The original artifacts were archived outside the project; their location and hashes are recorded locally in `outputs/history_adp_development_20261027/clean_restart.json`. The budgets above describe the completed rerun.

## Reproduce and inspect

The complete phase sequence is in [RUNNING.md](../RUNNING.md#historical-model-comparison). To print the saved comparison without rerunning policies:

```bash
source .venv/bin/activate
python experiments/run_history_adp_study.py --phase show \
  --study-dir outputs/history_adp_study_20261027
```

The local `outputs/` directory is excluded from Git. The study preserves the original quote logs, fitted model, training histories, all validation candidates, frozen test inputs, per-episode outcomes, planning traces and exact source snapshot. The original known-model controllers remain available for reproducing earlier studies.

## Correctness and reproduction

The full unit suite passes all 351 tests. Three complementary audits also pass:

| Audit | Independent check |
|---|---|
| `audit.json` | Refits the historical model exactly; reconstructs price-search updates, PPO budgets and selection; checks all 3,840 final episodes, profit ledgers and paired intervals |
| `physics_audit.json` | Replays nine complete selected episodes and four historical quote days; checks nine candidate plans using actual simulator transitions and 792 fitted-model future transitions |
| `training_replay_audit.json` | Replays four recorded pricing-training episodes without opening new validation or test scenarios |

The seed audit verifies 26,056 distinct declared environment seeds and 658,944 distinct planned future-query seeds, separated from historical and development use. The physics replay observes 4,856 production planning queries; every one uses the fitted scenario, while the nine executed episodes use the correct private environments. Four historical days reproduce all 4,165 quote records exactly. The largest candidate-score or training-return discrepancy is 2.27e-13, below the 1e-8 tolerance. Frozen sources and experiment records remain unchanged throughout the audits.

Audit replays check saved behavior and do not add independent performance observations. [VALIDATION.md](../reference/VALIDATION.md) describes the evidence files and what the checks establish.
