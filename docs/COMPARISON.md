# Compare ADP, PPO, and the benchmark rules

The newest experiment adds learned mobile prices to rollout ADP and gives the inventory rule its own price learner. The four-way comparison separates the benefit of pricing from the choice of dispatch controller. Learned pricing gives a consistent gain across the three tested seeds on the 25-zone network. Its gain is much smaller in the four-zone case, where the original exhaustive inventory rule remains stronger.

For how the controller works, read [Rollout ADP](methods/GENERAL_ADP.md). For commands, read [TRAINING.md](TRAINING.md#learn-prices-for-rollout-adp). Earlier studies remain below and use different scenario banks; their numbers should not be combined with the newest results.

## Learned pricing on independent scenarios

The price policy shares twelve coefficients across zones. It can react to time, prescribed demand rates, fleet battery and position, and current electricity prices. Realized current requests remain hidden until after pricing. Cross-entropy policy search fits the coefficients using complete discounted business profit. The same pricing policy governs actual operations and every imagined future period inside ADP.

Each network uses training seeds **7, 19 and 37**, 32 separate validation days, and **256 new final test days**. All compared controllers face the same test days. The learned rows average three separately executed policies; they are not ensembles. No final test result changes training, pricing choices or benchmark settings.

| Controller | Four zones / two vehicles | 25 zones / ten vehicles |
|---|---:|---:|
| Rollout ADP, validated fixed price | 130.26 | 911.52 |
| Rollout ADP, selected pricing | 130.95 | 941.91 |
| Bounded inventory, validated fixed price | 124.76 | 914.50 |
| Bounded inventory, selected pricing | 126.84 | 924.80 |

Entries are mean discounted business profit in dollars per twelve-hour episode, including both mobile and fixed-station business. **Selected pricing** means validation chooses among the fixed-price fallback, the initial learned policy and its dispatch-specific refinement. It does not imply that every selected seed uses an adaptive policy.

The additional strong small-case reference, the original exhaustive half/full inventory rule, earns **133.63**. Its joint action enumeration does not scale to the ten-vehicle case. It retains its historical settings: price 0.60 and battery values 0.30/0.15 early/late. It was specified before testing and was not retuned on these test days.

### What changed in profit?

- **Four-zone case:** selected ADP improves on fixed-price ADP by **0.70 [0.25, 1.13]**. That gain comes from seed 19; seeds 7 and 37 retained fixed pricing. ADP exceeds the separately price-trained bounded inventory rule by **4.12 [2.63, 5.58]**, but remains **2.67 below the exhaustive reference**, with ADP-minus-reference interval **[-3.83, -1.52]**.
- **25-zone case:** selected ADP improves on fixed-price ADP by **30.39 [27.68, 33.08]**, a **3.33%** gain. It exceeds fixed-price inventory by **27.41 [24.58, 30.38]**, or **3.00%**, and separately price-trained inventory by **17.11 [15.34, 18.92]**, or **1.85%**. Each of the three seeds has a positive paired interval for all three comparisons.

Intervals are paired 95% bootstrap intervals from 5,000 resamples of the test days. They describe scenario sampling uncertainty conditional on the evaluated policies and three fixed seeds. They do not establish the success rate over arbitrary future retraining runs. A positive average alone is not evidence that every seed improved.

The fixed-price rollout benchmark chooses among all seven prices using average validation profit across its three planning seeds. It selects **0.60** in the small network and **0.80** in the city. The bounded inventory benchmark checks 49 price/battery-value settings; it selects price **0.80** in both cases, with battery values **0.20/0.00** early/late in the small case and **0.20/0.10** in the city. Prices and battery values are in dollars per kWh. Those inventory schedules are also used during its price refinement and final evaluations.

### Which pricing policy was selected?

Every seed first trains prices with inexpensive service-first dispatch. Rollout and inventory then start from that same initial price policy and receive their own refinement. Validation makes the following choices:

| Network and dispatcher | Seed 7 | Seed 19 | Seed 37 |
|---|---|---|---|
| Small rollout ADP | Fixed | Initial learned | Fixed |
| Small inventory | Fixed | Refined | Refined |
| City rollout ADP | Initial learned | Refined | Initial learned |
| City inventory | Initial learned | Refined | Refined |

Refinement does not automatically improve a policy: its search uses noisy sampled returns and a limited budget. Keeping the initial learned policy and fixed-price alternatives makes that outcome visible. Fixed-price fallbacks reproduce the corresponding fixed controller exactly on the final test bank.

<details>
<summary>Per-seed results and the pricing-swap diagnostic</summary>

| Network | Seed | Fixed-price ADP | Selected ADP | Selected inventory | ADP minus selected inventory, 95% interval |
|---|---:|---:|---:|---:|---|
| Small | 7 | 130.32 | 130.32 | 124.76 | 5.56 [3.48, 7.60] |
| Small | 19 | 130.17 | 132.27 | 128.69 | 3.58 [1.98, 5.11] |
| Small | 37 | 130.27 | 130.27 | 127.06 | 3.21 [1.56, 4.80] |
| City | 7 | 911.89 | 939.01 | 930.33 | 8.68 [6.63, 10.68] |
| City | 19 | 911.02 | 942.75 | 920.93 | 21.81 [18.93, 24.73] |
| City | 37 | 911.65 | 943.99 | 923.14 | 20.85 [17.74, 24.07] |

The diagnostic gives inventory dispatch the selected ADP pricing policy while retaining its validation-selected battery-value schedule. The price function is shared; realized price paths can still differ because different dispatch decisions lead to different fleet states. This is a comparison of those complete dispatch controllers, not proof of optimality at a fixed realized sequence of prices.

In the small case, ADP earns **7.39 [6.15, 8.62]** more than that pricing-swap control. Separately, learning prices improves the bounded inventory rule over its own fixed-price version by **2.08 [1.39, 2.78]**. In the city, ADP exceeds the pricing-swap control by **18.44 [16.68, 20.21]**. Inventory's own pricing gain is **10.30 [8.15, 12.50]**.

City operating quantities are fleet totals in kWh per episode. Ending battery subtracts the ten vehicles' combined 50-kWh reserve.

| Controller | Mobile energy delivered | Recharge bought | Unmet mobile energy | Ending battery above reserve |
|---|---:|---:|---:|---:|
| Fixed-price ADP | 380.44 | 182.68 | 11.22 | 33.30 |
| Selected-pricing ADP | 605.53 | 396.60 | 42.49 | 38.31 |
| Fixed-price inventory | 369.28 | 207.85 | 20.21 | 47.70 |
| Selected-pricing inventory | 450.87 | 177.06 | 39.89 | 36.99 |

The learned city ADP policies mostly use prices 0.60 and 0.80, with an average of **0.7525** across available zone-period offers. Fixed-price ADP uses 0.80. Lower prices attract more mobile demand; ADP serves more energy and recharges more, but unmet mobile energy also rises. Its discounted mobile revenue increases by **100.62** relative to fixed-price ADP. Lower fixed-station revenue, less export revenue and higher procurement, travel and unmet-demand costs offset most of that increase, leaving the **30.39** profit gain. This reconciles the accounting; it does not isolate which learned feature caused each decision.

Against inventory with its own learned prices, ADP earns **79.78** more mobile revenue but also incurs more procurement and travel costs and earns less fixed-station revenue. The net gain is **17.11**. Neither minimum recharge nor minimum unused battery alone defines the best business policy.

</details>

### What this experiment establishes

The learner optimizes original business profit without reward shaping. Both dispatchers receive the same initial pricing fit and the same 128-episode refinement budget per seed, but rollout additionally simulates future fleet operations. This is not an equal-computation comparison. Observed mean city dispatch time was **492.60 ms** for selected ADP and **28.08 ms** for selected inventory; the learned pricing calls themselves took about **0.21 ms**. These are descriptive timings under local parallel evaluation, not hardware-independent speed guarantees.

These are independent simulated test days from the supplied models. They establish performance within those models, not robustness to an unknown real-world demand process or an incorrect simulator. Recorded pricing checkpoints use their training scenarios; transfer of the learned prices to a different scenario has not been tested. The original table ADP and PPO implementations remain available; they were not retrained or evaluated on this newest test bank.

**309 automated tests pass.** Both independent audits pass, and all final episodes contain **zero invalid actions**. The audits reconstruct training proposals, verify selections and ledgers, recompute paired intervals, replay 36 training episodes and six complete final episodes, and independently check rollout scores through the simulator. The largest replay/score discrepancy is **2.27e-13**. The study contains 29,184 training candidate episodes, 5,248 validation policy episodes and 6,912 final policy episodes. Candidates share training days within each search generation, so candidate-episode counts are not counts of independent demand days.

The completed local bundle is `outputs/priced_adp_study_20261025/`. It includes checkpoints, all proposed price coefficients, validation choices, independent episode records, planning traces, source snapshots and audits. Generated outputs are ignored by Git; this report is included in the repository. Display the saved scorecard without retraining:

```bash
python experiments/run_priced_adp_study.py --phase show \
  --study-dir outputs/priced_adp_study_20261025
```

[TRAINING.md](TRAINING.md#learn-prices-for-rollout-adp) includes a short standalone fit and the full reproduction procedure. [VALIDATION.md](reference/VALIDATION.md#evidence-behind-the-learned-pricing-comparison) explains the correctness checks. The following sections preserve the earlier studies.

## Rollout ADP on independent scenarios

**Earlier fixed-price study.** Each main case uses **256 fresh simulated days** shared by all compared policies. Rollout uses eight future simulations per candidate and planning seeds **7, 19 and 37**. These are planning replicates, not offline training runs. PPO uses the previously trained 500-update models for those three training seeds. Averaged rows describe three separately executed policies, not an action ensemble.

On 32 separate validation days per case, rollout selected a fixed price of **0.60** for the small network and **0.80** for the 25-zone network. The two rule families were retuned across 70 settings, including prices. All settings and source code were frozen before the main test.

| Controller | Four zones / two vehicles | 25 zones / ten vehicles |
|---|---:|---:|
| Rollout ADP | **127.78** | **910.21** |
| Saved Joint PPO | 126.53 | 729.89 |
| Retuned bounded inventory | 126.35 | 913.65 |
| Original exhaustive inventory, supplementary | 131.41 | Not scalable to this fleet |
| Retuned service-first | 123.20 | 896.22 |

Entries are mean discounted business profit in dollars per twelve-hour episode. Compare methods within a column. Both mobile and fixed-station revenue are included.

The distinction between inventory rows matters. The bounded rule uses two coordinate sweeps and maximum exchange amounts in both networks. Its selected battery values are 0.30/0.15 early/late in the small case and 0.20/0.10 in the city case. The supplementary small-case reference preserves the historical strongest validation-selected rule: price 0.60, battery values 0.30/0.15, and exhaustive joint search with half/full exchange. It was specified separately before evaluation and was not retuned on the test bank.

### What the profit differences establish

- **Four-zone case:** rollout minus PPO: 1.25 [0.18, 2.30]; bounded inventory: 1.43 [0.62, 2.29]; service-first: 4.58 [2.63, 6.41].
- **25-zone case:** rollout minus PPO: 180.32 [175.44, 185.30]; bounded inventory: -3.45 [-5.15, -1.70]; service-first: 13.99 [12.42, 15.65].
- **Original exhaustive inventory:** small-case rollout earns 3.63 less, with rollout-minus-rule interval [-4.64, -2.61].

These are paired 95% intervals from 5,000 scenario bootstrap resamples. They describe scenario/action sampling uncertainty conditional on these fixed policies and planning seeds. They do not establish arbitrary retraining performance or robustness to a different real-world distribution.

The original two-vehicle ADP table remains stronger in its tractable small case: **136.10** mean profit. The myopic current-profit control earns 105.64 in the small case and 816.39 in the city case. That control uses zero-shadow coordinate search; its different candidate search means the gap is not a pure ablation of future simulation.

<details>
<summary>Results for each seed and operating behavior</summary>

| Seed | Small rollout ADP | City rollout ADP | Small PPO | City PPO |
|---|---:|---:|---:|---:|
| 7 | 128.04 | 910.60 | 126.71 | 730.71 |
| 19 | 127.56 | 909.66 | 126.41 | 718.81 |
| 37 | 127.74 | 910.37 | 126.47 | 740.14 |

The seed controls online sampling for rollout and prior training for PPO. The same numerical seed does not imply equal training data or computation.

City operating quantities are fleet totals in kWh per episode:

| Controller | Mobile energy delivered | Recharge bought | Unmet mobile energy | Ending battery above reserve |
|---|---:|---:|---:|---:|
| Rollout ADP | 379.34 | 187.19 | 12.10 | 34.27 |
| PPO | 474.45 | 194.37 | 397.42 | 72.17 |
| Bounded inventory | 368.47 | 206.47 | 21.31 | 46.75 |
| Service-first | 390.25 | 214.88 | 2.68 | 189.40 |

Unused battery alone is not a measure of policy quality. Location, remaining whole requests, price-dependent customer choices and future service opportunities all matter. Different pricing policies attract different amounts of mobile demand.

</details>

### Why a general rollout controller can still lose

Rollout evaluates future actions under service-first, not an optimal continuation. After an inventory-rule candidate is selected for the current period, its imagined future still follows service-first. Including that current candidate therefore does not guarantee improvement over repeatedly executing the inventory rule. Finite simulation noise and the limited set of complete fleet plans add further approximation error. The original small-state ADP uses a richer joint action grid and reusable fitted values.

In the small case, rollout delivers 146.88 kWh of mobile service versus 137.69 for exhaustive inventory, but buys 102.64 versus 77.38 kWh for recharge. Its extra discounted mobile revenue of 4.93 is outweighed by 3.93 more procurement cost, 3.49 more travel cost, and 1.49 less fixed-station revenue. More mobile service does not necessarily maximize total profit. This is an accounting explanation, not an isolated test of which planning approximation caused the choices.

The methods also use different information and computation. Rollout queries the known generative model during every decision; PPO performs network inference and learns pricing as well as dispatch. This is a comparison of complete implemented controllers, not an isolated or equal-compute comparison of learning objectives.

Observed mean city dispatch times were **433.16 ms** for rollout, **20.99 ms** for bounded inventory, and **0.52 ms** for PPO. These are descriptive timings under local parallel evaluation, not hardware-independent complexity claims.

## Portability to other known configurations

The same rollout implementation was also tested with unequal initial batteries, three constrained stations, moving demand patterns and normally distributed station-price shocks with a 15% relative standard deviation. Reserve-level vehicles start at stations. The 36-zone case has a terminal inventory payment of 0.08 per stored kWh. No offline retraining or algorithm changes were needed.

All methods below use a predeclared fixed price of 0.80. Inventory uses battery values 0.30/0.15 early/late, and service-first uses threshold 0.20; they are **not retuned** for these additional cases. Each case has 32 fresh scenarios, with rollout averaged over planning seeds 7, 19 and 37.

| Configuration | Rollout ADP | Bounded inventory | Service-first | Rollout minus inventory, paired 95% interval |
|---|---:|---:|---:|---|
| 9 zones, 5 vehicles, 18 one-hour periods | 72.21 | 75.83 | 62.81 | -3.62 [-6.85, -0.39] |
| 36 zones, 12 vehicles, 8 half-hour periods | 104.64 | 105.19 | 93.92 | -0.55 [-2.13, 1.02] |

The additional intervals use the paired normal approximation across 32 scenarios. Rollout improves on service-first in both cases, loses to inventory in the longer constrained case, and is approximately tied with inventory in the shorter case. Compare within a row: the horizons and operating models differ.

Each changed model is supplied correctly to the planner. These are **known-model portability checks**, not tests of an unknown distribution shift, calibrated real-city data, waiting queues or multi-period trips. “General” means configurable within the existing simulator and computation limits; it does not mean universally optimal.

## Verification and saved results

**These fixed-price studies passed 274 automated tests.** The primary comparison contains 5,376 final policy episodes, the supplementary original-inventory check adds 256, and portability adds 384: **6,016 in total, with zero invalid actions**. Audits reconcile profit ledgers, check source and checkpoint hashes, reconstruct planning random streams and query budgets, verify candidate choices, recompute uncertainty intervals and replay selected complete episodes. Correctness does not imply superior profit.

Local evidence is stored under `outputs/general_adp_study_20261025/` and `outputs/adp_transfer_20261025/`. The supplementary inventory records are in the first directory’s `reference_inventory/`. Failed fitted-regression development trials remain archived under `outputs/general_adp_development_20261025/`; that implementation is not part of the public controller package. Generated outputs are ignored by Git; this report is included in the repository.

```bash
python experiments/run_general_adp_study.py --phase show \
  --study-dir outputs/general_adp_study_20261025
python experiments/run_adp_transfer.py --show \
  --output-dir outputs/adp_transfer_20261025
```

The first command displays the primary bounded-search scorecard; the supplementary exhaustive reference is reported above. [TRAINING.md](TRAINING.md) explains standalone runs and complete reproduction.

## Earlier studies — optional context

The remaining sections describe the original implementations and earlier scenario banks. Their reported values remain historical results, not additional observations in the new tables.

## Scaling to 25 synthetic zones

The larger scenario uses **25 zones, 10 mobile vehicles, and five fixed stations**. The zones form a 5-by-5 grid across a 10 km by 10 km area, with 2.5 km between adjacent zone centers. Each vehicle starts with 55 kWh in an 80-kWh battery and retains a 5-kWh reserve. Each station has separate 100-kWh hourly capacities for customer service and fleet exchange. The horizon remains twelve one-hour service periods, with `b=3`.

Demand varies across both space and time. Two spatial concentrations create busier areas. Residential-zone demand peaks earlier and later in the day; commercial-zone demand peaks around midday. Counts are sampled independently from the prescribed Poisson rates, with **1,032.5 expected potential requests per day** before customer choice. Request energies remain uniform between 5 and 15 kWh. The full scenario is [synthetic_25_zones.json](../configs/synthetic_25_zones.json).

This is a synthetic scaling test. Zones represent aggregated service areas, travel uses static Manhattan distances at 20 km/h, and no road network or demand records from a real city were used. Training and testing share the same prescribed distribution but use independent random draws. Increasing the network size does not by itself test distribution shift or validate real-world forecasts.

### What changes when the fleet gets larger

PPO already accepts larger observation and action dimensions, but it needs fresh networks: the four-zone checkpoints do not fit this scenario. The 25-zone model has 532,697 parameters across its four networks. The inventory rule's original exhaustive joint search becomes infeasible as the number of vehicles grows. Its replacement starts with two feasible fleet plans and improves one vehicle's action at a time, for up to two sweeps. Every candidate is scored as a complete fleet plan, preserving shared charging capacity and competition for requests. This bounded search can miss a better joint plan.

The original table-based approximate-DP implementation stores a value table for exactly **two vehicles**. It is therefore **not evaluated with this ten-vehicle fleet**. Increasing the number of zones in a JSON file does not remove that limitation. A larger-fleet ADP controller needs a different value representation and a scalable action optimizer; its two-vehicle profit ranking cannot be transferred to this experiment.

### Training and comparison protocol

Each PPO model, trained with seed **7, 19, or 37**, receives 256 demonstration days and 100 supervised passes, followed by **500 updates of 16 fresh days each**. This uses the existing ordinary imitation warm start and PPO objective. The teacher retains price 0.60 and inventory coefficients 0.30 early and 0.15 late. Its coordinate search follows PPO's action mask and maximum feasible exchange convention. The previous four-zone repair used 512 demonstration days, so these are separate experiments rather than an otherwise identical test of network size alone.

The final update is evaluated even if an earlier checkpoint has a higher monitoring score. Initialization-only checkpoints are evaluated separately. Three models are executed independently and averaged for reporting; their actions are not combined into an ensemble.

Two benchmark families are retuned on **32 validation days**, separate from training, monitoring, and testing:

| Rule | Validation search | Selected setting |
|---|---|---|
| Bounded inventory rule | 16 early/late inventory schedules; fixed price 0.60 | 0.40 per stored kWh early, 0.10 late; zero in the final period |
| Service-first | Seven prices and three recharge thresholds | Price 0.80; recharge below 20% of usable battery above reserve |

The inventory search uses two coordinate sweeps and maximum exchange amounts. It can use all physically admissible simulator actions; its demonstration-teacher version additionally obeys PPO's more restrictive target mask. Both rules retain their selected settings for the final tests. The inventory family's price is fixed in this search, so it is not a joint optimization over all pricing and dispatch rules.

### Results on 256 independent days

**PPO learns substantially after initialization, but all three seeds still lose to both rules.** All methods below face the same 256 previously unused scenario seeds. Profits are discounted dollars per day; energy quantities are fleet totals in kWh per day. The two learned rows average three separately executed models.

| Method | Profit | Mobile energy delivered | Recharge bought | Unmet mobile energy | End battery above reserve |
|---|---:|---:|---:|---:|---:|
| Initialization alone | 625.56 | 509.64 | 268.43 | 588.89 | 44.18 |
| Initialization + 500 PPO updates | **728.36** | 475.04 | 196.08 | 396.95 | 72.64 |
| Bounded inventory rule | 863.49 | 760.77 | 475.52 | 306.59 | 20.81 |
| Service-first | **893.04** | 387.40 | 210.17 | 3.73 | 188.66 |

The battery column subtracts the mandatory fleet reserve of 50 kWh. It is not a direct measure of avoidable waste: some remaining energy may be poorly located or unusable for a complete remaining request.

PPO gains **102.80 per day (16.4%)** over initialization, with a paired 95% interval of **[98.92, 106.67]**. Its deficit to the inventory rule is **135.12 per day (15.6%)**, with PPO-minus-rule interval **[−138.99, −131.40]**. The deficit to service-first is **164.68 per day (18.4%)**, with interval **[−169.67, −159.45]**.

| Training seed | Initialization profit | Final PPO profit | PPO minus inventory rule, with paired 95% interval |
|---|---:|---:|---|
| 7 | 626.34 | 733.30 | −130.19 [−134.47, −126.01] |
| 19 | 621.53 | 714.60 | −148.89 [−154.40, −143.59] |
| 37 | 628.81 | 737.19 | −126.29 [−132.02, −120.68] |

Every seed has a positive interval for its gain over initialization and a negative interval against each rule. These are 5,000-resample paired scenario bootstrap intervals, conditional on the fixed models. They do not include uncertainty from arbitrary retraining or a changed demand distribution. Monitoring profit still fluctuates late in training; 500 updates do not establish convergence.

### What the larger case reveals

PPO's remaining problem is not simply failure to buy energy. Relative to the inventory rule, it makes almost as many service commands per day (**63.45 versus 66.11**) but delivers less energy per command (**7.49 versus 11.51 kWh**). It recharges less, ends with 51.83 kWh more battery, and leaves more mobile demand unmet even though less demand chooses its mobile service. The discounted accounting attributes the deficit mainly to **160.42 less mobile revenue** and **30.70 more unmet-demand cost**, partly offset by procurement savings and higher fixed-station revenue. These identify operational differences; they do not isolate which individual routing or charging decisions caused them.

The gain over initialization comes with **less travel and lower unmet-demand penalties**, rather than more mobile energy delivery. Travel energy falls from 170.46 to 112.48 kWh per day. Customer choices also change: mobile-selected energy falls from 1,098.54 to 871.99 kWh. Thus the reduction in unmet demand cannot all be interpreted as serving the same demand more effectively.

Service-first is a useful counterexample to treating unused battery as the main failure. Its price of 0.80 attracts only 391.13 kWh of mobile-selected demand per day, compared with PPO's 871.99. It fulfills about 99% of selected mobile requests, serves more energy at fixed stations, and earns the most despite leaving the most battery unused. Profit includes both mobile and fixed-station operation. Forcing the fleet to empty its batteries would not follow that objective.

Fleet coordination and training remain plausible bottlenecks. Ten separately sampled vehicle action factors can compete for requests, and imitation errors can lead to states absent from the demonstration data. The joint KL stopping criterion also limits the number of optimization steps: seeds 7, 19, and 37 complete **2,046, 2,307, and 2,377** steps out of a maximum 3,000, while still collecting all 8,000 on-policy days each. These observations motivate future controlled experiments; this study does not prove a single cause or a universal limitation of PPO.

All **2,048 final policy episodes** have zero invalid actions. The audit checks episode profit ledgers, seed separation, source snapshots, model budgets, checkpoint replay, and paired intervals; **252 automated tests pass**. Mean dispatch callback time in the final evaluation is **0.33 ms for PPO, 31.50 ms for bounded inventory, and 0.17 ms for service-first** on this machine. These are observed timings at this one network size, not general complexity guarantees.

The completed study is stored locally in `outputs/scale_25_study_20261024/`. `analysis.json` contains the scorecard and intervals, `audit.json` records checks, `training/` holds all checkpoints, and `test/` contains episode records. To display it without retraining:

```bash
python experiments/run_scale_study.py --phase show --study-dir outputs/scale_25_study_20261024
```

The [training guide](TRAINING.md#a-larger-network-25-synthetic-zones) explains how to reproduce the study in a new directory. Results under `outputs/` are ignored by Git; the tables on this page are included in the repository.

## What every controller is trying to achieve

Each physical period has two decisions. The operator first quotes mobile charging prices. Customers then choose mobile charging, a fixed station, or the outside option. After seeing those choices, the operator dispatches the mobile charging vehicles (MCVs).

All reported scores use the same original business profit: mobile and fixed-station revenue, plus electricity export revenue, minus procurement, travel, operating, and unmet-demand costs. The primary score is **discounted profit per episode**, in dollars. Period 1 has weight 1, period 2 has weight 0.99, and so on over the 12-period horizon. The recorded scenario gives no terminal payment for leftover battery energy. A higher score means a larger expected discounted operating profit under this accounting; it does not necessarily mean more customers served.

The main scenario is `configs/utility_b3.json`: two MCVs, four locations, and `b=3`. Here `b` is the charging-utility intercept: it makes charging options more attractive relative to leaving. All methods in the comparison use the same `b=3` customer-choice model and physical constraints.

## How approximate DP and PPO make decisions

A **policy** is a rule for choosing an action from the information currently available. A **value** estimates the profit that can still be earned from a state. The controllers differ in how they use these two ideas.

### Original table-based ADP: learn future value, then compare current actions

Fitted-value dispatch fixes the mobile price at **\$0.60/kWh** and learns how valuable each combination of time, vehicle locations, and battery levels is. It builds a value table backward from the end of the episode. Battery levels lie on a grid during training; interpolation supplies values between grid points.

When a dispatch decision arrives, the controller evaluates feasible joint actions for both vehicles using the known one-period model. It selects the action with the largest sum of:

1. the business profit earned in the current period; and
2. the discounted learned value of the resulting fleet state.

This is **approximate dynamic programming (ADP)** because the future value is estimated from sampled demand and a battery grid. Current action search is exact over its finite candidate grid, but that does not make the full-horizon controller globally optimal. The implementation supports exactly two MCVs. It learns dispatch continuation value; it does not learn prices or a neural-network actor.

### PPO: learn which actions to choose from experience

PPO runs complete episodes, samples actions from its current policy, and learns from the resulting rewards. Its actors produce action probabilities; its critics estimate future return and help determine whether a sampled action did better or worse than expected. Its clipped learning objective reduces the incentive for large changes in sampled-action probabilities; it does not impose a hard bound on those changes.

Both PPO variants learn **mobile pricing and dispatch**. At execution time, their actors choose actions directly from observations. They use feasibility masks and the simulator's physical limits, but they do not enumerate all alternative joint actions and calculate each one's current profit before choosing.

For example, consider a vehicle with a low battery early in the day. Serving now can earn immediate revenue; recharging costs money but may enable more service later. ADP explicitly compares the modeled current profit plus learned future value for these options. An inventory rule makes a similar calculation using a simple assigned value per remaining kWh. PPO must learn a probability of recharging from experiences in which this immediate cost eventually paid off. This example explains the approaches; it is not a claim that any particular saved policy makes the right choice in every low-battery state.

| Question | Fitted-value dispatch / approximate DP | Joint PPO | Earlier SA-PPO/MAPPO |
|---|---|---|---|
| What is learned? | A continuation-value table | Pricing actor, dispatch actor, and two critics | Pricing actor, shared vehicle dispatch actor, and two critics |
| How is pricing chosen? | Fixed at \$0.60/kWh where mobile service is available | Learned categorical price at each available location | Learned categorical price at each available location |
| How is dispatch chosen? | Search joint fleet actions using modeled current profit and learned continuation value | Sample vehicle action factors from a global post-choice observation | Sample a shared actor for each vehicle from its local observation, which also includes peer and demand information |
| What can training query? | Synthetic demand at arbitrary feasible fleet-grid states, plus every candidate's one-period outcome | Outcomes of actions sampled during episode rollouts | Outcomes of actions sampled during episode rollouts |
| How much recharge/export can an action request? | Half or all of its individually feasible amount | All of its individually feasible amount | A learned continuous fraction of a fixed nominal energy scale; physical limits then reduce the request |
| What happens at a busy station? | The environment applies shared station capacity | The environment applies shared station capacity | The environment applies shared station capacity |

The table describes the actual implementations and recorded settings. “All of its individually feasible amount” is a request before shared capacity allocation; it does not guarantee that the vehicle receives that amount. Earlier PPO instead multiplies its sampled fraction by `max(mobile_charge_kw, mobile_discharge_kw) * period_hours`, which is 30 kWh here, before the simulator applies travel, battery, power, and station limits. None of the controllers receives future realized demand or future random price shocks.

## Differences between the two PPO versions

Both retain the pricing-then-dispatch structure, separate stage critics, and a shared business objective. Joint PPO changes more than one detail:

| Aspect | Earlier PPO used in this comparison | Joint PPO used in this comparison |
|---|---|---|
| Dispatch update | Clip each vehicle's likelihood ratio separately, then average the losses | Multiply the vehicle likelihood ratios and clip the resulting fleet ratio once |
| Information before pricing | Encodes realized current requests before their choices are resolved | Uses fleet state, arrival rates, time, and current prices; hides realized requests from both pricing actor and critic |
| Dispatch observation | Shared actor receives each vehicle's local observation with included peer information | Actor receives the complete current post-choice state and outputs a factor for each vehicle |
| Dispatch actions | Five simulator modes, a target, and a continuous recharge/export fraction | Four operation labels and a target; idle means waiting or repositioning, and recharge/export requests use the full feasible amount |
| Advantage trace | Trace factor 1 within a period; configured inter-period lambda is 1.0 | Lambda 0.95 applies at both decision stages |
| Training signals | Normalized advantages, period-weighted actor losses, and inventory-potential reward shaping | Raw advantages, uniform period weighting in the surrogate loss, and original rewards |
| Exploration bonus | Pricing bonus plus a dispatch bonus that decreases during training | Dispatch bonus only, summed over vehicles |

An **advantage** estimates how much better a sampled action performed than its critic expected. The trace parameter, lambda, controls how advantage estimates combine successive prediction errors. Both implementations discount future business rewards once per physical period. The extra lambda between stages in Joint PPO is not an extra business-profit discount.

Joint PPO also uses its own SiLU networks and four independent optimizers. Its dispatch critic uses dispatch advantage plus the old dispatch value, as specified in Equation (11) of the [algorithm specification](MDP_RL_2.pdf). The optional `pricing_advantage` target is an experimental alternative and was not used in the reported full runs.

These simultaneous changes mean the measured improvement belongs to **Joint PPO and its recorded settings**. The experiment does not isolate joint clipping, SiLU, lambda, or any other single change as the cause. A controlled ablation would change one factor while holding the others fixed.

## The six methods in the main comparison

The three learned controllers are compared with three useful rules. The rules were tuned on **128 separate validation scenarios**, with 82 candidate settings in total. One setting per family was frozen before testing.

| Benchmark | Decision principle | Setting selected on validation |
|---|---|---|
| Service-first | Serve current requests; replenish when usable inventory is low | Price \$0.80/kWh; recharge threshold 20% of usable battery above reserve |
| Constant inventory value | Search joint actions using current profit plus a fixed value for stored energy | Price \$0.60/kWh; inventory coefficient \$0.25 per stored kWh |
| Time-dependent inventory value | Use a higher inventory coefficient early and a lower one late | Price \$0.60/kWh; coefficient \$0.30 early, \$0.15 late; switch halfway through the horizon |

The artificial inventory coefficient is zero in the final period, when no future service remains. The two inventory rules use the same joint action grid and one-period model as fitted-value dispatch. They are strong planning baselines, rather than arbitrary or random behavior. Validation selected the **time-dependent inventory rule** as the primary heuristic baseline; the high-demand result did not change that selection afterward.

## Results on the same independent scenarios

The **2026-10-21 joint study** is the original six-method scorecard. The newer recharge study is reported separately below. It reevaluates all methods on the same fresh scenarios. Earlier reports used different test seeds and remain historical records; their numbers should not be inserted into this table.

Each learned-method row averages three separately executed models, trained with seeds **7, 19, and 37**. It is not an ensemble that chooses or combines their actions. Both PPO rows use the final **500-update** checkpoints and stochastic execution, fixed before testing. Earlier PPO and fitted-value checkpoints were reused as frozen controls; the Joint PPO models were newly trained.

| Method | Base: 512 scenarios | High demand: 256 scenarios | Low inventory: 256 scenarios |
|---|---:|---:|---:|
| Approximate DP / fitted-value dispatch | **137.76** | 140.56 | **126.67** |
| Joint PPO | 123.19 | 131.93 | 106.90 |
| Earlier SA-PPO/MAPPO | 117.42 | 127.09 | 105.70 |
| Service-first | 123.25 | **144.48** | 113.56 |
| Constant inventory value | 129.02 | 129.77 | 115.90 |
| Time-dependent inventory value | 133.41 | 133.63 | 121.75 |

Entries are mean discounted business profit per episode, in dollars. High demand increases arrival means by 50%. Low inventory starts each vehicle with 35 kWh instead of 55 kWh. The learned parameters and rule settings remain frozen in both stress cases. Each column uses its own scenario bank, so compare methods within a column.

The paired base-test comparisons show:

- **Joint PPO improves on earlier PPO.** Its mean gain is **\$5.77 per episode, or 4.91%**, with a paired 95% interval of **[\$4.95, \$6.55]**. All three matched training seeds improve.
- **Approximate DP outperforms both PPO variants.** It earns **\$14.57 more than Joint PPO**, with a paired 95% interval of **[\$13.54, \$15.64]** for that advantage. Every fitted-value seed exceeds its matched Joint PPO seed.
- **The time-dependent inventory rule outperforms Joint PPO.** Joint PPO earns **\$10.22 less** than the time-dependent inventory rule, with a paired interval of **[-\$11.35, -\$9.12]** for PPO minus the rule. Its base mean is approximately tied with service-first.

### Variation across training runs

| Training seed | Earlier PPO | Joint PPO | Approximate DP | Joint PPO minus earlier PPO, with paired 95% interval |
|---|---:|---:|---:|---|
| 7 | 115.55 | 124.82 | 137.95 | +9.27 [8.05, 10.56] |
| 19 | 116.44 | 121.40 | 137.57 | +4.96 [3.69, 6.19] |
| 37 | 120.28 | 123.36 | 137.76 | +3.07 [1.71, 4.43] |

Joint PPO improves on earlier PPO in all three base runs. Under high demand, Joint PPO also improves on earlier PPO for all three seeds. Under low inventory, however, seed 19 declines by **\$0.86**, and its difference interval includes zero. The positive average low-inventory gain of \$1.20 is therefore not evidence of improvement for every training seed. Fitted-value dispatch leads the base and low-inventory tables, but service-first leads the high-demand table.

A **paired interval** is calculated by subtracting two methods' profits on each shared test scenario, then resampling those scenario differences. The reported intervals use 5,000 bootstrap resamples. They describe scenario/action sampling uncertainty for these fixed trained models. They do not include arbitrary retraining uncertainty, guarantee a win on every episode, or establish performance in a different city or fleet size. Three training seeds provide useful but limited evidence about training variability.

## Operating behavior behind the profit gap

The operating records show a substantial difference in battery replenishment. The following are fleet totals per base episode, averaged over the actual runs:

| Method | Mobile energy delivered (kWh) | Grid energy bought for MCV recharge (kWh) | Battery energy left at the end (kWh) |
|---|---:|---:|---:|
| Earlier PPO | 63.27 | 0.21 | 31.07 |
| Joint PPO | 69.57 | 0.17 | 28.67 |
| Time-dependent inventory rule | 140.34 | 81.17 | 16.89 |
| Approximate DP | 159.06 | 108.78 | 17.58 |

Both PPO variants recharge very little. Joint PPO uses the initial battery stock more productively and travels less: travel energy falls from 11.63 to 7.76 kWh. Its accounting gain over earlier PPO includes \$2.60 in lower discounted travel cost, \$2.42 in additional mobile revenue, and \$1.14 in lower unmet-demand cost, partly offset by other revenue and cost changes. It improves current operation while leaving the replenishment problem largely unresolved.

Approximate DP and the inventory rule repeatedly buy energy and use it for later service. Their higher mobile delivery is not explained by simply ending with larger batteries. ADP's learned continuation value explicitly considers how useful a recharged fleet state could be later; the current PPO runs have not learned comparable replenishment behavior.

This is an explanation supported by observed operating records, not a causal proof that any additional recharge would improve PPO. Prices, action sets, information, and planning capabilities also differ. The [recharge investigation below](#why-ppo-learned-almost-never-to-recharge) subsequently measured feasible recharge probabilities and intervened in states where PPO waits. It uses separate development scenarios and reserves a new bank for the training repair.

## Scope and limitations of the comparison

The comparison is fair for the practical question, **“Which of these implemented controllers earns more under this simulator and test protocol?”** All controllers face common scenario seeds, identical physical and customer-choice rules, and the same profit accounting. The main test executes **12,288 policy episodes** over 1,024 distinct scenarios, with **zero invalid actions**. Training, monitoring, rule selection, and final testing use separate scenario banks.

It is not a comparison with equal information or equal computation:

| Per trained model | Approximate DP | Earlier PPO | Joint PPO |
|---|---|---|---|
| Main learning budget | 124,416 fleet-state/draw queries | 8,000 episodes; 96,000 physical periods | 8,000 episodes; 96,000 physical periods |
| Additional computation | Evaluate many candidate joint actions for every queried state | Reuse rollout batches in PPO updates | Reuse rollout batches for six epochs per update |
| Saved learned representation | 16,848 table values, including the terminal layer | Four neural networks | Four neural networks |
| Execution work | Model-based joint action search plus value interpolation | Actor forward passes and sampling | Actor forward passes and sampling |

ADP reuses 96 distinct period/draw seeds across the fleet grid in each training run; its 124,416 queries are not that many independent demand realizations or episodes. Across its three previously trained models, it evaluated about **53.7 million candidate outcomes**. PPO receives rewards for the actions it actually samples, while ADP and the coordinated rules can evaluate alternative current actions with the known model. No matched deployment-latency experiment was performed.

ADP also solves a narrower pricing problem by fixing the price. Joint PPO follows a more restrictive information convention before pricing than earlier PPO, and their energy action spaces differ. Consequently, the results establish a ranking of these implementations under the recorded conditions, not a universal ranking of approximate DP against PPO.

For the current two-vehicle base scenario, fitted-value dispatch is the strongest learned reference. For research on joint learned pricing and dispatch, Joint PPO is the stronger of the two tested PPO implementations, with the remaining inventory gap made explicit. Keep the earlier PPO as a reproducible control and the three rules as interpretable benchmarks. Scaling ADP to more vehicles or claiming a general PPO advantage requires a separate implementation and experiment.

## View the evidence without training again

The tables and figure on this page are included in the repository. If you also have the saved study in `outputs/paper_ppo_study/`, run these commands from the project root to display its scorecard and figure:

```bash
python3 experiments/summarize_paper_ppo_study.py --show
open docs/assets/algorithm_comparison.png
```

The `--show` command reads the saved analysis. It does not train, rerun evaluation, or modify results. The `open` command opens the existing image on macOS.

![Current base comparison, PPO gains, stress results, and Joint PPO monitoring curves](assets/algorithm_comparison.png)

The monitoring curves use separate, repeatedly observed scenarios. A late plateau indicates that measured learning has stabilized over that range; it does not prove global convergence or replace the independent tests above.

The files below are local records under `outputs/`, which is not included in Git. Use the saved study files or [reproduce the experiment](TRAINING.md#reproduce-the-six-method-study) to inspect them.

For exact numbers and reproducibility, use the saved analysis (`outputs/paper_ppo_study/analysis.json`), frozen study protocol (`outputs/paper_ppo_study/protocol.json`), validation-selected rule settings (`outputs/paper_ppo_study/rule_selection/selection.json`), and test results (`outputs/paper_ppo_study/benchmarks/`). [TRAINING.md](TRAINING.md) explains how to run each controller, evaluate a checkpoint, and reproduce a fresh comparison.

## Why PPO learned almost never to recharge

The connected-day experiment has been removed from the active environment. An episode again contains the original 12 service periods, with a fresh reset afterward. Its saved experiment outputs remain available as historical evidence.

The recharge investigation examines the original **Joint PPO**, not the connected-day controller. The main finding is a coordination problem: **paying for energy is useful only if later prices attract customers and later dispatch uses that energy to serve them**. A random policy often pays the charging bill and then wastes the opportunity. PPO initially learns to avoid that cost. Once recharge becomes extremely unlikely, it rarely experiences the successful sequences needed to reverse that decision.

This explanation is supported by three checks:

1. **Recharge is available.** Across 96 fresh development episodes for each of the three saved PPO models, recharge is physically feasible in 98.64% of vehicle decisions. Its average learned probability nevertheless falls from 23.50% at initialization to 0.0329% after training. Even below 20 kWh, the final probability averages only 0.0861% when recharge is feasible. No action-mask failures were found.
2. **The useful experience disappears early.** In seed 7, validation recharge falls from 122.61 kWh initially to 62.23, 29.55, and 15.15 at updates 25, 50, and 75. During those early updates the dispatch critics explain almost none of the variation in returns. They fit the policy's eventual behavior much better later, after charging has largely disappeared.
3. **The payoff depends on subsequent decisions.** A paired intervention replaced one WAIT with feasible recharge in 30 early/midday low-battery states, with 12 matched future simulations per state. The immediate extra cost averaged \$6.80. The total change in remaining discounted profit was:

| Policy used after that one intervention | Charge minus wait | 95% interval across sampled states |
|---|---:|---:|
| Initial random PPO | −\$7.13 | [−\$10.19, −\$4.07] |
| Trained PPO | +\$0.28 | [−\$1.18, +\$1.73] |
| Approximate DP | +\$2.42 | [\$1.33, \$3.50] |
| Fixed \$0.60 pricing with PPO dispatch | +\$2.20 | [\$0.31, \$4.09] |

These are approximate normal intervals calculated from the 30 state-level mean gains. The same 12 future random seeds are reused across states, so these intervals are conditional on that fixed future-simulation bank; they do not capture all future-sampling uncertainty. These deliberately chosen WAIT states diagnose the mechanism; they do not estimate the benefit of charging in every state. Charging late in a day can still be a bad decision because leftover energy has no terminal value.

This is **not evidence that the critic always undervalues batteries**. The final dispatch critics predict returns under their own policies fairly well: explained variance is about 0.91 on fresh rollouts. In a broader intervention that sometimes replaces a service action, the critics actually overestimate recharge returns. A critic estimates what the current policy will do next, not what a better controller could achieve with the same battery.

### A targeted training change

An optional **inventory-rule warm start** gives both actors coherent examples of pricing, recharge, and service before ordinary PPO begins. The teacher uses the existing time-dependent inventory rule, restricted to exactly the action masks and maximum exchange quantities available to Joint PPO. It quotes \$0.60/kWh, values inventory at \$0.30 early and \$0.15 late, and gives it zero artificial value in the terminal period.

Supervised initialization fits the actors to those actions and the critics to the teacher's sampled returns. These transitions are never passed through PPO's on-policy probability-ratio objective. Subsequent PPO updates use fresh rollouts generated by the neural policy. At execution time the policy uses its networks; it does not call the teacher or search candidate actions.

The experiment also uses return scaling by 20, normalized actor advantages, lambda 1, and a smaller actor learning rate. Scaling changes the units of the critic targets; reported business profits remain in dollars. Lambda 1 recovers complete-episode returns and avoids the extra trace decay across the pricing and dispatch substages. There is no extra payment for recharging and no change to the simulator's profit objective. These are a combined training repair, not a controlled proof that one hyperparameter caused the improvement.

The original settings remain the defaults. The warm start is explicitly model-informed: the teacher evaluates possible current actions using the simulator's known transition model. Its demonstration budget and computation must be disclosed when comparing it with PPO trained from scratch. A separate cloning-only checkpoint shows how much is gained by the initialization itself and how much is gained or lost in later PPO training.

### Independent results after the recharge repair

The 2026-10-24 study trains seeds **7, 19, and 37** for **500 PPO updates each**, following 512 teacher episodes and 100 supervised passes per model. Each PPO update uses 16 fresh episodes. Both PPO versions and the cloning-only control use stochastic execution. Every controller, including the frozen original ADP models, faces the same **512 previously unused test scenarios**, each covering one 12-hour service day. The inventory rule retains its previously selected settings. No test score chooses a checkpoint, seed, or setting.

| Method | Discounted profit per day | Recharge bought (kWh/day) | Mobile energy delivered (kWh/day) |
|---|---:|---:|---:|
| Original Joint PPO | 123.86 | 0.21 | 69.25 |
| Inventory initialization only, before PPO | 127.17 | 78.44 | 132.43 |
| Inventory initialization + 500 PPO updates | **129.20** | **90.27** | **141.72** |
| Time-dependent inventory rule | 134.70 | 80.79 | 140.26 |
| Approximate DP | 138.58 | 109.86 | 159.77 |

The learned rows average three separately executed models, not an ensemble. These fresh-test numbers replace neither the scenarios nor the accounting in the earlier table; compare methods within this new table.

**The recharge collapse is removed in these runs, and profit improves consistently over original Joint PPO.** The mean gain is **\$5.33/day, or 4.31%**, with a paired 95% bootstrap interval of **[\$4.22, \$6.42]**. All three matched training seeds improve with positive interval lower bounds:

| Seed | Original PPO | Warm-started PPO | Gain, with paired 95% interval |
|---|---:|---:|---|
| 7 | 125.09 | 128.37 | +3.28 [1.90, 4.64] |
| 19 | 122.27 | 129.93 | +7.67 [6.24, 9.06] |
| 37 | 124.24 | 129.29 | +5.05 [3.68, 6.37] |

PPO fine-tuning also improves on initialization alone: **+\$2.03/day [\$1.36, \$2.71]** on average, with positive paired interval lower bounds for every seed. Thus part of the improvement comes from imitation and part comes from subsequent on-policy learning. The experiment does not attribute the combined repair to warm start, scaling, or lambda individually.

**The inventory rule still wins.** The remaining mean deficit is **\$5.50/day**, with PPO-minus-rule interval **[−\$6.25, −\$4.76]**. The repair closes about half of the original \$10.84 gap to the rule, but does not establish a PPO advantage over planning. ADP still earns \$9.38/day more than the repaired PPO.

The residual behavior differs from the original failure. Repaired PPO delivers roughly as much mobile energy as the rule, yet buys 9.48 kWh more recharge and ends with 25.02 kWh in the batteries versus 15.93 for the rule. Its discounted accounting shows \$2.05 more unmet-demand cost, \$1.61 more procurement cost, and \$1.55 less fixed-station revenue, partly offset by higher mobile revenue. Timing, routing, and customer allocation still need improvement. These accounting differences identify where profit is lost; they do not prove that blindly reducing recharge would recover it.

All **6,656 policy test episodes** contain zero invalid actions. Checkpoint reloads reproduce evaluation exactly, and **231 automated tests** pass. The paired intervals use 5,000 scenario bootstrap resamples and condition on these fixed models; three training seeds do not establish performance under arbitrary retraining or changed demand distributions. Five hundred updates demonstrate this improvement, not global convergence.

The code and results are retained in `outputs/recharge_study_20261024/`: `analysis.json` contains the scorecard and per-seed intervals; `protocol.json` fixes the experiment; `training/` holds the initialization and final checkpoints; `test/` holds every episode; `development_summary.json` and `development/` preserve successful and unsuccessful diagnostic experiments. `audit.json` records the completed checks. These generated artifacts remain outside Git. [TRAINING.md](TRAINING.md#joint-ppo-with-an-inventory-rule-warm-start) shows how to display or reproduce the results.
