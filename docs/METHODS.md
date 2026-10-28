# How the controllers decide

A **controller** makes the operator's two decisions every hour: the mobile price at each zone, then one action for every vehicle. This page explains each controller in the project, what it is allowed to see, and how it chooses. Read [MODEL.md](MODEL.md) first if you haven't; it defines the physical system.

| Controller | Pricing | Dispatch | Preparation before operating | Fleet sizes |
|---|---|---|---|---|
| Service-first rule | Fixed | Serve now, recharge when low | Tune two settings on validation days | Any |
| Inventory rule | Fixed or learned | Current profit plus a dollar value per stored kWh | Tune price and battery values | Exhaustive search: 2 vehicles. Bounded search: any |
| **Rollout ADP** | Fixed or learned | Simulate a few futures for each candidate plan | Supply or fit a market model; optionally fit prices | Any |
| Table ADP | Fixed | Current profit plus a learned value table | One backward pass over a battery grid | Exactly 2 |
| Joint PPO | Learned | Learned neural network | 500 updates of policy-gradient training | Any (retrain per scenario) |
| Earlier PPO | Learned | Learned shared vehicle network | Same as Joint PPO | 2 in the saved models |

## What each decision may see

The two decisions happen at different points in the hour, so they see different information.

| | Pricing decision | Dispatch decision |
|---|---|---|
| When | Before customers choose | After customers choose |
| Always visible | Time, vehicle locations and batteries, expected arrival rates, station and grid prices | Everything visible to pricing, plus posted prices |
| Current requests | **Hidden** for rollout ADP, learned pricing and Joint PPO. Earlier PPO sees them | Visible: who arrived, how much energy each needs, and what each chose |

Hiding current requests from pricing is a deliberate modeling choice. The operator knows that a zone usually gets four requests at noon, not that seven arrived today. The simulator draws the requests before pricing, so the code projects the raw observation down to the permitted fields (`paper_pricing_observation()` in Joint PPO, `pricing_observation()` in learned pricing).

No controller sees future arrivals or future price shocks.

**Where do the forecasts come from?** The earlier studies give controllers the simulator's demand and customer-choice parameters. The [historical-model study](studies/12_historical_model.md) instead estimates them from past quote records. All three controllers then receive forecasts from that fitted model, even when the private test environment changes. Current requests and choices still become visible at dispatch. This separates a forecast from knowledge of the actual future.

## The rules

The rules are benchmarks: simple enough to explain, strong enough to be hard to beat.

**Service-first.** Each vehicle, in index order, goes to the zone where serving waiting mobile customers earns the most right now. That score is revenue plus the unmet-demand penalty avoided, minus travel cost and the cost of replacing the energy used. A vehicle with nothing profitable to serve recharges if its usable battery is below a threshold (20% in the main studies), and otherwise waits. It never recharges in the last hour.

**Inventory rule.** Choose the joint fleet action that maximizes

> profit this hour + (dollar value per kWh) × (change in stored battery energy).

The dollar value per kWh, also called a shadow price, stands in for "how useful will this energy be later". The **time-dependent** version uses a higher value early in the day and a lower one late, for example $0.30 then $0.15, and zero in the final hour. The **constant** version uses one value all day.

Two search versions exist:

- **Exhaustive** (`benchmarks/policies.py`): tries every combination of vehicle actions, with recharge and export amounts at half or all of what each vehicle can physically take. This only works for two vehicles.
- **Bounded** (`benchmarks/scalable.py`): starts from two complete plans (everyone waits; service-first) and improves one vehicle's action at a time, for up to two passes. Every trial is scored as a complete fleet plan, so shared station capacity and competition for customers are respected. It scales to large fleets but can miss a better plan that needs several vehicles to change together.

## Rollout ADP

**Idea.** Before committing to an action, imagine the rest of the day a few times and see how each choice plays out. ADP stands for approximate dynamic programming: compare actions by *profit now + estimated profit later*.

At each dispatch decision (code: `rl/rollout_adp.py`):

1. **Propose a few complete fleet plans.** By default, up to five: service-first with recharge thresholds 0.2, 0.4 and 0.6; the bounded inventory rule ($0.30 early, $0.15 late); and the same search with zero battery value, which maximizes current profit only. Duplicates are removed.
2. **Score each plan's current hour exactly**, using the known physics: profit, next locations, next batteries.
3. **Simulate the rest of the day eight times** from each resulting fleet state. In these imagined futures the fleet follows service-first (the **base policy**). All candidates face the same eight sampled futures (**common random numbers**), so their differences reflect the plans rather than luck.
4. **Pick the plan with the highest** current profit + 0.99 × average future profit.
5. **Execute only this hour's actions**, observe what really happens, and plan again next hour.

In symbols, with base policy $`\mu`$, $`K`$ sampled futures, and next fleet state $`x_{t+1}=f(\widetilde S_t,\mathbf a)`$:

```math
\widehat Q_t(\widetilde S_t,\mathbf a)
= R_t(\widetilde S_t,\mathbf a)
+ \gamma\,\frac1K\sum_{k=1}^{K}\sum_{j=t+1}^{T}\gamma^{\,j-t-1}R_j^{(k,\mu)}.
```

**Why it supports different fleet sizes.** It never stores a value for every possible fleet state; it only simulates from the few states it is comparing. The planner itself has no value network to retrain. A learned demand model or price policy may still need fitting for a changed operating environment, and computation grows with the problem.

**What limits it.** The future is valued as if the fleet followed service-first, not the best possible policy. Only a handful of plans are compared. Eight samples leave noise. Planning is also slower than network inference or the inventory rule: roughly half a second per dispatch in the 25-zone city. [Study 12](studies/12_historical_model.md) reports measured decision times, including its information-boundary checks.

Deep dive: [methods/ROLLOUT_ADP.md](methods/ROLLOUT_ADP.md).

## Learned pricing

**Idea.** Replace the fixed price with a small, interpretable rule that reacts to the situation: raise prices when the fleet is short of energy or far away; lower them when it has spare capacity near demand.

**The rule** (code: `rl/adaptive_pricing.py`). For each zone, compute 12 features, each scaled to [−1, 1]:

| Feature group | Features |
|---|---|
| Baseline and time | bias, time of day |
| Demand and supply | expected arrival rate, fleet capacity relative to expected demand, share of the fleet that can reach the zone, average travel time of reachable vehicles |
| Battery | average fleet battery |
| Prices and stations | cheapest nearby station price including travel cost, cost of replacement energy, expected station load relative to capacity |
| End-of-day interactions | time × battery, time × capacity balance |

The price target is a weighted sum of these features using 12 coefficients shared by all zones. It is then rounded to the nearest allowed price. Pricing sees the same restricted information as Joint PPO: no current requests or choices.

**How the coefficients are fitted.** Cross-entropy search, a simple form of direct policy search:

1. Sample 24 coefficient vectors around the current guess.
2. Run each for 16 complete simulated days. All candidates in a round face the same days.
3. Keep the best 6 by total discounted profit, and move the guess toward them.
4. Repeat for 12 rounds.

Training first uses cheap service-first dispatch for those 12 rounds. It then refines the prices separately with the dispatcher that will actually use them, rollout ADP or the inventory rule, using a smaller search: 8 candidates, 4 days and 4 rounds. On separate validation days, each seed then picks one of three options: the best fixed price, the first fitted rule, or the refined rule. Seeds that pick a fixed price simply run the fixed-price controller.

**Consistency.** Rollout ADP calls the same learned pricing rule inside its imagined futures. Otherwise it would value future fleet states as if prices were still fixed.

Code that combines the two: `rl/priced_rollout.py`. Deep dive: [methods/ROLLOUT_ADP.md](methods/ROLLOUT_ADP.md#learning-prices-as-well-as-planning-dispatch).

## Table ADP

**Idea.** For the two-vehicle city, the future depends only on the hour and on each vehicle's location and battery. That is small enough to tabulate. Learn how much profit is still available from every combination, then use that table at decision time.

**Learning** (code: `rl/fitted_value.py`). Start at the last hour, where future value is zero, and work backward. For each hour and each grid state (4 × 4 location pairs × 9 × 9 battery levels on a 10 kWh grid), sample eight possible hours of demand. For each sample, find the best action by current profit plus the next hour's table value, then average. Battery levels between grid points are interpolated.

**Deciding.** Try every joint action on the half/full exchange grid and pick the one with the best current profit + 0.99 × table value of the next state. The price is fixed at $0.60.

**Limits.** The table grows exponentially with the number of vehicles, so it supports exactly two. It also needs to query the simulator from arbitrary fleet states, which is stronger access than PPO gets.

Deep dive: [methods/TABLE_ADP.md](methods/TABLE_ADP.md).

## Joint PPO

**Idea.** Learn both decisions with neural networks from experience. Proximal Policy Optimization (PPO) repeatedly runs simulated days, then nudges each network toward actions that turned out better than expected.

**Four networks** (code: `rl/paper_ppo.py`):

| Network | Sees | Outputs |
|---|---|---|
| Pricing actor | Restricted pre-choice information | A price probability for each zone |
| Pricing critic | Same | Expected profit for the rest of the day, before customers choose |
| Dispatch actor | Full post-choice information | An action probability for each vehicle |
| Dispatch critic | Same | Expected profit for the rest of the day, after customers choose |

**Training signal.** Each hour has two steps, pricing then dispatch, so the advantage calculation (generalized advantage estimation) runs through both. Profit is discounted once per hour, not once per step. The dispatch update treats the fleet's action as one joint choice: it multiplies the vehicles' probability ratios and clips the product once. Dispatch actions are service, idle (wait or move), charge, or discharge, each with a destination. Charge and discharge request the most energy the vehicle can physically exchange.

**Warm start (optional).** PPO trained from scratch learned to almost never recharge. With `--warmstart-episodes N`, the networks first imitate the inventory rule on N simulated days (code: `rl/paper_warmstart.py`), then train with ordinary PPO. The warm-started runs also used reward scaling, advantage normalization and λ = 1; see [RUNNING.md](RUNNING.md#joint-ppo).

The equations are in [MDP_RL_2.pdf](MDP_RL_2.pdf). Deep dive: [methods/JOINT_PPO.md](methods/JOINT_PPO.md).

## Earlier PPO

The first implementation, the SA-PPO–MAPPO design described in [MDP_RL.pdf](MDP_RL.pdf), is kept so its saved models remain valid controls. It differs from Joint PPO in several ways at once:

- Pricing sees current requests.
- One shared network acts for each vehicle from that vehicle's own view.
- Each vehicle's probability ratio is clipped separately.
- The amount of energy to exchange is a learned continuous fraction.

Deep dive: [methods/EARLIER_PPO.md](methods/EARLIER_PPO.md).

## Comparing them fairly

These controllers do not get equal information or equal computation, and the results should be read that way:

- **Model access.** ADP and the inventory rules calculate alternative current actions with the known physical rules. Rollout ADP also samples futures from its market model. In the historical-model study that model is estimated from records; PPO trains by interacting with the same fitted simulator. This is simulator-trained PPO, not direct offline PPO training on logged actions.
- **Computation at decision time.** Rollout ADP simulates futures at every decision. PPO runs one network pass.
- **Pricing.** Some controllers learn prices and others use a fixed price chosen on validation days.

Results therefore compare complete implemented controllers under one simulator. They do not say that "ADP beats RL" in general.

## Terms used throughout

| Term | Meaning here |
|---|---|
| Episode, day | One 12-period run of the simulator |
| Discounted profit | Sum of hourly profits, the hour-$`t`$ profit weighted by $`0.99^{t-1}`$; the main score |
| Base policy | The simple controller rollout ADP assumes for the imagined future (service-first) |
| Continuation value | Estimated profit from the next hour to the end of the day |
| Shadow price, battery value | Dollar value per stored kWh used by the inventory rule |
| Training / validation / test days | Separate random seeds for fitting, for choosing settings, and for the final score. Test days are used once, after everything else is frozen |
| Seed 7, 19, 37 | Three optimization repetitions (and planning streams for rollout ADP). Table rows average separately run policies. Study 12 shares one historical model across them, so these do not measure uncertainty from collecting new histories |
| Paired 95% interval | Uncertainty of a profit difference, computed from the per-day differences of two controllers on the same test days (5,000 bootstrap resamples). It does not include retraining uncertainty |
