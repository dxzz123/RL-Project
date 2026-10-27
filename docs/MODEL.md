# How the charging model works

This project models an operator that owns fixed charging stations and mobile charging vehicles. In each period, the operator sets mobile prices, customers choose a charging option, and the mobile vehicles decide where to go. The simulator then calculates how much energy is delivered, how vehicle batteries change, and how much profit the operator earns.

This guide explains the common simulation used by every controller: how customers arrive and choose, how vehicles deliver energy, and how profit is counted. The numbers below describe the small city (`configs/utility_b3.json`). The large city (`configs/synthetic_25_zones.json`) follows exactly the same rules with 25 zones, 10 vehicles and 5 stations. Neither is calibrated to real customer or fleet data. The controllers are explained in [METHODS.md](METHODS.md), and the commands are in [RUNNING.md](RUNNING.md).

For a first reading, follow the order of events: arrivals, customer choice, vehicle dispatch, then energy and profit. The equations make each rule precise; the surrounding explanations describe what it means in operation.

## 1. The city, energy units, and order of decisions

A **node** represents a location or service zone in the city. A customer's electric vehicle is an **EV**, a mobile charging vehicle is an **MCV**, and a fixed charging station is an **FCS**. In the equations, $`i`$ identifies a node, $`m`$ a mobile vehicle, $`j`$ a node containing a station, and $`t=1,\ldots,T`$ an operating period. Python counts periods from `0` to `T-1` and stores station arrays by slot `f`; station slot `f` is at node `fcs_zones[f]`. These are the conventions in [NOTATION.md](reference/NOTATION.md).

The small city has four nodes, two stations, two mobile vehicles, and 12 one-hour periods. Node coordinates are `(0,0)`, `(4,0)`, `(0,4)`, and `(4,4)` km. Stations are at nodes 0 and 3; mobile vehicles start at nodes 1 and 2, each with 55 kWh in its battery. Distance is Manhattan distance, which is the sum of horizontal and vertical distances, as on a rectangular street grid. This distance is used when checking station access and calculating travel time, energy, and cost.

| Quantity | Unit |
|---|---|
| Energy inside a mobile vehicle's battery, including its reserve and capacity | Stored kWh |
| Request energy and customer delivery | Delivered kWh |
| MCV service power | Delivered kW |
| MCV recharge / export amounts and the station's MCV energy limit | Grid-side kWh |
| Charging / discharging power | Grid-side kW |
| Maximum energy a station can deliver to customer EVs | Delivered kWh per period |
| Unit price | Dollars per applicable kWh |
| Travel and waiting | km and hours |
| Customer inconvenience | Dollars |

Stored energy is measured inside an MCV battery; delivered energy is what reaches the customer's EV; grid-side energy is measured at the grid connection. Charging and discharging losses mean these amounts can differ.

Every period contains two decisions—pricing and dispatch—in this order:

1. **Pricing, stage 0:** generate the arriving customers, their energy needs, and station prices. Estimate waiting times. The raw simulator observation contains this information and the fleet's locations and batteries. The pricing controller uses its permitted inputs to choose one mobile price per node from a fixed list.
2. **Customer choice:** each customer chooses mobile service at their origin, a station they can access, or the outside option, which means leaving the operator's system. These choices produce the information available for dispatch, called the post-choice state.
3. **Dispatch, stage 1:** all mobile vehicles submit their actions together. The simulator determines which customers are served and how shared station capacity is used, executes the actions, and calculates the period's profit.
4. **Next period:** move to the next time period, generate new customer requests and station prices, and return to stage 0. After all `T` periods, the episode ends at stage 2.

Customers do not try another provider after making their choice. Unserved requests do not carry over to later periods.

## 2. How customer requests and station prices are generated

One **request** means one customer asking for a certain amount of charging energy. For each node and period, the simulator independently draws the number of arriving customers from a Poisson distribution. The configured average arrival count is `lambda`; the actual count is `N`. Each customer requests an energy amount `q_r` drawn uniformly between the configured lower and upper bounds. The total energy requested at that node is the sum of the individual amounts:

```math
\begin{aligned}
N_{i,t} &\sim \mathrm{Poisson}(\lambda_{i,t}),\\
q_r &\sim \mathrm{Uniform}(q_{\min}, q_{\max}),\\
\overline d_{i,t}&=\sum_{r\in\mathcal R_{i,t}}q_r.
\end{aligned}
```

The average arrival count normally equals the zone's **baseline demand multiplied by a time-dependent multiplier**. For example, a baseline of 4 customers and a multiplier of 0.5 give an average of 2 customers in that period. The actual count can still be 0, 1, 2, 3, or more. There is no artificial upper limit on arrivals.

The configuration calls the time-dependent multiplier `arrival_profile` and the zone's baseline demand `mean_requests_by_zone`. Alternatively, `arrival_rates` can directly provide a table of average arrival counts, with one row per period and one column per zone. In that case, this table replaces the baseline-times-multiplier calculation.

The default bounds $`q_{\min}`$ and $`q_{\max}`$ are 5 and 15 kWh. Each customer's requested energy is drawn uniformly within this range, so equal-width intervals are equally likely. These bounds apply to each customer, rather than to the zone's total requested energy.

**The request list groups customers by their origin zone.** Within a period, each customer keeps the same position in this list while their provider choice and service outcome are calculated. For example, if zone 0 has two customers and zone 1 has one, the list contains the two zone-0 customers followed by the zone-1 customer.

For controlled examples, `reset(options={"requests": counts})` lets you specify the first period's customer counts yourself. Those customers all request the configured constant `request_kwh`, unless you also supply their individual energies in `request_energy_kwh`. A supplied energy list must follow the same customer order and stay within the allowed energy bounds.

You can also supply `accessible_fcs` to say which stations each customer may consider. This is a table with one row per customer and one column per station: `True` means accessible and `False` means inaccessible. Its shape is `[sum(counts), n_fcs]`. These optional inputs require explicit customer counts. The manual inputs affect only the first period; later periods return to random generation.

For each station and period, the simulator draws one random percentage change, $`\epsilon_{j,t}`$, and applies it to both the customer charging price and the price paid for MCV energy exported through that station:

```math
\begin{aligned}
p^{F,C}_{j,t}&=\max\{0,\overline p^{F,C}_{j,t}(1+\epsilon_{j,t})\},\\
p^{F,D}_{j,t}&=\max\{0,\overline p^{F,D}_{j,t}(1+\epsilon_{j,t})\}.
\end{aligned}
```

Baseline station prices can be set separately for each period and station using `fcs_charge_price_profile` and `fcs_discharge_price_profile`. Each table has one row per period and one column per station. Without these tables, each station's customer-price baseline stays constant over time, while the export-price baseline follows the same time pattern at every station.

By default, the random percentage change is uniformly distributed between -5% and +5%, so both station prices move up or down together. The configuration can instead use normally distributed changes or turn randomness off. `price_noise_scale` sets the uniform distribution's half-width or the normal distribution's standard deviation. A calculated price below zero is set to zero. Prices are generated once at the start of a period and stay fixed through pricing and dispatch. For a given scenario, the baseline patterns stay fixed across episodes, while random arrivals and price changes can differ.

The separate wholesale electricity purchase price, `grid_buy_per_kwh[period]`, is what the operator pays for energy from the external grid. Transfers between its own stations and mobile vehicles do not create an additional payment in the profit calculation.

## 3. Which services customers can use, and their estimated delays

A mobile vehicle is **eligible** for a node if it can reach that node within the period, keep the required battery reserve, and still deliver at least the smallest allowed request, `request_min_kwh`. Requiring capacity for at least the smallest request is an availability rule; it is separate from the stored battery reserve. An eligible vehicle may still lack enough time or energy for a larger request.

When at least one vehicle is eligible, the estimated mobile delay is the average travel time of those vehicles. Here $`\mathcal M_{i,t}`$ is the set of eligible vehicles, $`d(l_{m,t},i)`$ is the distance from vehicle $`m`$ to node $`i`$, and $`v`$ is vehicle speed:

```math
\widehat w_{i,t}=\frac{1}{|\mathcal M_{i,t}|}\sum_{m\in\mathcal M_{i,t}}\frac{d(l_{m,t},i)}{v}.
```

If no vehicle is eligible, customers at that node cannot choose mobile service. The code records a delay of zero in this case, so it must also check `mobile_available` and `eligible_mcvs`; zero delay alone does not mean immediate service. This delay is an estimate made before dispatch and does not promise that a particular vehicle will serve the customer.

By default, a customer may consider a fixed station if it is within the allowed travel distance and can deliver at least the smallest allowed request. In a controlled example, a manually supplied accessibility table can replace the distance rule and allow a more distant station. The station must still have enough capacity for the minimum request. A station with no customer-service capacity is unavailable.

We also need to estimate station waiting times. To estimate a station's possible demand, $`L_{j,t}`$, the simulator divides each customer's requested energy equally among the stations that customer may consider. This is only a calculation for estimating congestion; it does not split an actual customer's charging across stations. With station customer-service capacity $`K_j`$, the estimated wait is

```math
w^F_{j,t}=\frac{\Delta t}{2}\max\left\{0,\frac{L_{j,t}}{K_j}-1\right\},\qquad K_j>0.
```

A station with no customer-service capacity stores zero wait but remains unavailable. Other stations use this simple waiting estimate before customers choose, and keep it fixed for the whole period. The estimate is not recalculated after seeing their choices. Mobile-vehicle recharge and export use separate station capacity and do not affect these customer waiting estimates.

## 4. How prices influence each customer's choice

The operator chooses one mobile price, $`p^M_{i,t}`$, for each node from the fixed list `Scenario.price_levels`. All customers at that node see the same price. In the Python interface, `quote_prices` accepts dollar-per-kWh amounts from this list, while `quote_price_levels` and Gymnasium's `price_levels` action accept positions in the list, such as index 0 for its first entry. A node without mobile availability receives a valid placeholder price, but customers there still cannot choose mobile charging.

A customer chooses mobile service at their origin, rather than choosing a named mobile vehicle. For request $`r`$ at node $`i`$, the customer compares payment plus inconvenience, with inconvenience expressed in dollars:

```math
\begin{aligned}
c^M_{r,t}&=p^M_{i,t}q_r+a_M\widehat w_{i,t},\\
c^j_{r,t}&=p^{F,C}_{j,t}q_r+b_F d(i,j)+a_Fw^F_{j,t}.
\end{aligned}
```

Here `a_M` is the customer's cost per hour of waiting for mobile service, `a_F` is the cost per hour of waiting at a station, and `b_F` is the inconvenience cost per kilometre of traveling to a station. Mobile waiting cost uses the average arrival time of eligible vehicles. For example, 10 kWh at 0.60 dollars/kWh costs 6 dollars before any waiting inconvenience is added.

Let $`\mathcal A_{r,t}`$ contain the charging options this customer may consider: mobile charging, if available, and accessible fixed stations. The outside option means leaving without using the operator's service, and its baseline preference score is zero. The positive parameter `eta` determines how strongly cost reduces the appeal of a charging option. The parameter `b` is a common charging benefit that makes all charging options more attractive relative to leaving. The following logit model converts these scores into probabilities:

```math
\begin{aligned}
u_{r,t}^k&=b-\eta c_{r,t}^k,\\
u_{r,t}^O&=0,\\
\rho_{r,t}^k&=\frac{\exp(b-\eta c_{r,t}^k)}{1+\sum_{a\in\mathcal A_{r,t}}\exp(b-\eta c_{r,t}^a)},\quad k\in\mathcal A_{r,t},\\
\rho_{r,t}^O&=\frac{1}{1+\sum_{a\in\mathcal A_{r,t}}\exp(b-\eta c_{r,t}^a)}.
\end{aligned}
```

The simulator randomly selects exactly one option using these probabilities. The possible outcomes include leaving, written as `O`, as well as the charging options in $`\mathcal A_{r,t}`$. An unavailable service has zero chance of selection. If no charging option is available, the customer always leaves.

The default `b=0` is the original choice rule. Under this rule, nonnegative charging costs give each charging option a weight no greater than the outside option's weight of one, so many customers may leave.

The optional `b=3` scenario adds a charging benefit relative to leaving. `b=2` and `b=4` let us compare smaller and larger benefits. These are changes to the customer model, with values chosen for experiments rather than estimated from real customer data. For unchanged costs and available options, increasing `b` makes leaving less likely, while keeping customers' relative preferences among charging providers unchanged. With no available provider, the customer still leaves. The simulator and adaptive pricing benchmark use the same probability calculation in `choice.py`. See [OUTSIDE_OPTION.md](reference/OUTSIDE_OPTION.md) for details.

Each customer's choice stays attached to their own request. Customers choosing mobile charging are grouped by origin node, and any vehicle visiting that node may serve those requests. Customers choosing a fixed station are grouped by their chosen station. Customers who leave are counted separately and do not incur an unmet-service penalty.

## 5. How vehicles serve customers and share station capacity

Each mobile vehicle chooses an operation, called its **mode**, and a destination node. Recharge and export actions also specify how much grid-side energy the vehicle requests:

| Mode | Operation |
|---|---|
| `WAIT` | Remain at the current node |
| `SERVE` | Travel to a node and serve its mobile request pool |
| `REPOSITION` | Travel without serving or exchanging energy |
| `RECHARGE` | Travel to an FCS and receive grid energy |
| `DISCHARGE` | Travel to an FCS and export battery energy |

An **action mask** marks which combinations of operation and destination a vehicle can physically carry out. It checks each vehicle separately, so a valid action does not reserve customer demand or station capacity. Another vehicle may use those resources first. An action that violates the physical rules is carried out as `WAIT` and reported in `invalid_actions`.

Travel takes both time and battery energy. If distance is `d` and speed is `v`, the time left for charging operations is the period length minus travel time. The energy used for travel is distance multiplied by the vehicle's energy use per kilometre, `kappa`:

```math
h=\Delta t-d/v,\qquad E^{\mathrm{travel}}=\kappa d.
```

After travel, a vehicle's maximum delivery is limited by both its battery energy above the required reserve and how much it can deliver in the remaining time. In the formula below, `eta_M` is service efficiency and `P_M` is service power:

```math
Q^{\max}_{m,i,t}=\max\left\{0,\min\left[\eta_M(e_{m,t}-E^{\mathrm{travel}}-e_{\min}),hP_M\right]\right\}.
```

The simulator processes vehicles in vehicle-number order: vehicle 0 first, then vehicle 1, and so on. At a chosen node, it checks the remaining customer requests in their original list order. A request is served in full if it fits the vehicle's remaining energy-delivery limit. A request that is too large is skipped, while a smaller later request may still fit. Served requests are removed so another vehicle cannot serve them again.

For example, with a 24 kWh delivery limit and requests of `[15, 12, 8]` kWh, the vehicle serves 15, skips 12, and serves 8, delivering 23 kWh. This rule is called **first-fit allocation**. It does not search for the most profitable combination or partially serve requests. Fixed stations check requests in the same way, up to their customer-service energy limit. No customer request is divided across vehicles.

Each fixed station has separate limits for charging customer EVs and exchanging energy with mobile vehicles. First, each mobile vehicle's requested recharge or export amount is reduced as needed to fit its remaining time, power, battery limits, and efficiency. The station then adds the recharge and export amounts together. If their total exceeds its mobile-vehicle energy limit, every amount is reduced by the same proportion. Energy drawn and energy exported both count toward this limit; one does not cancel the other.

The model limits total energy over the period without scheduling individual charging ports or tracking total power at every instant. Fixed stations have no storage batteries.

## 6. How batteries change and profit is calculated

For mobile vehicle $`m`$ in period $`t`$, let $`C_{m,t}`$ be energy drawn from the grid, $`X_{m,t}`$ energy exported to the grid, and $`Q_{m,t}`$ energy delivered to customers. Let $`e_{m,t}`$ be its stored battery energy before acting and $`e_{m,t+1}`$ its energy afterward. Its battery loses energy through travel, service, and export, and gains energy through recharge. Charging and discharging efficiencies account for losses:

```math
\begin{aligned}
e_{m,t+1}&=e_{m,t}-E^{\mathrm{travel}}_{m,t}+\eta_C C_{m,t}
-\frac{Q_{m,t}}{\eta_M}-\frac{X_{m,t}}{\eta_D},\\
e_{\min}&\leq e_{m,t+1}\leq B.
\end{aligned}
```

The simulator reports these amounts in an **energy ledger**, a record of where energy came from and where it went. It also reports the difference between the recorded battery change and the sum of the energy flows. A difference near zero confirms that the accounting balances. Stored battery energy, grid-side energy, and energy delivered to customers are measured at different points and must be converted using the relevant efficiencies.

The operator owns both mobile and fixed charging, so the reward includes profit from both. Let $`Q_{m,i,t}`$ be mobile energy delivered at a node, $`Q^F_{j,t}`$ station energy delivered to customers, and $`X^F_{j,t}`$ mobile energy exported through that station. Let $`U_t`$ be the requested energy left unserved among customers who chose the operator's service. Write $`d_{m,t}`$ for distance traveled, $`\mathcal M`$ for the fleet, and $`\nu`$ for the optional terminal value per stored kWh. The period's profit is

```math
\begin{aligned}
r_t={}&\sum_{m,i}p^M_{i,t}Q_{m,i,t}+\sum_j p^{F,C}_{j,t}Q^F_{j,t}+\sum_j p^{F,D}_{j,t}X^F_{j,t}\\
&-g^{\mathrm{buy}}_t\left(\sum_m C_{m,t}+\sum_j Q^F_{j,t}/\eta_F\right)\\
&-c_{\mathrm{travel}}\sum_m d_{m,t}-c_{\mathrm{op}}|\mathcal M|-c_{\mathrm{unmet}}U_t\\
&+\mathbf{1}_{\{t=T\}}\nu\sum_m e_{m,t+1}.
\end{aligned}
```

Read the equation as customer charging revenue plus grid-export revenue, minus electricity purchase, travel, operating, and unmet-service costs. Export revenue uses the price paid at the station through which energy is exported. Electricity purchase cost uses the external wholesale price, so internal transfers between the operator's stations and vehicles are not charged again. Every mobile vehicle incurs an operating cost each period, even when idle. The unmet-service penalty depends on the actual unserved kWh of customers who chose the operator's service.

The model can optionally assign a value to battery energy remaining at the end of the episode. This is called **terminal inventory value** and defaults to zero. Set `terminal_value_per_stored_kwh` to a positive value to change this assumption. Export through a station sends energy to the grid; it does not fill station storage or increase that station's capacity for serving customer EVs.

The vehicles start with energy already in their batteries. The operator is not charged for buying this initial energy during the episode. This convention matters when comparing scenarios with different starting battery levels.

The output `reward_terms` lists each revenue and cost separately. `served_request_mask` records which customers were served, and `assigned_mcv` identifies the mobile vehicle assigned to each mobile-served request. These outputs describe the period just completed, even when the next observation already contains a new period's customers.

The output `info["metrics"]` reports the shares of customers who leave or receive service, the fraction of mobile vehicles that are active, and how much service capacity the vehicles and stations use. Mobile service utilization is measured against the vehicles' full-period nominal delivery capacity. These describe what happened in a period; they do not by themselves establish whether training improved the policy. Customers who leave are tracked separately from customers who choose a service but remain unserved.

## 7. What the policies can see and how future profit is valued

The **raw pricing observation** contains: the current period, vehicle locations and batteries, configured average arrivals, actual customer requests and their accessible stations, station charging and export prices, the wholesale electricity price, and estimated waiting times and service availability. The **raw dispatch observation** adds the posted mobile prices, the option each customer actually chose, and totals of chosen customer counts and requested energy.

Each controller uses its permitted inputs. Earlier PPO can use current requests before pricing; Joint PPO hides those requests until dispatch. The table below summarizes the differences. Array shapes, action masks, and Python field names are collected in the [environment API reference](reference/ENVIRONMENT.md).

The common economic objective is to maximize expected total profit across the episode, discounting later profit by a factor `gamma` per period. Two value functions estimate the remaining discounted profit: `V^P` just before pricing, and `V^D` after customers choose but before dispatch. They are related as follows:

```math
\max\mathbb E\left[\sum_{t=1}^{T}\gamma^{t-1}r_t\right],
```

```math
\begin{aligned}
V^P(s_t)&=\mathbb E[V^D(\widetilde s_t)\mid s_t],\\
V^D(\widetilde s_t)&=\mathbb E[r_t+\gamma V^P(s_{t+1})\mid\widetilde s_t].
\end{aligned}
```

Pricing and dispatch are two decisions within the same physical period. Future profit is therefore discounted only when moving to the next period, rather than at both decisions. The simulator reports this in `info["discount"]`: `1.0` after pricing, `discount_per_period` after a nonfinal dispatch, and `0.0` when the episode ends.

The information a controller uses and the way it estimates future profit are separate design choices:

| Controller | Pricing input and decision | Dispatch and future-value estimate |
|---|---|---|
| [Rollout ADP](methods/ROLLOUT_ADP.md) | Fixed price, or a learned price rule that sees time, fleet, arrival means and current prices but not current requests | Compares a few complete fleet plans by exact current profit plus simulated future profit under service-first |
| [Table ADP](methods/TABLE_ADP.md) | Fixed price | Compares every joint action by current profit plus a learned table value for the next fleet state |
| [Joint PPO](methods/JOINT_PPO.md) | Learned; hides current request details, counts, total requested energy and waiting estimates from both pricing networks | Global post-choice observation; clips the fleet action ratio jointly; separate neural critics |
| [Earlier PPO](methods/EARLIER_PPO.md) | Learned from the full pre-choice observation, including current requests | Per-vehicle observations; separate neural pricing and dispatch critics |

Both PPO methods discount only between periods, never between pricing and dispatch within one period. Their generalized advantage estimation (GAE) differs: earlier PPO uses trace coefficient 1 between pricing and dispatch, while Joint PPO uses `gae_lambda` there as well. This changes the training targets, not physical time or the simulator's reward. [METHODS.md](METHODS.md) compares all controllers side by side.
