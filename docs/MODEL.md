# How the charging model works

This project models an operator that owns fixed charging stations and mobile charging vehicles. In each period, the operator sets mobile prices, customers choose a charging option, and the mobile vehicles decide where to go. The simulator then calculates how much energy is delivered, how vehicle batteries change, and how much profit the operator earns.

This guide explains those rules. The default numerical values describe an example city and have not been fitted to real-world, more complicated configurations. [TRAINING.md](TRAINING.md) explains how SA-PPO–MAPPO learns to make the pricing and vehicle decisions.

## 1. The city, energy units, and order of decisions

A **node** represents a location or service zone in the city. A customer's electric vehicle is an **EV**, a mobile charging vehicle is an **MCV**, and a fixed charging station is an **FCS**. We use `z` to identify a node, `m` a mobile vehicle, `f` a station, and `t` a time period. Periods run from `0` to `T-1`, where `T` is the number of periods in an episode.

The default city has four nodes, two stations, two mobile vehicles, and 12 one-hour periods. Node coordinates are `(0,0)`, `(4,0)`, `(0,4)`, and `(4,4)` km. Stations are at nodes 0 and 3; mobile vehicles start at nodes 1 and 2, each with 55 kWh in its battery. Distance is Manhattan distance, which is the sum of horizontal and vertical distances, as on a rectangular street grid. This distance is used when checking station access and calculating travel time, energy, and cost.

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

1. **Pricing, stage 0:** generate the arriving customers, their energy needs, and station prices. Estimate waiting times. The operator sees this information and the fleet's locations and batteries, then chooses one mobile price per node from a fixed list.
2. **Customer choice:** each customer chooses mobile service at their origin, a station they can access, or the outside option, which means leaving the operator's system. These choices produce the information available for dispatch, called the post-choice state.
3. **Dispatch, stage 1:** all mobile vehicles submit their actions together. The simulator determines which customers are served and how shared station capacity is used, executes the actions, and calculates the period's profit.
4. **Next period:** move to the next time period, generate new customer requests and station prices, and return to stage 0. After all `T` periods, the episode ends at stage 2.

Customers do not try another provider after making their choice. Unserved requests do not carry over to later periods.

## 2. How customer requests and station prices are generated

One **request** means one customer asking for a certain amount of charging energy. For each node and period, the simulator independently draws the number of arriving customers from a Poisson distribution. The configured average arrival count is `lambda`; the actual count is `N`. Each customer requests an energy amount `q_r` drawn uniformly between the configured lower and upper bounds. The total energy requested at that node is the sum of the individual amounts:

```math
N_{tz} \sim \mathrm{Poisson}(\lambda_{tz}),
\qquad q_r \sim \mathrm{Uniform}(q_{\min}, q_{\max}),
\qquad \overline d_{tz}=\sum_{r\in R_{tz}}q_r.
```

The average arrival count normally equals the zone's **baseline demand multiplied by a time-dependent multiplier**. For example, a baseline of 4 customers and a multiplier of 0.5 give an average of 2 customers in that period. The actual count can still be 0, 1, 2, 3, or more. There is no artificial upper limit on arrivals.

The configuration calls the time-dependent multiplier `arrival_profile` and the zone's baseline demand `mean_requests_by_zone`. Alternatively, `arrival_rates` can directly provide a table of average arrival counts, with one row per period and one column per zone. In that case, this table replaces the baseline-times-multiplier calculation.

The default bounds $q_{\min}$ and $q_{\max}$ are 5 and 15 kWh. Each customer's requested energy is drawn uniformly within this range, so equal-width intervals are equally likely. These bounds apply to each customer, rather than to the zone's total requested energy.

**The request list groups customers by their origin zone.** Within a period, each customer keeps the same position in this list while their provider choice and service outcome are calculated. For example, if zone 0 has two customers and zone 1 has one, the list contains the two zone-0 customers followed by the zone-1 customer.

For controlled examples, `reset(options={"requests": counts})` lets you specify the first period's customer counts yourself. Those customers all request the configured constant `request_kwh`, unless you also supply their individual energies in `request_energy_kwh`. A supplied energy list must follow the same customer order and stay within the allowed energy bounds.

You can also supply `accessible_fcs` to say which stations each customer may consider. This is a table with one row per customer and one column per station: `True` means accessible and `False` means inaccessible. Its shape is `[sum(counts), n_fcs]`. These optional inputs require explicit customer counts. The manual inputs affect only the first period; later periods return to random generation.

For each station and period, the simulator draws one random percentage change, `epsilon`, and applies it to both the customer charging price and the price paid for MCV energy exported through that station:

```math
p^{F,C}_{tf}=\max\{0,\overline p^{F,C}_{tf}(1+\epsilon_{tf})\},
\qquad p^{F,D}_{tf}=\max\{0,\overline p^{F,D}_{tf}(1+\epsilon_{tf})\}.
```

Baseline station prices can be set separately for each period and station using `fcs_charge_price_profile` and `fcs_discharge_price_profile`. Each table has one row per period and one column per station. Without these tables, each station's customer-price baseline stays constant over time, while the export-price baseline follows the same time pattern at every station.

By default, the random percentage change is uniformly distributed between -5% and +5%, so both station prices move up or down together. The configuration can instead use normally distributed changes or turn randomness off. `price_noise_scale` sets the uniform distribution's half-width or the normal distribution's standard deviation. A calculated price below zero is set to zero. Prices are generated once at the start of a period and stay fixed through pricing and dispatch. For a given scenario, the baseline patterns stay fixed across episodes, while random arrivals and price changes can differ.

The separate wholesale electricity purchase price, `grid_buy_per_kwh[t]`, is what the operator pays for energy from the external grid. Transfers between its own stations and mobile vehicles do not create an additional payment in the profit calculation.

## 3. Which services customers can use, and their estimated delays

A mobile vehicle is **eligible** for a node if it can reach that node within the period, keep the required battery reserve, and still deliver at least the smallest allowed request, `request_min_kwh`. This minimum-energy reserve is a simulator assumption which can be changed later. An eligible vehicle may still lack enough time or energy for a larger request.

When at least one vehicle is eligible, the estimated mobile delay is the average travel time of those vehicles. Here `M_z` is the set of eligible vehicles, `d(L_m,z)` is the distance from vehicle `m` to the node, and `v` is vehicle speed:

```math
\widehat w^M_{tz}=\frac{1}{|M_z|}\sum_{m\in M_z}\frac{d(L_m,z)}{v}.
```

If no vehicle is eligible, customers at that node cannot choose mobile service. The code records a delay of zero in this case, so it must also check `mobile_available` and `eligible_mcvs`; zero delay alone does not mean immediate service. This delay is an estimate made before dispatch and does not promise that a particular vehicle will serve the customer.

By default, a customer may consider a fixed station if it is within the allowed travel distance and can deliver at least the smallest allowed request. In a controlled example, a manually supplied accessibility table can replace the distance rule and allow a more distant station. The station must still have enough capacity for the minimum request. A station with no customer-service capacity is unavailable.

We also need to estimate station waiting times. To estimate a station's possible demand, `L_f`, the simulator divides each customer's requested energy equally among the stations that customer may consider. This is only a calculation for estimating congestion; it does not split an actual customer's charging across stations. With station customer-service capacity `K_f`, the estimated wait is

```math
\widehat w^F_{tf}=\frac{\Delta t}{2}\max\left\{0,\frac{L_f}{K_f}-1\right\},\qquad K_f>0.
```

A station with no customer-service capacity stores zero wait but remains unavailable. Other stations use this simple waiting estimate before customers choose, and keep it fixed for the whole period. The estimate is not recalculated after seeing their choices. Mobile-vehicle recharge and export use separate station capacity and do not affect these customer waiting estimates.

## 4. How prices influence each customer's choice

The operator chooses one mobile price, `p_z`, for each node from the fixed list `Scenario.price_levels`. All customers at that node see the same price. In the Python interface, `quote_prices` accepts dollar-per-kWh amounts from this list, while `quote_price_levels` and Gymnasium's `price_levels` action accept positions in the list, such as index 0 for its first entry. A node without mobile availability receives a valid placeholder price, but customers there still cannot choose mobile charging.

A customer chooses mobile service at their origin, rather than choosing a named mobile vehicle. For request `r` at node `z`, the customer compares payment plus inconvenience, with inconvenience expressed in dollars:

```math
c^M_r=p_zq_r+a_M\widehat w^M_{tz},
\qquad c^F_{rf}=p^{F,C}_{tf}q_r+b_F d(z,z_f)+a_F\widehat w^F_{tf}.
```

Here `a_M` is the customer's cost per hour of waiting for mobile service, `a_F` is the cost per hour of waiting at a station, and `b_F` is the inconvenience cost per kilometre of traveling to a station. Mobile waiting cost uses the average arrival time of eligible vehicles. For example, 10 kWh at 0.60 dollars/kWh costs 6 dollars before any waiting inconvenience is added.

Let `A_r` contain the charging options this customer may consider: mobile charging, if available, and accessible fixed stations. The outside option means leaving without using the operator's service, and its baseline preference score is zero. The positive parameter `eta` determines how strongly cost reduces the appeal of a charging option. The parameter `b` is a common charging benefit that makes all charging options more attractive relative to leaving. The following logit model converts these scores into probabilities:

```math
u_r^k=b-\eta c_r^k,\quad u_r^O=0,\qquad
P_r(k)=\frac{\exp(b-\eta c_r^k)}{1+\sum_{j\in A_r}\exp(b-\eta c_r^j)},\quad k\in A_r,
\qquad
P_r(O)=\frac{1}{1+\sum_{j\in A_r}\exp(b-\eta c_r^j)}.
```

The simulator randomly selects exactly one option using these probabilities. The possible outcomes include leaving, written as `O`, as well as the charging options in `A_r`. An unavailable service has zero chance of selection. If no charging option is available, the customer always leaves.

The default `b=0` is the original choice rule. Under this rule, nonnegative charging costs give each charging option a weight no greater than the outside option's weight of one, so many customers may leave.

The optional `b=3` scenario adds a charging benefit relative to leaving. `b=2` and `b=4` let us compare smaller and larger benefits. These are changes to the customer model, with values chosen for experiments rather than estimated from real customer data. For unchanged costs and available options, increasing `b` makes leaving less likely, while keeping customers' relative preferences among charging providers unchanged. With no available provider, the customer still leaves. The simulator and adaptive pricing benchmark use the same probability calculation in `choice.py`. See [OUTSIDE_OPTION.md](OUTSIDE_OPTION.md) for details.

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
Q^{\max}_{mz}=\max\left\{0,\min\left[\eta_M(E_m-E^{\mathrm{travel}}-E_{\min}),hP_M\right]\right\}.
```

The simulator processes vehicles in vehicle-number order: vehicle 0 first, then vehicle 1, and so on. At a chosen node, it checks the remaining customer requests in their original list order. A request is served in full if it fits the vehicle's remaining energy-delivery limit. A request that is too large is skipped, while a smaller later request may still fit. Served requests are removed so another vehicle cannot serve them again.

For example, with a 24 kWh delivery limit and requests of `[15, 12, 8]` kWh, the vehicle serves 15, skips 12, and serves 8, delivering 23 kWh. This rule is called **first-fit allocation**. It does not search for the most profitable combination or partially serve requests. Fixed stations check requests in the same way, up to their customer-service energy limit. No customer request is divided across vehicles.

Each fixed station has separate limits for charging customer EVs and exchanging energy with mobile vehicles. First, each mobile vehicle's requested recharge or export amount is reduced as needed to fit its remaining time, power, battery limits, and efficiency. The station then adds the recharge and export amounts together. If their total exceeds its mobile-vehicle energy limit, every amount is reduced by the same proportion. Energy drawn and energy exported both count toward this limit; one does not cancel the other.

The model limits total energy over the period without scheduling individual charging ports or tracking total power at every instant. Fixed stations have no storage batteries.

## 6. How batteries change and profit is calculated

For mobile vehicle `m`, let `C_m` be the energy drawn from the grid, `X_m` the energy exported to the grid, and `Q_m` the energy delivered to customers. Its battery loses energy through travel, service, and export, and gains energy through recharge. Charging and discharging efficiencies account for losses:

```math
E'_m=E_m-E^{\mathrm{travel}}_m+\eta_C C_m-Q_m/\eta_M-X_m/\eta_D,
\qquad E_{\min}\leq E'_m\leq E_{\max}.
```

The simulator reports these amounts in an **energy ledger**, a record of where energy came from and where it went. It also reports the difference between the recorded battery change and the sum of the energy flows. A difference near zero confirms that the accounting balances. Stored battery energy, grid-side energy, and energy delivered to customers are measured at different points and must be converted using the relevant efficiencies.

The operator owns both mobile and fixed charging, so the reward includes profit from both. Let `Q_mz` be energy delivered by a mobile vehicle at a node, `Q_f` energy delivered to customers at a station, and `X_f` energy exported by mobile vehicles through that station. Let `U` be the requested energy left unserved among customers who chose the operator's service. The period's profit is

```math
\begin{aligned}
r_t={}&\sum_{m,z}p_zQ_{mz}+\sum_f p^{F,C}_{tf}Q_f+\sum_f p^{F,D}_{tf}X_f\\
&-g^{\mathrm{buy}}_t\left(\sum_m C_m+\sum_f Q_f/\eta_F\right)
-c_{\mathrm{travel}}\sum_m d_m-c_{\mathrm{op}}|M|-c_{\mathrm{unmet}}U\\
&+\mathbf{1}_{\{t=T-1\}}s\sum_m E'_m.
\end{aligned}
```

Read the equation as customer charging revenue plus grid-export revenue, minus electricity purchase, travel, operating, and unmet-service costs. Export revenue uses the price paid at the station through which energy is exported. Electricity purchase cost uses the external wholesale price, so internal transfers between the operator's stations and vehicles are not charged again. Every mobile vehicle incurs an operating cost each period, even when idle. The unmet-service penalty depends on the actual unserved kWh of customers who chose the operator's service.

The model can optionally assign a value to battery energy remaining at the end of the episode. This is called **terminal inventory value** and defaults to zero which can also be changed to other values. Export through a station sends energy to the grid; it does not fill station storage or increase that station's capacity for serving customer EVs.

The vehicles start with energy already in their batteries. The operator is not charged for buying this initial energy during the episode. This convention matters when comparing scenarios with different starting battery levels.

The output `reward_terms` lists each revenue and cost separately. `served_request_mask` records which customers were served, and `assigned_mcv` identifies the mobile vehicle assigned to each mobile-served request. These outputs describe the period just completed, even when the next observation already contains a new period's customers.

The output `info["metrics"]` reports the shares of customers who leave or receive service, the fraction of mobile vehicles that are active, and how much service capacity the vehicles and stations use. Mobile service utilization is measured against the vehicles' full-period nominal delivery capacity. These describe what happened in a period; they do not by themselves establish whether training improved the policy. Customers who leave are tracked separately from customers who choose a service but remain unserved.

## 7. What the policies can see and how future profit is valued

The **pricing state** is the information available before prices are chosen: the current period, vehicle locations and batteries, configured average arrivals, actual customer requests and their accessible stations, station charging and export prices, the wholesale electricity price, and estimated waiting times and service availability. The **dispatch state** adds the posted mobile prices, the option each customer actually chose, and totals of chosen customer counts and requested energy.

The `request_details` table has one row per customer. Its columns record origin zone, requested kWh, provider choice, and accessibility to each station. With `N` customers and `n_fcs` stations, its shape is `[N, 3+n_fcs]`, stored as `float64`. The choice code is `-1` before the customer chooses, `0` for mobile service, `1..n_fcs` for fixed stations, and `n_fcs+1` for leaving.

The number of customer rows can change each period. Gymnasium therefore uses a variable-length sequence, `Sequence(Box(...), stack=True)`, rather than imposing a maximum request count. The learner encodes each request and combines the resulting features into a fixed-size input for its networks. It includes request order because service allocation checks requests in order. A trainer that expects a fixed-length input needs an adapter to handle these variable-length requests.

Some outputs give one value per zone: `mobile_demand` has shape `[n_zones]`. Others give one value for each vehicle and zone: `served_mobile` and `delivered_mobile_kwh` have shape `[n_mcv,n_zones]` and describe the completed period. `fcs_retail_prices` and `fcs_discharge_prices` each give one price per station, replacing the old single `grid_sell_price` observation. Price values may include any nonnegative result of the configured random price changes. Request counts use `int64` storage without an artificial demand cap.

The `pricing_mask` marks available price choices during stage 0, and `action_mask` marks available vehicle actions during stage 1. A node without mobile service has no available price choice. The policy supplies a valid placeholder index such as `0` and excludes that node from its probability calculation, rather than trying to choose from an empty list.

After pricing, each vehicle receives its own input through `local_observation(m)`. This includes its location, battery, available actions, and `agent_id`, as well as other vehicles' locations and batteries and the shared customer, price, and waiting-time information. This is the coordination information called `zeta_mt`. Every vehicle can use the same neural network while receiving different inputs, and all vehicles learn from the same total operator profit.

The aim is to maximize expected total profit across the episode, discounting later profit by a factor `gamma` per period. Two value functions estimate the remaining discounted profit: `V^P` just before pricing, and `V^D` after customers choose but before dispatch. They are related as follows:

```math
\max\mathbb E\left[\sum_{t=0}^{T-1}\gamma^t r_t\right],\qquad
V^P(s_t)=\mathbb E[V^D(\widetilde s_t)\mid s_t],\qquad
V^D(\widetilde s_t)=\mathbb E[r_t+\gamma V^P(s_{t+1})\mid\widetilde s_t].
```

Pricing and dispatch are two decisions within the same physical period. Future profit is therefore discounted only when moving to the next period, rather than at both decisions. The simulator reports this in `info["discount"]`: `1.0` after pricing, `discount_per_period` after a nonfinal dispatch, and `0.0` when the episode ends.

The learner has separate pricing and dispatch critics, which are networks that estimate these two values. When calculating learning targets, its trace coefficient is one between pricing and dispatch, and `gae_lambda` between periods. This is a training implementation choice. [TRAINING.md](TRAINING.md) explains the network architectures, PPO losses, learning-target calculations, and evaluation procedure.

The same environment also supports [fitted-value dispatch](FITTED_VALUE.md), which holds pricing fixed and learns the future value of fleet states. That controller uses the existing physical transitions and profit accounting; its learned value replaces the dispatch rule rather than changing customer behavior.
