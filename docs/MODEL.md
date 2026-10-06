<!-- # Decision model and simulator contract

This document describes the implemented model, rather than a learning algorithm. The basic procedure is pricing by the operator followed by endogenous user choices and decentralized MCV dispatch. This document specifies the demand distributions, choice utilities, service duration, capacities, or energy accounting. The choices below settle those details for an initial simulator and are adjustable research assumptions.

## Entities, units, and defaults

The operator owns fixed charging stations (FCSs) and mobile charging vehicles (MCVs). Zones have coordinates in kilometers; travel distance is Manhattan distance, the sum of the absolute coordinate differences between zones. The default scenario has four zones, two FCSs, two MCVs, and 12 one-hour periods. Vehicle and station indices start at zero.

| Zone | Coordinates (km) | Fixed station | Initial MCV |
|---|---|---|---|
| 0 | `(0, 0)` | FCS 0 | — |
| 1 | `(4, 0)` | — | MCV 0 |
| 2 | `(0, 4)` | — | MCV 1 |
| 3 | `(4, 4)` | FCS 1 | — |

Thus the MCVs initially occupy different locations from the FCSs. This is an initial-state choice, not a permanent separation constraint: vehicles can visit an FCS to recharge or discharge, then move elsewhere. Station locations stay fixed throughout an episode.

| Quantity | Unit / convention |
|---|---|
| Vehicle battery inventory, reserve, capacity | Stored kWh |
| EV request | Delivered kWh; every customer requests `request_kwh` |
| Service power | Delivered kW |
| Fleet charging power | Grid-input kW |
| Fleet discharge power | Grid-export kW |
| Fleet exchange action and station exchange budget | Grid-side kWh |
| FCS EV service capacity | Delivered kWh per period |
| Mobile and fixed retail price | Dollars per delivered kWh |
| Grid buy and sell price | Dollars per grid-side kWh |
| Travel | km, hours, and stored kWh consumed |
| Unmet-demand penalty | Dollars per unserved requested kWh |
| Terminal inventory value | Dollars per remaining stored kWh |

The numeric defaults are synthetic. They support inspection and testing, without establishing empirical realism or a profitable operating strategy.

## Physical period and information sequence

Let period indices be `t = 0, ..., T-1`. Let `L_m` and `E_m` be each vehicle's location and stored energy.

1. **Pricing state, stage 0.** The operator observes the fleet, current EV requests by zone, known FCS retail prices, and current grid buy/sell prices.
2. **Price action.** The operator sets one price `p_m` per MCV, bounded by `price_min` and `price_max`. Prices apply per delivered kWh and to the individual MCV, not to a zone.
3. **User choices.** Each request chooses a named MCV, an FCS, or the outside option. The sampled choices become fixed demand for this period.
4. **Dispatch state, stage 1.** Each MCV can inspect a local observation and select one mode and one target zone. The simulator receives all dispatch actions together.
5. **Operation.** Each MCV travels to its target and uses the remaining period for its selected operation. Joint fleet station limits are resolved. FCS EV service occurs in parallel.
6. **Reward and next period.** The operator receives period profit. New EV arrivals are drawn, unless the horizon is complete. The next pricing state begins at period `t+1`.

Pricing has no elapsed physical time and returns zero reward. Dispatch advances time by one period and returns the whole period reward. The final dispatch sets `terminated=True`; the simulator has no early termination or time-limit truncation. Calls after termination require a new `reset()`.

## Exogenous arrivals

For each zone `z`, independently,

```math
N_{tz}\sim\mathrm{Binomial}\left(N_{\max},
\frac{\mu_z h_t}{N_{\max}}\right).
```

Here `N_max = max_requests_per_zone`, `mu_z = mean_requests_by_zone[z]`, and `h_t = arrival_profile[t]`. Validation requires each probability to lie in `[0,1]`. Draws are independent over zones and periods, with period-specific means. Every customer has the same requested energy `q = request_kwh`.

The realized counts, rather than just their means, are observed before pricing. `reset(options={"requests": [...]})` can override the first-period counts for a controlled experiment. It does not replace the later arrival process.

## Endogenous user choices

Users within a zone are homogeneous. For a request in zone `z`, a mobile option `m` has deterministic utility

```math
u^M_{zm}=b_M-\alpha q p_m-\delta_M d(L_m,z).
```

An FCS `f`, located in zone `z_f`, has utility

```math
u^F_{zf}=b_F-\alpha q p^F_f-\delta_F d(z,z_f).
```

The outside utility is `u_out`. Utilities are dimensionless: `alpha` multiplies the request's dollar bill, and the distance coefficient multiplies kilometers. The choice probabilities are multinomial logit:

```math
P_{zj}=\frac{\exp(u_{zj})}{\sum_{k\in\mathcal A_z}\exp(u_{zk})}.
```

The outside option is always available. A mobile option is available only if that MCV could travel to this zone and serve at least one full request within the period, using its current battery and service power. An FCS is available if the zone is within its access radius and its EV capacity admits at least one full request. Unavailable options have probability zero.

The simulator samples a multinomial count vector for each zone. `mobile_demand[m,z]` counts customers who selected vehicle `m` in zone `z`; `fixed_demand[f]` aggregates customers who selected FCS `f`.

These availability checks do not guarantee eventual service. A vehicle can be attractive to customers in several zones but can serve only one zone in the period. Also, an FCS can attract more demand than it can serve. Dispatch and capacity determine the served quantities. Customers who are not served do not switch provider, carry demand forward, or join a queue. Outside-option customers generate no revenue or unmet-demand penalty.

## Dispatch actions and feasibility

Each MCV receives one `DispatchAction(mode, target_zone, energy_kwh)`.

| Mode | Effect |
|---|---|
| `WAIT` | Stay at the current zone; no travel or energy exchange. |
| `SERVE` | Travel to one zone and serve as many of this MCV's selected customers there as feasible. |
| `REPOSITION` | Travel to a zone without serving or exchanging energy. |
| `RECHARGE` | Travel to an FCS and request grid-input kWh. |
| `DISCHARGE` | Travel to an FCS and request grid-export kWh from the MCV battery through the FCS connection. |

`energy_kwh` is used only for recharge/discharge; it is ignored for the other modes. The simulator clips exchange requests to the feasible physical quantity. The mode/zone action mask does not encode a continuous energy bound or joint station competition.

For travel distance `d`, available operating time and travel energy are

```math
h=\Delta-d/v,\qquad e^{\mathrm{travel}}=\kappa d.
```

Travel must fit within the period and leave at least the battery reserve. A vehicle completes travel within the period; there is no in-transit state. Every vehicle performs at most one operation, and cannot serve multiple zones or recharge and serve in the same period.

Let `eta_M` denote mobile service efficiency and `P_M` the delivered service power. The maximum number of full requests vehicle `m` can serve in zone `z` is

```math
k_{mz}=\left\lfloor\frac{\min\{\eta_M(E_m-e^{\mathrm{travel}}-E_{\min}),\ hP_M\}_+}{q}\right\rfloor.
```

The served count is the smaller of `k_mz` and its selected demand in that zone. Partial EV requests are not served.

A `WAIT` action must name the current zone. `SERVE` must have positive selected demand and admit at least one full request. Recharge/discharge requires an FCS, positive remaining operating time, positive station exchange capacity, and room to charge or available energy to export. Well-formed but individually infeasible actions execute `WAIT` and are listed in `info["invalid_actions"]`. Malformed actions raise an error before state changes.

## FCS capacities and joint fleet exchange

Each FCS has two separate per-period resource banks:

- `fcs_ev_capacity_kwh[f]`: energy delivered to EVs.
- `fcs_fleet_exchange_capacity_kwh[f]`: total grid-input charging plus grid-export discharge of the MCV fleet.

FCS EV service and fleet exchange do not compete for a common limit in this version. FCS EVs are served up to `floor(ev_capacity / q)` requests. The number of charging ports, queues, and customer waiting times are not modeled.

The exchange bank is a per-period energy limit. It does not implement an instantaneous station power limit or schedule exchange start times among vehicles with different travel times.

For `DISCHARGE`, the vehicle supplies energy to the station-side grid through the FCS connection. The exported quantity earns the current grid sell price. This implements the proposal's MCV discharge/grid-support option as a grid-connected energy transaction. It does not charge a modeled FCS battery or directly allocate MCV output to individual EV orders. The FCS has no storage state, and the simulator does not model instantaneous electrical power flows. FCS retail service and MCV export are therefore accounted for as separate grid purchases and exports rather than netted station flows.

After travel, a recharge request is capped by the requested grid kWh, remaining time times charging power, and `(battery_capacity - energy) / charge_efficiency`. A discharge request is capped by requested grid kWh, remaining time times discharge power, and `(energy - reserve) * discharge_efficiency`.

For station `f`, let the resulting grid-side requests be `x_m`, including both charging and export. If their total exceeds station capacity `C_f`, each receives the same proportional factor:

```math
\widehat x_m=x_m\min\left\{1,\frac{C_f}{\sum_{j\text{ at }f}x_j}\right\}.
```

The factor is one if total demand is zero. Charging and discharging share the gross exchange budget; they are not netted. Simultaneous charging and export at an FCS are permitted. There are no station storage dynamics. Grid sell prices cannot exceed grid buy prices under the current scenario validation.

## Energy conservation

Let `Q_m` be EV energy delivered by MCV `m`, `C_m` its allocated grid-input charging, and `X_m` its allocated grid-export discharge. For every period,

```math
E'_m=E_m-e_m^{\mathrm{travel}}+\eta_C C_m
-\frac{Q_m}{\eta_M}-\frac{X_m}{\eta_D}.
```

All final vehicle inventories must be between reserve and battery capacity. The simulator checks this equation numerically on every dispatch and reports its residual in the energy ledger.

The ledger reports fleet exchange both per MCV (`charge_grid_kwh`, `export_grid_kwh`) and aggregated by FCS (`fcs_charge_grid_kwh`, `fcs_export_grid_kwh`). Station vectors follow FCS index order, which is distinct from zone indices.

FCS delivery `Q^F_f` purchases `Q^F_f / eta_F` grid kWh. It does not draw from a modeled FCS battery. Travel energy is drawn from MCV batteries and is separate from the dollar travel cost.

## Profit and objective

Define mobile and FCS retail prices `p_m`, `p^F_f`, and period grid buy/sell prices `g^buy_t`, `g^sell_t`. Let `U_t` be the number of customers who selected an operator service but were not served. Period profit is

```math
\begin{aligned}
r_t={}&\sum_m p_m Q_m+\sum_f p^F_f Q^F_f
+g_t^{\mathrm{sell}}\sum_m X_m\\
&-g_t^{\mathrm{buy}}\left(\sum_m C_m+\sum_f Q^F_f/\eta_F\right)
-c_{\mathrm{travel}}\sum_m d_m\\
&-c_{\mathrm{op}}|\mathcal M|-c_{\mathrm{unmet}}qU_t
+\mathbf1_{\{t=T-1\}}s\sum_m E'_m.
\end{aligned}
```

Operating cost is charged for every MCV every period, including vehicles that wait. Unmet cost includes unserved selected MCV and FCS demand. Outside choices are excluded. `terminal_value_per_stored_kwh = s` values all remaining battery inventory, including reserve, at the final dispatch.

Initial stored energy is provided without a purchase charge: its historical acquisition cost is treated as sunk. Terminal salvage is an explicit modeling choice and changes the incentive to deplete batteries near the horizon. Comparisons of scenarios with different initial batteries should account for that convention. The simulator reports each revenue and cost term separately.

The FCSs and MCVs belong to one integrated operator. Payments between an FCS and an MCV would be internal transfers and are not additional external service revenue. `DISCHARGE` revenue comes from selling exported energy to the grid; `RECHARGE` incurs the grid purchase cost. No separate FCS-to-MCV retail charging fee is included in total operator profit.

The intended objective is

```math
\mathbb E\left[\sum_{t=0}^{T-1}\gamma^t r_t\right],
\qquad \gamma=\texttt{discount\_per\_period}.
```

`cumulative_profit` is the raw sum; `cumulative_discounted_profit` uses the expression above. Pricing transitions return continuation discount `1`; nonterminal dispatch transitions return `gamma`; terminal dispatch returns `0`. A future learner must handle these stage-specific continuation discounts. Applying a fixed gamma to both calls introduces an extra discount within every physical period. Gymnasium's `info` dictionary exposes these values, but a generic RL library will not necessarily consume them automatically.

## Observations and shared dynamics

The full observation includes stage, period, all vehicle locations and energies, current requests, all mobile prices, selected demand, outside count, FCS retail prices, current grid prices, and a dispatch mode/zone mask. The mask is active at dispatch and zero at pricing and termination. Arrays returned to the caller are copies.

`local_observation(m)` is available only at dispatch. It retains vehicle `m`'s location, energy, and action mask, while omitting other MCV locations and energies. It still includes the period, all quoted prices, and aggregate selected demand. Known scenario constants can be shared with agents.

The full state supports a finite-horizon Markov decision model under the specified independent arrival process. Local observations are partial: another MCV's hidden inventory and action can affect station sharing. Independently selected actions still feed into one joint transition and one operator reward. These partial views are not independent local MDPs. A decentralized training protocol, centralized critic, per-agent reward allocation, and learning implementation are not supplied here.

The `ChargingEnv` Gymnasium wrapper preserves these dynamics. Its action dictionary always has four fields: `prices` (floating-point vector), `modes` (integer vector with values 0–4), `targets` (integer zone vector), and `energy_kwh` (floating-point vector). Each vector has one entry per MCV. Pricing reads only `prices`; dispatch reads the other fields. All four must pass the advertised action-space validation, including at stages where some are ignored. One `step()` therefore means one decision stage, not one complete physical period.

## Simplifications to review before research experiments

The initial implementation assumes homogeneous request energy and preferences; independent bounded arrivals; deterministic travel time and energy; a single-period operation per vehicle; no customer retry, backlog, or queue; known grid and fixed retail price schedules; separate FCS service and fleet exchange capacity; proportional fleet exchange allocation; common vehicle physical parameters; no battery degradation; and synthetic choice/cost coefficients.

The proposal does not establish these assumptions. Change the corresponding model and checks before claiming results for a richer physical system. In particular, heterogeneous EV needs, en-route vehicles, shared station power, and waiting customers would require additional state and transition logic rather than just a new parameter value. -->

# Decision framework for a hybrid charging network

This framework describes an integrated operator managing Fixed Charging Stations (FCSs) and Mobile Charging Vehicles (MCVs) to maximize expected cumulative operational profit. MCVs can deliver charging energy to EV customers or export battery energy to the grid through an FCS connection.

The policy symbols below describe proposed learning components. The current simulator implements the system dynamics, observations, and rewards; it does not implement or train these policies.

## 1. Two-stage decision architecture

Each physical period follows a pricing decision, customer choice, and a dispatch decision:

```math
s_t
\xrightarrow{\text{Stage 1: pricing}} p_t^M
\xrightarrow{\text{Customer choice}} D_t
```

```math
\widetilde{s}_t
\xrightarrow{\text{Stage 2: dispatch}} a_t^D
\xrightarrow{\text{System transition}} (r_t,s_{t+1}).
```

### Stage 1: Pricing phase

An abbreviated pricing state is

```math
s_t =
\left(
\{(l_{mt},e_{mt})\}_{m\in\mathcal{M}},
\lambda_t,p_t^F
\right).
```

Here $l_{mt}$ and $e_{mt}$ are vehicle locations and stored energies, $\lambda_t$ denotes realized EV request counts by zone, and $p_t^F$ denotes FCS retail prices. The full implemented state also includes the period index and known grid buy and sell prices.

The operator quotes a price for each MCV using a proposed pricing policy:

```math
p_m\in[p_{\min},p_{\max}],
\qquad
p_t^M\sim\pi^P_{\theta_1}(\cdot\mid s_t).
```

The bounds $p_{\min}$ and $p_{\max}$ correspond to `price_min` and `price_max`.

### Customer choice

Customers choose among available MCVs, FCSs, and an outside option after observing the quoted prices. Their random choices induce endogenous demand:

```math
D_t=D(p_t^M;\xi_t).
```

Here $\xi_t$ represents customer-choice randomness.

### Stage 2: Dispatch phase

After customer choices, the post-choice state is

```math
\widetilde{s}_t=(s_t,p_t^M,D_t).
```

An abbreviated local observation for MCV $m$ and its proposed dispatch policy are

```math
o_{mt}=(l_{mt},e_{mt},p_t^F,p_t^M,D_t),
\qquad
a^D_{mt}\sim\pi^D_{\theta_{2m}}(\cdot\mid o_{mt}).
```

The full local observation also includes the period index and feasibility information. Dispatch advances time by one physical period and returns period profit $r_t$.

## 2. Exogenous request arrivals

EV charging requests arrive independently across zones $z$ and periods $t$. Every customer requests the same energy quantity $q$, specified by `request_kwh`.

The number of requests in zone $z$ at period $t$ follows

```math
N_{tz}\sim\mathrm{Binomial}
\left(
N_{\max},\frac{\mu_z h_t}{N_{\max}}
\right).
```

- $N_{\max}$: maximum requests per zone, `max_requests_per_zone`.
- $\mu_z$: baseline mean requests in zone $z$, `mean_requests_by_zone[z]`.
- $h_t$: arrival-profile multiplier, `arrival_profile[t]`.

Validation requires $0\leq\mu_z h_t/N_{\max}\leq1$.

## 3. Endogenous customer choice model

Customers evaluate available options using a multinomial logit model based on price and distance.

### Utility equations

For a customer in zone $z$, mobile option $m$ has utility

```math
u^M_{zm}=b_M-\alpha q p_m-\delta_M d(L_m,z).
```

Fixed station $f$, located in zone $z_f$, has utility

```math
u^F_{zf}=b_F-\alpha q p^F_f-\delta_F d(z,z_f).
```

The outside option has fixed utility $u_{\mathrm{out}}$.

- $b_M,b_F$: base utility constants.
- $\alpha$: sensitivity to the customer's total charging bill.
- $p_m,p^F_f$: retail prices per delivered kWh.
- $\delta_M,\delta_F$: distance-disutility coefficients.
- $d(\cdot,\cdot)$: Manhattan distance in kilometers.
- $L_m$: current zone of MCV $m$.

For zone coordinates $(x_i,y_i)$ and $(x_j,y_j)$, the distance is

```math
d(i,j)=\left\|(x_i,y_i)-(x_j,y_j)\right\|_1
=|x_i-x_j|+|y_i-y_j|.
```

This assumes travel along an axis-aligned street grid. For example, the distance from $(0,0)$ to $(4,4)$ is $4+4=8$ km. The same distance matrix is used for customer distance penalties, the FCS access radius, MCV travel time, travel energy, travel costs, and dispatch feasibility.

### Choice probabilities

The probability that a customer in zone $z$ chooses option $j$ is

```math
P_{zj}=
\frac{\exp(u_{zj})}
{\sum_{k\in\mathcal{A}_z}\exp(u_{zk})}.
```

The set $\mathcal{A}_z$ contains the available options, including the outside option. An MCV is available only if its battery energy and remaining operating time allow at least one full request. An FCS is available if it lies within the access radius and its EV-service capacity permits at least one full request. Unavailable options have probability zero.

Availability does not guarantee that all customers who select an option will receive service. An FCS can attract more requests than it can fulfill. An MCV can attract requests in several zones but serve only one zone in the period. Unserved customers do not queue, retry another provider, or carry demand forward.

## 4. Dispatch actions and physical constraints

Each MCV chooses one mode and one target zone per period.

| Mode | Function |
| --- | --- |
| `WAIT` | Stay in the current zone without movement or energy exchange. |
| `SERVE` | Travel to a target zone and deliver energy to selected EV customers. |
| `REPOSITION` | Move to a zone without serving or exchanging energy. |
| `RECHARGE` | Travel to an FCS and purchase grid energy to charge the MCV battery. |
| `DISCHARGE` | Travel to an FCS and export battery energy to the grid. |

### Travel time and energy

For distance $d$, the remaining operating time and travel energy consumption are

```math
h=\Delta-\frac{d}{v},
\qquad
e^{\mathrm{travel}}=\kappa d.
```

Here $\Delta$ is period duration (one hour by default), $v$ is speed in km/h, and $\kappa$ is travel energy consumption in kWh/km. Travel must fit within the period and leave the battery reserve intact.

### Serving capacity

The maximum number of full requests MCV $m$ can serve in zone $z$ is

```math
k_{mz}=
\left\lfloor
\frac{
\min\left\{
\eta_M(E_m-e^{\mathrm{travel}}-E_{\min}),\,hP_M
\right\}_+
}{q}
\right\rfloor.
```

Here $E_m$ is stored battery energy, $E_{\min}$ is the reserve, $\eta_M$ is service efficiency, and $P_M$ is delivered service power in kW. The notation $x_+=\max(x,0)$ clips negative capacity to zero. Actual service is limited by both this capacity and the customers who selected that vehicle in the target zone.

## 5. Station capacity and proportional scaling

Fixed Charging Stations (FCSs) manage their energy using two entirely separate budgets per period: one for regular EV customers, and one for Mobile Charging Vehicles (MCVs).

### 1. The two separate energy budgets

- **EV-service capacity:** Energy reserved strictly for regular EV customers.
- **Fleet exchange capacity:** Grid energy reserved for MCVs to either recharge their batteries or discharge energy back to the grid.

These budgets do not interact. An MCV discharging energy back to the grid does not add to the EV-service budget, because stations in this model have no storage batteries; they act as connections to the power grid. Both budgets are measured in kWh per period.

### 2. Serving regular EV customers (all or nothing)

Regular EV customers require a fixed, full amount of energy per charge. There are no partial charges.

If a station has an EV budget of 40 kWh and each customer needs 10 kWh, the station can serve up to four cars per period. If six customers choose that station, four are served and two are turned away as unmet demand. Unused capacity does not roll over to the next period, and unserved customers do not wait in line.

### 3. Processing MCV requests (two-step verification)

When MCVs arrive at a station to charge or discharge, the system calculates their energy allowance in two steps.

#### Step 1: Individual physical limits

Before checking the station's budget, the system clips each MCV's request down to what is physically possible. The energy it can exchange depends on:

- The time it has left in the period, after accounting for travel time.
- Its maximum charging or discharging power. Power multiplied by the remaining time gives the energy it can exchange during that time.
- Its battery limits: it cannot charge past its maximum capacity or discharge below its required minimum reserve. Charging and discharging efficiency are included when converting between grid energy and stored battery energy.

#### Step 2: Proportional station sharing

Once all individual MCV requests are verified, the system adds them up.

Crucially, **charging and discharging both consume the station's fleet exchange budget**. They are added together as **gross energy**, rather than canceling each other out. If the total verified energy request is at or below the station's fleet budget, everyone gets their full verified request.

If total requests exceed the station's fleet budget, every MCV gets scaled back by the exact same percentage.

#### Example of proportional scaling

Suppose the following MCV requests have already passed the individual physical limits in Step 1:

- **Station budget:** 40 kWh.
- **MCV A wants to charge:** 30 kWh.
- **MCV B wants to discharge:** 20 kWh.
- **Total requested:** 50 kWh.

Because the total request (50 kWh) is larger than the budget (40 kWh), the station can only fulfill 80% of the total request: 40 divided by 50.

- MCV A gets exactly 80% of its charging request: **24 kWh**.
- MCV B gets exactly 80% of its discharging request: **16 kWh**.

The station is now maxed out at **40 kWh of gross exchange capacity**. Even though its net grid import is only **8 kWh** (24 kWh pulled from the grid minus 16 kWh pushed back), the system applies the per-period budget to the sum of energy exchanged in both directions. This is a per-period energy limit; the model does not schedule charging ports or enforce a shared instantaneous station power limit.

## 6. Energy conservation and state update

Stored battery energy at the end of the period obeys

```math
E'_m=
E_m-e_m^{\mathrm{travel}}
+\eta_C C_m
-\frac{Q_m}{\eta_M}
-\frac{X_m}{\eta_D}.
```

- $C_m$: allocated grid-input energy used to charge the MCV; $\eta_C$ is charging efficiency.
- $Q_m$: energy delivered to EV customers; $\eta_M$ is service efficiency.
- $X_m$: energy exported to the grid; $\eta_D$ is discharging efficiency.

The feasible final inventory satisfies

```math
E_{\min}\leq E'_m\leq E_{\max},
```

where $E_{\max}$ is the configured battery capacity. Energy quantities are measured in kWh; power quantities are measured in kW.

## 7. Financial objective and reward dynamics

Period profit includes retail revenue, grid transactions, operating costs, unmet-demand penalties, and final-period salvage value:

```math
\begin{aligned}
r_t={}&
\sum_m p_m Q_m+\sum_f p^F_f Q^F_f
+g_t^{\mathrm{sell}}\sum_m X_m\\
&-g_t^{\mathrm{buy}}
\left(\sum_m C_m+\sum_f\frac{Q^F_f}{\eta_F}\right)
-c_{\mathrm{travel}}\sum_m d_m\\
&-c_{\mathrm{op}}|\mathcal{M}|
-c_{\mathrm{unmet}}qU_t
+\mathbf{1}_{\{t=T-1\}}s\sum_m E'_m.
\end{aligned}
```

### Revenue and cost terms

- $\sum_m p_mQ_m+\sum_f p^F_fQ^F_f$: retail revenue from energy delivered to EVs.
- $g_t^{\mathrm{sell}}\sum_mX_m$: grid-export revenue.
- $g_t^{\mathrm{buy}}(\sum_mC_m+\sum_fQ^F_f/\eta_F)$: electricity purchase cost, accounting for FCS service efficiency $\eta_F$.
- $c_{\mathrm{travel}}\sum_md_m$: travel cost.
- $c_{\mathrm{op}}|\mathcal{M}|$: operating cost for every MCV in every period, including vehicles that wait.
- $c_{\mathrm{unmet}}qU_t$: penalty for customers who selected an operator service but were not served; outside-option customers are excluded.
- $\mathbf{1}_{\{t=T-1\}}s\sum_mE'_m$: terminal inventory value at $s$ dollars per stored kWh.

Here $Q^F_f$ is energy delivered to EVs by FCS $f$, $d_m$ is MCV travel distance, and $g_t^{\mathrm{buy}}$ and $g_t^{\mathrm{sell}}$ are the known grid buy and sell prices.

### Long-term optimization objective

The intended learning objective is expected cumulative discounted profit:

```math
\max_{\pi^P,\pi^D}
\mathbb{E}\left[
\sum_{t=0}^{T-1}\gamma^t r_t
\right],
\qquad 0\leq\gamma\leq1.
```

The parameter $\gamma$ corresponds to `discount_per_period`. Discounting applies once per physical period. The pricing transition has continuation discount one; a nonterminal dispatch transition has continuation discount $\gamma$, and the terminal dispatch transition has continuation discount zero.
