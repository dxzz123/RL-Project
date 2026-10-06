<!-- # Decision model and simulator contract

This document describes the implemented model, rather than a learning algorithm. The basic procedure is pricing by the operator followed by endogenous user choices and decentralized MCV dispatch. This document specifies the demand distributions, choice utilities, service duration, capacities, or energy accounting. The choices below settle those details for an initial simulator and are adjustable research assumptions.

## Entities, units, and defaults

The operator owns fixed charging stations (FCSs) and mobile charging vehicles (MCVs). Zones have coordinates in kilometers; travel distance is Euclidean distance between zone coordinates. The default scenario has four zones, two FCSs, two MCVs, and 12 one-hour periods. Vehicle and station indices start at zero.

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

This decision framework models an integrated operator managing a hybrid network of Fixed Charging Stations (FCSs) and Mobile Charging Vehicles (MCVs) to maximize long-term operational profit. MCVs serve a dual role: delivering flexible charging directly to EV customers or discharging power back to the grid for extra revenue.

---

## 1. Two-Stage Decision Architecture

Each period $t$ operates through a two-stage sequential decision process:

```
   [State s_t] ───> Stage 1: Pricing p_t^M ───> Customer Choice D_t ───> Stage 2: Dispatch a_t^D ───> [Reward r_t, State s_{t+1}]

```

1. **Stage 1: Pricing Phase**
* The operator observes system state $s_t = (\{l_{mt}, e_{mt}\}_{m \in \mathcal{M}}, \lambda_t, p^F_t)$, where $l_{mt}$ and $e_{mt}$ are vehicle locations and energies, $\lambda_t$ are EV arrivals, and $p^F_t$ are FCS prices.


* The operator sets retail price $p_m \in [\text{price\_min}, \text{price\_max}]$ for each MCV via pricing policy $\pi^P_{\theta_1}(\cdot \vert{} s_t)$.




2. **Customer Decision**
* Customers observe quoted prices and decide among available MCVs, FCSs, or an outside option. This induces endogenous demand $D_t = D(p^M_t; \xi_t)$.




3. **Stage 2: Dispatch Phase**
* After choice realization, the post-choice state is $\tilde{s}_t = (s_t, p^M_t, D_t)$.


* Each MCV $m$ receives local observation $o_{mt} = (\{l_{mt}, e_{mt}\}, p^F_t, p^M_t, D_t)$ and selects an action $a^D_{mt} \sim \pi^D_{\theta_{2m}}(\cdot \vert{} o_{mt})$.


* Time advances by 1 period, returning physical reward $r_t$.





---

## 2. Exogenous Request Arrivals

EV charging demand arrives independently across zones $z$. Each customer requests a uniform energy quantity $q = \text{request\_kwh}$.

The request count $N_{tz}$ in zone $z$ at period $t$ follows a Binomial distribution:

$$N_{tz} \sim \mathrm{Binomial}\left(N_{\max}, \frac{\mu_z h_t}{N_{\max}}\right)$$

* $N_{\max}$: Maximum allowed requests per zone (`max_requests_per_zone`).
* $\mu_z$: Zone mean request baseline (`mean_requests_by_zone[z]`).
* $h_t$: Time-of-day arrival profile multiplier (`arrival_profile[t]`).

---

## 3. Endogenous Customer Choice Model

Customers evaluate available choices using a multinomial logit framework based on price and travel distance.

### Utility Equations

* **Mobile Option ($m$) for customer in zone $z$:**

$$u^M_{zm} = b_M - \alpha q p_m - \delta_M d(L_m, z)$$


* **Fixed Station ($f$) in zone $z_f$ for customer in zone $z$:**

$$u^F_{zf} = b_F - \alpha q p^F_f - \delta_F d(z, z_f)$$


* **Outside Option:** Fixed utility $u_{\text{out}}$.

**Parameters:**

* $b_M, b_F$: Base utility constants for mobile and fixed options.
* $\alpha$: Price-sensitivity coefficient.
* $p_m, p^F_f$: Retail prices per delivered kWh.
* $\delta_M, \delta_F$: Distance friction coefficients.
* $d(\cdot)$: Euclidean distance (km) between locations.
* $L_m$: Zone location of MCV $m$.

### Choice Probability (Multinomial Logit)

The probability $P_{zj}$ that a customer in zone $z$ selects option $j$ is:

$$P_{zj} = \frac{\exp(u_{zj})}{\sum_{k \in \mathcal{A}_z} \exp(u_{zk})}$$

Where $\mathcal{A}_z$ is the set of **available** choices. A mobile option is available only if the MCV has enough battery power and time to serve at least one full request $q$. An FCS is available if within access radius and its EV capacity admits at least one request $q$. Unavailable options have probability zero.

---

## 4. Dispatch Actions & Physical Constraints

Each MCV selects one dispatch mode per period:

| Mode | Function |
| --- | --- |
| `WAIT` | Remain in current zone; zero movement or energy exchange. |
| `SERVE` | Travel to target zone and deliver power to selected EV demand. |
| `REPOSITION` | Relocate to a zone without serving or exchanging energy. |
| `RECHARGE` | Travel to an FCS and draw charging power from the grid. |
| `DISCHARGE` | Travel to an FCS and sell battery energy back to the grid.

 |

### Travel Time & Energy Consumption

For travel distance $d$:

* **Available operating time ($h$):** $h = \Delta - \frac{d}{v}$
* **Travel energy consumed ($e^{\text{travel}}$):** $e^{\text{travel}} = \kappa d$

Where $\Delta$ is period duration (1 hour), $v$ is travel speed (km/h), and $\kappa$ is consumption rate (kWh/km).

### Serving Capacity

The maximum number of full customer requests $k_{mz}$ vehicle $m$ can fulfill in zone $z$ is:

$$k_{mz} = \left\lfloor \frac{\min\left\{ \eta_M (E_m - e^{\text{travel}} - E_{\min}), \, h P_M \right\}_+}{q} \right\rfloor$$

Where $E_m$ is battery inventory, $E_{\min}$ is battery reserve, $\eta_M$ is service delivery efficiency, and $P_M$ is service delivery power output (kW). Unserved customers do not queue, retry, or carry forward.

---

## 5. Station Capacity & Proportional Scaling

Each FCS $f$ maintains a fixed grid-exchange limit $C_f$ (`fcs_fleet_exchange_capacity_kwh`) per period for MCV fleet charging and discharging.

If total requested grid energy from arriving MCVs exceeds capacity $C_f$, requests $x_m$ are scaled down proportionally:

$$\widehat{x}_m = x_m \min\left\{1, \, \frac{C_f}{\sum_{j \text{ at } f} x_j}\right\}$$

Charging and discharging share this gross energy budget without netting.

---

## 6. Energy Conservation State Update

Vehicle stored energy $E'_m$ at the end of the period strictly obeys mass-balance conservation:

$$E'_m = E_m - e_m^{\text{travel}} + \eta_C C_m - \frac{Q_m}{\eta_M} - \frac{X_m}{\eta_D}$$

* $C_m$: Allocated grid energy charged into MCV ($\eta_C$ is charge efficiency).
* $Q_m$: Delivered energy to EV customers ($\eta_M$ is service efficiency).
* $X_m$: Allocated grid discharge energy ($\eta_D$ is discharge efficiency).

Feasibility requires $E_{\min} \le E'_m \le \text{battery\_capacity}$.

---

## 7. Financial Objective & Reward Dynamics

The period profit $r_t$ consolidates all operator revenue and cost streams:

$$\begin{aligned} r_t = {}& \sum_m p_m Q_m + \sum_f p^F_f Q^F_f + g_t^{\text{sell}} \sum_m X_m \\ &- g_t^{\text{buy}} \left(\sum_m C_m + \sum_f \frac{Q^F_f}{\eta_F}\right) - c_{\text{travel}} \sum_m d_m \\ &- c_{\text{op}} \vert{}\mathcal{M}\vert{} - c_{\text{unmet}} q U_t + \mathbf1_{\{t=T-1\}} s \sum_m E'_m \end{aligned}$$

### Revenue & Cost Breakdown

* **$\sum_m p_m Q_m + \sum_f p^F_f Q^F_f$**: Retail revenue from mobile and fixed EV service.


* **$g_t^{\text{sell}} \sum_m X_m$**: Revenue from grid energy discharge sales at grid sell price $g_t^{\text{sell}}$.


* **$g_t^{\text{buy}} \left(\sum_m C_m + \sum_f \frac{Q^F_f}{\eta_F}\right)$**: Wholesale electricity purchase cost for MCVs and FCSs.
* **$c_{\text{travel}} \sum_m d_m$**: Vehicle travel/wear-and-tear costs.
* **$c_{\text{op}} \vert{}\mathcal{M}\vert{}$**: Fixed operational fee per active MCV.
* **$c_{\text{unmet}} q U_t$**: Penalty fine for total unserved customer demand $U_t$.
* **$\mathbf1_{\{t=T-1\}} s \sum_m E'_m$**: Terminal salvage valuation ($s$ dollars/kWh) applied strictly at the final period $t = T-1$.

### Long-Term Optimization Objective

The reinforcement learning policy is trained to maximize expected cumulative discounted profit:

$$\mathbb{E} \left[ \sum_{t=0}^{T-1} \gamma^t r_t \right]$$

Where $\gamma = \text{discount\_per\_period} \in [0, 1]$.