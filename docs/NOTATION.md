# Notation and the link to Python

This guide explains how to read the equations and find the same quantities in the code. The physical-model notation comes from the earlier attachment, *Dynamic Pricing and Dispatching for Mobile Charging Services: An MDP-Based Two-stage RL Framework* (`MDP_RL (1).pdf`), with the clarifications in Section 5. The later four-page algorithm attachment is saved as [MDP_RL_2.pdf](MDP_RL_2.pdf); [PAPER_PPO.md](PAPER_PPO.md) explains its training equations. These are two different source documents. Physical assumptions added in the simulator are identified below rather than attributed to either manuscript.

Start with [MODEL.md](MODEL.md) for the system's story and [ENVIRONMENT.md](ENVIRONMENT.md) for the Python interface. Use this page as a reference when a symbol or array index is unclear. In the tables, **Symbol** is the notation used in equations, **Meaning** explains it, **Unit or type** says what kind of quantity it is, and the Python column shows where to find it in code. A set is a collection of objects; an index identifies one object in that collection.

## 1. Four distinctions that prevent most confusion

**A node number and a station-array position can refer to different places.** A node is a place where demand may originate. Some nodes also contain a fixed charging station (FCS). In equations, $i$ is a general node and $j$ is a station's node. Python stores station data in a shorter array indexed by station slot `f`. The conversion is `j = scenario.fcs_zones[f]`.

**Arriving requests, choices, and completed service are different events.** A request first arrives, then chooses mobile charging, an FCS, or the outside option. Choosing a service does not guarantee that capacity will be available to fulfill it. A request count and the energy requested by those customers are also different quantities.

**A period contains two decisions.** The operator first posts mobile prices. Customers then choose. The vehicles then receive dispatch decisions. Only after dispatch does the simulator compute the period's profit and advance time.

**Where energy is measured matters.** Energy delivered to a customer, energy entering a vehicle from the grid, and energy stored in its battery are all measured in kWh. Charging and discharging losses make their numerical values different. For example, 10 kWh drawn from the grid increases stored energy by 9.5 kWh when charging efficiency is 0.95.

## 2. Places, vehicles, time, and prices

| Symbol | Meaning | Unit or type | Python name or representation |
|---|---|---|---|
| $\mathcal I$; $i$ | All locations or service zones; one such location | Set; node label | `range(n_zones)`; node index |
| $\mathcal J\subseteq\mathcal I$; $j$ | Nodes containing an FCS; one such node | Set; node label | `fcs_zones`; `fcs_zones[f]` |
| $\mathcal M$; $m$ | Mobile charging vehicles (MCVs); one vehicle | Set; vehicle label | `range(n_mcv)`; vehicle index |
| $\mathcal T=\{1,\ldots,T\}$; $t$ | All operating periods in an episode; the current period in equations | Period index | `horizon`; `period = t - 1` while an episode is active |
| $\Delta t$ | Duration of one period | Hours | `period_hours` |
| $l_{m,t}$ | Vehicle $m$'s node at the start of period $t$ | Node label | `locations[m]` |
| $e_{m,t}$; $B$ | Stored battery energy; battery capacity | kWh | `energy_kwh[m]`; `battery_capacity_kwh` |
| $\tau_{i,i'}$ | Travel time between any two nodes $i$ and $i'$ | Hours | `distances[i, destination] / speed_kmph` |
| $\mathcal L$; $\ell$ | Allowed mobile price levels; one level | Set; level index | `price_levels`; index into that tuple |
| $p_\ell$ | Unit price associated with level $\ell$ | Currency/kWh | `scenario.price_levels[level_index]` |
| $p^M_{i,t}$ | Mobile charging price posted at node $i$ | Currency/kWh | `mobile_prices[i]` |
| $p^{F,C}_{j,t}$ | FCS $j$'s customer charging price | Currency/kWh | `fcs_retail_prices[f]` |
| $p^{F,D}_{j,t}$ | Price paid for fleet energy exported through FCS $j$ | Currency/kWh | `fcs_discharge_prices[f]` |

Here $l$ is the letter “ell” for a vehicle location. We write the price-level index as $\ell$ to make the two easier to distinguish. Neither one is the number 1.

The station mapping deserves a concrete example. With `fcs_zones = (0, 3)`, station slot `f = 1` refers to the station at node `j = 3`. Therefore `fcs_retail_prices[1]` is the price at node 3; it is not the price at node 1. Node arrays, such as `mobile_prices`, still use the full node index.

Similarly, a **price-level index** is not a price. With the default menu, level index `4` means a price of `0.60` per kWh. `quote_price_levels(...)` accepts indices, whereas `quote_prices(...)` accepts the corresponding monetary prices.

The mathematical first period is $t=1$ and the Python first period is `period=0`. After the final dispatch, Python reports `period=horizon` and `stage=2` to mark termination. That observation does not describe another operating period.

## 3. Requests, waiting estimates, and customer choice

| Symbol | Meaning | Unit or type | Python name or representation |
|---|---|---|---|
| $\lambda_{i,t}$ | Expected number of new requests at node $i$ in period $t$ | Requests per period | `arrival_rates[i]` in an observation; `resolved_arrival_rates[t - 1, i]` in the scenario |
| $\mathcal R_{i,t}$; $N_{i,t}$ | Actual arriving requests; their number | Set; count | Rows of `request_details` with origin $i$; `requests_by_zone[i]` |
| $r$; $q_r$ | One actual request; its required energy | Request label; kWh | Request row; `request_energy_kwh[r]` internally |
| $\bar d_{i,t}$ | Total energy requested at node $i$ before customers choose | kWh | `potential_energy_kwh[i]` |
| $\mathcal J_{r,t}$ | FCSs accessible to request $r$ | Set of station nodes | Accessibility columns in `request_details`, ordered by station slot |
| $\mathcal M_{i,t}$ | Vehicles meeting the physical requirements to offer service at node $i$ | Set of vehicles | `eligible_mcvs[:, i]` |
| $\widehat w_{i,t}$ | Estimated mobile arrival delay before dispatch | Hours | `mobile_wait_hours[i]` |
| $w^F_{j,t}$ | Estimated customer queueing delay at FCS $j$ | Hours | `fcs_wait_hours[f]` |
| $h^M_{i,t}$; $h^F_{r,j,t}$ | Mobile waiting inconvenience in money; FCS travel and waiting inconvenience in money | Currency | Computed inside `choice_probabilities(...)` |
| $c^M_{r,t}$; $c^j_{r,t}$ | Request $r$'s total perceived cost of mobile service; of FCS $j$ | Currency | `mobile_cost`; station column of `fixed_cost` |
| $\rho^k_{r,t}$; $z^k_{r,t}$ | Probability of choosing option $k$; flag recording whether it was chosen | Probability; 0 or 1 | Probability matrix; test on `request_choices` |
| $\mathcal R^M_{i,t}$; $\mathcal R^F_{j,t}$ | Requests that chose mobile service at $i$; requests that chose FCS $j$ | Sets of requests | Choice/origin filters on request rows |

Requests are sampled before prices are posted:

$$
N_{i,t}\sim\operatorname{Poisson}(\lambda_{i,t}),\qquad
N_{i,t}=|\mathcal R_{i,t}|,\qquad
\bar d_{i,t}=\sum_{r\in\mathcal R_{i,t}}q_r.
$$

Thus $\lambda_{i,t}=4$ means four requests **on average**, rather than exactly four requests. The actual count may be 2 or 7, for example. The raw simulator observation contains both the configured average and the current actual requests. The earlier PPO can use both for pricing. The manuscript PPO intentionally hides actual requests and request-dependent waiting estimates from its pricing actor and critic; it uses actual requests only after customer choice for dispatch. The simulator does not cap the random count using `max_requests_per_zone`; that setting is retained only for compatibility with older configurations.

For a request originating at node $i$, the manuscript's perceived costs are

$$
c^M_{r,t}=p^M_{i,t}q_r+h^M_{i,t},\qquad
c^j_{r,t}=p^{F,C}_{j,t}q_r+h^F_{r,j,t}.
$$

The superscript $j$ in $c^j_{r,t}$ identifies a particular FCS; it is the manuscript's notation for station cost. The code converts waiting hours and travel kilometres into money before adding them to the charging bill. Customers compare these total costs, not unit prices alone.

Let $\mathcal A_{r,t}$ contain the **available operator services**: accessible FCSs and mobile service if mobile service is available at the request's origin. The outside option $O$ is always available. **Utility** is a preference score used to calculate choice probabilities. The manuscript takes leaving as the reference option and sets its utility to zero; this choice of reference is called normalization. Then

$$
\rho^k_{r,t}=\frac{\exp(-\eta c^k_{r,t})}
{1+\sum_{a\in\mathcal A_{r,t}}\exp(-\eta c^a_{r,t})},
\quad k\in\mathcal A_{r,t},\qquad
\rho^O_{r,t}=\frac{1}
{1+\sum_{a\in\mathcal A_{r,t}}\exp(-\eta c^a_{r,t})}.
$$

The parameter $\eta>0$ is **cost sensitivity**: a larger value makes an added dollar of perceived cost reduce a service's preference score more strongly. Its units are inverse currency, so multiplying it by a monetary cost gives a score with no physical unit. An unavailable service has probability zero. Each request chooses exactly once:

$$
\sum_{k\in\mathcal A_{r,t}\cup\{O\}}z^k_{r,t}=1.
$$

The later **common charging-benefit** experiment replaces every available service utility $-\eta c^k_{r,t}$ with $b-\eta c^k_{r,t}$ and leaves outside utility at zero. Here $b$ is dimensionless, meaning it has no physical unit, and maps to `charging_utility_intercept`. **Intercept** means the amount added to a score before subtracting the cost effect. The default $b=0$ reproduces the manuscript's equation; $b=3$ is an explicit extension. See [OUTSIDE_OPTION.md](OUTSIDE_OPTION.md) for an example of how it changes participation.

Python encodes choices as follows:

| Stored choice | Meaning |
|---|---|
| `-1` | The request has not chosen yet |
| `0` | Mobile service at the request's origin |
| `f + 1` | The FCS in station slot `f`, at node `fcs_zones[f]` |
| `n_fcs + 1` | Outside option |

`mobile_demand` and `fixed_demand` are **counts of customers who chose** each service. `mobile_requested_kwh` and `fixed_requested_kwh` sum those customers' energy requirements. Completed service is recorded separately in `served_request_mask` and the delivered-energy fields. Outside requests have declined the operator's services; unmet requests chose an operator service but were not fulfilled.

## 4. States, actions, profit, and learning

| Symbol | Meaning | Code counterpart |
|---|---|---|
| $s_t$ | Information used before mobile prices and customer choices | Raw observation has `stage=0`; each controller selects its permitted inputs |
| $\mathbf p^M_t$ | Posted mobile prices across nodes | Pricing action, then `mobile_prices` |
| $D_t$ | Customers' actual choices of mobile service or an FCS | Request choices and demand fields with `stage=1` |
| $\widetilde s_t$ | Information after customer choice and before vehicle dispatch, including requests, posted prices, and choices | Observation with `stage=1` |
| $o_{m,t}$; $\zeta_{m,t}$ | Earlier PPO: dispatch observation of vehicle $m$; information it receives about peers | `local_observation(m)`; `other_locations` and `other_energy_kwh` |
| $a^D_{m,t}$; $\mathbf a^D_t$ | One vehicle's dispatch action; joint fleet action | `DispatchAction`; one action per vehicle |
| $r_t$ | Total operator profit for the completed period | Dispatch reward; itemized in `reward_terms` |
| $\gamma$ | Discount applied when advancing one physical period | `discount_per_period` |
| $\pi^P_{\theta_1}$; $\pi^D_{\theta_2}$ | Learned pricing and dispatch decision rules | Separate pricing/dispatch actors; architecture depends on the PPO implementation |
| $V^P$; $V^D$ | Expected remaining discounted profit before choice; after choice | Pricing critic; dispatch critic |
| $\lambda_{\mathrm{GAE}}$ | Setting controlling how training estimates whether an action did better than expected | `gae_lambda`; unrelated to request intensity $\lambda_{i,t}$ |

The first decision changes customer choices; the second changes the physical fleet and produces profit:

$$
s_t\ \xrightarrow{\ \mathbf p^M_t\ }\ 
\text{customer choice}\ \longrightarrow\ \widetilde s_t
\ \xrightarrow{\ \mathbf a^D_t\ }\ (r_t,s_{t+1}).
$$

The manuscript calls these **Stage 1** and **Stage 2**. Python uses integer `stage=0` for pricing, `stage=1` for dispatch, and `stage=2` only for the terminal observation. A period is not the same as one call to `step(...)`.

The objective and stage values are

$$
J=\mathbb E\!\left[\sum_{t=1}^{T}\gamma^{t-1}r_t\right],\qquad
V^P(s_t)=\mathbb E[V^D(\widetilde s_t)\mid s_t],\qquad
V^D(\widetilde s_t)=\mathbb E[r_t+\gamma V^P(s_{t+1})\mid\widetilde s_t].
$$

The **value** is the expected remaining return under the policies: the average sum of future profits, accounting for discounting. A critic network learns to predict it. The **horizon** is the episode's final operating period; nothing is earned after it, so the remaining value is zero. **Discounting** gives later profit less weight. There is no discount between pricing and dispatch within the same period. In Python the identical objective is `sum(gamma ** period * reward)` over `period=0,...,horizon-1`. The manuscript PPO writes these stage values as $V_1$ and $V_2$. Training details are in [PAPER_PPO.md](PAPER_PPO.md) and [LEGACY_PPO.md](LEGACY_PPO.md); optional reward shaping belongs to the earlier PPO implementation. Reported economic profits continue to use $r_t$.

### Translating the newer PPO algorithm

| Later manuscript | Meaning in this project |
|---|---|
| Stage 1 / Stage 2 | Pricing / dispatch; Python uses `stage=0` / `stage=1` |
| $R_t^p$ / $R_t^d$ | Pricing reward / dispatch reward; the simulator uses $0$ / $r_t$ |
| $S_t$ / $\widetilde S_t$ | Newer manuscript notation for the pricing / dispatch information, also written $s_t$ / $\widetilde s_t$ above |
| $V_1$ / $V_2$ | Pricing critic / dispatch critic, also written $V^P$ / $V^D$ above |
| $\hat A_{1,t}$ / $\hat A_{2,t}$ | Pricing / dispatch advantage estimates used to update the two actors |
| $\lambda$ in GAE | `gae_lambda`, written $\lambda_{\mathrm{GAE}}$ here to distinguish it from Poisson arrival means |
| Product of vehicle probability ratios | One joint fleet ratio, clipped once in the manuscript PPO dispatch loss |

Equation (11) in the later attachment uses the dispatch advantage $\hat A_{2,t}$ for the dispatch critic target. That is the default implementation. The optional `pricing_advantage` target substitutes $\hat A_{1,t}$ as an experiment; it is not the manuscript equation.

The newer PPO's four operations are **service, idle/move, charge, and discharge**. The simulator uses five mode codes because it separates idle-at-current-node (`WAIT`) from idle-at-another-node (`REPOSITION`). The adapter in `paper_ppo.py` performs this translation. The physical simulator has not acquired a second set of dynamics.

Approximate DP uses a different object: a table of expected future profit indexed by period, fleet locations, and a battery-energy grid. Its notation, interpolation, and Bellman update are explained in [FITTED_VALUE.md](FITTED_VALUE.md). It has no PPO advantage or policy-ratio loss.

### Additional symbols for the implemented energy accounting

The manuscript describes the profit categories but does not fully specify how to account for each energy flow or allocate limited capacity. The following symbols describe the physical details added in the simulator; these equations are not attributed to the manuscript.

| Symbol | Meaning | Unit |
|---|---|---|
| $C_{m,t}$ | Grid energy drawn to recharge vehicle $m$ | Grid-side kWh |
| $X_{m,t}$ | Energy exported from vehicle $m$ to the grid | Grid-side kWh |
| $Q_{m,i,t}$; $Q_{m,t}$ | Energy delivered by vehicle $m$ at node $i$; its total delivery $\sum_i Q_{m,i,t}$ | Customer-side kWh |
| $Q^F_{j,t}$ | Energy delivered to customers at FCS $j$ | Customer-side kWh |
| $E^{\mathrm{travel}}_{m,t}$ | Stored battery energy consumed by travel | Battery-side kWh |
| $\eta_C,\eta_M,\eta_D,\eta_F$ | Charging, mobile service, discharging, and FCS service efficiencies | Fractions with no physical unit |
| $e_{\min}$ | Minimum stored battery reserve | Battery-side kWh |
| $g^{\mathrm{buy}}_t$ | Price the operator pays to buy electricity from outside the fleet | Currency/grid-side kWh |
| $U_t$ | Energy requested by operator-service customers who remain unserved | Customer-side kWh |
| $\nu$ | Optional value assigned to stored energy at the end of the horizon | Currency/battery-side kWh |

For example, the fleet battery update uses

$$
e_{m,t+1}=e_{m,t}-E^{\mathrm{travel}}_{m,t}
+\eta_C C_{m,t}-\frac{Q_{m,t}}{\eta_M}
-\frac{X_{m,t}}{\eta_D}.
$$

An efficiency always has a subscript here, such as $\eta_C$. The unsubscripted $\eta$ is customer cost sensitivity. The lower battery bound $e_{\min}$ maps to `reserve_kwh`, and $\nu$ maps to `terminal_value_per_stored_kwh`. Serving requests only in full and assigning a value to battery energy left at the episode's end are also implementation assumptions. Their effects are explained in [MODEL.md](MODEL.md).

## 5. Clarifications of the earlier model attachment

The section/page references below refer only to `MDP_RL (1).pdf`, not the newer PPO algorithm attachment. They record how the original simulation interprets that model. Neither source PDF has been changed.

1. **Use $\lambda_{i,t}$ for the average arrival count.** Section 4.1 describes $\boldsymbol\lambda_t$ as potential requests, whereas Section 5 defines it as the expected count used in a Poisson draw. These are different objects. This documentation uses $\mathcal R_{i,t}$ for actual requests. The raw simulator exposes their contents before pricing, an implementation choice used by the earlier PPO. The newer manuscript PPO restricts pricing to arrival means and other permitted state variables.

2. **Include outside choice in the indicator sum.** On page 2, $\mathcal A_{r,t}$ is defined to contain available charging services, and the probability denominator adds the outside option as a separate `1`. Therefore the subsequent sum over choice indicators must include $O$, as written above. Otherwise a customer who leaves would make that sum zero. Another valid convention would include $O$ in the set of options and change the probability formula to match. This guide keeps the original definition, in which that set contains charging services only.

3. **An FCS receives customers from multiple origin nodes.** The page 2 expression $\mathcal R^F_{j,t}\subseteq\mathcal R_{i,t}$ refers to an origin $i$ without specifying which one. A station's customers instead come from the full collection of requests:

   $$
   \mathcal R^F_{j,t}\subseteq\mathcal R_t,\qquad
   \mathcal R_t:=\bigcup_{i\in\mathcal I}\mathcal R_{i,t}.
   $$

   A particular FCS request retains its own origin, which determines travel cost and accessibility. In contrast, $\mathcal R^M_{i,t}\subseteq\mathcal R_{i,t}$ is correct: mobile service occurs at the request's origin.

4. **Keep the calendar in the state.** Section 3.1 says the current period is observed, but the displayed state in Section 4.1 omits it. The implementation explicitly observes the period. This matters because the number of periods left, the time-dependent demand multiplier, and electricity prices depend on time. Whether a pricing policy may observe actual requests is a separate information assumption. The raw simulator and earlier PPO expose them; the newer manuscript PPO does not.

5. **Separate outside participation from failed fulfillment.** Section 5.3 uses “unserved demand” beside outside-option share, which can confuse two different outcomes. `outside_option_share` counts customers who selected $O$. Unmet-demand costs apply to customers who selected mobile or FCS service but were not served. They are separate populations.

6. **Identify the added charging benefit and physical details as extensions.** The attachment uses service utility $-\eta c$ and does not specify $b$, the rule for serving the first request that fits the remaining capacity, the detailed accounting for energy losses, the battery reserve, or a formula valuing energy left at the end. Their documentation must identify them as explicit implementation choices or later experiments. The default intercept and terminal value are both zero.

7. **Use one mathematical convention while documenting the API translation.** The manuscript's $t=1,\ldots,T$, lower-case battery energy $e_{m,t}$, and station-node price subscripts are retained in equations. The tables explain how these symbols correspond to Python periods starting at zero and arrays indexed by station position. The equations keep the manuscript's notation. This clarification changes the documentation, not the simulator's indexing.

The earlier model manuscript's numerical sections contain placeholders and an illustrative comparison table. Those entries are not results produced by this repository. Reproducible empirical results are recorded in the study documents and saved outputs linked from the README.
