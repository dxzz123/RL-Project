# Why customers may choose not to charge

The **outside option** means a customer chooses neither mobile charging nor a fixed charging station (FCS). That customer leaves the operator's market for the current period. The model does not specify what they do instead.

The common charging-benefit parameter $b$ controls how attractive charging is compared with leaving. **Utility** is a preference score: a higher score makes an option more likely to be chosen. The default scenario uses charging utility $-\eta\times\text{cost}$, corresponding to $b=0$. Positive values add a benefit to all charging services. The supplied values of $b$ are experimental assumptions; they have not been estimated from customer data.

Read [MODEL.md](MODEL.md) for the full period sequence and [NOTATION.md](NOTATION.md) for the mathematical symbols and Python names.

## 1. What the charging benefit changes

Suppose one customer can use one charging provider. If charging costs money and leaving has a preference score of zero, the original model gives charging a negative preference score. A positive $b$ adds a benefit to receiving charging service. It can therefore make the customer more willing to charge, even when the price and physical service stay the same.

For example, a perceived charging cost of 5 dollars and cost sensitivity $\eta=0.7$ give charging utility $-3.5$ when $b=0$, and $-0.5$ when $b=3$. The choice formula compares each score with the outside score of zero and assigns a probability to each option. This example concerns a choice probability: it does not guarantee that the customer chooses charging or that capacity will be available to serve them.

The common benefit applies to **both mobile and FCS charging**. It does not enter the operator's revenue, electricity-purchase cost, or profit. Customers who choose the outside option remain lost demand and incur no unmet-service penalty. Customers who choose an operator service but cannot be served are a different case; the existing unmet-demand penalty still applies to them.

## 2. Customer costs, utilities, and probabilities

Consider request $r\in\mathcal R_{i,t}$, which arrives at node $i$ in period $t$ and needs $q_r$ kWh. A node is a location or service zone. The stations that this customer can access form $\mathcal J_{r,t}\subseteq\mathcal J$. Mobile charging is available only if at least one mobile charging vehicle (MCV) meets the physical requirements to offer service at node $i$.

The **perceived cost** combines the charging bill with the customer's travel and waiting inconvenience, expressed in money. For mobile service and FCS service, respectively:

$$
c^M_{r,t}=p^M_{i,t}q_r+h^M_{i,t},
\qquad
c^j_{r,t}=p^{F,C}_{j,t}q_r+h^F_{r,j,t}.
$$

Here $p^M_{i,t}$ is the mobile price, $p^{F,C}_{j,t}$ is station $j$'s customer charging price, and the $h$ terms express inconvenience in dollars. Mobile inconvenience uses the estimated time until a vehicle arrives, $\widehat w_{i,t}$. FCS inconvenience includes travel and the estimated station wait $w^F_{j,t}$. These payment, waiting, and travel costs are unchanged in this experiment.

Let $\mathcal A_{r,t}$ contain the **available charging providers**: mobile option $M$, when available, and accessible FCS nodes $j$. The outside option $O$ is added separately and is always available. Write $c^k_{r,t}$ for the perceived cost of charging option $k\in\mathcal A_{r,t}$. The preference scores are

$$
u^k_{r,t}=b-\eta c^k_{r,t}
\quad(k\in\mathcal A_{r,t}),
\qquad
u^O_{r,t}=0.
$$

$\eta$ is the customer's **cost sensitivity**: a larger value makes an added dollar of cost reduce the preference score more strongly. It is named `price_sensitivity_per_dollar` in Python; its value remains **0.7**. It is not an energy-conversion efficiency. $b$ is dimensionless, meaning it has no physical unit, because utility is a preference score rather than money or energy.

The **multinomial logit** rule converts preference scores into choice probabilities. It applies the exponential function, exp, to turn each score into a positive number, then divides by the total over all options.

$$
\rho^k_{r,t}=
\frac{\exp(b-\eta c^k_{r,t})}
{1+\sum_{k'\in\mathcal A_{r,t}}\exp(b-\eta c^{k'}_{r,t})},
\qquad
\rho^O_{r,t}=
\frac{1}
{1+\sum_{k'\in\mathcal A_{r,t}}\exp(b-\eta c^{k'}_{r,t})}.
$$

The denominator's `1` is $\exp(u^O_{r,t})=\exp(0)$. Unavailable charging providers have probability zero. When there is no charging provider, the sum is empty and the outside probability is one. Exactly one alternative is sampled from $\mathcal A_{r,t}\cup\{O\}$. Leaving is therefore included in the same calculation as every charging option.

### More customers may charge, but the same customer's provider ranking stays the same

For the same customer, costs, and available options, increasing $b$ makes leaving less likely whenever at least one charging provider is available. Among outcomes in which that customer chooses **some** charging service, the common $\exp(b)$ cancels from the formula. The relative chances of choosing the providers still follow $\exp(-\eta c^k_{r,t})$.

This statement applies to one customer under fixed conditions. Across all customers who choose charging, the mobile/FCS shares can still change: increasing $b$ can attract customers with different costs and accessible stations. In a full simulation, those extra customers can also change battery inventory, service availability, and later prices. More customers choosing charging does not guarantee more completed service or greater profit when capacity is limited.

## 3. How it is configured and implemented

| Configuration | Benefit $b$ | Purpose |
|---|---:|---|
| `Scenario()` or an older JSON/checkpoint without this field | 0.0 | Original customer model |
| `configs/toy_city.json` | 0.0 | Explicit original-model scenario |
| `configs/utility_b2.json` | 2.0 | Sensitivity case |
| `configs/utility_b3.json` | 3.0 | Main outside-option experiment |
| `configs/utility_b4.json` | 4.0 | Sensitivity case |

The field is `Scenario.charging_utility_intercept`. **Intercept** means the amount added to the preference score before subtracting the cost effect. It must be a finite real number. Negative values are also supported and make charging less attractive compared with leaving. The supplied values are experimental assumptions and have not been fitted to customer data.

Both `ChargingSimulator.choice_probabilities` and the adaptive pricing benchmark use the same helper in `src/charging_env/choice.py`. It calculates **softmax**, the conversion from scores to probabilities used above. It first shifts the scores so exponentiation does not produce numbers too large for the computer. It handles empty request lists and several candidate prices together, and gives unavailable providers exactly zero probability. This customer-choice calculation is separate from the activations inside the neural networks and the probabilities those networks use to choose actions.

A **checkpoint** is a saved snapshot of a trained model and its training state. It includes the scenario and $b$, which are restored when training resumes. The command-line interface (CLI) cannot replace a checkpoint's scenario with `--scenario`, and the benchmark code rejects changing the intercept when evaluating a learned policy. A b=0 checkpoint therefore remains a b=0 model; train a new b=3 model for a comparison within the changed customer market.

## 4. What has been checked

The tests cover the probability formula, unavailable providers, empty request lists, numerical stability at very large positive and negative benefits, scenario loading, and checkpoint/scenario compatibility. A controlled experiment also evaluated the same 5,334 reference requests at several values of $b$, holding prices, accessibility, waiting estimates, and fleet states fixed:

| Charging benefit $b$ | Expected outside share | Expected charging participation |
|---|---:|---:|
| 0 | 92.3744% | 7.6256% |
| 2 | 67.1085% | 32.8915% |
| 3 | 47.1786% | 52.8214% |
| 4 | 27.9692% | 72.0308% |

These are averages of request-level choice probabilities from 32 reference episodes under the original market. They isolate the formula's effect. They are not measurements of completed service or profit in new b=3 episodes. The saved [summary](../outputs/choice_intercept_check/summary.json) and [protocol](../outputs/choice_intercept_check/protocol.json) record the check; [VALIDATION.md](VALIDATION.md) records the current test suite.

## 5. Use the intended market in a new run

Pass the scenario explicitly. Creating or editing a JSON file does not make a script use it automatically. For example, this small run trains the joint PPO in the b=3 market:

```bash
source .venv/bin/activate
python -m charging_env.rl.paper_train \
  --scenario configs/utility_b3.json \
  --updates 10 --episodes-per-update 4 --seed 7 \
  --output-dir outputs/joint_ppo
```

The standalone training commands automatically select a numbered sibling directory if the requested directory contains an earlier run. You can reuse the same command. A complete controlled study uses stricter output rules so its selected models and test evidence stay together; [TRAINING.md](TRAINING.md) explains both workflows.

For other methods, follow the commands in [FITTED_VALUE.md](FITTED_VALUE.md) and [LEGACY_PPO.md](LEGACY_PPO.md), again passing `configs/utility_b3.json` when training a new model. Resuming a checkpoint restores its saved scenario instead of substituting a different customer model.

## 6. Compare controllers within the same market

Increasing $b$ changes customer participation. Higher profit in b=3 than in b=0 would therefore combine a market change with any improvement in the controller. To compare controllers, train them for the same scenario, choose rule settings using separate validation episodes, then evaluate all methods on the same fresh test scenarios.

The current comparison uses b=3 throughout and includes approximate DP, joint PPO, earlier PPO, and three useful rules. [COMPARISON.md](COMPARISON.md) explains what each method knows, what it optimizes, and how it performed. The earlier [SiLU + b=3 study](archive/SILU_B3_STUDY.md) and [targeted PPO improvements](archive/B3_TRAINING_IMPROVEMENTS.md) remain available as research history. Their separate test sets should not be mixed into the current comparison.
