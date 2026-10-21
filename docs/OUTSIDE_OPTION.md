# Why customers may choose not to charge

The **outside option** means a customer chooses neither mobile charging nor a fixed charging station (FCS). That customer leaves the operator's market for the current period. The model does not specify what they do instead.

This page explains a later experiment that changes how attractive charging is compared with leaving. **Utility** is a preference score: a higher score makes an option more likely to be chosen. The attached manuscript and the original scenario use charging utility $-\eta\times\text{cost}$, equivalent to the special case $b=0$ below. On 2026-10-17, the project added a common charging-benefit parameter $b$. This is an added assumption for experiments; its value was not estimated from customer data or supplied by the manuscript.

Read [MODEL.md](MODEL.md) for the full period sequence and [NOTATION.md](NOTATION.md) for the mathematical symbols and Python names.

## 1. What the new parameter changes

Suppose one customer can use one charging provider. If charging costs money and leaving has a preference score of zero, the original model gives charging a negative preference score. A positive $b$ adds a benefit to receiving charging service. It can therefore make the customer more willing to charge, even when the price and physical service stay the same.

For example, a perceived charging cost of 5 dollars and cost sensitivity $\eta=0.7$ give charging utility $-3.5$ when $b=0$, and $-0.5$ when $b=3$. The choice formula compares each score with the outside score of zero and assigns a probability to each option. This example concerns a choice probability: it does not guarantee that the customer chooses charging or that capacity will be available to serve them.

The common benefit applies to **both mobile and FCS charging**. It does not enter the operator's revenue, electricity-purchase cost, or profit. Customers who choose the outside option remain lost demand and incur no unmet-service penalty. Customers who choose an operator service but cannot be served are a different case; the existing unmet-demand penalty still applies to them.

## 2. Customer costs, utilities, and probabilities

Consider request $r\in\mathcal R_{i,t}$, which arrives at node $i$ in period $t$ and needs $q_r$ kWh. A node is a location or service zone. The stations that this customer can access form $\mathcal J_{r,t}\subseteq\mathcal J$. Mobile charging is available only if at least one mobile charging vehicle (MCV) meets the physical requirements to offer service at node $i$.

The **perceived cost** combines the charging bill with the customer's travel and waiting inconvenience, expressed in money. Using the manuscript's symbols:

$$
c^M_{r,t}=p^M_{i,t}q_r+h^M_{i,t},
\qquad
c^j_{r,t}=p^{F,C}_{j,t}q_r+h^F_{r,j,t}.
$$

Here $p^M_{i,t}$ is the mobile price, $p^{F,C}_{j,t}$ is station $j$'s customer charging price, and the $h$ terms express inconvenience in dollars. Mobile inconvenience uses the estimated time until a vehicle arrives, $\widehat w_{i,t}$. FCS inconvenience includes travel and the estimated station wait $w^F_{j,t}$. These payment, waiting, and travel costs are unchanged in this experiment.

Let $\mathcal A_{r,t}$ contain the **available charging providers**: mobile option $M$, when available, and accessible FCS nodes $j$. The outside option $O$ is added separately and is always available. Write $c^k_{r,t}$ for the perceived cost of charging option $k\in\mathcal A_{r,t}$. The new preference scores are

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

For the same customer, costs, and available options, increasing $b$ makes leaving less likely whenever at least one charging provider is available. If we consider only outcomes in which that customer chooses **some** charging service, the common $\exp(b)$ cancels from the formula. The relative chances of choosing the providers still follow $\exp(-\eta c^k_{r,t})$.

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

## 4. What the implementation checks established

The implementation change passed **108 tests**: 90 preceding tests, 11 new choice/checkpoint regressions, and 7 experiment-entry-point checks. These checked the formulas using customers with different energy needs and station access; unchanged provider proportions for a given customer who chooses charging; exclusion of unavailable services; reliable numerical calculation at $b=\pm10000$; JSON and older-configuration defaults; adaptive pricing at b=3; exact continuation from saved checkpoints, including the random-number generator (RNG); correct scenario loading; and rejection of mismatched customer models. The later three-seed study brought that historical suite to 112 tests. See [VALIDATION.md](VALIDATION.md) for the current suite count.

Tanh and SiLU also ran separately at b=3 for two updates each: 8 training episodes and 96 physical periods per activation. Both produced finite metrics, meaning no undefined or infinite measurements, zero invalid evaluation actions, and exactly matching results after checkpoint reload. These checks show that the code runs and saves correctly. They do not show stable learning or a profit advantage. Their saved record is [outside_option_smoke/verification.json](../outputs/outside_option_smoke/verification.json).

### A controlled check of customer participation

To isolate the choice formula, a separate check generated 32 complete b=0 reference episodes with the previously frozen inventory rule. This supplied **384 physical states and 5,334 requests**. The check then varied only $b$, keeping prices, requests, accessibility, waiting estimates, and vehicle states identical. The table averages each request's choice probabilities, giving every request equal weight:

| Charging benefit $b$ | Expected outside share | Expected charging participation |
|---|---:|---:|
| 0 | 92.3744% | 7.6256% |
| 2 | 67.1085% | 32.8915% |
| 3 | 47.1786% | 52.8214% |
| 4 | 27.9692% | 72.0308% |

These shares come from the choice probabilities for the same reference customers. They are not counts of customers actually choosing or receiving service in new b=3 episodes. They show how the formula changes participation in situations encountered under b=0. They do not establish how many customers the system can serve, whether capacity is sufficient, or whether training improves profit.

For the same customer choosing charging, the largest change in the relative provider probabilities was only `2.22e-16`; the largest error in probabilities summing to one was `3.33e-16`; unavailable-provider violations were zero. Reconstructed b=0 probabilities matched the simulator exactly. Source/selection hashes—file fingerprints used to detect changes—and the global NumPy RNG state were preserved. See the saved [summary](../outputs/choice_intercept_check/summary.json) and [protocol](../outputs/choice_intercept_check/protocol.json).

## 5. Run a new scenario

Activate the existing environment and choose a new output directory for every run:

```bash
source .venv/bin/activate

# Small combined SiLU + b=3 execution check.
python -m charging_env.rl.train --scenario configs/utility_b3.json \
  --dispatch-final-activation silu --updates 10 --episodes-per-update 4 --seed 7 \
  --output-dir "outputs/silu_b3_$(date +%Y%m%d_%H%M%S)"

# Established 500-update convergence preset with the changed customer model.
python experiments/run_convergence.py --seed 7 \
  --dispatch-final-activation silu --scenario configs/utility_b3.json \
  --output-root "outputs/silu_b3_full_$(date +%Y%m%d_%H%M%S)"

# Repeat the fixed-state probability check.
python experiments/check_choice_intercept.py \
  --selection-path outputs/policy_benchmarks/selection.json \
  --output-dir outputs/choice_intercept_check_repeat
```

Pass `--scenario` to the training and benchmark commands when choosing a customer model. Editing or creating a JSON file does not make a script use it automatically; a script may otherwise create its default `Scenario()`.

## 6. Compare controllers fairly within b=3

Changing $b$ changes the customer market. Higher absolute profit than in the original b=0 market would not show that the learned controller improved. To assess the controller, compare b=3 RL against rules selected again for b=3, using the same independent test scenes, meaning the same new simulated episodes kept separate from training and rule selection.

The general benchmark script checks all three final checkpoints and all three checkpoints chosen using validation results against the requested base scenario **before** searching rule settings or creating output. A b=0/b=3 mismatch fails that check.

The command below illustrates the flat checkpoint layout produced by `run_convergence.py` when all three seeds share an output root. Its example root, `outputs/silu_b3_full_flat`, must contain `seed_7`, `seed_19`, and `seed_37`, each with `checkpoint.pt` and `best_checkpoint.pt`. The exact command was not run as part of the original outside-option implementation change:

```bash
python experiments/run_policy_benchmarks.py \
  --scenario configs/utility_b3.json --checkpoint-root outputs/silu_b3_full_flat \
  --validation-seed-base 3300001 --test-seed-base 3500001 \
  --stress-seed-base 3700001 --action-seed 214733 \
  --output-dir outputs/benchmarks_silu_b3
```

The completed dedicated studies save models under a different `training/seed_*` folder structure. Follow their own commands; this generic example expects the flat structure described above.

## 7. What the later training studies found

The [SiLU + b=3 study](SILU_B3_STUDY.md) completed 500 updates for seeds 7, 19, and 37, retuned the rules, and evaluated independent base and stress tests. Its dedicated entry points are `run_b3_training.py` and `run_b3_benchmarks.py`.

On 512 base-test scenes, final stochastic RL—sampling actions from the learned probabilities—averaged **116.42** discounted profit per episode, compared with **125.26** for the retuned inventory rule. Every seed improved over its own initialization, but every seed remained below that rule. Comparing RL with the rule on matching scenes gave an average difference of **−8.84**, with a 95% interval of **[−10.06, −7.58]**. The interval describes uncertainty across test scenes for these fixed trained models. Both stress tests also favored the rule in mean profit.

The later [targeted b=3 training report](B3_TRAINING_IMPROVEMENTS.md) investigates the remaining gap and evaluates new training variants on a separate set of fresh scenes. These results show why more charging participation and better network features must be tested in the full pricing–dispatch system. Neither change alone shows that RL manages battery inventory well or earns more than a strong rule.
