# SiLU with b=3: learning improved profit, but the inventory rule still won

> Historical experiment. Its numbers belong to the scenarios and methods recorded here. Use [the current comparison](../COMPARISON.md) for approximate DP, both PPO versions, and the three main rules on shared tests. See [the archive index](README.md) for context.

This completed **2026-10-17** study trained or retained three 500-update models with seeds **7, 19, and 37**, retuned the benchmark rules for the same customer model, and compared them on independent scenarios. All three policies improved over their untrained initialization. Their final mean discounted profit was **116.42**, below the inventory rule's **125.26**. Thus, learning was successful relative to initialization, but the practical benchmark gap remained.

A **seed** is a number that makes a random sequence repeatable. Different training seeds can produce different learned policies, even with the same settings. A **policy** is the learned rule for choosing actions.

This is the first b=3 study. It follows the historical b=0 [training analysis](TRAINING_ANALYSIS.md) and [benchmark comparison](POLICY_BENCHMARKS.md). The later [targeted training study](B3_TRAINING_IMPROVEMENTS.md) investigates why the gap remained and evaluates another set of fresh policies and test scenarios.

## What do “SiLU” and “b=3” mean here?

The two terms refer to different parts of the system:

- **SiLU** is an activation function: it changes the values passed between layers inside a neural network. In this study it replaces only the **final hidden Tanh activation** in the dispatch actor, the network that chooses vehicle actions. Earlier Tanh layers, the pricing actor, and the rest of the network structure keep their established settings. Only this part of the network uses SiLU.
- **b=3** changes the customer model. Each available mobile or fixed charging station (FCS) option receives a baseline boost of 3 in its customer-choice score, called its systematic utility; the outside option of leaving the system keeps a score of zero. In the notation guide's extension, a charging utility is $b-\eta c^k_{r,t}$. At a fixed cost, this makes charging more attractive relative to leaving the system. The attached manuscript's cost-only specification is the **b=0** reference; b is an explicitly labeled implementation extension.

These changes affect different things. A larger profit under b=3 than in an old b=0 table does not by itself show a better controller, because the customer model also makes charging more attractive. This study measures learning and benchmark performance **within b=3**. It does not include otherwise-matching experiments that change only SiLU versus Tanh, or only b=3 versus b=0. It therefore cannot establish that either change caused the improvement. See [OUTSIDE_OPTION.md](../OUTSIDE_OPTION.md) for the choice model and [NOTATION.md](../NOTATION.md) for symbols.

Throughout this report, profit means the original business reward discounted by 0.99 per period over a 12-period episode: profit earned later receives slightly less weight. It is measured in model dollars and includes FCS as well as mobile operations. A *physical period* includes both pricing and dispatch; it is not one RL stage.

## How were the three models trained?

An **actor** is a network that chooses actions; a **critic** is a network that predicts future profit to help train the actor. PPO (Proximal Policy Optimization) adjusts both from simulated operating episodes. An **update** collects a batch of episodes and makes several training passes, called **epochs**, over that batch. Each pass processes smaller groups called **minibatches**. The **learning rate** controls the size of the adjustments. The **entropy coefficient** controls an incentive to keep trying a range of actions. Each model used the established 500-update settings:

| Setting | Value |
|---|---:|
| Updates | 500 |
| Complete episodes per update | 16 |
| PPO epochs | 6 |
| Minibatch size | 192 physical periods |
| Hidden width | 64 |
| Actor / critic learning rate | 0.0003 / 0.001 |
| Value clipping | 0, disabled |
| Entropy coefficient | 0.005 |
| Probability-ratio clip / gradient limit | 0.2 / 0.5 |
| GAE trace parameter, $\lambda_{\mathrm{GAE}}$ | 0.95 |
| Discount | 0.99 |
| Outer discounted policy-loss weighting | Enabled |

The GAE trace parameter controls how the training target combines observed rewards with the critic's predictions; it is not the Poisson arrival intensity $\lambda_{i,t}$. One run used **8,000 training episodes and 96,000 physical periods**. The total exposure of the three delivered models was **288,000 periods**.

Seed 7 was an existing completed run that matched the required scenario, settings, and budget. It was retained based on metadata **before new test results were observed**; it was not selected because it performed well. Seeds 19 and 37 were trained afresh. Therefore, **192,000 training periods were newly collected**, and **96,000 were reused** in this study.

The retained run is `outputs/silu_b3_20261017_215542/seed_7`. Its final weights, the numbers learned inside its networks, were copied unchanged. Its initial model was reconstructed from saved scenario/configuration and reproduced both recorded initial validation modes exactly. That old run had no file-by-file code record saved at the time, so compatibility with the current code was checked afterward. Before testing, we recorded hashes of the new training code and all final checkpoints. A hash is a file fingerprint that lets us check whether its contents changed. The [protocol](../../outputs/silu_b3_study/protocol.json) and [seed-7 reuse record](../../outputs/silu_b3_study/seed_7_reuse.json) document these limits.

All runs used 32 fixed training-validation scenarios, separate examples used to monitor learning, `100001 + 2*i`, with action seed base **12345**, every 25 updates. Validation selected each `best_checkpoint.pt`, a saved copy of the model at its best validation score. The final update-500 checkpoint remained the primary test model. Training environment seeds were even. The runs used many of the same training scenarios, so differences between their results mainly reflect the initial network weights, randomly chosen actions, and training adjustments. They do not represent three completely separate training datasets.

## How were rules tuned and policies tested?

A rule with settings chosen for b=0 need not perform well under b=3. We therefore compared the existing **61-candidate** set of rule settings on b=3 validation scenarios. The rule families were random actions, fixed-price WAIT, service first, myopic joint dispatch, inventory-value joint dispatch, and adaptive-price inventory dispatch.

An inventory-value rule scores an action by immediate profit plus an estimated value for the change in stored battery energy. This estimated value, also called a **shadow value**, can make recharging attractive before that energy earns service revenue. It helps choose actions and is not added to reported business profit. The coordinated rules use the known physical model to compare combinations of vehicle actions together. They know the current state, but not future demand or random outcomes. They provide strong feasible comparisons using a finite set of candidate energy amounts; they do not prove which policy would be best over the whole episode.

| Split | Scenarios per case | Environment seeds | Purpose |
|---|---:|---|---|
| Training validation | 32 | `100001 + 2*i` | Monitor learning; select secondary checkpoints |
| Rule validation | 128 | `3300001 + 2*i` | Retune the 61 rule candidates under b=3 |
| Base test | 512 | `3500001 + 2*i` | Make the primary independent comparison |
| Each stress case | 256 | `3700001 + 2*i` | Check robustness of frozen policies |

Rule-validation and test action seed base was **271991**. Within a case, policies used the same underlying random draws for potential demand, requested energies, random station-price changes, and customer choices. Random action selection used a separate random number generator (RNG). This sharing makes comparisons fairer, while still allowing customers to choose differently when policies charge different prices.

Rule tuning cost **61 × 128 = 7,808 policy-episode executions**. Both parameters within each rule family and the strongest nonrandom family were selected only on rule validation. After all choices were fixed, the base test evaluated final policies in stochastic and deterministic modes, validation-selected policies in both modes, and initial stochastic policies. **Stochastic** execution samples actions from the policy's learned probabilities. **Deterministic** execution chooses the highest-probability discrete actions and the mean energy fraction. Stress tests evaluated final policies in both modes with the same frozen rules.

The two stress cases multiplied average arrival counts by **1.5** or set initial energy to **35 kWh per vehicle**. Within each case, methods ran on the same scenarios, so each profit difference compares matching scenarios. Base and stress seed ranges differed. A difference between cases therefore describes what happened in those samples; it cannot isolate the effect of changing only the scenario parameter.

### What counts as a consistent improvement?

The primary RL result gives equal weight to **three separately executed final stochastic policies**. We average their reported results; the vehicles do not combine the policies' action choices. We also report every training seed. No seed, execution mode, checkpoint, or hyperparameter is chosen using test profit.

The benchmark-advantage criterion was fixed before testing: **every seed must have a positive final-minus-rule mean difference, and the aggregate paired 95% interval must have a positive lower endpoint**. A separate criterion applies the same test to initial-to-final learning gains. Improving from initialization and beating a strong benchmark are different achievements.

A **paired bootstrap** repeatedly draws new samples from the shared test scenarios and recalculates the policy differences. Its **confidence interval** describes how much the measured difference could vary with the scenario sample, while keeping these already-trained models fixed. This is the meaning of an interval that is **conditional on the trained models**. There are **512 paired scenario values**, not 1,536 independent training outcomes, in the primary aggregate. Variation across the three training seeds is reported separately. Three seeds provide limited evidence about what would happen across other training runs.

The probability of losing money and the profit in the worst episodes are calculated for each actual policy, then averaged. Averaging policies' episode outcomes before calculating these risks could hide the losses of individual policies. Likewise, a flat late validation curve is a **plateau**, meaning that the measured score has stopped changing much. It does not prove that further training could never change the policy, or that the best possible policy has been found.

## What did the independent base test show?

Every policy below ran the same 512 new base scenarios. Values are mean discounted business profit per 12-period episode. Each interval refers to a paired difference, rather than uncertainty in one table cell considered alone.

| Training seed | Initial stochastic profit | Final stochastic profit | Final minus inventory rule | Paired 95% interval for difference |
|---|---:|---:|---:|---|
| 7 | −47.48 | 117.74 | −7.51 | [−8.93, −6.07] |
| 19 | −47.87 | 113.55 | −11.71 | [−13.19, −10.17] |
| 37 | −47.49 | 117.96 | −7.29 | [−8.67, −5.93] |
| Three-model mean | −47.61 | **116.42** | **−8.84** | **[−10.06, −7.58]** |
| Retuned inventory rule | — | **125.26** | 0.00 | — |

The final mean was **7.06% below the rule**. Every seed difference and every individual interval was negative, so the criterion for consistent benchmark advantage failed. The sample standard deviation across the three seeds' mean profits was **2.49**, a measure of how far their results spread around the mean.

Learning from initialization was clear: the mean gain was **164.03 [161.03, 166.98]**, with a positive gain and interval for every seed. The learning-gain criterion passed. That result demonstrates useful learning within this setting; it does not isolate either SiLU or b as its cause.

### Were the rules selected fairly?

The following settings were all selected on the 128-scenario rule-validation set, before any final test results were used. The inventory rule won validation before testing. Its shadow value **0.30** was the upper end of the retained finite grid; we did not extend that grid after seeing the results.

| Rule family | Validation-selected settings | Validation profit | Independent base-test profit |
|---|---|---:|---:|
| Random | Existing random-action rule | −40.83 | −49.47 |
| Fixed-price WAIT | Price \$1.00/kWh | 82.61 | 81.80 |
| Service first | Price \$0.80/kWh; recharge threshold 0.20 | 124.16 | 124.01 |
| Myopic joint dispatch | Price \$0.60/kWh; shadow value 0.00 | 105.86 | 104.60 |
| Inventory joint dispatch | Price \$0.60/kWh; shadow value 0.30 | **127.17** | **125.26** |
| Adaptive-price inventory dispatch | Shadow value 0.20 | 112.78 | 112.99 |

Final stochastic RL beat random, WAIT, myopic dispatch, and adaptive pricing in mean test profit. It trailed both inventory dispatch and service first. Secondary three-model means were **114.17** for final deterministic policies, **116.32** for validation-selected stochastic policies, and **114.41** for validation-selected deterministic policies. These additional results did not replace the primary policy chosen before testing.

## What happened under stress?

The models and rule settings remained fixed throughout testing. Each stress case used 256 scenarios, paired between policies within that case.

| Test case | Final stochastic three-model mean | Inventory rule | RL minus rule | Paired 95% interval |
|---|---:|---:|---:|---|
| Base: 512 scenarios | 116.42 | 125.26 | −8.84 | [−10.06, −7.58] |
| Arrival intensity ×1.5 | 125.85 | 132.26 | −6.41 | [−8.85, −4.10] |
| Initial vehicle energy 35 kWh | 103.95 | 115.42 | −11.47 | [−13.46, −9.56] |

Every training seed remained below the inventory rule in mean profit in both stress cases. That does not mean every risk measure was worse: under high demand, RL's average individual-policy worst-10% mean was **97.16**, versus **95.52** for the rule. In the base case, the corresponding values were **88.75 and 94.71**. Neither policy recorded a loss in the tested base scenarios.

## Why did a stable policy still leave profit on the table?

Late validation was stable, but at a level below the strong rule. The six scheduled stochastic scores at updates **375, 400, 425, 450, 475, and 500** gave these summaries:

| Seed | Late validation range | Late validation sample SD | Final validation profit | Validation-selected update |
|---|---:|---:|---:|---:|
| 7 | 117.56–119.57 | 0.81 | 117.56 | 475 |
| 19 | 114.19–114.77 | 0.21 | 114.25 | 300 |
| 37 | 117.07–119.32 | 0.83 | 117.07 | 450 |

Profit did not collapse late in training, but it also did not keep rising from one check to the next. More importantly, the inventory rule delivered **158.87 mobile kWh per episode**, versus **64.27 for final stochastic RL**. RL served **42.55%** of potential requests, versus **45.65%** for the rule. Their outside-choice shares were **52.24% and 48.53%**. These are outcomes from complete episodes, where each policy creates its own sequence of prices, locations, and battery levels. They need not match a customer-choice calculation at one fixed state.

The rule's extra delivery increased revenue and also increased procurement and travel costs. Breaking the discounted profit difference into revenues and costs shows why the rule's net advantage remained positive:

| Contribution to rule-minus-RL profit | Amount |
|---|---:|
| Higher mobile-service revenue | +48.99 |
| Lower fixed-station service revenue | −16.63 |
| Higher discharge revenue | +1.38 |
| Higher procurement costs | −21.31 |
| Higher travel costs | −2.46 |
| Higher unmet-demand costs | −1.14 |
| Operating costs and terminal value | 0.00 |
| Net, using unrounded components | **+8.84** |

The additional mobile revenue more than offset the extra costs and the reduction in FCS revenue. This explains which revenue and cost differences make up the gap. It does not identify a particular network or training defect as its cause. It motivated the subsequent [behavior and training diagnosis](B3_TRAINING_IMPROVEMENTS.md). Longer training alone had not closed the gap.

## What was verified, and where is the evidence?

The suite at this study's completion passed **112 tests**. Rule selection executed **7,808 policy episodes**, and independent base/stress evaluation executed **16,896 policy episodes**, with zero invalid evaluation actions. These are execution counts across policies sharing scenarios, not counts of independent statistical samples. Checks that the code and selected settings had not changed passed. All nine initial/final/validation-selected checkpoint hashes stayed unchanged, and checkpoint reload reproduced recorded evaluation results. [VALIDATION.md](../VALIDATION.md) gives the current project's later checks.

![Training and independent benchmark results](../../outputs/silu_b3_study/silu_b3_study.png)

Saved evidence includes the [machine-readable analysis](../../outputs/silu_b3_study/analysis.json), [benchmark summary](../../outputs/silu_b3_study/benchmarks/summary.json), [frozen evaluation protocol](../../outputs/silu_b3_study/benchmarks/protocol.json), [rule selection](../../outputs/silu_b3_study/rule_selection/selection.json), and [training/source audit](../../outputs/silu_b3_study/reviews/training_audit.md). Final weights are retained for [seed 7](../../outputs/silu_b3_study/training/seed_7/checkpoint.pt), [seed 19](../../outputs/silu_b3_study/training/seed_19/checkpoint.pt), and [seed 37](../../outputs/silu_b3_study/training/seed_37/checkpoint.pt).

## Reproduction

Run from the project root in its existing virtual environment, using a new study directory. The commands below train all three seeds from scratch; the delivered study reused its already-completed seed 7 as described above.

```bash
source .venv/bin/activate
for seed in 7 19 37; do
  python experiments/run_b3_training.py --seed "$seed" \
    --study-dir outputs/silu_b3_repeat
done
python experiments/run_b3_benchmarks.py --phase select \
  --study-dir outputs/silu_b3_repeat
python experiments/run_b3_benchmarks.py --phase evaluate \
  --study-dir outputs/silu_b3_repeat
python experiments/summarize_b3_study.py --study-dir outputs/silu_b3_repeat
python experiments/plot_b3_study.py --study-dir outputs/silu_b3_repeat
```

Only plotting needs optional Matplotlib. Rule selection can run during training. Evaluation requires all nine initial/final/validation-selected checkpoint files. It checks that scenarios, network structures, training budgets, selected settings, and recorded code versions match the saved plan. These reproduction commands reuse the published test scenarios to check the recorded results. Any further tuning needs a fresh final test set that has not influenced model selection.
