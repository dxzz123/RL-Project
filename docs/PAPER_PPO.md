# Interleaved joint PPO from the new manuscript

This controller implements the four-page [new algorithm manuscript](MDP_RL_2.pdf). It learns both mobile prices and fleet dispatch. Its central idea is to connect the two decisions through their advantages, then judge a dispatch update by the probability of the **whole fleet action**.

The earlier [SA-PPO/MAPPO](TRAINING.md) and [fitted-value controller](FITTED_VALUE.md) remain separate methods. This guide explains the new implementation, the choices needed to connect the manuscript to this simulator, and the independent performance comparison.

The completed base test gives **123.19 for new PPO versus 117.42 for earlier PPO**, while fitted value remains higher at **137.76**. [Section 6](#6-train-or-evaluate-the-new-controller) gives terminal commands; [Section 8](#8-recorded-results) gives all results and limitations.

## 1. What happens in a period

The mathematical period is $t=1,\ldots,T$; the simulator's Python `period` is `t-1`.

1. The pricing actor sees time, fleet locations and batteries, background arrival rates, and current fixed-station prices. It chooses mobile prices.
2. Customers choose mobile charging, an accessible fixed station, or the outside option.
3. The dispatch actor sees the realized choices and chooses one operation and destination for each vehicle. All vehicles' commands are submitted together.
4. The simulator executes travel, service, and energy exchange. It reports profit and advances to the next period.

The simulator internally draws requests before pricing, as it did previously. The new pricing actor **and pricing critic** deliberately omit those realized requests and all demand-derived waiting estimates. This reproduces the manuscript's information boundary without changing the physical simulator. Observing the distribution's arrival rate is different from observing how many customers actually arrived.

There is no separate pricing-stage payment in this environment: $R^p_t=0$. Dispatch receives the original business reward $R^d_t$. Revenues, procurement costs, travel costs, penalties, and optional terminal inventory payment retain their existing definitions. This implementation uses no imitation learning, inventory reward shaping, or learned-value dispatch search.

## 2. Four networks with distinct jobs

| Network | Information used | Job |
|---|---|---|
| Pricing actor | The restricted pre-choice state | A categorical price distribution at each available node |
| Pricing critic | The same restricted state | Remaining return before prices and customer choice |
| Dispatch actor | The global post-choice state | One masked categorical operation/destination distribution per vehicle |
| Dispatch critic | The global post-choice state | Remaining return after current customer choice |

Each network owns its parameters and optimizer. The central dispatch actor has a global state encoder and separate vehicle outputs. Its fleet distribution factorizes over those outputs, conditional on the same global observation. This is centralized fleet dispatch, as described in the new manuscript.

An encoder converts physical quantities into scaled numerical features. The post-choice encoder also processes every request, including requested energy, location, and choice, with no artificial request-count cap. Time is included because the finite horizon changes the value of battery inventory. The chosen hidden width, activation, normalization, learning rates, and numerical safeguards are implementation settings: the PDF does not specify their numerical values.

## 3. Credit passes through both decision stages

An **advantage** estimates how much better a sampled action was than its critic's prediction. A **TD error** compares one step's reward plus the following critic prediction with the current prediction. With critic estimates recorded during rollout collection, Equations (4)-(7) give

$$
\delta^{(1)}_t=R^p_t+V_2(\widetilde S_t)-V_1(S_t),\qquad
\delta^{(2)}_t=R^d_t+\gamma V_1(S_{t+1})-V_2(\widetilde S_t),
$$

$$
\widehat A_{2,t}=\delta^{(2)}_t+\gamma\lambda\widehat A_{1,t+1},\qquad
\widehat A_{1,t}=\delta^{(1)}_t+\lambda\widehat A_{2,t}.
$$

At episode termination the next critic value and next advantage are zero. No return or advantage crosses into another episode.

There is **no discount within a physical period**. The factor $\gamma$ applies only between dispatch and the next period's pricing. However, the manuscript applies the trace parameter $\lambda$ at **both** decision stages. The recursive coefficient across two pricing states is therefore $\gamma\lambda^2$. This differs from the earlier trainer, which applied $\lambda$ only between physical periods. At $\lambda=1$, both consistent critic targets reduce to ordinary discounted returns.

### Critic targets and the optional A1 experiment

Equations (9) and (11) use the corresponding stage's advantage plus its old critic estimate:

$$
y_{1,t}=\widehat A_{1,t}+V^{\mathrm{old}}_1(S_t),\qquad
y_{2,t}=\widehat A_{2,t}+V^{\mathrm{old}}_2(\widetilde S_t).
$$

The main implementation follows these formulas directly. Equation (11) in the supplied PDF uses the dispatch advantage A2; no correction to that equation is applied.

The CLI option `--dispatch-critic-target pricing_advantage` retains the alternative $\widehat A_{1,t}+V^{\mathrm{old}}_2(\widetilde S_t)$ discussed during implementation. Its checkpoint records that choice. This is an experimental A1-target variant, not the formula printed in the supplied PDF and not the main reported algorithm.

## 4. One PPO ratio for a complete fleet action

For each stored physical period, dispatch log probability is the sum of the sampled vehicles' log probabilities. The new-to-old ratio is

$$
r^{(2)}_t=
\exp\left(\sum_m[\log\pi^m_{\mathrm{new}}(a^m_t\mid\widetilde S_t)
-\log\pi^m_{\mathrm{old}}(a^m_t\mid\widetilde S_t)]\right).
$$

The PPO objective clips this **one fleet ratio once**, using the team dispatch advantage. It does not clip each vehicle's ratio and average those losses. For example, two individual ratios of 1.15 produce a joint ratio of 1.3225; a clip range of 0.8-1.2 clips the joint change even though neither individual ratio exceeds 1.2. The earlier trainer's per-vehicle objective treats that example differently.

Dispatch entropy, which rewards exploration, is also **summed over vehicles** before averaging over periods. Pricing uses the probability of the full node-price vector. An unavailable pricing node contributes probability one, log probability zero, and entropy zero.

The main experiment follows the displayed objectives with raw advantages, uniform averages over collected physical periods, no pricing entropy bonus, and ordinary mean-squared critic losses. It does not use the earlier trainer's additional outer time-discount weights or value clipping. Networks remain fixed while episodes are collected; their old log probabilities, critic values, advantages, and targets remain fixed during the subsequent minibatch updates. The buffer is replaced on each iteration, despite the PDF's generic name “replay buffer.”

## 5. Discrete actions and physical feasibility

The paper specifies four operations: **service, idle, charge, discharge**, paired with a destination. It does not specify a separate recharge/export quantity action. The adapter therefore uses the following explicit interpretation:

| Paper operation | Simulator command |
|---|---|
| Service | `SERVE` at the chosen mobile-request node |
| Idle at the current node | `WAIT` |
| Idle at a different node | `REPOSITION`, meaning travel there and do no service or exchange |
| Charge | `RECHARGE` with the maximum individually feasible grid-energy request |
| Discharge | `DISCHARGE` with the maximum individually feasible grid-energy request |

The idle interpretation supplies the manuscript's destination choice without adding a fifth paper operation. Maximum exchange is constrained by travel time, remaining operating time, power, efficiency, battery capacity, and reserve. If vehicles share a station, the simulator applies its existing joint station-capacity allocation. This deterministic quantity rule is an integration choice; the manuscript does not state an exchange-quantity rule.

Destinations are current mobile-request nodes together with fixed-station nodes. Staying at the current node is always retained as a safe idle option, including when no demand exists. Service and exchange choices also obey the simulator's physical action masks. Invalid categorical logits are set to negative infinity before the distribution is formed.

This gives one small distribution per vehicle, rather than enumerating every possible fleet action. The earlier PPO learns a continuous energy fraction, while inventory rules and fitted value search half/full requested-energy choices. Those different action sets are recorded when comparing the complete controllers.

## 6. Train or evaluate the new controller

To view the saved comparison immediately, run these commands from the project root:

```bash
python3 experiments/summarize_paper_ppo_study.py \
  --study-dir outputs/paper_ppo_study --show
open outputs/paper_ppo_study/paper_ppo_comparison.png
```

The first command prints all six methods under all three conditions. It reads the existing result file, requires only standard Python, and does not execute a policy or change any saved result. The second opens the figure on macOS.

To train a new model, activate the existing virtual environment. The same `[train]` dependency extra used for the earlier PPO supplies PyTorch.

```bash
source .venv/bin/activate
python -m charging_env.rl.paper_train \
  --scenario configs/utility_b3.json \
  --updates 10 --episodes-per-update 4 --seed 7 \
  --output-dir outputs/paper_joint_ppo
```

This short command checks execution; it is too small to establish strong performance. Repeated CLI training commands choose a fresh numbered output directory automatically and print its path. The original scenario remains `b=0` when `--scenario` is omitted, so keep the explicit `utility_b3.json` argument for the current market model.

A checkpoint records which algorithm and critic-target interpretation produced it, the complete configuration, scenario, all four networks and optimizers, counters, and random-number state. New checkpoints have a distinct format and cannot silently be loaded as the earlier PPO. Reported profit always uses the original environment reward.

To evaluate a saved model on a separate set of scenarios, with no further training:

```bash
python -m charging_env.rl.paper_train \
  --evaluate-only outputs/paper_ppo_study/training/seed_7/checkpoint.pt \
  --eval-episodes 128 --eval-seed-base 193000001 --eval-policy stochastic
```

This prints an evaluation JSON containing `mean_discounted_profit`, episode details, and invalid-action counts. The checkpoint supplies its `b=3` scenario; `--scenario` cannot replace it during evaluation. These example scenarios differ from the recorded primary test, so their mean need not be identical.

## 7. Compare performance independently

The main comparison keeps six methods: new joint PPO, earlier PPO, fitted-value dispatch, service-first, constant inventory value, and time-dependent inventory value. The earlier methods' saved checkpoints are frozen controls. The three heuristic families are tuned again on 128 separate validation scenarios.

New PPO models use training seeds 7, 19, and 37, each with 500 updates, 16 complete episodes per update, and 12 physical periods per episode. That is 8,000 episodes or 96,000 periods per model, matching the earlier PPO's rollout-period budget. Fitted value uses known-model state queries instead of episode-based training; equal information or equal computation is not implied.

The primary evaluation uses each model's **final 500-update checkpoint**, without picking a favorable snapshot or a favorable sampling mode on the test data. Stochastic policy evaluation uses fixed recorded action seeds. Performance checks during training use separate validation scenarios and do not change that final-checkpoint rule.

The final test contains 512 independent base scenarios and 256 scenarios in each stress case. High demand increases arrival means by 50%; low inventory starts each vehicle at 35 kWh. All model weights, learned value tables, and rule parameters remain frozen. Different cases use different scenario seeds, while controllers within a case share the same scenarios.

A paired comparison subtracts two controllers' profits on each shared scenario. Its 95% bootstrap interval resamples those paired differences 5,000 times. These intervals describe scenario uncertainty for the saved policies. They do not include arbitrary retraining uncertainty; three training seeds provide only limited evidence about that variation.

### Frozen training settings

| Setting | Main experiment |
|---|---|
| Updates / episodes per update | 500 / 16 |
| Epochs / minibatch size | 6 / 192 physical periods |
| Main state encoder | Two layers of width 64, SiLU and LayerNorm; post-choice request processing uses its own two-layer encoder |
| Actor / critic learning rates | 0.0003 / 0.001 |
| PPO clip range | 0.8-1.2 |
| Physical-period discount / GAE parameter | 0.99 / 0.95 |
| Pricing / dispatch entropy coefficient | 0 / 0.01 |
| Advantage normalization / value clipping | Both disabled |
| Gradient norm limit | 0.5 separately for each of four networks |
| KL early stopping | Disabled; all six epochs run |
| Validation monitoring | 64 fixed scenarios every 25 updates |
| Final checkpoint selection | Final update 500, stochastic execution |

The paper specifies the structure and objectives but not these numerical settings. They are one fixed implementation of the method, rather than an exhaustive search for its best possible hyperparameters.

### Reproduce the complete experiment

Use a fresh study directory; it keeps fixed checkpoint paths and frozen protocols for reliable comparison. Individual training commands automatically add numbered destinations, while a dedicated study deliberately refuses an existing study path.

```bash
python experiments/run_paper_ppo_study.py --phase init \
  --study-dir outputs/paper_ppo_repeat \
  --control-root outputs/fitted_value_study
python experiments/run_paper_ppo_study.py --phase select \
  --study-dir outputs/paper_ppo_repeat
for seed in 7 19 37; do
  python experiments/run_paper_ppo_study.py --phase train --seed "$seed" \
    --study-dir outputs/paper_ppo_repeat
done
python experiments/run_paper_ppo_study.py --phase evaluate \
  --study-dir outputs/paper_ppo_repeat
python experiments/summarize_paper_ppo_study.py \
  --study-dir outputs/paper_ppo_repeat
python experiments/plot_paper_ppo_study.py \
  --study-dir outputs/paper_ppo_repeat
```

Plotting additionally needs Matplotlib (`python -m pip install matplotlib`). The six old control checkpoints must exist under the supplied `--control-root`; the init phase copies them and records their hashes before training. It also saves an unchanged copy of the algorithm PDF and executed source files. Rule parameters are selected using validation only, before the first final-test episode.

## 8. Recorded results

The experiment completed on **2026-10-21**. The new joint PPO is **better than the earlier PPO on the base test**, with consistent gains across seeds 7, 19, and 37. It still earns less than the inventory rules and fitted-value dispatch.

Each table entry is mean discounted business profit per episode, in dollars. Learned-method rows average three separately executed models; they are not ensembles. Controllers within a column face the same scenario seeds.

| Method | Base: 512 scenarios | High demand: 256 | Low inventory: 256 |
|---|---:|---:|---:|
| New joint PPO | 123.19 | 131.93 | 106.90 |
| Earlier PPO/MAPPO | 117.42 | 127.09 | 105.70 |
| Fitted-value dispatch | **137.76** | 140.56 | **126.67** |
| Service first | 123.25 | **144.48** | 113.56 |
| Constant inventory value | 129.02 | 129.77 | 115.90 |
| Time-dependent inventory value | 133.41 | 133.63 | 121.75 |

### Consistency of the PPO improvement

The base-test gain over earlier PPO is **5.77 dollars, or 4.91%**, with a paired 95% bootstrap interval **[4.95, 6.55]**. Each individual training seed also has a positive gain interval:

| Seed | New PPO | Earlier PPO | Gain | Paired 95% gain interval |
|---|---:|---:|---:|---|
| 7 | 124.817 | 115.549 | +9.269 | [8.048, 10.564] |
| 19 | 121.396 | 116.440 | +4.956 | [3.687, 6.193] |
| 37 | 123.357 | 120.284 | +3.072 | [1.707, 4.431] |

The predeclared base criterion for improvement over earlier PPO passes. The criterion for beating the strongest validation-selected heuristic fails: new PPO minus the time-dependent rule is **-10.22**, with interval **[-11.35, -9.12]**. Its gap to fitted value is **-14.57**, with interval **[-15.64, -13.54]**. New PPO and simple service-first are effectively tied on this test: difference **-0.06**, interval **[-1.07, 0.94]**.

This experiment compares complete implementations. Observation sets, action quantities, architecture, and numerical settings also differ. It does not isolate joint clipping as the cause of the measured improvement or establish that every implementation of the new algorithm will beat every earlier-PPO configuration.

### Changes in operating behavior

Compared with earlier PPO, the new controller delivers **69.57 versus 63.27 mobile kWh** per base episode. Travel energy falls from **11.63 to 7.76 kWh**, and ending fleet stock falls from **31.07 to 28.67 kWh**. It serves slightly more energy while spending less on movement.

The reward ledger accounts for the gain: lower travel cost adds 2.60 dollars, mobile revenue adds 2.42, lower unmet-demand cost adds 1.14, and lower procurement cost adds 0.31. Reduced fixed-station revenue subtracts 0.66 and reduced export revenue subtracts 0.05. Together the unrounded terms give the measured **5.77-dollar** improvement. These are observed operating changes, not a causal ablation of the algorithm's individual components.

The remaining inventory-management gap is substantial. New PPO buys only **0.17 grid kWh** for recharge per episode, compared with **81.17** for the time-dependent rule and **108.78** for fitted value. Those stronger controllers deliver **140.34** and **159.06 mobile kWh**, respectively. The PPO improvement mainly uses initial inventory more effectively; it has still not learned the profitable recharge-and-serve behavior seen in the stronger methods.

### Stress tests and training stability

Under higher demand, new PPO improves on earlier PPO by **4.83**, with interval **[3.50, 6.21]**. Service-first is nevertheless strongest at **144.48**, above new PPO's **131.93**.

Under low initial inventory, new PPO's average gain over earlier PPO is **1.20**, with interval **[0.11, 2.25]**. This gain is not consistent across training seeds: seed 19 declines by **0.86**, with interval **[-2.51, 0.76]**. The base-case consistency result therefore must not be extended to every stress condition.

All three training runs completed 500 updates with finite metrics and no early-stopped epochs. Monitoring profit rose from roughly -46 to 123-125. Over updates 375-500, the respective monitored ranges were **121.65-125.07**, **119.50-122.54**, and **124.09-125.04**. The curves settle toward a plateau in this run. That is evidence of numerical stability, not proof of global convergence or the best possible policy. No test result triggered more tuning or replacement of the primary final checkpoint.

![New joint PPO, earlier learners and rules, paired PPO improvements, stress cases, and validation curves](../outputs/paper_ppo_study/paper_ppo_comparison.png)

Panel A shows the six base means and individual learned-model means. Panel B compares each new PPO seed with its earlier-PPO counterpart, followed by their mean comparison. Panel C shows both stress conditions. Panel D uses separate monitoring scenarios, not the final test set.

### Execution budget and evidence

The three new models collected **24,000 training episodes**, or **288,000 physical periods**. Each model performed 3,000 minibatch updates, stepping its four independent optimizers each time. Each took about **five minutes** on the CPU, measured while all three seeds ran concurrently; their elapsed times should not be added and described as overall wall time.

Monitoring executed **4,224 policy episodes**, and checkpoint-reload verification executed another **192**. Rule selection used **10,496 validation policy episodes**. Final testing used **12,288 policy episodes** over 1,024 distinct scenarios, with **zero invalid actions**. Two current-source execution checks used another 96 training periods and 40 evaluation episodes; earlier preflight checks used the same separate budget. Numerical unit checks are outside these experiment budgets. The optional A1 variant was execution-tested, but was not a main performance-comparison method.

All **211 automated tests** passed. The independent audit recomputed paired means and intervals, actual-model risk statistics, reward accounting, and budgets from the saved records, and verified checkpoints, source files, and the unchanged source PDF.

The [protocol](../outputs/paper_ppo_study/protocol.json), [selected rules](../outputs/paper_ppo_study/rule_selection/selection.json), [test summary](../outputs/paper_ppo_study/benchmarks/summary.json), [independent audit](../outputs/paper_ppo_study/analysis.json), [211-test log](../outputs/paper_ppo_study/logs/unit_tests.log), and [executed-source snapshot](../outputs/paper_ppo_study/provenance/executed_training_source/) provide the exact settings and evidence. New models are saved under [training/seed_7](../outputs/paper_ppo_study/training/seed_7/), [training/seed_19](../outputs/paper_ppo_study/training/seed_19/), and [training/seed_37](../outputs/paper_ppo_study/training/seed_37/). The three earlier PPO controls remain under [legacy/](../outputs/paper_ppo_study/legacy/), and fitted-value controls under [fitted/](../outputs/paper_ppo_study/fitted/).

