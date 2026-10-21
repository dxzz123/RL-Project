# Joint PPO: learning prices and fleet dispatch

Joint PPO follows the two-stage [algorithm specification](MDP_RL_2.pdf) and learns **both mobile prices and vehicle dispatch**. A pricing decision changes which customers request service; the later dispatch decision determines how the fleet serves them. The method trains these two decisions together while keeping a separate value estimate for each stage.

The project also keeps [earlier SA-PPO/MAPPO](LEGACY_PPO.md) and [approximate DP](FITTED_VALUE.md). The two PPO implementations differ in their information, action representation, advantage calculation, and dispatch loss. [COMPARISON.md](COMPARISON.md) explains these differences side by side and reports their performance on the same independent test scenarios.

Use [TRAINING.md](TRAINING.md) for the common setup and comparison workflow. This guide explains Joint PPO, its assumptions, and how to run a single model.

## Follow one physical period

The model has two decisions within each physical period $`t=1,\ldots,T`$. Python stores this as `period = t - 1`.

1. **Set prices.** The pricing actor sees time, fleet locations and batteries, background arrival rates, and current station prices. It chooses a mobile price for each available node.
2. **Observe customer choices.** Customers select mobile charging, an accessible fixed charging station (FCS), or the outside option.
3. **Dispatch the fleet.** The dispatch actor sees the realized choices and selects an operation and destination for each mobile charging vehicle (MCV). The commands are submitted together.
4. **Execute and earn profit.** The simulator applies travel, service, and energy exchanges, reports business profit, and advances to the next period.

The simulator draws requests internally before pricing. For this controller, both the pricing actor and pricing critic hide the realized requests and demand-derived waiting estimates. Knowing that a node expects four requests is different from knowing that seven actually arrived. The dispatch networks receive the realized information after choice.

The pricing-stage reward is $`R_t^p=0`$. All current business profit appears as the dispatch-stage reward $`R_t^d`$: revenues minus procurement, travel, operating costs, and unmet-demand penalties, plus any configured terminal inventory payment. Pricing still matters because it changes subsequent choices and profit. See [MODEL.md](MODEL.md) for the accounting.

## Four networks, four jobs

An **actor** selects actions. A **critic** predicts remaining return and supplies a baseline for learning. There are four independent networks and optimizers:

| Network | Input | Output |
|---|---|---|
| Pricing actor | Restricted pre-choice state $`S_t`$ | Price probabilities at each available node |
| Pricing critic $`V_1`$ | The same state $`S_t`$ | Expected remaining return before pricing and customer choice |
| Dispatch actor | Global post-choice state $`\widetilde S_t`$ | Operation/destination probabilities for every vehicle |
| Dispatch critic $`V_2`$ | The same state $`\widetilde S_t`$ | Expected remaining return after current customer choice |

The dispatch actor uses one global encoder with a separate output for each vehicle. Given that global observation, its vehicle distributions factorize: the probability of a fleet action is the product of the probabilities of its component vehicle actions. The actor therefore considers fleet information without enumerating every joint action.

The critics are needed for training; dispatch decisions are taken from the actor rather than by searching a critic's predictions. This distinguishes the method from approximate DP, which evaluates action candidates using a continuation table at every decision.

The encoders scale physical quantities, include time, and process all observed requests without truncating the request list. The networks use SiLU activations and LayerNorm in the recorded experiment. The architecture and numerical settings are implementation choices beyond the [algorithm specification](MDP_RL_2.pdf).

## How a later outcome teaches an earlier decision

A **temporal-difference error** compares a critic's prediction with immediate reward plus the following prediction. An **advantage** estimates whether a sampled action did better or worse than the critic expected. A positive advantage encourages the actor to make that action more likely; a negative one discourages it, subject to PPO's clipped objective.

The two stages have the following errors, using the critic values saved when collecting the episode:

```math
\delta_t^{(1)}=R_t^p+V_2(\widetilde S_t)-V_1(S_t),
```

```math
\delta_t^{(2)}=R_t^d+\gamma V_1(S_{t+1})-V_2(\widetilde S_t).
```

The first asks whether the post-choice situation is better than expected before pricing. The second asks whether dispatch profit and the next fleet state are better than expected after choice. The physical-period discount $`\gamma`$ appears only when moving to the next period; pricing and dispatch within one period are not discounted relative to one another.

Generalized advantage estimation (GAE) combines these errors backward through the episode. Joint PPO uses

```math
\begin{aligned}
\widehat A_{2,t}&=\delta_t^{(2)}+\gamma\lambda\widehat A_{1,t+1},\\
\widehat A_{1,t}&=\delta_t^{(1)}+\lambda\widehat A_{2,t}.
\end{aligned}
```

The parameter $`\lambda`$ controls how much later errors enter the current advantage. Here it is applied at **both stages**, so the recursive coefficient from one pricing advantage to the next is $`\gamma\lambda^2`$. The earlier PPO trainer applies $`\lambda`$ only between physical periods. At termination, both the next critic value and next advantage are zero; information never crosses an episode boundary.

Each critic is trained toward its own saved value plus its stage's advantage:

```math
\begin{aligned}
y_{1,t}&=V_1^{\mathrm{old}}(S_t)+\widehat A_{1,t},\\
y_{2,t}&=V_2^{\mathrm{old}}(\widetilde S_t)+\widehat A_{2,t}.
\end{aligned}
```

These targets follow Equations (9) and (11) of the [algorithm specification](MDP_RL_2.pdf). With $`\lambda=1`$, they reduce to ordinary discounted returns.

An optional CLI setting, `--dispatch-critic-target pricing_advantage`, substitutes $`\widehat A_1+V_2^{\mathrm{old}}`$ for the dispatch target. This experimental alternative is recorded in the checkpoint and was not used in the main comparison.

## Why the dispatch loss uses one fleet probability ratio

PPO compares the probability of a recorded action under the updated actor with its probability under the actor that collected it. For dispatch, Joint PPO uses the probability of the **whole fleet action**:

```math
r_t^{(2)}=
\exp\left(\sum_m
\left[\log\pi^m_{\mathrm{new}}(a_t^m\mid\widetilde S_t)
-\log\pi^m_{\mathrm{old}}(a_t^m\mid\widetilde S_t)\right]\right).
```

The dispatch surrogate to maximize is

```math
\min\left(r_t^{(2)}\widehat A_{2,t},
\mathrm{clip}(r_t^{(2)},1-\epsilon,1+\epsilon)\widehat A_{2,t}\right),
```

averaged over stored physical periods, with a dispatch entropy bonus to encourage exploration. Vehicle entropies are summed because the fleet distribution factorizes.

For example, if each of two vehicle-action probabilities increases by a factor of 1.15, the fleet ratio is $`1.15^2=1.3225`$. For a positive advantage and $`\epsilon=0.2`$, the joint surrogate limits the rewarded increase to 1.2. Neither individual ratio would be clipped by the earlier per-vehicle objective. Clipping limits the incentive in the surrogate; it is not a hard guarantee that every resulting probability ratio stays inside the interval.

Pricing similarly uses the probability of the complete node-price vector. Unavailable nodes contribute probability one and no gradient. The main experiment uses no pricing entropy bonus.

## Action meanings and physical limits

Dispatch selects one of four operations and a destination. The adapter maps these choices to simulator commands as follows:

| Operation | Meaning in this implementation |
|---|---|
| Service | Travel if needed, then serve whole requests at the chosen node |
| Idle at the current node | Wait without moving |
| Idle at another admissible node | Reposition there without service or exchange |
| Charge | Request the maximum individually feasible grid energy at an FCS |
| Discharge | Request the maximum individually feasible grid export at an FCS |

The action space has no separate energy-quantity action. This implementation requests the maximum individually feasible exchange. It accounts for travel, remaining time, power, battery capacity, reserve, and efficiency. When vehicles share a station, the simulator may allocate less than their requested amount under the existing joint capacity rule.

Candidate destinations are current mobile-request nodes and FCS nodes. Each vehicle's current node is always retained for idle, including when there are no requests. Physically invalid operation/destination combinations are masked before sampling by setting their logits to negative infinity. This prevents individually invalid commands; shared-resource conflicts are handled by the simulator's allocation rules.

These actions differ from the earlier PPO's learned continuous exchange fraction and approximate DP's half/full exchange grid. Performance comparisons therefore concern the complete implemented controllers, rather than an isolated change to the PPO loss.

## What happens in one training update

1. Freeze all four networks and collect complete episodes using the current actors.
2. Save states, sampled actions, rewards, old log probabilities, and old critic predictions.
3. Calculate advantages and critic targets once from those saved predictions.
4. Run several minibatch epochs to update the two actors and two critics. The old probabilities and targets stay fixed throughout these epochs.
5. Discard the collected buffer and collect fresh episodes for the next update.

Each update uses fresh **on-policy** data collected by the current actors. It uses the original rewards, with no inventory reward shaping or imitation learning.

The completed comparison used the following settings:

| Setting | Value |
|---|---|
| Training seeds | 7, 19, 37 |
| Updates / complete episodes per update | 500 / 16 |
| Epochs / minibatch size | 6 / 192 physical periods |
| Hidden layers | Two layers of width 64, SiLU and LayerNorm; separate request encoder after choice |
| Actor / critic learning rates | 0.0003 / 0.001 |
| Clip parameter $`\epsilon`$ | 0.2 |
| Physical-period discount $`\gamma`$ / GAE parameter $`\lambda`$ | 0.99 / 0.95 |
| Pricing / dispatch entropy coefficient | 0 / 0.01 |
| Advantage normalization / critic-value clipping | Both disabled |
| Gradient norm limit | 0.5 separately for each network |
| KL-based early stopping | Disabled; all six epochs run |
| Monitoring | 64 separate scenarios every 25 updates |
| Primary tested checkpoint | Final update 500, stochastic execution |

KL divergence measures how much action probabilities change. Here it is recorded for diagnosis but does not stop an update. Actor losses use uniform averages over collected periods and raw advantages; the earlier trainer's additional outer time-discount weights are not used. Critics use mean-squared error. These are the settings tested in the recorded comparison; further hyperparameter studies may produce different results.

## Run a single model

From the project root, use the existing virtual environment and the `[train]` dependencies described in [TRAINING.md](TRAINING.md):

```bash
source .venv/bin/activate
python -m charging_env.rl.paper_train \
  --scenario configs/utility_b3.json \
  --updates 10 --episodes-per-update 4 --seed 7 \
  --output-dir outputs/paper_joint_ppo
```

This small run checks execution; it does not establish good performance. Keep the explicit scenario argument because omitting it selects the original `b=0` defaults. If the requested output directory exists, the CLI chooses a numbered sibling automatically and prints the path.

| Saved file | Use |
|---|---|
| `checkpoint.pt` | Final model, all optimizers, configuration, scenario, counters, and random-number state |
| `best_checkpoint.pt` | Best checkpoint according to this run's monitoring scenarios |
| `training_metrics.csv` | Per-update training profit, losses, entropy, and numerical diagnostics |
| `validation_history.json` | Monitoring scores before, during, and after training |
| `final_evaluation.json` | Final monitoring episodes; this is not the independent benchmark test |
| `summary.json` and `protocol.json` | Run summary, exact settings, source hashes, and replay verification |

The recorded study uses `checkpoint.pt` at update 500, not the most favorable monitoring checkpoint. The checkpoint format is separate from earlier PPO's format, so use the matching CLI.

To evaluate a saved model on additional scenarios without training:

```bash
python -m charging_env.rl.paper_train \
  --evaluate-only outputs/paper_ppo_study/training/seed_7/checkpoint.pt \
  --eval-episodes 128 --eval-seed-base 193000001 --eval-policy stochastic
```

This prints JSON containing mean discounted profit, individual episode results, and invalid-action counts. The saved checkpoint supplies its scenario. These scenarios differ from the recorded test, so their mean need not match the reported result. `stochastic` samples from the actor; `deterministic` takes its most likely actions. Choose the evaluation mode before comparing results, rather than choosing whichever looks better on the test set.

For a full three-seed comparison with separate rule selection and test scenarios, follow [TRAINING.md](TRAINING.md). Its study runner also freezes the earlier PPO and approximate-DP controls. Running the single-model command above does not automatically produce that comparison.

## Results and code references

On the shared base test, this implementation earns **123.19 dollars per episode**, compared with **117.42** for earlier PPO and **137.76** for approximate DP. All three Joint PPO seeds improve over their earlier-PPO counterparts on the base test, but the inventory rule and approximate DP remain stronger. The principal observed gap is that PPO still recharges very little. See [COMPARISON.md](COMPARISON.md) for the paired intervals, operating evidence, stress tests, and limitations.

These records are local files under `outputs/paper_ppo_study/`, which is not included in Git. The saved protocol (`protocol.json`) records exact settings. The independent audit (`analysis.json`) contains recomputed comparisons and source checks, and the executed-source snapshot (`provenance/executed_training_source/`) preserves the code used for those results. Use the saved study files or follow [the reproduction workflow](TRAINING.md#reproduce-the-six-method-study) to inspect them.

| Code or test | What to read there |
|---|---|
| [paper_ppo.py](../src/charging_env/rl/paper_ppo.py) | Information boundary, action adapter, four networks, GAE, joint loss, training, checkpoints |
| [paper_train.py](../src/charging_env/rl/paper_train.py) | Train, resume, and evaluation CLI |
| [run_paper_ppo_study.py](../experiments/run_paper_ppo_study.py) | Frozen comparison protocol and data splits |
| [test_paper_ppo.py](../tests/test_paper_ppo.py) | Equation checks, masks, information isolation, network updates, and reproducibility |
| [test_paper_study.py](../tests/test_paper_study.py) | Split separation, frozen-source checks, control validation, and result viewer |
