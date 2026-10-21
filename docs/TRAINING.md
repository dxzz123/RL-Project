# Training the PPO pricing and dispatch policies

Training teaches the operator two connected decision rules: **what mobile price to offer, and how to use its vehicles after customers respond**. A learned decision rule is called a **policy**. Here, a policy usually assigns probabilities to several possible actions, rather than always choosing one fixed action. A good pricing rule attracts demand the fleet can serve profitably. A good vehicle rule also considers battery energy and locations needed in later periods.

The policies are implemented by **neural networks**: calculations with adjustable numbers called **parameters**, or weights. Training changes those numbers using experience collected from the simulator. Sampling diferent actions lets the policy **explore**, meaning try alternatives to discover useful behavior. Settings we choose before a run, such as the learning rate or number of episodes, are called **hyperparameters**. They control how learning proceeds.

The basic loop is: run complete episodes, record the choices and profits, adjust the networks, then run new episodes. Periodically, the trainer also measures performance without changing the networks. The rest of this guide explains each part of that loop. Read [the model walkthrough](MODEL.md) for the physical system and customer behavior, and use [the notation guide](NOTATION.md) to match equations to the manuscript or Python names.

The framework is **Stage-Aware PPO–Multi-Agent PPO (SA-PPO–MAPPO)**. **PPO**, short for Proximal Policy Optimization, is a method for adjusting action probabilities cautiously using collected experience. **Stage-aware** means the implementation accounts for the different information and timing of pricing and dispatch. **Multi-agent** means several vehicles learn to act toward the same team objective. The implementation uses PPO's clipped learning rule, described in [the PPO paper](https://arxiv.org/abs/1707.06347), and shared vehicle policies with global value estimates, following [MAPPO](https://arxiv.org/abs/2103.01955). There is some choice freedom in the network and training settings; this guide records the choices made in this project.

The most essential parts are Section 1, the four-network table in Section 3, the six-step update loop and PPO explanation at the start of Section 4, and Section 6. Other further details include network details, equations, configuration tables. Completing a run confirms that the code executes.

The latest manuscript specifies a separate **interleaved joint PPO** controller. Its [complete guide](PAPER_PPO.md) covers the new equations, training, and comparison with the earlier PPO, fitted value, and three heuristics. The model-based fitted-value dispatcher has its own [training and comparison guide](FITTED_VALUE.md). This page describes the retained earlier PPO/MAPPO implementation.

To **see the latest six-method performance comparison**, run this from the project root:

```bash
python3 experiments/summarize_paper_ppo_study.py \
  --study-dir outputs/paper_ppo_study --show
open outputs/paper_ppo_study/paper_ppo_comparison.png
```

These commands display the saved results. The [new PPO guide](PAPER_PPO.md) gives its separate training command and complete comparison protocol. The sections below describe the earlier PPO trainer and its settings.

## 1. Understand one training episode

An **episode** is one complete run of the simulated system from its initial state to its final period. The default scenario has 4 nodes, 2 mobile charging vehicles (MCVs), 2 fixed charging stations (FCSs), and 12 physical periods. One period represents one hour, so this episode covers 12 simulated hours.

At the beginning of a period, the operator sees the customers who have arrived, their energy needs, station prices, vehicle locations, and battery levels. The pricing policy chooses a mobile price at each available node. Customers then choose mobile charging, an accessible station, or the outside option, meaning they leave. The dispatch policy sees these choices and chooses an action for each vehicle. The simulator executes the actions and calculates the operator's profit.

That period's profit is the **reward**. The **return** is the sum of rewards from the current period through the end of the episode, with later profit given less weight when discounting is used. Vehicles share the operator's total reward. A vehicle that recharges may reduce the current period's profit but enable more profitable service later, which is why training considers return as well as immediate reward.

The following table connects the manuscript's sequence to the **API**, the Python methods and input/output fields used to interact with the simulator. A **state** is the information available at a decision point, and an **actor** is the network that chooses the action. Mathematical periods are numbered $t=1,\ldots,T$; the observation's code field `period` is `t - 1`.

| Point in the period | Mathematical object | Code behavior |
|---|---|---|
| Before prices and customer choice | Pre-choice state $s_t$ | `stage == 0`; pricing actor acts |
| After prices and customer choice | Post-choice state $\widetilde{s}_t$, including the customers' actual choices $D_t$ | `stage == 1`; dispatch actors act |
| After physical execution | Period profit $r_t$, then the next state | `dispatch(...)` returns the reward; next stage is `0`, or `2` at termination |

Thus, the paper's **first and second decision stages** map to API stages **0 and 1**. API stage `2` means that the episode has ended; it is not a third decision stage. [ENVIRONMENT.md](ENVIRONMENT.md) gives the complete API fields, reset behavior, and action masks.

The simulator runs periods; the trainer organizes learning into the following units:

| Term | Meaning in this project |
|---|---|
| **Rollout or collected batch** | The recorded complete episodes collected before changing the policies |
| **Update** | Collect one batch, then use it to adjust the networks |
| **Epoch** | One pass through that collected batch during learning |
| **Minibatch** | A smaller group of recorded physical periods processed together; each includes both stages and all vehicles |
| **Optimizer step** | One round of adjusting network parameters using a minibatch; the optimizer is the calculation that makes those adjustments |

The policies stay fixed while the whole batch is collected. For example, 4 episodes of 12 periods produce 48 recorded periods. With minibatches of 12 periods, one epoch has 4 learning steps. Running 2 epochs gives 8 steps over the same recorded experience, unless an early-stop check intervenes. It does not create additional simulated episodes. One training update can therefore contain several optimizer steps.

## 2. Run a small execution check

Open a terminal in the project root—the folder containing `pyproject.toml`. Activate the existing **virtual environment**, which keeps this project's Python packages together, and install the project with its optional training packages:

```bash
source .venv/bin/activate
python -m pip install -e '.[train]'
python -m charging_env.rl.train \
  --updates 10 --episodes-per-update 4 --seed 7 \
  --output-dir outputs/sa_ppo_mappo
```

If `.venv` does not exist, create it first with `python3 -m venv .venv`. In the install command, `-e` makes the project editable, so saved source changes are picked up without reinstalling it; `[train]` adds the packages needed for training, including PyTorch. Training uses the CPU by default and needs no GPU, online service, or dataset download. To use the direct package versions tested here, run `python -m pip install -r requirements-training-tested.txt` before installation. That file records tested NumPy, Gymnasium, and PyTorch versions for Python 3.12.4, rather than fixing every package version on every platform.

Here `--updates 10` means 10 collect-and-learn cycles, and `--episodes-per-update 4` means 4 episodes in each cycle. The command therefore collects 40 training episodes: 480 physical periods, containing 960 decision-stage transitions because each period has pricing and dispatch. It also measures the initial and final policies without learning. This checks that collection, network adjustment, saving, and loading work; the run is too small to establish that profit has improved reliably.

The CLI automatically uses a fresh output directory. If `outputs/sa_ppo_mappo` already exists, it chooses `outputs/sa_ppo_mappo_2`, then `_3`, and prints the actual path. You can repeat the same command while preserving previous results. A **checkpoint** is a saved snapshot of the networks and training state; continuing from one also writes to a fresh destination, as explained in Section 8. Direct Python training calls and the dedicated study runners still require an unused destination.

Two other ways to start training use the same command options:

- `python examples/train_sa_ppo_mappo.py` calls the same training entry point.
- In VS Code, select `.venv/bin/python`, then choose **Run and Debug → Train SA-PPO–MAPPO (small run)** to run with breakpoints.

### Choose the customer model explicitly

Without `--scenario`, the trainer uses `Scenario()`. To use a JSON configuration, supply its path:

```bash
python -m charging_env.rl.train --scenario configs/toy_city.json \
  --updates 100 --episodes-per-update 8 --eval-episodes 20 \
  --output-dir outputs/sa_ppo_mappo_longer
```

The original customer-choice model has a common charging benefit $b=0$, called `charging_utility_intercept` in the configuration. `configs/utility_b3.json` sets $b=3$ for both mobile and fixed-station service, making charging more attractive relative to leaving. The outside option's preference score stays zero, and cost sensitivity remains $\eta=0.7$. The `utility_b2.json` and `utility_b4.json` files let us compare smaller and larger charging benefits. These change customer behavior in the environment, rather than how the networks learn. See [the outside-option explanation](OUTSIDE_OPTION.md).

The scenario is saved inside each checkpoint. A changed customer model requires a new run: `--scenario` cannot replace the scenario of a resumed or evaluated checkpoint. Both `run_convergence.py` and `run_policy_benchmarks.py` accept `--scenario`, but their defaults do not change merely because a JSON file exists.

## 3. What the four networks do

An **actor** is a neural network that chooses actions. A **critic** is a neural network that predicts the team's remaining discounted return from the current situation. It does not choose actions or know the correct answer in advance; it learns its predictions from collected experience. The actor uses those predictions as a reference when judging outcomes. This project has two actors and two critics because pricing happens before customer choice and dispatch happens afterward.

| Network | Input | Output and purpose |
|---|---|---|
| Pricing actor $\pi^P_{\theta_1}$ | Global pre-choice state $s_t$: the full information available before pricing | Probabilities for the allowed prices at each node; it selects all node prices together |
| Pricing critic $V^P$ | Global pre-choice state $s_t$ | A prediction of the remaining discounted team return before pricing |
| Shared dispatch actor $\pi^D_{\theta_2}$ | `local_observation(m)` after customer choice | Vehicle $m$'s operation, destination, and recharge/export fraction when needed |
| Dispatch critic $V^D$ | Global post-choice state $\widetilde{s}_t$: the full information after customers choose | A prediction of the remaining discounted team return before dispatch |

The dispatch actor is **shared**: both vehicles use the same learned decision rule, but each supplies its own observation. They can therefore choose different actions. During training, the dispatch critic sees the full shared state to help assess the outcome. When vehicles act, each uses the actor with its own input and does not need the critic. This arrangement is called **centralized training with decentralized execution**. A vehicle's input still includes other vehicles' public locations and batteries, plus shared demand and prices; “local observation” does not mean it sees only itself.

### Pricing: choose from a list at each node

At each available service node $i\in\mathcal I$, the actor assigns probabilities to the allowed price levels and samples one level $\ell\in\mathcal L$. The resulting unit price is $p^M_{i,t}$. A probability distribution over a fixed list of choices is called a **categorical distribution**.

The whole price action contains all node prices. Its **joint probability** is the product of the individual node probabilities. The code stores logarithms of probabilities, called **log probabilities**, so it can add them instead of multiplying many small numbers. PPO compares the new and old probability of this whole action, using **one joint pricing probability ratio**. Section 4 explains what this ratio does.

If mobile service is unavailable at a node, the actor supplies placeholder price index `0` and excludes that node from its learning and exploration calculations. If no node has mobile service, the pricing action is still valid, but that state contributes no pricing-action learning signal to the update.

### Dispatch: choose a compatible mode and target together

For each vehicle $m\in\mathcal M$, the dispatch actor chooses from a list of **mode–target pairs**. A mode is the operation, such as SERVE or RECHARGE; the target is the destination node. An **action mask** identifies the pairs allowed by the physical model. Choosing the pair together prevents combinations such as RECHARGE at a node with no station.

For `RECHARGE` or `DISCHARGE`, the actor also chooses an energy amount. It uses a **Beta distribution**, a flexible distribution for a continuous fraction between zero and one. The fraction's distribution depends on the chosen operation and destination; this is what **conditional** means here. The trainer multiplies that fraction by

```text
max(mobile_charge_kw, mobile_discharge_kw) * period_hours
```

to obtain the requested energy at the grid connection in kWh. With the default scale of 30 kWh, a fraction of 0.6 requests 18 kWh. For every other operation, exchange energy is zero. The network branch that predicts the fraction, called the **energy head**, then contributes nothing to the action's log probability.

Travel time, battery limits, energy losses, and competition for station capacity can reduce the amount actually exchanged. PPO scores the **requested action**, before the simulator reduces it to fit those limits. Each vehicle has its own new-versus-old action probability ratio, but all vehicles use the same team assessment of the outcome, called the dispatch advantage. The code does not multiply these ratios across vehicles.

**Technical detail: scoring a continuous energy fraction.** A continuous distribution uses a **probability density**, which describes how likely a small interval around a number is. It is not the probability of choosing that one exact number. The code saves the sampled fraction and evaluates its Beta log density. Converting fractions to kWh changes density by a constant scale factor, sometimes called a Jacobian factor. This factor cancels when comparing the new and old policies because both use the same scale. The exploration calculation also uses fractions, rather than kWh.

By default, the input-processing code uses `agent_id` to align availability information with the current vehicle and the other vehicles. With `--dispatch-agent-id`, it also gives the actor an explicit identity feature. This is a **one-hot vector**: a 1 at this vehicle's position and zeros elsewhere, such as `[1, 0]` for vehicle 0 and `[0, 1]` for vehicle 1. It allows otherwise identical vehicles to learn different roles while using the same actor.

### How variable numbers of requests enter a network

An **encoder** converts raw observations into numeric features useful for the network. Each of the four networks has its own encoder for customer requests. It processes every request, then combines the resulting features into a fixed-length summary. Combining them is called **pooling**. This lets the networks handle different numbers of customers, including periods with no requests, without imposing a maximum request count.

Request position in the list is included because service allocation checks requests in order. The summary can retain useful order information, but compressing a long list into a limited number of features cannot preserve every detail. Other inputs are scaled using fixed physical or scenario quantities so their numbers are easier for the networks to work with. Scaling changes their units or magnitude; it does not cap customer demand.

### Optional architecture experiments

An **activation function** transforms values inside a neural network, allowing it to learn relationships more complex than a straight-line calculation. **Hidden layers** are the intermediate calculations between inputs and outputs. Tanh compresses values into the range from -1 to 1. SiLU smoothly rescales its input and allows large positive outputs. `--dispatch-final-activation silu` replaces only the dispatch actor's last hidden activation, `dispatch_actor.encoder.body[-1]`, with `nn.SiLU()`. The default is `tanh`; this flag does not change the other activations, actors, critics, masks, or action distributions. [The activation report](ACTIVATION_EXPERIMENT.md) describes the controlled comparison.

**Layer normalization**, or LayerNorm, rescales the values within a hidden layer before its activation. This can make their scale easier to learn from, but does not guarantee a better policy. `--encoder-layer-norm` adds it before both activations in the main dense layers of all four networks, while leaving their request-processing networks unchanged. A **dense layer** is a layer whose outputs each use all its inputs. These choices of network structure are called the **architecture** and are saved in checkpoints. Compare different architectures using new training runs with clearly recorded budgets.

## 4. How an update learns from both stages

One training update follows this sequence:

1. **Collect experience.** Run the configured number of complete episodes with the current actors. Save what they saw, which actions were allowed, what they chose, the log probabilities assigned by the collecting policies, the critics' predictions, and the actual period profits. The collecting policy is also called the **behavior policy**.
2. **Assess the outcomes.** Estimate how much better or worse the outcomes were than the critics expected. These estimates are called **advantages**.
3. **Adjust the actors.** Use the advantages to adjust action probabilities. The pricing actor receives a pricing advantage, while each vehicle's dispatch decision receives the same team dispatch advantage.
4. **Adjust the critics.** Train each critic to predict its **return target**, the estimate of remaining return calculated from the episode and the critic predictions saved during collection. These targets are fixed before the minibatch learning steps start.
5. **Repeat over the recorded data.** Process minibatches for up to the configured number of epochs. If the action distribution has changed too much, stop the remaining optimization work for this update.
6. **Collect new experience.** Run another batch with the adjusted policies. Old batches are not carried into later updates. This use of newly collected experience is called **on-policy learning**.

The intuition behind an advantage is “better or worse than expected.” If a critic predicted 20 units of remaining return and the estimated outcome is 25, the outcome is 5 better than expected. Training tends to make the chosen action more likely. A worse-than-expected outcome tends to make it less likely. The actual calculation below combines both decision stages and later outcomes; this example only explains the comparison.

The networks are adjusted using a **loss**: a numerical score that the learning procedure tries to reduce. Actor losses encourage actions with positive advantages. Critic losses measure errors in return predictions. A loss is not the business profit, so its numerical value should not be read as money earned by the operator.

### How PPO keeps learning changes cautious

PPO compares how strongly the current actor favors each recorded action with how strongly the collecting actor favored it. This is the **new-policy/old-policy probability ratio**. For example, an old probability of 0.25 and a new probability of 0.30 give a ratio of 1.2.

With `--clip-ratio 0.2`, the learning score compares the ordinary ratio with a version restricted to the interval 0.8–1.2, then uses the more cautious score for the action's advantage. This is the **clipped surrogate objective**: a practical score for adjusting the policy using recorded data. It reduces the incentive to make an already large favorable change even larger. It does **not** guarantee that actual probabilities or their ratios stay inside that interval.

The trainer also monitors **KL divergence**, short for Kullback–Leibler divergence, which measures how different action distributions are. It estimates this change using the recorded actions. If either actor's approximate KL exceeds `--target-kl`, the trainer skips the remaining minibatches and epochs for both actors and critics in that update. It keeps changes already made and then moves on to collecting a fresh batch. This does not stop the whole training run.

**Entropy** measures how spread out a policy's choices are. An entropy bonus encourages the actor to keep trying alternatives rather than immediately concentrating on one action. Its coefficient sets the strength of that encouragement. More exploration does not guarantee more profitable exploration. For the continuous Beta fraction, entropy is a density-based quantity and can be negative; its sign alone does not indicate a bug.

**Value clipping** is a separate option for critics. It compares prediction error with an error calculated from a prediction restricted to a range around its old value, and uses the larger error. This discourages large changes in critic predictions; it does not impose a strict bound on them. `--value-clip 0` turns this option off. These learning safeguards do not change the simulator's charging rules or profit formula.

### Discount time once per physical period

The objective follows the manuscript's period numbering:

$$
J(\theta_1,\theta_2)
=\mathbb E\!\left[\sum_{t=1}^{T}\gamma^{t-1}r_t\right].
$$

**Discounting** means giving later profit slightly less weight when calculating return. Here $\gamma$ is `Scenario.discount_per_period`, default `0.99`. For example, profits of 10 and 20 in a two-period episode give $10+0.99\times20=29.8$. Pricing and dispatch within the first period are two decisions, not two periods, so they do not use two discounts.

For learning, the pricing-to-dispatch transition uses reward zero and discount one. Moving from dispatch to the next period uses the period's reward $r_t$ and discount $\gamma$. To assess a decision, the trainer compares the current prediction with the reward just observed plus the prediction of what remains. This prediction error is called a **temporal-difference residual**, written as delta below. Pricing has no immediate reward, so its residual only compares the before-choice and after-choice predictions:

$$
\delta_t^P=V^D(\widetilde{s}_t)-V^P(s_t),
\qquad
\delta_t^D=r_t+\gamma(1-d_t)V^P(s_{t+1})-V^D(\widetilde{s}_t),
$$

Here $d_t=1$ at the final physical period and zero otherwise. At the end, no later profit remains, so the next-period value is excluded.

One prediction error can be noisy or incomplete. **Generalized advantage estimation (GAE)** combines it with errors from later periods to assess the earlier decision. The code calculates these advantages backward from the end of each episode:

$$
A_t^D=\delta_t^D+
\gamma\lambda_{\mathrm{GAE}}(1-d_t)A_{t+1}^P,
\qquad
A_t^P=\delta_t^P+A_t^D.
$$

$\lambda_{\mathrm{GAE}}$, set by `--gae-lambda`, controls how much later prediction errors influence an earlier advantage. A smaller value relies more on short-range estimates using the critics. A value nearer one gives later recorded outcomes more influence, which can also make the estimates noisier. This lambda is different from $\lambda_{i,t}$, the average customer-arrival count in the physical model.

The **trace coefficient** is the factor carrying a later error back toward an earlier decision. It is one between pricing and dispatch in the same period, and uses $\lambda_{\mathrm{GAE}}$ between periods. This placement is an implementation choice left unspecified by the manuscript. At $\lambda_{\mathrm{GAE}}=1$, both stage targets $A+V$ equal the discounted return calculated from that period through the end of the episode. If the optional inventory signal below is enabled, this is the return from the modified learning rewards. **Bootstrapping** means using a critic prediction for future return; there is no such prediction after the final period. Credit calculations also stop at each episode's end, so one episode cannot affect another's targets.

### Why the policy loss also uses a period weight

There are two related timing questions. First, what are the rewards after the current decision worth? Discounting inside GAE answers that. Second, how much weight should this decision have when learning the total profit measured from the start of the episode? New runs use a period weight for that second question: both actor learning scores and entropy bonuses are multiplied by $\gamma^{t-1}$, implemented as `gamma ** observation['period']`.

Both decisions within a period receive the same weight. Before using advantages in actor losses, the trainer subtracts their average and divides by their standard deviation, a measure of spread. This is **advantage normalization**: it keeps their scale easier to optimize. The averages and standard deviations use the same period weights. Pricing includes only states where at least one node has mobile service. Critics still predict targets in the original reward units; their targets are not normalized this way. The period weights match the intended objective, but clipping and normalization are practical approximations; they do not guarantee exact optimization of that objective.

`--no-discount-policy-loss` gives every period equal weight in the actor loss, as an earlier implementation did. Older checkpoints keep that behavior unless it is explicitly changed when continuing training.

### Optimizers and the optional inventory signal

An **optimizer** is the procedure that changes network parameters to reduce a loss. This project uses **Adam**, which uses current gradients and a running history of past gradients to choose adjustments. A **gradient** describes how changing each parameter would change the loss. The **learning rate** sets the overall scale of the adjustments. One Adam optimizer handles both actors, and another handles both critics, so the two groups can use different learning rates.

**Gradient clipping** reduces an unusually large gradient before Adam uses it. The **gradient norm** measures the combined size of that gradient. By default, the trainer clips the combined actor gradients and the combined critic gradients separately. `--separate-network-grad-clip` instead clips each of the four networks separately. This limits gradient size, not the exact change in parameters or action probabilities; it is different from PPO probability-ratio clipping.

The optional `--inventory-potential c` changes the signal used for learning while retaining the original profit calculation for reporting. A **potential** is a score assigned to a state; here it is total stored MCV energy multiplied by a coefficient `c`. **Reward shaping** uses changes in this score to redistribute the learning signal between periods. It defines

$$
\Phi(s_t)=c\sum_{m\in\mathcal M}e_{m,t},
\qquad
r'_t=r_t+\gamma\Phi(s_{t+1})-\Phi(s_t),
$$

The final potential is set to $\Phi(s_{T+1})=0$. GAE and critic targets use the modified learning reward $r'_t$, while all reported training profit and evaluation results still use the actual business profit $r_t$. Pricing's zero-reward transition is unchanged. When this option is enabled, the critics predict modified training returns rather than raw profit returns.

Across a complete discounted episode, the added terms cancel except for the initial potential. This cancellation is called **telescoping**: the modified total equals the original discounted total minus the initial state's potential. For a fixed starting state, that subtraction is the same whatever the policy does. The method therefore does not add business profit, give an extra final-battery reward, or guarantee that the vehicles learn to recharge well. It can change how learning proceeds. The coefficient $c$ has currency units per stored kWh, giving $\Phi$ the same units as profit. Its default is zero, which leaves learning rewards unchanged.

## 5. Configuration reference

The **command-line interface (CLI)** is the set of options supplied after the training command. Run `python -m charging_env.rl.train --help` to list them. These options are training hyperparameters: settings you choose, rather than numbers learned by the networks. Each is also a field of `TrainingConfig`, with underscores in place of hyphens in Python names. `None` means a setting was not supplied and the related setting should be used; it does not mean zero.

A **seed** is a number used to reproduce a sequence of random draws. Separate settings control training randomness and evaluation randomness. Options written as `--[no-]...` have an on and an off form: for example, `--encoder-layer-norm` enables the option and `--no-encoder-layer-norm` disables it.

### Data collection and optimization

| CLI option | New-run default | Meaning |
|---|---:|---|
| `--seed` | 7 | Controls the initial network parameters and training random draws |
| `--updates` | 10 | Number of collect-and-learn cycles to run; when resuming, this is the number to add |
| `--episodes-per-update` | 4 | Complete episodes to collect before each update |
| `--epochs` | 4 | Maximum number of passes through that collected data |
| `--minibatch-size` | 48 | Recorded physical periods processed in each learning step, including both stages and every vehicle |
| `--learning-rate` | 0.0003 | Scale of actor parameter adjustments; also used for critics unless a separate rate is set |
| `--critic-learning-rate` | `None` | Optional separate adjustment rate for the critics |
| `--clip-ratio` | 0.2 | PPO uses the interval 1 minus this value to 1 plus this value in its clipped learning score |
| `--value-clip` | 0.2 | Range around old critic predictions used in the clipping loss, measured in reward units; zero disables it |
| `--gae-lambda` | 0.95 | $\lambda_{\mathrm{GAE}}$: how much later prediction errors affect earlier advantages |
| `--value-coef` | 0.5 | Multiplier on the critics' prediction-error losses |
| `--entropy-coef` | 0.01 | Strength of the exploration bonus for both actors, unless dispatch has its own setting |
| `--max-grad-norm` | 0.5 | Maximum combined gradient size before the optimizer uses it |
| `--target-kl` | 0.03 | Stop the rest of an update if estimated action-distribution change exceeds this threshold; zero disables this check |
| `--[no-]discount-policy-loss` | `True` | Apply the period weights $\gamma^{t-1}$ to actor scores and exploration bonuses, and use them when normalizing advantages |
| `--training-seed-base` | 0 | Starting offset for training environment seeds; the formula is `2*(base + seed + episode_count)` |
| `--device` | `cpu` | Run on the CPU, or a CUDA GPU if available |
| `--torch-threads` | 1 | Number of CPU processing threads PyTorch may use |

The discount factor comes from the scenario's `discount_per_period`; it is applied per physical period and has no separate training CLI option.

### Architecture and optional training variants

| CLI option | New-run default | Meaning |
|---|---:|---|
| `--hidden-size` | 64 | Number of units in the networks' main hidden layers |
| `--dispatch-final-activation` | `tanh` | Function used at the dispatch actor's final hidden activation: `tanh` or `silu` |
| `--[no-]dispatch-agent-id` | `False` | Include an explicit identity feature so the shared actor can distinguish vehicles |
| `--[no-]encoder-layer-norm` | `False` | Rescale values before both main hidden activations in every actor and critic |
| `--inventory-potential` | 0.0 | Stored-energy score per kWh used to modify learning rewards; the score after termination is zero |
| `--dispatch-entropy-coef` | `None` | Separate strength for dispatch exploration; otherwise use `entropy_coef` |
| `--dispatch-entropy-final-coef` | `None` | Optional final dispatch exploration strength reached by a gradual linear change |
| `--dispatch-entropy-anneal-updates` | 0 | Number of updates over which that linear change runs; must be positive when a final strength is supplied |
| `--[no-]separate-network-grad-clip` | `False` | Limit the gradients of each actor and critic separately |

Changing a coefficient gradually over training is called **annealing**. The dispatch entropy schedule keeps its own saved number of updates, so adding updates later does not restart or stretch it. These options are experiments, not automatically enabled improvements; loading a recent report does not change the CLI defaults.

### Validation and checkpoint selection

| CLI option | New-run default | Meaning |
|---|---:|---|
| `--eval-episodes` | 4 | Episodes used to measure each execution mode at each validation checkpoint |
| `--eval-interval` | 0 | Number of total updates between validation checks; zero checks only the start and end |
| `--eval-seed-base` | 100001 | First seed for validation scenarios; must be a nonnegative odd integer |
| `--eval-action-seed` | 12345 | Seed offset controlling sampled actor choices during stochastic evaluation |
| `--eval-policy` | `deterministic` | Execution mode whose score selects `best_checkpoint.pt` and supplies the main summary; both modes are still monitored |

A minimal Python equivalent is:

```python
from charging_env import Scenario
from charging_env.rl.trainer import Trainer, TrainingConfig

trainer = Trainer(Scenario(), TrainingConfig(updates=10, gae_lambda=0.95))
summary = trainer.train("outputs/sa_ppo_mappo_python")
```

A different scenario can change the number of network inputs or available actions—for example, by adding vehicles or nodes. Checkpoints therefore save the exact scenario as well as the learned parameters.

## 6. Read the saved results correctly

A completed run saves the files below. Start with `summary.json` for the outcome, `training_config.json` and `scenario.json` for the exact setup, and `validation_metrics.csv` for performance over training. A **CSV** is a table you can open in a spreadsheet; a **JSON** file stores named fields and values.

| File | What it tells you |
|---|---|
| `training_metrics.csv` | Actual training profit plus learning diagnostics for each update: losses, policy-change estimates, clipping, gradient sizes, parameter changes, and critic prediction accuracy |
| `validation_metrics.csv` | Initial, periodic, and final validation for deterministic and stochastic execution, including service and reward components |
| `evaluation_before.json`, `evaluation_after.json` | Initial/final validation under the configured `eval_policy` |
| `validation_before.json`, `validation_after.json` | Complete initial/final validation for both execution policies |
| `checkpoint.pt` | Latest saved networks, optimizer history, settings, scenario, progress counters, and random-generator states; this is the final training state after a completed run |
| `best_checkpoint.pt` | Saved snapshot with the highest average validation discounted profit among those evaluated during this run segment, using `eval_policy` |
| `best_validation.json` | Selected update, selection policy, and validation result |
| `scenario.json`, `training_config.json` | Exact settings used |
| `summary.json` | Run summary and final evaluation |

The loss columns describe how learning is proceeding, not how much business profit the policy earns. To judge profit improvement, read the validation and independent test results. A smaller learning loss alone does not establish a better controller.

Two metrics describe how well a critic predicts targets from its current collected batch:

- **Root mean squared error (RMSE)** is a typical prediction-error size in the target's units. Smaller means a closer fit to those targets.
- **Explained variance** measures how much variation in targets remains unexplained by the prediction errors. Values near one mean little residual variation; zero or negative values indicate a poor fit by that measure. A constant prediction can score zero even if some of its errors are small. The code reports zero when the targets have almost no variation.

These describe fit to the current training targets, not performance on new scenarios. Explained variance can miss a constant prediction offset, so read it alongside RMSE. Logged entropy also includes period weights when `discount_policy_loss` is enabled. It is the exploration quantity used in the loss, so its scale need not match an unweighted entropy calculation.

### Deterministic and stochastic execution answer different questions

**Validation** means measuring a policy on designated scenarios without learning from those episodes. At each validation check, the trainer monitors two ways of using the same learned actors:

- **Stochastic execution** samples prices, vehicle actions, and energy fractions from the learned distributions, as training does.
- **Deterministic execution** selects the highest-probability price and mode–target choices, and uses the mean of the corresponding Beta energy fraction. This makes the actors' choices repeatable for the same input. The environment still generates customer arrivals, choices, and price changes randomly from its seed.

For example, probabilities `[0.49, 0.51]` make deterministic execution choose the second option; `[0.51, 0.49]` make it choose the first. That small network change switches the chosen action completely. Stochastic execution samples both options in either case, using those slightly different probabilities. This helps explain why the two performance curves can differ. Compare each execution mode with its own initial evaluation.

The training and validation scenarios use separate random seeds. Training environment seeds are even, while validation uses odd seeds starting from `eval_seed_base` and increasing by two. The same validation seeds are reused at each check so changes can be compared on the same scenarios. Validation episodes do not update the networks, although their scores can influence checkpoint selection.

**Technical detail: reproducing random evaluation choices.** A **random-number generator (RNG)** produces random draws from a repeatable starting state. Validation environment seeds are `eval_seed_base + 2*i`, with `i` from zero to `eval_episodes - 1`. Stochastic evaluation uses `(eval_action_seed + environment_seed) % (2**63 - 1)` as its action seed. Evaluation saves and restores the training process's CPU/CUDA Torch RNG state, so a monitoring check does not change the random actions drawn when training resumes.

### “Final” and “best” are different checkpoints

The trainer always evaluates before and after learning. If periodic checks are enabled, their timing uses the total update count, including updates completed before a resume. A **run segment** means the work written to one output directory, whether it starts from scratch or continues a checkpoint. Best-checkpoint selection compares only the evaluations in that segment, including its starting policy. Equal scores keep the earlier checkpoint.

Consequently, `best_checkpoint.pt` may contain the starting policy if learning never beats it on validation. `checkpoint.pt`, `evaluation_after.json`, and the summary's final score describe the latest update. `best_validation.json` describes the snapshot selected by validation. After training, the trainer reloads the final checkpoint and checks that evaluation in the selected mode reproduces the result exactly.

### Decide whether a policy is good using fresh tests

The **validation set** is the group of scenarios used repeatedly to monitor learning and select a checkpoint. Because those scores help choose the model, the best validation score can be overly optimistic. An **independent test set** uses new scenarios after training settings, checkpoint choice, and execution mode have been fixed. Choose its environment and action seeds separately, and do not use its results to choose whichever model happens to look best.

Compare the final and validation-selected policies with **benchmarks**: alternative decision rules, including simple rules and stronger inventory-aware rules. Repeat training with several seeds to check whether a result depends on one particular training run. When policies use the same test scenarios, subtract their profits scenario by scenario before averaging. These are **paired differences**, which make the comparison easier to interpret than differences between unrelated sets of episodes. Also read mobile energy delivery, the share of customers leaving, and unmet demand; higher total operator profit can occur alongside less mobile charging.

Evaluation JSON reports actual profit, once-per-period discounted profit, revenue and cost components, action counts, and service measures. It also reports a **standard error**, which estimates sampling uncertainty in the average across evaluated episodes, and an approximate **95% confidence interval** based on a normal approximation. This interval describes episode-sampling uncertainty for the fixed evaluated policy. It does not include variation from retraining with different seeds or correct for choosing a model using the same results. With only one episode, the interval is unavailable and stored as `null`.

Four evaluation episodes are enough for a basic execution check but too few for a reliable performance conclusion. **Convergence** means learning has settled toward a limiting solution. A flat observed profit curve only shows that measured performance has stabilized over that range; it does not prove the policy has found the best solution.

[VALIDATION.md](VALIDATION.md) records execution checks. [POLICY_BENCHMARKS.md](POLICY_BENCHMARKS.md) explains the comparisons, and [TRAINING_ANALYSIS.md](TRAINING_ANALYSIS.md) reports the earlier longer study.

## 7. Run a longer study

A **preset** is a named or recorded collection of settings. The following preset was used in an earlier longer-training study. It collects more episodes before each update, uses a larger learning rate for critics than actors, turns off value clipping, and uses a smaller exploration bonus. It measures both execution modes every 25 updates and uses stochastic validation profit to select the best checkpoint:

```bash
python -m charging_env.rl.train \
  --updates 500 --episodes-per-update 16 --epochs 6 --minibatch-size 192 \
  --learning-rate 3e-4 --critic-learning-rate 1e-3 \
  --value-clip 0 --entropy-coef 0.005 --discount-policy-loss \
  --eval-episodes 32 --eval-interval 25 --eval-policy stochastic \
  --eval-seed-base 100001 --eval-action-seed 12345 --seed 7 \
  --output-dir outputs/sa_ppo_mappo_longer_500
```

For the default 12-period scenario, 500 updates times 16 episodes per update means 8,000 training episodes and 96,000 physical periods. These are the fresh simulated periods used for learning; the 6 epochs reuse each collected batch and do not multiply the amount of simulated experience. Evaluation episodes are additional. This is a recorded experiment setup, not a promise that more training will converge or beat a benchmark.

The dedicated runner uses the same preset:

```bash
python experiments/run_convergence.py --seed 7
```

It writes to `outputs/convergence_study/seed_7/` and also saves `checkpoint_250.pt` after update 250. That snapshot has used 48,000 learning periods, making it possible to compare policies given the same amount of simulated training experience. Use `--output-root outputs/convergence_repeat` for a new directory. The script learns from training episodes and chooses checkpoints using validation; independent testing is a separate step.

Later studies ask different questions. [The SiLU + b=3 report](SILU_B3_STUDY.md) changes both a network activation and customers' willingness to charge. [The targeted b=3 report](B3_TRAINING_IMPROVEMENTS.md) investigates pricing, battery replenishment, and coordination, then trains three fresh seeds and tests on independent scenarios. Its runner explicitly turns on its chosen experimental options. The ordinary training command still uses the defaults in Section 5; those options do not turn on just because a study used them.

The optional `charging_env.rl.warmstart.warm_start_from_inventory` helper tries **behavior cloning**, also called imitation learning. It first teaches the actors to copy examples from the inventory rule, rather than learning those examples from profit feedback. This initialization is called a **warm start**. Report the teacher-demonstration episodes and supervised learning steps separately from PPO's training budget. The helper never mixes teacher actions into an on-policy PPO batch. This experiment was tested in a development pilot but was not selected for the latest three full training runs.

## 8. Continue or evaluate a saved checkpoint

To restore training state and run 10 **additional** updates, write to a new directory:

```bash
python -m charging_env.rl.train \
  --resume outputs/sa_ppo_mappo/checkpoint.pt --updates 10 \
  --output-dir outputs/sa_ppo_mappo_continued
```

**Resume** means continue the same training process from a saved state. It restores the scenario, learned networks, optimizer history, progress counters, and random-generator states. Settings you do not supply keep their saved values. If a checkpoint has already completed 100 updates, `--updates 10` continues to 110. The saved dispatch exploration schedule keeps its original duration; adding updates does not restart or stretch it.

Some settings must match the checkpoint. The CLI rejects a replacement `--scenario` or changes to saved `--hidden-size`, `--seed`, `--dispatch-final-activation`, `--dispatch-agent-id`, `--encoder-layer-norm`, `--inventory-potential`, or `--training-seed-base`. Other listed settings can be changed explicitly. For example, supplying `--learning-rate` changes the actor rate and also the critic rate if no separate critic rate was set. Supplying `--critic-learning-rate` changes the critic rate independently; a previously saved separate critic rate is retained when only the actor rate changes. Loading uses the CPU unless an available `--device cuda` is requested.

For **evaluation only**, keep the saved parameters fixed and run new episodes to measure performance. Use fresh seeds when these episodes are intended as independent tests:

```bash
python -m charging_env.rl.train \
  --evaluate-only outputs/sa_ppo_mappo_longer_500/best_checkpoint.pt \
  --eval-episodes 128 --eval-seed-base 900001 --eval-action-seed 54321 \
  --eval-policy stochastic --output-dir outputs/sa_ppo_mappo_test_stochastic

python -m charging_env.rl.train \
  --evaluate-only outputs/sa_ppo_mappo_longer_500/best_checkpoint.pt \
  --eval-episodes 128 --eval-seed-base 900001 --eval-action-seed 54321 \
  --eval-policy deterministic --output-dir outputs/sa_ppo_mappo_test_deterministic
```

Evaluation-only mode loads the saved policy and scenario, runs the requested execution mode, and prints the result. If `--eval-policy` is omitted, it uses the saved mode. With `--output-dir`, it also saves `evaluation.json`, refusing to overwrite an existing file of that name. It does not change network parameters, create a checkpoint, write training metrics, or select a new best model. The environment and action seeds may be changed without changing how the original networks were initialized.

### Loading checkpoints saved by older versions

Older version-1 checkpoints remain loadable. If `dispatch_final_activation` was not recorded, the loader uses Tanh. Tanh and SiLU have no adjustable parameters themselves, but apply different calculations to the same network outputs. The activation choice must therefore be saved even when all learned weight shapes match.

If an old checkpoint has no field for period-weighted actor losses, the loader uses `discount_policy_loss=False` to preserve its earlier behavior of weighting all periods equally. If its validation seed base is missing, the loader uses the earlier formula `2*saved_seed + 1`. Other missing new settings receive their defaults. This keeps earlier results tied to the settings actually used at the time.

To use the current period weighting and regular stochastic validation while continuing an older run, explicitly add:

```bash
--discount-policy-loss --eval-seed-base 100001 --eval-interval 25 --eval-policy stochastic
```

New runs already default to `discount_policy_loss=True` and `eval_seed_base=100001`. Use `--eval-interval 25` and `--eval-policy stochastic` explicitly if you also want the monitoring choices shown above.
