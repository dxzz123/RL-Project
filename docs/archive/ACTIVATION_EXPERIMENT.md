# Does SiLU help the dispatch policy respond to its state?

> Historical experiment. Its numbers belong to the scenarios and methods recorded here. Use [the current comparison](../COMPARISON.md) for approximate DP, both PPO versions, and the three main rules on shared tests. See [the archive index](README.md) for context.

Paths under `outputs/` refer to local experiment records and are not included in Git. Use the saved study files or reproduce the experiment to inspect them; [TRAINING.md](../TRAINING.md) explains the setup and current workflow.

A dispatch policy is the learned rule for choosing vehicle actions. It should make different decisions when vehicle energy, locations, or customer requests change. The original trained policy appeared to respond very little to those changes. This experiment asked whether a small neural-network change could help: replace the dispatch actor's **last hidden Tanh activation with SiLU**.

This **pilot experiment** used a fixed training budget and was completed on **2026-10-17** under the baseline customer model, $`b=0`$. It explains what changed, what the comparison found, and what the result does not establish. Read [Earlier PPO reference](../LEGACY_PPO.md) for the full architecture, [MODEL.md](../MODEL.md) for the environment, and [NOTATION.md](../NOTATION.md) for symbols.

## 1. Why test a different activation?

An **activation function** transforms values inside a neural network before the next layer uses them. **Hidden features** are the intermediate numbers the network calculates from an observation. Tanh maps its input to a value between −1 and 1. For sufficiently large positive or negative inputs, its output changes very little: it is **saturated**. Different observations can then produce almost the same hidden features. The **local derivative**, which measures how much the activation's output changes for a small input change, is also small. This can weaken the signal used to adjust earlier network weights during training.

For example, two observations with different remaining vehicle energy may enter the network but produce almost the same saturated final hidden vector. The later layers then have little information with which to give those states different action scores. This motivates checking the network's internal features; it does not prove that an activation caused poor profit.

A diagnostic that left the saved models unchanged supported this concern. It examined the three final convergence-study checkpoints, which are saved model snapshots. It supplied **384 shared local observations** from 16 inventory-rule episodes, so each model received the same inputs. All final dispatch hidden values satisfied `abs(tanh) > 0.99`, and all 64 features had across-state standard deviation below `0.001`. Standard deviation measures how much a feature varies across those observations; a very small value means it is almost constant.

| Training seed | Mean local Tanh derivative |
|---|---:|
| 7 | 0.002559 |
| 19 | 0.001676 |
| 37 | 0.002735 |

Action probabilities also changed very little among states with identical **feasibility masks**, the lists of physically allowed actions. Inputs are normalized, meaning scaled using scenario quantities before entering the network. The largest absolute scaled input was `2.3214`; this check did not reveal an obvious extreme input scale. These findings motivated a controlled trial, without establishing that an activation change alone would improve decisions. The saved probe is `outputs/activation_diagnostics/legacy_dispatch_shared_state_probe.json`.

## 2. What changed, and how checkpoints record it

SiLU computes

```math
\mathrm{SiLU}(x)=x\,\mathrm{sigmoid}(x).
```

It does not bound positive outputs at one. Consequently, `abs(h) > 0.99` is a useful diagnostic of near-boundary **Tanh** values, but is not a SiLU saturation test. The experiment instead measured local activation derivatives and how hidden features and action scores varied across a fixed collection of states. An action score is the number the network produces before converting it into an action probability. The [PyTorch reference](https://docs.pytorch.org/docs/stable/generated/torch.nn.SiLU.html) describes the activation.

`DispatchActor(..., final_activation='silu')` replaces only `encoder.body[-1]` with `torch.nn.SiLU()`. In the historical pilot's layout, with no LayerNorm, that element was `encoder.body[3]`. Referring to the final element also works with the current optional LayerNorm layout. **Layer normalization**, or LayerNorm, rescales the values inside a hidden layer before its activation.

The following parts retained their definitions in this isolated comparison:

| Component | Pilot setting |
|---|---|
| Dispatch actor's earlier hidden activation and request encoder | Tanh |
| Pricing actor and both critics | Existing architectures |
| Dispatch categorical and Beta distributions | Existing definitions |
| Action masks and requested-energy mapping | Existing definitions |
| Customer choice, outside option, physical limits, and profit | Original model, including $`b=0`$ |
| Benchmark rule parameters | Previously selected and frozen |

`TrainingConfig.dispatch_final_activation` and CLI option `--dispatch-final-activation` accept `tanh` or `silu`. The default remains Tanh. Each checkpoint explicitly saves its activation; old checkpoints missing the field load as Tanh.

The command-line interface (CLI) rejects changing this setting when resuming training or evaluating a saved model. Tanh and SiLU have no adjustable weights of their own, so the networks' weight arrays have the same shapes. Replacing one activation with the other still changes how observations become actions. Comparing the two network designs therefore requires starting both training runs from scratch.

## 3. How the comparison avoided choosing a favorable result

The pilot fixed its budget before testing: **training seed 7, 100 updates, 16 complete episodes per update**, or **19,200 physical periods per variant**. Both variants started from identical weight arrays and used the same sequence of environment random seeds. A seed determines the sequence of simulated random events, allowing a run to be reproduced.

The learning settings were the existing convergence-study settings, with no additional tuning:

| Setting | Value in both variants |
|---|---:|
| Hidden width | 64 |
| PPO epochs per update | 6 |
| Minibatch size in physical periods | 192 |
| Actor learning rate | 0.0003 |
| Critic learning rate | 0.001 |
| Value clipping | Disabled |
| Entropy coefficient | 0.005 |
| PPO ratio clipping | 0.2 |
| Gradient-norm limit | 0.5 |
| GAE trace parameter $`\lambda_{\mathrm{GAE}}`$ | 0.95 |
| Physical-period discount $`\gamma`$ | 0.99 |
| Outer discounted policy loss | Enabled |

The GAE parameter $`\lambda_{\mathrm{GAE}}`$ controls how the trainer estimates an action's advantage: how much better its outcome was than the critic expected. It is distinct from the average request-arrival count $`\lambda_{i,t}`$ in the model. The training guide explains the remaining settings, including PPO's clipped learning score, which discourages overly large probability changes, and entropy's encouragement to explore.

Three separate sets of scenes served different purposes:

| Set | Environment seeds | Use |
|---|---|---|
| Validation | 32 scenes, `2300001 + 2*i` | Monitor both models every 25 updates; action-seed base `173971` |
| Independent test | 256 scenes, `2500001 + 2*i` | Compare frozen policies after training; action-seed base `193939` |
| Representation diagnostic | `2700001 + 2*i` | Generate an identical bank of local observations using the frozen inventory rule |

Here `i` starts at zero. The diagnostic collection contained 768 local observations. It was used only to inspect the networks, with no learning or checkpoint selection. Giving both networks the same observations lets us compare their responses directly. Otherwise, they might appear different simply because their own actions led them into different situations.

The main comparison was **final stochastic SiLU minus final stochastic Tanh**: each policy sampled actions from its learned probabilities after the full training budget. Results from checkpoints chosen using validation scores and from deterministic execution, which chooses the highest-probability discrete actions and the mean energy fraction, were additional comparisons. Neither could replace the main comparison because it happened to score better on the test. The frozen inventory rule ran on the same test scenes. It required no retuning because this experiment retained the same customer model.

This pilot used one training seed. **Paired** means both models were tested on matching scenes. A **bootstrap interval** is calculated by repeatedly resampling those paired scenes to estimate uncertainty in their mean profit difference. It describes variation across scenes for these fixed models, not how much the result might change if we trained again. The written experiment plan did not permit more training or changed learning settings after the test results were seen.

## 4. What the pilot found

Both variants completed the planned 100 updates and 19,200 physical periods. Their initial weight arrays matched. File fingerprints, called **hashes**, confirmed that the source, experiment plan, selected rule, observation collection, and final checkpoints stayed fixed. All training measurements were finite, with no undefined or infinite values, and checkpoint reload checks matched exactly.

The test ran five policy modes on each of 256 shared scenes, for **1,280 policy-episodes**, with **zero invalid actions**. A policy-episode is one policy running on one scene. Because the modes shared scenes, the comparison had 256 paired test scenes rather than 1,280 independent statistical samples. Local CPU runtime was **143.5 seconds**.

### Profit improved in this short matched comparison

The table reports modeled dollars per 12-period episode, discounted once per physical period:

```math
G=\sum_{t=1}^{12}0.99^{t-1}r_t.
```

Served share is the fraction of all potential requests actually served. Loss probability is the fraction of test episodes with negative discounted profit.

| Final stochastic policy | Mean discounted profit | Potential-request served share | Mobile delivery, kWh/episode | Loss probability |
|---|---:|---:|---:|---:|
| Tanh | 12.663 | 6.13% | 12.43 | 1.95% |
| SiLU | 13.898 | 7.02% | 23.87 | 1.17% |
| Frozen inventory rule | 22.558 | 7.39% | 30.10 | 0.00% |

The primary SiLU-minus-Tanh difference was **+1.235**, with a paired 5,000-bootstrap 95% interval of **[0.485, 2.010]**. That is approximately **9.8%** above the matched Tanh control. SiLU still earned **8.660 less** than the inventory rule per episode; its paired difference interval was **[−9.408, −7.932]**.

The additional deterministic comparison, specified before testing, earned **14.130** for Tanh and **15.606** for SiLU: a difference of **+1.475 [0.818, 2.114]**. Its loss probability increased from **0.00% to 1.56%**. A higher mean therefore did not mean better outcomes by every risk measure. Deterministic execution remained a secondary result.

### The representation became more responsive, with limits

This diagnostic asks how much the network's internal features and action scores change when supplied with the same collection of input states. These measurements help describe network behavior; profit is still assessed separately.

- **Local derivative** measures how responsive the final activation is to a small change in its own input. It does not measure the whole policy's training signal.
- **Feature standard deviation** measures how much intermediate network features vary across states.
- **Centered raw-score variation** measures changes in action scores before impossible actions are excluded. Centering means subtracting the mean action score within each state.
- **Actual minus mask-only probability RMS** measures how much action probabilities change beyond the effect of changing which actions are allowed. It compares the actual probabilities with those obtained using constant average scores and each state's mask. **RMS**, or root mean square, summarizes the typical size of those differences.

After training, measurements on the same collection of 768 observations were:

| Diagnostic | Tanh | SiLU |
|---|---:|---:|
| Mean absolute local activation derivative | 0.005552 | 0.528117 |
| Mean hidden-feature standard deviation across states | 0.000828 | 0.027669 |
| Centered raw-score variation across states, RMS | 0.002037 | 0.049660 |
| Actual minus mask-only probability RMS | 0.00005166 | 0.00036367 |

Tanh had **99.74%** of final activations above `0.99` in absolute value. SiLU's larger derivative and greater variation in features and scores fit the reason for trying it. However, differences in action probabilities after applying masks were still small, and raw-score measurements also included scores for impossible actions. These measurements do not prove that greater responsiveness caused the profit gain, solved vehicle coordination, or would last through longer training.

The pilot supported further testing of SiLU. It did **not** show that learning settled at a reliably good policy, that SiLU was better across training seeds, or that it beat the strong rule. Its 19,200-period budget also differs from the historical 96,000-period study; comparing its absolute profit with the older **16.49** result would not isolate the activation effect.

Saved evidence includes the full summary (`outputs/activation_pilot/summary.json`), frozen protocol (`outputs/activation_pilot/protocol.json`), frozen model hashes (`outputs/activation_pilot/frozen_models.json`), SiLU final checkpoint (`outputs/activation_pilot/silu/checkpoint.pt`), and Tanh final checkpoint (`outputs/activation_pilot/tanh/checkpoint.pt`). No additional tuning or budget extension followed testing.

## 5. Reproduce the pilot or start a separate study

Activate the existing project environment. Each command requires a new output directory:

```bash
source .venv/bin/activate

# Repeat the bounded paired trial using the previously frozen rule selection.
python experiments/run_activation_pilot.py \
  --selection-path outputs/policy_benchmarks/selection.json \
  --output-dir outputs/activation_pilot_repeat

# Start a separate study using the established 500-update convergence preset.
python experiments/run_convergence.py --seed 7 \
  --dispatch-final-activation tanh --output-root outputs/activation_full_tanh
python experiments/run_convergence.py --seed 7 \
  --dispatch-final-activation silu --output-root outputs/activation_full_silu

# Run a small SiLU execution check through the general CLI.
python -m charging_env.rl.train --dispatch-final-activation silu \
  --updates 10 --episodes-per-update 4 --seed 7 \
  --output-dir outputs/activation_silu_smoke
```

The two full-convergence commands retain the established 500-update configuration. Repeating them with seeds 19 and 37 would be a separate, larger study. Those commands were not part of this bounded pilot.

[The later SiLU + b=3 study](SILU_B3_STUDY.md) changes the customer-participation model as well as using SiLU. Its profit levels therefore should not be treated as an activation-only comparison with this b=0 pilot.

## 6. Verification record

The activation change retained the original **81 passing tests** and added **9 tests**. They checked that only the intended dispatch activation changed, starting weights otherwise matched, invalid settings were rejected, older checkpoints still loaded, and resumed SiLU training exactly restored both the optimizer and random-number generator (RNG). They also checked unchanged action masks and likelihood calculations, finite learning signals, rejection of architecture changes through the CLI when loading a checkpoint, and a small train/save/reload run. This is the historical verification count for this change; [VALIDATION.md](../VALIDATION.md) reports the current suite.
