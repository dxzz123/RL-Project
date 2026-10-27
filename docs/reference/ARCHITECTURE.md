# Find your way around the code

The project has three layers. The **simulator** defines the charging system. A **controller** chooses prices and fleet actions. An **experiment** trains or evaluates controllers under recorded settings. Keeping these roles separate lets several algorithms face the same physical system and profit calculation.

```text
Scenario settings
      |
      v
ChargingSimulator <---- prices and dispatch actions ---- Controller
      |                                                   |
      +---- observations, profit, next state ------------+
      |
      v
Episode records --> shared evaluation --> comparison reports
```

Start with [MODEL.md](../MODEL.md) for the operating rules, [TRAINING.md](../TRAINING.md) for commands, and [COMPARISON.md](../COMPARISON.md) for the controllers. This page maps those explanations to code.

## A first pass through the code

Read `config.py` to see the city settings, `choice.py` to see how customers choose, then follow `quote_prices()` and `dispatch()` in `simulator.py`. Those three files define the physical problem. Next, choose one controller: `rl/rollout_adp.py` for rollout ADP or `rl/paper_ppo.py` for Joint PPO. Read the original two-vehicle ADP and earlier PPO files only when investigating those comparison methods. You can leave experiment orchestration and tests until you need to reproduce or change something.

## Follow one physical period

1. `ChargingSimulator.reset()` starts an episode; `_prepare_period()` draws requests and station-price shocks. `observe()` exposes the raw simulator state.
2. The controller chooses a mobile price at each node. `quote_prices()` or `quote_price_levels()` applies those prices and samples customer choices using `choice.py`. It returns the post-choice state. This transition earns no reward and does not advance physical time.
3. The controller sends one `DispatchAction` per vehicle. `dispatch()` resolves travel, whole-request service, energy exchange, shared station limits, and the period profit. It advances to the next period, or terminates after the horizon.
4. Training code records the transitions to learn from them. Evaluation code records the business outcomes without updating the policy.

The mathematical model labels pricing as stage 1 and dispatch as stage 2; Python uses `stage=0` and `stage=1`. Python `stage=2` means termination. Mathematical period `t=1` is Python `period=0`. These translations are documented in [NOTATION.md](NOTATION.md).

**The raw simulator observation is not the input to every controller.** Earlier PPO can use the realized pre-choice requests. Joint PPO's `paper_pricing_observation()` gives its pricing actor and critic only the specified background and fleet information. Learned ADP pricing uses an equivalent NumPy projection in `adaptive_pricing.py`. All ADP dispatchers see current choices when searching dispatch. Keep that distinction when adding features or interpreting experiments.

## The environment files

| File | Responsibility | Read it when you want to… |
|---|---|---|
| [config.py](../../src/charging_env/config.py) | `Scenario`, physical/economic settings, loading and validation | Change the city, demand, prices, battery sizes, or horizon |
| [choice.py](../../src/charging_env/choice.py) | Numerically stable customer-choice probabilities | Understand the outside option and charging benefit `b` |
| [simulator.py](../../src/charging_env/simulator.py) | Direct two-stage API, feasible actions, energy and profit ledgers | Trace a physical transition or inspect accounting |
| [gym_env.py](../../src/charging_env/gym_env.py) | Gymnasium observation/action spaces and `step()` adapter | Connect software that expects a Gymnasium environment |
| [ENVIRONMENT.md](ENVIRONMENT.md) | API walkthrough, fields, masks, and a hand calculation | Start using the environment without reading every function |

The trainers use the direct simulator API. The Gymnasium adapter is useful for integration and interface checks; it does not supply the custom two-stage learning algorithm automatically. Requests are variable-length lists, so an off-the-shelf fixed-array policy needs an appropriate encoder.

`configs/toy_city.json` records the original `b=0` city. `utility_b2.json`, `utility_b3.json`, and `utility_b4.json` change the common charging benefit while keeping the same physical settings. The newest comparison uses `utility_b3.json` and the larger `synthetic_25_zones.json`. Some older configuration fields remain for compatibility; [ENVIRONMENT.md](ENVIRONMENT.md) explains which quantities actually limit demand and actions.

## The controllers

### Approximate dynamic programming

For a new scenario, start with [rollout_adp.py](../../src/charging_env/rl/rollout_adp.py). At each dispatch decision, it generates a small set of feasible joint plans, simulates their future consequences under service-first, and executes the plan with the highest estimated total profit. It uses the complete remaining horizon and supports different fleet and network dimensions. There is no offline fit or learned value checkpoint.

`RolloutADPConfig` sets the planning budget and fixed price; `RolloutADPPolicy` supplies pricing and dispatch callbacks. `evaluate_rollout_policy()` resets the private planning random stream per episode and records a detailed planning trace. The same module provides the CLI. It reuses the existing generative sampler and full-fleet outcome model, preserving shared station capacity, request competition, costs and terminal inventory value. See [GENERAL_ADP.md](../methods/GENERAL_ADP.md) for the equations and limits.

The learned-price extension has two files. [adaptive_pricing.py](../../src/charging_env/rl/adaptive_pricing.py) defines twelve shared pricing features, `LearnedPricingPolicy`, safe pricing checkpoints, and `fit_price_policy()` for cross-entropy search on full episode returns. It accepts a dispatch-backend factory, so the same learner can optimize pricing for rollout or inventory. [priced_rollout.py](../../src/charging_env/rl/priced_rollout.py) combines a frozen price policy with the existing rollout candidates. Its future-period sampler calls the same actor before customer choice. This ensures that the prices used to value future states agree with the pricing policy used in operation.

The original fixed-price sources remain unchanged. The learned-price subclass reuses their candidate generation and physics while supplying its own future-pricing transition. A constant-price actor is tested for exact outcome and score parity with the original implementation. Pricing is learned offline; dispatch continuation values are still estimated online.

The original two-vehicle implementation remains available as a reference and comparison control:

[fitted_value.py](../../src/charging_env/rl/fitted_value.py) contains the battery grid, synthetic observation sampling, interpolated continuation values, backward fitting, and checkpoint format. `FittedValueTrainer.train()` fills the table from the final period backward. `FittedValuePolicy.dispatch()` searches the current finite joint-action grid using immediate profit plus discounted continuation value.

[fitted_train.py](../../src/charging_env/rl/fitted_train.py) exposes its CLI. This table-based implementation uses NumPy and supports exactly two ordered vehicles. The guide is [FITTED_VALUE.md](../methods/FITTED_VALUE.md). Both implementations are ADP. The original fits a table before operation; rollout estimates a base policy’s continuation value at each decision.

### Joint PPO

[paper_ppo.py](../../src/charging_env/rl/paper_ppo.py) contains the joint PPO implementation:

| Part | Function or class |
|---|---|
| Restricted pricing information | `paper_pricing_observation()`, `pricing_features()` |
| Discrete fleet operations and masks | `PaperOperation`, `paper_dispatch_mask()`, `paper_dispatch_commands()` |
| Two-stage advantages and critic targets | `interleaved_gae()`, `critic_targets()` |
| One clipped probability ratio for the fleet | `joint_ppo_loss()` |
| Collection, optimization, evaluation, saving | `PaperPPOTrainer` |

[paper_train.py](../../src/charging_env/rl/paper_train.py) handles new training, resume, and evaluation commands. [JOINT_PPO.md](../methods/JOINT_PPO.md) walks through the equations and network roles. Its `.pt` checkpoint is distinct from an earlier-PPO `.pt` checkpoint.

[paper_warmstart.py](../../src/charging_env/rl/paper_warmstart.py) optionally initializes Joint PPO from feasible inventory-rule demonstrations. It is separate from PPO rollout updates and is never called during policy execution. The original defaults disable it.

### Earlier PPO/MAPPO

| File | Responsibility |
|---|---|
| [networks.py](../../src/charging_env/rl/networks.py) | Observation encoders, pricing actor, shared vehicle actor, critics, continuous energy fractions |
| [returns.py](../../src/charging_env/rl/returns.py) | Earlier stage-aware return and advantage calculation |
| [trainer.py](../../src/charging_env/rl/trainer.py) | Rollouts, per-vehicle PPO updates, monitoring, checkpoints |
| [train.py](../../src/charging_env/rl/train.py) | Earlier PPO CLI |
| [warmstart.py](../../src/charging_env/rl/warmstart.py) | Optional imitation experiment; not used in the current saved controls |

The separate implementation is retained so an existing checkpoint continues to mean the same thing. [LEGACY_PPO.md](../methods/LEGACY_PPO.md) is its detailed reference. [COMPARISON.md](../COMPARISON.md) explains the substantive differences from joint PPO.

## Benchmarks and common measurements

[benchmarks/policies.py](../../src/charging_env/benchmarks/policies.py) defines the heuristic rules, feasible candidate actions, and one-period candidate outcome calculation. Approximate DP and the coordinated inventory rules use this model to search actions. The candidate outcome implementation is checked against actual simulator execution in the benchmark tests; if physical rules change, both paths need to remain consistent.

[benchmarks/compact.py](../../src/charging_env/benchmarks/compact.py) defines the three rule families and 82 validation candidates used by the historical fitted-value and six-method comparisons: service first, constant inventory value, and time-dependent inventory value. The newest learned-pricing study uses 49 bounded inventory settings from the general-ADP study runner. Other rule classes remain available to reproduce earlier studies.

[benchmarks/scalable.py](../../src/charging_env/benchmarks/scalable.py) provides bounded coordinate search for larger fleets. It changes one vehicle's action at a time while evaluating the entire fleet through the same one-period outcome model. It preserves shared capacity and request competition without allocating the Cartesian joint-action grid. It is an approximation to the inventory rule's action search, not a learned ADP value function. Select `warmstart_dispatch_solver='coordinate'` to use it for larger-fleet PPO demonstrations; the default exact teacher still requires two vehicles.

[run_scale_study.py](../../experiments/run_scale_study.py) preserves the earlier comparison of fresh PPO models and two retuned rules using [synthetic_25_zones.json](../../configs/synthetic_25_zones.json). It excludes the original two-vehicle ADP table. [run_general_adp_study.py](../../experiments/run_general_adp_study.py) compares rollout ADP, a current-profit control, and frozen learned policies on fresh independent tests of both network sizes. [run_adp_transfer.py](../../experiments/run_adp_transfer.py) checks the same rollout implementation under two additional known configurations.

[benchmarks/evaluation.py](../../src/charging_env/benchmarks/evaluation.py) runs frozen policies and calculates profit, service, action, and energy summaries. `paired_comparison()` compares profits on matching scenario seeds. [benchmarks/learned.py](../../src/charging_env/benchmarks/learned.py) adapts earlier PPO to this common policy interface. The joint-PPO trainer and fitted-value policy expose the same `prices()` and `dispatch()` callbacks directly.

## Experiments, outputs, and provenance

The newest orchestration is [run_priced_adp_study.py](../../experiments/run_priced_adp_study.py). It learns pricing for both rollout and inventory, selects policies on validation, and freezes them before shared independent tests. The earlier [run_general_adp_study.py](../../experiments/run_general_adp_study.py) retains the fixed-price rollout comparison with saved learned controls. [run_paper_ppo_study.py](../../experiments/run_paper_ppo_study.py) retains the original six-method training study. The [experiment index](../../experiments/README.md) distinguishes these workflows.

The original six-method study, `outputs/paper_ppo_study/`, contains:

| Location | Meaning |
|---|---|
| `protocol.json` | Settings, seed banks, model-selection rule, and frozen source fingerprints |
| `training/seed_*/` | Three new joint-PPO runs |
| `legacy/seed_*/`, `fitted/seed_*/` | Copies of the fixed comparison models |
| `rule_selection/` | Validation candidates and selected heuristic parameters |
| `benchmarks/` | Individual test episodes and comparison summaries |
| `analysis.json` | Audited numerical evidence used by the comparison guide |
| `provenance/executed_training_source/` | The code and algorithm specification used for the recorded experiment |

`outputs/` is ignored by Git. Keep its checkpoints, protocols, and source snapshots when a reported result must remain reproducible. Historical reports live in [docs/archive](../archive/README.md); their script paths are retained because some scripts import others and recorded protocols refer to exact source paths. Removing an older experiment from the default reading path does not require discarding its evidence.

## When extending the project

Change environment rules in the simulator and scenario model, then check accounting and the benchmark candidate model against them. Change learning behavior in the matching learner and start a separate experiment with a recorded configuration. Add a benchmark through the common evaluation callbacks so profit is calculated consistently.

Use [VALIDATION.md](VALIDATION.md) to select existing checks. For claims about a better controller, reserve new test scenarios after choosing the implementation and settings. Replaying the current test bank is useful for reproduction; once its results guide development, it is no longer an untouched test for that development.

The recharge investigation uses [run_recharge_study.py](../../experiments/run_recharge_study.py). Its separate `outputs/recharge_study_20261024/` stores the frozen protocol, copied controls, initialization and final checkpoints, independent tests, and a readable scorecard in `analysis.json`. See [TRAINING.md](../TRAINING.md#joint-ppo-with-an-inventory-rule-warm-start) for commands.
