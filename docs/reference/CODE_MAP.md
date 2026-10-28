# Code map

Which file does what, and where to start reading. All source paths are under `src/charging_env/`.

The code has three layers. The **simulator** defines the charging system. A **controller** chooses prices and fleet actions through two callbacks, `prices(obs, rng)` and `dispatch(obs, rng)`. An **evaluation** or **study** runs controllers on fixed simulated days and records profit.

```text
Scenario (configs/*.json)
      │
      ▼
ChargingSimulator ◄── prices, then dispatch actions ── Controller
      │                                                    ▲
      └── observation, profit, next state ─────────────────┘
      │
      ▼
evaluate_policy() ──► episode records ──► study analysis and reports
```

## First reading

1. `config.py`: every scenario setting, as the `Scenario` dataclass.
2. `choice.py`: how a customer picks mobile, a station, or leaving.
3. `simulator.py`: follow `reset()`, `quote_prices()` and `dispatch()`.
4. One controller: `rl/rollout_adp.py` for the current best method, or `rl/paper_ppo.py` for Joint PPO.

You can ignore `experiments/` and `tests/` until you need to reproduce or change something.

## One simulated hour in code

1. `reset()` starts a day. `_prepare_period()` draws the hour's requests and station price shocks.
2. **Pricing.** `quote_prices()` takes dollar prices, or `quote_price_levels()` takes indices. It samples each customer's choice and returns the post-choice observation. It gives no reward and doesn't advance time.
3. **Dispatch.** `dispatch()` takes one `DispatchAction(mode, target_zone, energy_kwh)` per vehicle. It resolves travel, whole-request service, station sharing, battery changes and profit, then advances to the next hour.
4. After the last hour the stage becomes 2 (terminal).

Python counts from zero: stage 0 is pricing, stage 1 is dispatch, and period 0 is the first hour. [NOTATION.md](NOTATION.md) maps every symbol to its Python name.

**The raw observation contains more than some controllers may use.** Pricing for Joint PPO and learned pricing goes through a projection that hides current requests: `paper_pricing_observation()` in `rl/paper_ppo.py` and `pricing_observation()` in `rl/adaptive_pricing.py`. Keep that boundary when adding features.

## Simulator

| File | Contents |
|---|---|
| `config.py` | `Scenario`: layout, fleet, demand, prices, costs, customer model; `from_json()`, `validate()` |
| `choice.py` | `charging_choice_probabilities()`: the logit choice model, shared by the simulator and the rules |
| `simulator.py` | `ChargingSimulator`, `Mode`, `DispatchAction`; feasibility masks; energy and profit ledgers |
| `gym_env.py` | `ChargingEnv`: a Gymnasium wrapper for outside tools. The controllers here call the simulator directly |

[ENVIRONMENT.md](ENVIRONMENT.md) walks through the API with examples.

## Controllers

| Method | Files | Key names |
|---|---|---|
| Rollout ADP | `rl/rollout_adp.py` | `RolloutADPConfig`, `RolloutADPPolicy`, `evaluate_rollout_policy()`; CLI `python -m charging_env.rl.rollout_adp` |
| Learned pricing | `rl/adaptive_pricing.py` | 12 pricing features, `LearnedPricingPolicy`, `PriceSearchConfig`, `fit_price_policy()` |
| Rollout ADP + learned pricing | `rl/priced_rollout.py` | `PricedRolloutADPPolicy`, which also applies the price rule inside imagined futures; CLI `python -m charging_env.rl.priced_rollout` |
| Historical market model | `rl/history_model.py` | `HistoricalLog`, `fit_history_model()`, `HistoricalModel`; safe saved logs/model and held-out prediction diagnostics |
| Historical-model evaluation | `rl/history_control.py` | `collect_historical_log()`, `evaluate_history_policy()`; controllers receive fitted forecasts while the evaluator owns the private environment |
| Table ADP | `rl/fitted_value.py`, `rl/fitted_train.py` | `FittedValueTrainer`, `FittedValuePolicy`; CLI `python -m charging_env.rl.fitted_train`. `sample_dispatch_observation()` is also used by rollout ADP |
| Joint PPO | `rl/paper_ppo.py`, `rl/paper_train.py` | `PaperPPOConfig`, `PaperPPOTrainer`, `interleaved_gae()`, `joint_ppo_loss()`, `paper_dispatch_mask()`; CLI `python -m charging_env.rl.paper_train` |
| Joint PPO warm start | `rl/paper_warmstart.py` | `fit_inventory_warmstart()`, which imitates the inventory rule before PPO |
| Earlier PPO | `rl/trainer.py`, `rl/networks.py`, `rl/returns.py`, `rl/train.py` | `Trainer`, `TrainingConfig`, `stage_gae()`; CLI `python -m charging_env.rl.train` |
| Earlier PPO imitation | `rl/warmstart.py` | An optional experiment, not used by the saved controls |

Why the file names differ from the method names: "paper" PPO is Joint PPO, named after its specification [MDP_RL_2.pdf](../MDP_RL_2.pdf). "Fitted value" is Table ADP. The study runner for rollout ADP is called `run_general_adp_study.py`. The files keep their original names because saved studies record exact source paths and hashes.

Each method has a deep dive in [docs/methods/](../methods/).

## Rules and evaluation

| File | Contents |
|---|---|
| `benchmarks/policies.py` | `fixed_prices()`, `service_first_dispatch()`, `coordinated_dispatch()` (exhaustive inventory rule), and `joint_dispatch_outcomes()` / `dispatch_outcome()`, the fast one-hour outcome model used by Table ADP, rollout ADP and the inventory rules |
| `benchmarks/scalable.py` | `coordinate_inventory_dispatch()`: the bounded, one-vehicle-at-a-time inventory search for large fleets |
| `benchmarks/compact.py` | The 82-setting rule grid of studies 06–07, and `inventory_shadow()` for time-dependent battery values |
| `benchmarks/evaluation.py` | `evaluate_policy()` runs any `prices`/`dispatch` pair on a list of day seeds; `paired_comparison()` computes paired intervals |
| `benchmarks/learned.py` | Wraps earlier-PPO checkpoints in the common callback interface |

The fast outcome model in `policies.py` has to agree exactly with `simulator.dispatch()`. The tests check this. If you change the physics, change both and rerun `tests/test_benchmarks.py`.

## Experiments, results and tests

| Location | Contents |
|---|---|
| `experiments/` | Study runners for studies 06–12; see [experiments/README.md](../../experiments/README.md) |
| `experiments/archive/` | Runners for studies 01–05 |
| `outputs/<study>/` | One study's frozen `protocol.json`, checkpoints, test episodes, `analysis.json`, audit, and a snapshot of the exact source that ran. Not in Git |
| `tests/` | Unit and regression tests; [VALIDATION.md](VALIDATION.md) maps them to features |

## Changing the code safely

- **Studies 09–11 froze a hash of every file in `src/`.** Any edit there, even a comment, means those studies no longer match the current code exactly. Their source snapshots in `outputs/` still allow exact reproduction. Edit freely, but rerun a study in a new directory before claiming its results for the new code.
- **Physics changes** go in `simulator.py` and `config.py`, mirrored in the outcome model in `benchmarks/policies.py`.
- **New controllers** only need `prices(obs, rng)` and `dispatch(obs, rng)` callbacks. Then `evaluate_policy()` scores them like everything else.
- **Performance claims** need new test days that were not used while developing the change.
