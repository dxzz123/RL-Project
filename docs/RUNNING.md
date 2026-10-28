# Running the code

Commands for installing, running each controller, and reproducing the studies. Run everything from the project root (the folder with `pyproject.toml`).

For commands that accept a new `--scenario`, pass it explicitly. Leaving it out selects the original `b = 0` toy city, which has a different customer model from the studies. Saved checkpoints and study runners supply their own scenarios.

| Scenario file | Use |
|---|---|
| `configs/utility_b3.json` | Small city: 4 zones, 2 vehicles. Used in every study since 04 |
| `configs/synthetic_25_zones.json` | Large city: 25 zones, 10 vehicles |

[configs/README.md](../configs/README.md) describes every scenario file.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[train]'
```

- `-e` means edits to the source take effect without reinstalling.
- `[train]` adds PyTorch, needed for both PPO methods and for every study runner in `experiments/`.
- The simulator, the rules, rollout ADP, learned pricing and Table ADP need only `python -m pip install -e .`.
- Tested dependency versions are in `requirements-tested.txt` and `requirements-training-tested.txt`.

In VS Code, choose `.venv/bin/python` with **Python: Select Interpreter**. The Run and Debug menu has a ready-made entry for each command marked ▶ below.

## Check that everything works

```bash
python -m unittest discover -s tests          # run the complete test suite
python examples/manual_walkthrough.py         # ▶ one scripted day, printed step by step
python examples/fcs_discharge_walkthrough.py  # ▶ a vehicle sells energy at a station
python examples/check_gym.py                  # ▶ Gymnasium interface check
```

The walkthrough writes `outputs/manual_episode.csv`. Its actions are scripted to explain the simulator, not chosen to earn profit.

## Run a controller

The single-controller commands in this section print the output directory they use. If the directory you name already exists, they pick a numbered sibling such as `outputs/rollout_adp_2` instead of overwriting. Study phases instead share one study directory, as explained below.

### Rollout ADP

No policy training. The planner simulates futures at every decision using the supplied scenario as its known model. ▶

```bash
python -m charging_env.rl.rollout_adp --scenario configs/utility_b3.json \
  --seed 7 --price-index 4 --rollout-samples 8 --episodes 16 \
  --eval-seed-base 801000001 --output-dir outputs/rollout_adp
```

| Option | Meaning |
|---|---|
| `--price-index` | Position in the scenario's price list. In both supplied cities, 4 = $0.60 and 5 = $0.80. Validation chose 4 for the small city and 5 for the large one |
| `--seed` | Planning randomness. Nothing is trained |
| `--rollout-samples` | Imagined futures per candidate plan |
| `--episodes`, `--eval-seed-base` | How many simulated days to evaluate, and which ones |

For the large city, use `--scenario configs/synthetic_25_zones.json --price-index 5`. Each decision there takes about half a second, so 16 days take a few minutes.

**Output files:**

- `summary.json`: mean profit, interval and timing.
- `evaluation.json`: every day's results.
- `planning_trace.json`: every candidate score and sampled future.
- `protocol.json` and `provenance.json`: exact settings and source hashes.

### Learned pricing

Evaluate a saved price rule with rollout dispatch. The checkpoint supplies the scenario used for both planning and evaluation. This command alone does not test a private environment with different demand; use the [historical-model comparison](#historical-model-comparison) for that.

```bash
python -m charging_env.rl.priced_rollout \
  --pricing-checkpoint outputs/priced_adp_study_20261025/training/city/warm/seed_7/checkpoint.npz \
  --seed 7 --rollout-samples 8 --episodes 16 --eval-seed-base 2410000001 \
  --output-dir outputs/priced_rollout
```

This path needs the saved study folder. Which checkpoint each seed selected is recorded in `outputs/priced_adp_study_20261025/selection/<city>.json`. The command also writes `pricing_trace.json` with every live price decision.

Fit a price rule from scratch with a tiny budget. This checks that the workflow runs; the budget is far too small to support a performance claim:

```python
from charging_env.config import Scenario
from charging_env.rl.adaptive_pricing import PriceSearchConfig, fit_price_policy
from charging_env.rl.fitted_train import _new_run_directory
from charging_env.rl.priced_rollout import PricedRolloutADPPolicy
from charging_env.rl.rollout_adp import RolloutADPConfig

scenario = Scenario.from_json("configs/utility_b3.json")

def backend(pricer, seed):          # which dispatcher the prices are fitted for
    return PricedRolloutADPPolicy(scenario, pricer, RolloutADPConfig(seed=seed, rollout_samples=2))

output = _new_run_directory("outputs/pricing_smoke")
config = PriceSearchConfig(seed=7, generations=2, population=12, elite_count=3,
                           episodes_per_generation=2, training_seed_base=2450000000)
policy, summary = fit_price_policy(scenario, config, backend, output)
print("Saved", output / "checkpoint.npz", "after", summary["environment_episodes"], "episodes")
```

The saved `checkpoint.npz` works with the `priced_rollout` command above.

### Table ADP (two vehicles only)

Fit the value table. This takes a few minutes and needs no PyTorch. ▶

```bash
python -m charging_env.rl.fitted_train --scenario configs/utility_b3.json \
  --seed 7 --grid-step 10 --samples 8 --output-dir outputs/table_adp
```

`--grid-step` is the battery grid spacing in kWh. `--samples` is the number of sampled hours per grid state. The price is fixed at index 4 ($0.60). Then evaluate a saved table:

```python
from charging_env.rl.fitted_value import FittedValuePolicy
from charging_env.benchmarks.evaluation import evaluate_policy

policy = FittedValuePolicy.load_checkpoint("outputs/table_adp/checkpoint.npz")
days = [193000001 + 2 * i for i in range(64)]
result = evaluate_policy(policy.scenario, days, policy.prices, policy.dispatch, label="Table ADP")
print(result["mean_discounted_profit"], "invalid actions:", result["invalid_actions"])
```

### Joint PPO

A short run that checks training works (40 days; far too short to learn much): ▶

```bash
python -m charging_env.rl.paper_train --scenario configs/utility_b3.json \
  --updates 10 --episodes-per-update 4 --seed 7 --output-dir outputs/joint_ppo
```

The full original setting is `--updates 500 --episodes-per-update 16 --eval-episodes 64 --eval-interval 25`, which is 8,000 training days.

The warm-started setting from [study 08](studies/08_recharge_warm_start.md), which fixes the recharge collapse:

```bash
python -m charging_env.rl.paper_train --scenario configs/utility_b3.json --seed 7 \
  --warmstart-episodes 512 --warmstart-epochs 100 \
  --reward-scale 20 --normalize-advantages --gae-lambda 1 \
  --actor-learning-rate 0.0001 --dispatch-entropy-coef 0.005 --target-kl 0.03 \
  --updates 500 --episodes-per-update 16 --output-dir outputs/joint_ppo_warmstart
```

For the large city, add `--warmstart-dispatch-solver coordinate --warmstart-coordinate-sweeps 2`. The default teacher's exhaustive search only works for two vehicles.

Continue training, or evaluate without training:

```bash
python -m charging_env.rl.paper_train --resume outputs/joint_ppo/checkpoint.pt \
  --updates 10 --output-dir outputs/joint_ppo_continued
python -m charging_env.rl.paper_train --evaluate-only outputs/joint_ppo/checkpoint.pt \
  --eval-episodes 128 --eval-seed-base 193000001 --eval-policy stochastic
```

**Resume and evaluation notes:**

- Resuming restores networks, optimizers, counters and random state.
- A resumed run cannot change its scenario, reward scale or warm-start settings.
- `--updates 10` adds ten updates to the saved count.
- Choose `stochastic` (sample actions) or `deterministic` (most likely action) before comparing results, not after.

**Output files:**

- `checkpoint.pt`: the final model.
- `best_checkpoint.pt`: the best model on the monitoring days; it may be the initial policy if later updates never improve that score.
- `warmstart_checkpoint.pt`: the imitation-only model, when a warm start was used.
- `training_metrics.csv`: losses and diagnostics for each update.
- `validation_history.json`: monitoring profit over training.
- `summary.json` and `protocol.json`: run summary and exact settings.

Checkpoint choice depends on the study protocol. The historical-model study validates the final, monitoring-best and warm-start checkpoints on separate fitted-model days, then freezes one before testing. It does not choose a checkpoint using private test profit.

### Earlier PPO

Kept as a comparison control. ▶

```bash
python -m charging_env.rl.train --scenario configs/utility_b3.json \
  --dispatch-final-activation silu --updates 10 --episodes-per-update 4 --seed 7 \
  --output-dir outputs/earlier_ppo
```

This is not the exact configuration of the saved control models. Those also used LayerNorm, a vehicle-ID input, inventory reward shaping and λ = 1, and their settings are stored in each checkpoint. All flags are documented in [methods/EARLIER_PPO.md](methods/EARLIER_PPO.md). Earlier-PPO checkpoints load only with `charging_env.rl.train`, not with `paper_train`.

### A benchmark rule

Rules have no command of their own. Evaluate one in a few lines:

```python
from charging_env.config import Scenario
from charging_env.benchmarks import policies
from charging_env.benchmarks.compact import inventory_shadow
from charging_env.benchmarks.evaluation import evaluate_policy

scenario = Scenario.from_json("configs/utility_b3.json")
days = [901000001 + 2 * i for i in range(32)]
schedule = {"family": "time_inventory", "early_shadow_price": 0.30,
            "late_shadow_price": 0.15, "switch_period": scenario.horizon // 2}

def price(obs, rng):                      # $0.60 everywhere mobile service is available
    return policies.fixed_prices(obs, scenario, 4)

def dispatch(obs, rng):                   # exhaustive time-dependent inventory rule
    return policies.coordinated_dispatch(obs, scenario, inventory_shadow(obs, scenario, schedule))

print(evaluate_policy(scenario, days, price, dispatch, label="inventory rule")["mean_discounted_profit"])
```

For the bounded version, use `benchmarks.scalable.coordinate_inventory_dispatch`. For service-first, use `policies.service_first_dispatch(obs, scenario, 0.2)`.

## View or reproduce a study

To see a finished study's numbers without rerunning anything, use the commands in [RESULTS.md, section 5](RESULTS.md#5-saved-evidence). The historical-model command below shows the latest comparison. The existing VS Code entries ▶ "Show current scorecard (study 11)" and ▶ "Show saved algorithm comparison (study 07)" still show those earlier studies.

To rerun a study, follow the **Reproduce** section of its report in [studies/](studies/README.md). The study runners follow the same pattern:

1. `--phase init` freezes the settings, day banks and a hash of every source file in a new study directory.
2. Middle phases tune rules, train, validate and select, depending on the study.
3. `--phase evaluate` opens the test days once and refuses to overwrite a finished test.
4. `--phase show` prints the saved scorecard.

The source hashes are checked again before training and testing. **Don't edit frozen source files in the middle of a study.** To test a code change, start a new study directory. Once a test bank has guided a change, a new claim needs new test days.

### Historical-model comparison

This workflow estimates demand and customer choice from synthetic quote histories, then compares rollout ADP, PPO and the inventory rule. All three prepare their policies using the same fitted simulator. Final evaluation uses private environments with the original demand pattern, higher arrival volume, and shifted demand locations and times. The fitted model stays frozen throughout those tests.

If the installed project includes the finished study, read its saved results from the project root:

```bash
python experiments/run_history_adp_study.py --phase show \
  --study-dir outputs/history_adp_study_20261027
```

The full sequence below starts the declared study on a clean checkout, or with a saved study history that has not used these scenario banks:

```bash
source .venv/bin/activate
python -m pip install -e '.[train]'

python experiments/run_history_adp_study.py --phase init \
  --study-dir outputs/history_adp_study_20261027 --control-root .
python experiments/run_history_adp_study.py --phase history \
  --study-dir outputs/history_adp_study_20261027
python experiments/run_history_adp_study.py --phase fit \
  --study-dir outputs/history_adp_study_20261027
python experiments/run_history_adp_study.py --phase rules \
  --study-dir outputs/history_adp_study_20261027
python experiments/run_history_adp_study.py --phase train \
  --study-dir outputs/history_adp_study_20261027
python experiments/run_history_adp_study.py --phase select \
  --study-dir outputs/history_adp_study_20261027
python experiments/run_history_adp_study.py --phase evaluate \
  --study-dir outputs/history_adp_study_20261027
python experiments/run_history_adp_study.py --phase show \
  --study-dir outputs/history_adp_study_20261027
```

**Use one directory for all phases.** `init` requires an unused directory; subsequent phases reuse it and verify saved evidence. The runner declares fixed scenario banks in its source. Once this study is installed, `--control-root .` detects those banks in the saved history and rejects another initialization, even with a different directory name. A new independent study requires new, unused seed-bank constants and a newly frozen protocol; renaming the output directory alone is insufficient. Keep the original source and results intact when preparing that separate study.

Run one phase at a time and continue only after it succeeds. Completed processing phases verify and reuse saved files where supported; `init` always requires a new directory. Incomplete artifacts are preserved rather than automatically resumed. Diagnose and archive a failed study before planning a documented restart.

This is the full comparison, including three PPO seeds with 500 updates each, not a quick smoke test. The defaults use one worker. The `rules`, `train`, `select` and `evaluate` phases accept `--workers 2` or `--workers 3` when machine resources allow.

| Output | What to inspect |
|---|---|
| `history/fit.npz`, `history/holdout.npz` | Complete quote records, including outside choices and zero-arrival count cells; 120 fitting days and 32 separate diagnostic days |
| `model/fit_report.json`, `model/holdout_diagnostics.json` | Parameter fitting and historical prediction checks |
| `model/checkpoint.npz` | The frozen fitted demand and customer-choice model |
| `training/`, `selection/` | Policies trained in that model and the validation-based choices |
| `analysis.json` | Paired profit comparisons for each private test regime |
| `test/<regime>/` | Per-day results and saved planning or pricing traces where applicable |

The histories assume that all quote opportunities, including declined offers, are observable. Poisson arrivals, uniform request sizes and logit customer choice are assumed model families whose parameters are estimated. Current accepted requests remain visible at dispatch; true future demand rates do not. See the [method explanation](methods/ROLLOUT_ADP.md#learning-the-model-from-quote-histories) and [study 12](studies/12_historical_model.md) for the assumptions and protocol.

## Judging a result

- **Compare on shared days.** Compare discounted profit on the same days, and prefer the paired difference to two separate means.
- **Keep the three roles apart.** Training days fit the method, validation days choose settings, and test days are opened once at the end. Using one set for all three gives optimistic numbers.
- **Check every seed.** A positive average can hide a seed that got worse.
- **Look beyond profit.** Check mobile energy delivered, recharge bought, unmet demand and travel. No single one of them defines a good policy.
- **Losses are not profit.** A lower critic loss or a flat training curve doesn't show a controller beats a benchmark.
