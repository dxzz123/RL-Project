# Run training and understand the results

Use this page for the practical workflow: choose a controller, run it, and compare its profit with other methods. [MODEL.md](MODEL.md) explains the environment. [COMPARISON.md](COMPARISON.md) explains why the algorithms differ and what the completed experiments show.

All commands below run from the project root, the folder containing `pyproject.toml`. The main examples use the supplied small or 25-zone `b=3` scenario files. For standalone commands that accept `--scenario`, omitting it uses the original `b=0` toy scenario unless a loaded checkpoint supplies the scenario. Changing `b` changes customer behavior, not just a training setting.

## Set up once

Activate the existing virtual environment and install the training dependencies:

```bash
source .venv/bin/activate
python -m pip install -e '.[train]'
```

If the folder is missing, first run `python3 -m venv .venv`. Python 3.10 or later is required. `-e` means source edits take effect without reinstalling; `[train]` adds PyTorch for the two PPO methods. The simulator and standalone ADP commands need only `python -m pip install -e .`. The comparison runners, including the learned-pricing study, currently import shared PPO comparison utilities and require `[train]`, even when they do not train PPO. Optional tested direct dependency versions are recorded in `requirements-tested.txt` and `requirements-training-tested.txt`; these are not complete cross-platform lock files.

In VS Code, choose `.venv/bin/python` with **Python: Select Interpreter**. The Run and Debug menu contains clearly named entries for each learner, the simulator walkthroughs, and the saved comparison.

## View the results already available

The [comparison report](COMPARISON.md) is included in the repository and can be read without installing anything. The newest comparison tests learned pricing with both rollout ADP and inventory dispatch on the small network and the 25-zone fleet. Display its saved results with:

```bash
python experiments/run_priced_adp_study.py --phase show \
  --study-dir outputs/priced_adp_study_20261025
```

This reads the saved scorecard and reports which pricing option was selected for each seed; it does not train or evaluate policies again. The full local study includes pricing checkpoints, planning traces, validation choices, independent episode records, and source snapshots. A Git checkout alone does not include ignored `outputs/`; the comparison report contains the main numerical results.

The earlier fixed-price rollout comparison is still available with `python experiments/run_general_adp_study.py --phase show --study-dir outputs/general_adp_study_20261025`.

For the original six-method study, the [figure](assets/algorithm_comparison.png) remains available. If you also have `outputs/paper_ppo_study/`, these commands display its older scorecard and figure:

```bash
python3 experiments/summarize_paper_ppo_study.py --show
open docs/assets/algorithm_comparison.png
```

`--show` reads `outputs/paper_ppo_study/analysis.json` and prints the six-method scorecard. It does not run policies or rewrite results. The figure shows base profits, the joint-PPO improvement over earlier PPO, stress cases, and monitoring curves. `open` is the macOS command; elsewhere, open the image in your file browser.

The [comparison guide](COMPARISON.md) explains each row, why approximate DP is not a PPO variant, and what the uncertainty intervals mean. In this local project the completed study is present. A Git checkout alone does not contain ignored `outputs/`; copy the study artifacts or follow the reproduction workflow below.

## Choose the method you want to run

| Your purpose | Entry point | Detailed guide |
|---|---|---|
| Learn prices, then compare rollout and inventory dispatch | `experiments/run_priced_adp_study.py` | [Learned pricing](methods/GENERAL_ADP.md#learning-prices-as-well-as-planning-dispatch) |
| Plan dispatch for different fleet and network sizes using future simulations | `charging_env.rl.rollout_adp` | [Rollout ADP](methods/GENERAL_ADP.md) |
| Reproduce the original two-vehicle continuation-value table | `charging_env.rl.fitted_train` | [Tabular ADP](methods/FITTED_VALUE.md) |
| Train pricing and fleet dispatch with joint PPO | `charging_env.rl.paper_train` | [Joint PPO](methods/JOINT_PPO.md) |
| Study or continue the earlier SA-PPO/MAPPO implementation | `charging_env.rl.train` | [Earlier PPO/MAPPO](methods/LEGACY_PPO.md) |

A **policy** is the controller's decision rule. A **checkpoint** saves a learned policy and the scenario/settings needed to load it. The two PPO trainers save `.pt` files, but their checkpoints are different formats and must be loaded with the matching trainer. The original table-based ADP saves `.npz` arrays and metadata. Fixed-price rollout needs no learned checkpoint. Its learned-price extension saves a separate `.npz` pricing policy while continuing to estimate dispatch values online.

### Learn prices for rollout ADP

The new learner fits twelve shared coefficients that choose a legal mobile price at each zone. Prices can depend on time, forecast demand, fleet battery and position, and observed station prices. Current realized requests are hidden from pricing, as in Joint PPO. The same learned price policy is used in actual operation and inside every simulated future period.

Training first searches pricing policies with inexpensive service-first dispatch, then refines them with the intended dispatch controller. The objective is complete discounted business profit. Inventory receives its own refinement with the same episode budget. This is direct policy search for prices, followed by online ADP planning for dispatch; it is not another PPO implementation.

To evaluate the validation-selected city pricing policy for seed 7 with rollout dispatch:

```bash
python -m charging_env.rl.priced_rollout \
  --pricing-checkpoint outputs/priced_adp_study_20261025/training/city/warm/seed_7/checkpoint.npz \
  --seed 7 --rollout-samples 8 \
  --episodes 16 --eval-seed-base 2410000001 \
  --output-dir outputs/priced_rollout
```

The checkpoint supplies the scenario and price coefficients. Here `--seed` controls future planning draws, while the checkpoint already contains the result of training seed 7. The command automatically chooses a numbered output directory if necessary. It saves episode outcomes in `evaluation.json`, live price decisions in `pricing_trace.json`, candidate scores and future draw seeds in `planning_trace.json`, and aggregate results in `summary.json`. `protocol.json` and `provenance.json` identify the settings and source/checkpoint fingerprints.

The city seed-7 selection uses the initial learned policy (`warm`); its refinement scored lower in validation. Other seeds can select refinement or a fixed-price fallback. Read the saved scorecard or `selection/city.json` for each choice. The final comparison uses those validation choices unchanged. The command above evaluates the selected policy on a new illustrative bank; it does not reproduce the study scorecard.

<details>
<summary>Try a small pricing fit without saved study files</summary>

This standalone example works with a valid scenario JSON and the base dependencies. It searches prices directly with rollout dispatch, using a deliberately small budget to check the workflow:

```bash
python - <<'PYTHON'
from charging_env.config import Scenario
from charging_env.rl.adaptive_pricing import PriceSearchConfig, fit_price_policy
from charging_env.rl.fitted_train import _new_run_directory
from charging_env.rl.priced_rollout import PricedRolloutADPPolicy
from charging_env.rl.rollout_adp import RolloutADPConfig

scenario = Scenario.from_json("configs/utility_b3.json")

def backend(pricer, seed):
    return PricedRolloutADPPolicy(
        scenario, pricer, RolloutADPConfig(seed=seed, rollout_samples=2))

output = _new_run_directory("outputs/pricing_smoke")
config = PriceSearchConfig(
    seed=7, generations=2, population=max(12, len(scenario.price_levels) + 2),
    elite_count=3, episodes_per_generation=2, training_seed_base=2450000000)
policy, summary = fit_price_policy(scenario, config, backend, output)
print("Pricing checkpoint:", output / "checkpoint.npz")
print("Training candidate episodes:", summary["environment_episodes"])
PYTHON
```

The supplied seven-price scenario uses 48 candidate episodes. This is an execution check, not enough training or independent testing to support a performance claim. The printed checkpoint can be evaluated with the `priced_rollout` command above. Replace its checkpoint path with the one printed here; repeated runs automatically receive a numbered output directory. The full study below uses a larger initial search, eight planning samples, validation selection and independent benchmarks.

</details>

<details>
<summary>Reproduce the learned-pricing comparison</summary>

If the completed local study is available and source code is unchanged, use an explicit replay in a new directory:

```bash
python experiments/run_priced_adp_study.py --phase init \
  --replay-of outputs/priced_adp_study_20261025 \
  --study-dir outputs/priced_adp_repeat
python experiments/run_priced_adp_study.py --phase rules --workers 3 \
  --study-dir outputs/priced_adp_repeat
python experiments/run_priced_adp_study.py --phase train --workers 3 \
  --study-dir outputs/priced_adp_repeat
python experiments/run_priced_adp_study.py --phase validate --workers 3 \
  --study-dir outputs/priced_adp_repeat
python experiments/run_priced_adp_study.py --phase select \
  --study-dir outputs/priced_adp_repeat
python experiments/run_priced_adp_study.py --phase evaluate --workers 3 \
  --study-dir outputs/priced_adp_repeat
python experiments/run_priced_adp_study.py --phase show \
  --study-dir outputs/priced_adp_repeat
```

Replay deliberately reuses the original banks and is not new independent evidence. For an initial study, use `--control-root .` instead of `--replay-of`; initialization checks historical seed use and the saved original-inventory reference. That project root must contain `outputs/general_adp_study_20261025/reference_inventory/protocol.json`. A clean Git checkout does not include this ignored evidence; use the completed local bundle or the standalone pricing-fit example above. If the declared banks have already been used, a new independent experiment requires new banks and a new frozen protocol.

The study uses seeds 7, 19 and 37 in both networks. Each seed first evaluates 24 pricing candidates on 16 common days in each of 12 generations. It then refines prices separately for rollout and inventory, with eight candidates on four common days in each of four generations. That is 4,608 initial candidate episodes plus 128 per dispatch-specific refinement. Candidates within a generation share days; these episode counts are not counts of distinct demand days.

Validation uses 32 separate days per network. It checks all seven constant prices for rollout and 49 price/inventory settings for the bounded rule, then selects fixed, initial learned or refined pricing for each dispatch controller and seed. Final testing uses 256 new days per network, for 6,912 policy episodes including the price-swap control and the original exhaustive small-case rule. Rollout uses eight future trajectories per candidate throughout actual-backend refinement, validation and testing.

This is a substantial simulation job. Training prints a line after each search generation; validation and testing print after each complete policy evaluation. A 256-day city rollout job can therefore remain quiet for many minutes. Use `--phase show` after the completed analysis has been saved; it does not display an unfinished experiment.

Completed jobs are checked and reused when a phase is invoked again. An incomplete training directory is preserved and reported; it is not silently erased or resumed from partial optimizer state. `--case city --seed 7 --training-phase rollout` restricts training to one refinement, after its rule selection and initial pricing training exist. The complete comparison requires every declared job.

</details>

### Rollout ADP: compare future consequences while operating

```bash
python -m charging_env.rl.rollout_adp \
  --scenario configs/synthetic_25_zones.json \
  --seed 7 --price-index 5 --rollout-samples 8 \
  --episodes 16 --eval-seed-base 1000001 \
  --output-dir outputs/rollout_adp
```

This runs the policy on 16 simulated days. At each dispatch decision, it compares several feasible fleet plans using current profit plus simulated future profit. Future simulations follow a simple service-first policy through the remaining horizon. Only the selected current plan is executed; the next decision is planned again from the newly observed state.

There is **no offline training step**. `--seed` controls planning randomness, `--rollout-samples` controls future simulations per candidate, and `--episodes` controls evaluated days. None is a PPO update count. Price index 5 is 0.80 dollars/kWh in the supplied scenarios. Pricing remains fixed; this method uses future simulation to choose dispatch actions.

The same command accepts another valid scenario JSON. It reads that scenario's demand assumptions, travel, capacities, costs and horizon while planning. Supporting different configurations does not establish robustness to an inaccurate model. See [GENERAL_ADP.md](methods/GENERAL_ADP.md) for the algorithm, computational limits and differences from PPO.

Outputs include `evaluation.json` (episode outcomes), `planning_trace.json` (candidate scores and future draw seeds), `provenance.json` (settings, counters and source hashes), and `summary.json`. The CLI automatically adds a numeric suffix when an output directory already contains a run. Eight simulations is a computational setting, not a guarantee of a precise value estimate or better profit.

### Compare rollout ADP with PPO and rules

The dedicated study evaluates planning seeds 7, 19 and 37 at fixed prices 0.60 and 0.80 on 32 validation days per network. It chooses one price using the average validation profit across all three seeds. It separately retunes service-first and inventory rules. Final evaluation uses 256 independent days per network, shared by all compared policies.

PPO is a **saved control**, using the completed 500-update models from the recharge and 25-zone studies. The small case also includes the original ADP table. The primary inventory baseline uses bounded coordinate search and maximum exchange amounts in both networks. The report additionally checks the historical exhaustive half/full inventory rule on the small case; its separate records are under `reference_inventory/`. The current-profit control uses the selected rollout price and bounded inventory search with zero shadow value. It is a useful myopic reference, but its candidate search differs from rollout ADP's five-plan comparison, so their difference is not a pure ablation of future simulation.

<details>
<summary>Reproduce the primary rollout-ADP comparison</summary>

Use an unused study directory. `--control-root` points to the project containing the completed `outputs/recharge_study_20261024/`, `outputs/scale_25_study_20261024/`, and `outputs/paper_ppo_study/` control bundles. Initialization verifies and copies their checkpoints and provenance; it does not retrain PPO.

```bash
python experiments/run_general_adp_study.py --phase init \
  --study-dir outputs/general_adp_repeat --control-root .
python experiments/run_general_adp_study.py --phase rules \
  --study-dir outputs/general_adp_repeat
python experiments/run_general_adp_study.py --phase validate \
  --study-dir outputs/general_adp_repeat
python experiments/run_general_adp_study.py --phase select \
  --study-dir outputs/general_adp_repeat
python experiments/run_general_adp_study.py --phase evaluate --workers 3 \
  --study-dir outputs/general_adp_repeat
python experiments/run_general_adp_study.py --phase show \
  --study-dir outputs/general_adp_repeat
```

`init` freezes source, settings and scenario banks. `rules` selects benchmarks; `validate` evaluates both rollout prices and all three planning seeds; `select` fixes the rollout price. `evaluate` checks those choices before opening the independent test bank. It refuses to overwrite completed tests; use `show` to read them again. The complete study runs 5,376 final policy episodes. Rollout ADP does substantially more computation per decision than a trained PPO policy.

These fixed seeds reproduce the recorded experiment. Once a test bank guides a new algorithm change, specify a new independent bank for the next performance claim.

The supplementary exhaustive-inventory check is separate from the primary workflow. Its runner is saved in the local result bundle, not tracked in Git. If you have that bundle, copy the runner into the new study and run it there:

```bash
mkdir -p outputs/general_adp_repeat/reference_inventory
cp outputs/general_adp_study_20261025/reference_inventory/run_reference.py \
  outputs/general_adp_repeat/reference_inventory/run_reference.py
python outputs/general_adp_repeat/reference_inventory/run_reference.py
```

It evaluates the historical fixed price and battery values on the new study's small-case test bank, then writes `evaluation.json` and `audit.json` in `reference_inventory/`. It refuses to overwrite an existing reference run. The primary `--phase show` scorecard does not include this separate reference.

</details>

For additional portability checks with different dimensions, horizons, battery states, station capacities and price noise:

```bash
python experiments/run_adp_transfer.py --workers 2 --output-dir outputs/adp_transfer
python experiments/run_adp_transfer.py --show --output-dir outputs/adp_transfer
```

This compares the same rollout algorithm and three rules on 9-zone/five-vehicle and 36-zone/twelve-vehicle models. All use a predeclared fixed price of 0.80; these controls are not retuned. Each changed model is known to the planner. This checks operation under different model settings, not transfer when the planner has the wrong model.

### Original approximate DP: fit a two-vehicle value table

```bash
python -m charging_env.rl.fitted_train \
  --scenario configs/utility_b3.json \
  --seed 7 --grid-step 10 --samples 8 \
  --output-dir outputs/fitted_value
```

This method learns dispatch continuation values for exactly two vehicles. It uses a fixed mobile price of 0.60 dollars/kWh under this scenario, then searches joint dispatch actions with that value. `--grid-step` sets battery-grid spacing in kWh; `--samples` sets the number of demand/price draws used at each time/state query. Smaller grid spacing or more samples increases training work. There is no PPO update count: fitting works backward once through the finite horizon. See [FITTED_VALUE.md](methods/FITTED_VALUE.md) for the equations and saved-policy example.

### Joint PPO: check a short training run

```bash
python -m charging_env.rl.paper_train \
  --scenario configs/utility_b3.json \
  --updates 10 --episodes-per-update 4 --seed 7 \
  --output-dir outputs/joint_ppo
```

An **episode** runs all 12 physical periods. An **update** collects fresh episodes and then adjusts the networks. This command collects 40 training episodes, or 480 physical periods, plus evaluation episodes. It checks that training and saving work; it is too small to establish reliable performance. The dispatch critic uses dispatch advantage A₂ by default. The separate A₁ option is retained for controlled experiments.

For a longer standalone run:

```bash
python -m charging_env.rl.paper_train \
  --scenario configs/utility_b3.json \
  --updates 500 --episodes-per-update 16 --seed 7 \
  --eval-episodes 64 --eval-interval 25 \
  --output-dir outputs/joint_ppo_500
```

This collects 8,000 training episodes and 96,000 physical periods. It uses the current trainer defaults for other settings. The dedicated study below additionally fixes separate seed banks, control checkpoints, and the complete comparison protocol. A standalone command is not a reproduction of that full study.

### A larger network: 25 synthetic zones

`configs/synthetic_25_zones.json` defines 25 service zones, 10 mobile vehicles, and five fixed stations over a 10 km by 10 km grid. It retains the 12-hour service day and `b=3`, with different residential and commercial demand profiles. This is a reproducible scaling test; it does not contain observations from a real city.

The earlier larger-fleet study trains fresh Joint PPO models with seeds 7, 19, and 37. Old four-zone checkpoints cannot be reused because the network dimensions depend on the scenario. It compares final PPO, initialization alone, a bounded inventory rule, and service-first. The original tabular ADP requires exactly two vehicles and is excluded from that study. The newer rollout ADP above supports this ten-vehicle configuration and has a separate comparison protocol.

The completed local run is `outputs/scale_25_study_20261024/`. Read the [25-zone results](COMPARISON.md#scaling-to-25-synthetic-zones), or display its saved table without training:

```bash
python experiments/run_scale_study.py --phase show --study-dir outputs/scale_25_study_20261024
```

From the project root, after activating `.venv` and installing `.[train]`:

```bash
python experiments/run_scale_study.py --phase init --study-dir outputs/scale_25
python experiments/run_scale_study.py --phase select --study-dir outputs/scale_25
for seed in 7 19 37; do
  python experiments/run_scale_study.py --phase train --seed "$seed" --study-dir outputs/scale_25
done
python experiments/run_scale_study.py --phase evaluate --study-dir outputs/scale_25
python experiments/run_scale_study.py --phase show --study-dir outputs/scale_25
```

`init` freezes the settings, seed banks, and implementation. `select` retunes the two rules on 32 separate validation days. Each training run uses 256 teacher days, 100 supervised passes, and 500 PPO updates of 16 fresh days each. Final evaluation uses 256 shared independent test days. The initialization-only models are evaluated separately so that the effect of subsequent PPO training is visible. The final update is used regardless of monitoring scores.

Exhaustively checking every joint action is infeasible for ten vehicles. The new inventory solver improves one vehicle's decision at a time, starting from two feasible fleet plans and making up to two passes. Every candidate is evaluated as a full fleet plan, including shared station capacity and competing requests. This bounds the search but does not guarantee the best joint action. The demonstration teacher additionally follows PPO's action mask. The original exhaustive two-vehicle solver remains the default for earlier experiments.

For a short standalone check, use:

```bash
python -m charging_env.rl.paper_train \
  --scenario configs/synthetic_25_zones.json \
  --warmstart-dispatch-solver coordinate --warmstart-coordinate-sweeps 2 \
  --warmstart-episodes 8 --warmstart-epochs 2 \
  --updates 2 --episodes-per-update 4 --seed 7 \
  --reward-scale 20 --normalize-advantages --gae-lambda 1 \
  --eval-episodes 4 --eval-interval 1 \
  --output-dir outputs/scale_smoke
```

This short command checks execution only. Use the study runner for a comparison with tuned rules and held-out scenarios. Study directories preserve their records and must be new; the standalone trainer automatically adds a suffix to an occupied output path.

### Joint PPO with an inventory-rule warm start

Use this optional workflow to address the near-zero recharge behavior diagnosed in the [comparison guide](COMPARISON.md#why-ppo-learned-almost-never-to-recharge). It first learns examples of coherent inventory-rule behavior, then trains with fresh PPO rollouts. This is **rule-guided initialization**, so its result should be compared with both ordinary PPO and the rule itself.

```bash
python -m charging_env.rl.paper_train \
  --scenario configs/utility_b3.json --seed 7 \
  --warmstart-episodes 512 --warmstart-epochs 100 \
  --reward-scale 20 --normalize-advantages --gae-lambda 1 \
  --actor-learning-rate 0.0001 --dispatch-entropy-coef 0.005 --target-kl 0.03 \
  --updates 500 --episodes-per-update 16 \
  --output-dir outputs/joint_ppo_warmstart
```

This adds 512 teacher episodes before the 8,000 PPO episodes. The teacher is used only during initialization. `warmstart_checkpoint.pt` saves the cloned policy before PPO; `checkpoint.pt` saves the final PPO policy. The warm-start summary records the demonstration and supervised-fitting budget. Resume restores this state and does not repeat initialization. Reward scale and initialization settings cannot be changed when resuming because the saved critics and training history already use those settings.

To inspect the completed recharge study:

```bash
python experiments/run_recharge_study.py --phase show \
  --study-dir outputs/recharge_study_20261024
```

Its scorecard compares final warm-started PPO, cloning alone, original Joint PPO, approximate DP, and the time-dependent inventory rule. All use the same 512 previously unused single-day test scenarios. The learned rows average separately executed models with training seeds 7, 19, and 37. Cloning alone is included to separate imitation gains from subsequent PPO gains.

To reproduce the protocol, start with an unused directory and the saved original controls:

```bash
python experiments/run_recharge_study.py --phase init \
  --study-dir outputs/recharge_repeat --control-root outputs/paper_ppo_study
for seed in 7 19 37; do
  python experiments/run_recharge_study.py --phase train --seed "$seed" \
    --study-dir outputs/recharge_repeat
done
python experiments/run_recharge_study.py --phase evaluate \
  --study-dir outputs/recharge_repeat
python experiments/run_recharge_study.py --phase show \
  --study-dir outputs/recharge_repeat
```

Initialization copies the controls and freezes the implementation, training settings, and test seeds. Evaluation uses the final update-500 checkpoints, irrespective of monitoring scores. Repeating this protocol reproduces an existing test bank; reserve a different untouched bank before using its results to develop another method. The runner preserves results and refuses to overwrite a completed test.

### Earlier PPO: keep it as a comparison method

```bash
python -m charging_env.rl.train \
  --scenario configs/utility_b3.json --dispatch-final-activation silu \
  --updates 10 --episodes-per-update 4 --seed 7 \
  --output-dir outputs/earlier_ppo
```

This selects `b=3` and the SiLU final dispatch activation. It does not reproduce every optional change used in the saved earlier-PPO control models. Their exact settings are stored with the checkpoints and described in [the historical targeted-training report](archive/B3_TRAINING_IMPROVEMENTS.md). The [earlier PPO reference](methods/LEGACY_PPO.md) documents all flags, network structure, and the older learning equations.

## Find the files from a run

Each standalone training CLI prints the directory it actually uses. If the requested directory exists, it chooses a numbered sibling such as `outputs/joint_ppo_2`. Repeating the same command therefore preserves earlier checkpoints without requiring you to invent a name. Direct Python training calls and the dedicated study runners have stricter unused-directory checks.

| What you need | Fixed-price rollout ADP | Original tabular ADP | Joint PPO | Earlier PPO |
|---|---|---|---|---|
| Learned controller | None; plans online | `checkpoint.npz` | `checkpoint.pt` | `checkpoint.pt` |
| Run overview | `summary.json` | `summary.json` | `summary.json` | `summary.json` |
| Decisions or training progress | `planning_trace.json` | `training_metrics.csv` | `training_metrics.csv` | `training_metrics.csv` |
| Validation over learning | No offline learning | No PPO-style monitoring curve | `validation_history.json` | `validation_metrics.csv` |
| Validation-selected snapshot | Not used | Not used | `best_checkpoint.pt` | `best_checkpoint.pt` |
| Exact environment | `protocol.json` | `scenario.json` and checkpoint | `scenario.json` and checkpoint | `scenario.json` and checkpoint |

Read the method guides for the remaining metadata files. `checkpoint.pt` is the final state; `best_checkpoint.pt` is the best monitored snapshot according to that run's validation rule. The completed six-method comparison uses the final 500-update PPO snapshots, fixed in advance.

A learning loss is an optimization diagnostic, not profit. Even a lower critic error or a flat training curve does not show that a controller beats a benchmark. Use the original business-profit metric on held-out scenarios.

## Continue or evaluate a PPO model

Use the same module that created the checkpoint. For joint PPO:

```bash
python -m charging_env.rl.paper_train \
  --resume outputs/joint_ppo/checkpoint.pt --updates 10 \
  --output-dir outputs/joint_ppo_continued
```

`--updates 10` adds ten updates to the saved counters. Resume restores the scenario, parameters, optimizer state, and random-generator state. Use the actual printed directory if the original run received a suffix. Do not supply a replacement scenario when resuming.

To evaluate one of the completed joint-PPO models without changing it:

```bash
python -m charging_env.rl.paper_train \
  --evaluate-only outputs/paper_ppo_study/training/seed_7/checkpoint.pt \
  --eval-episodes 128 --eval-seed-base 193000001 \
  --eval-policy stochastic
```

This prints a fresh evaluation; it does not reproduce the recorded scorecard. Repeating the same seed bank repeats those scenarios. To evaluate earlier PPO, use `charging_env.rl.train` and one of its checkpoints; to evaluate approximate DP, use the short Python example in [FITTED_VALUE.md](methods/FITTED_VALUE.md). Each method guide explains its evaluation options.

## Reproduce the six-method study

This earlier comparison trains three joint-PPO models and evaluates them alongside three frozen approximate-DP models, three frozen earlier-PPO models, and three tuned heuristic families. The learned-method mean rows average the three model results; they do not combine the models into an ensemble.

The commands below require the existing control files in `outputs/fitted_value_study/fitted/seed_{7,19,37}/` and `outputs/fitted_value_study/ppo/seed_{7,19,37}/`. These are present in this local project. If starting without those artifacts, the [fitted-value guide](methods/FITTED_VALUE.md) and [archive index](archive/README.md) explain the earlier studies that produced them. This workflow deliberately reuses fixed controls, rather than silently retraining different ones.

<details>
<summary>Full reproduction procedure — expand after preparing the control models</summary>

Choose an unused study directory and keep it unchanged through all phases:

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
python experiments/summarize_paper_ppo_study.py \
  --study-dir outputs/paper_ppo_repeat --show
```

`init` records the settings, copies the controls, and freezes source hashes. `select` tunes the three rules on validation scenarios. `train` fits one new PPO seed at a time. `evaluate` runs all fixed policies on the same independent test scenarios. The summary step audits the saved evidence and writes `analysis.json`; `--show` then displays it. Do not edit frozen source files midway through a study. To change the implementation, start a new study directory.

Plotting is optional and needs Matplotlib:

```bash
python -m pip install matplotlib
python experiments/plot_paper_ppo_study.py \
  --study-dir outputs/paper_ppo_repeat
```

The recorded study used 512 base scenarios and 256 scenarios for each of two stress cases. No test result selected the models or heuristic settings. These banks reproduce that experiment; if you use their results to guide a new algorithm, reserve a new untouched bank before claiming new independent performance. The current runner fixes its seed banks in code, so a newly designed study must explicitly change and document them before initialization.

</details>

## Judge a result

Compare **discounted business profit per episode** on the same scenario seeds. Keep customer model, physical settings, checkpoint choice, and execution mode explicit. Training randomness, validation for selection, and final testing have different roles; using one set for all three produces an optimistic comparison.

Read profit alongside service, recharge, travel, and unmet-demand measures. Then check whether improvements hold across training seeds and changed operating conditions. [COMPARISON.md](COMPARISON.md) applies that reasoning to the saved results; [VALIDATION.md](reference/VALIDATION.md) separates implementation correctness from performance evidence.
