# What the checks establish

The project needs two kinds of evidence: the implementation must follow its rules, and a controller must earn useful profit on new scenarios. A checkpoint that reloads correctly can still contain a weak policy. This page covers correctness and reproducibility; [COMPARISON.md](../COMPARISON.md) covers measured performance.

The current full suite contains **309 tests** covering the simulator, interfaces, learned policies, rollout planning, price learning, and study protocols. Passing correctness tests does not establish that a controller earns more profit than a benchmark.

## Run the checks

From the project root:

```bash
source .venv/bin/activate
python -m pip install -e '.[train]'
python -m unittest discover -s tests -v
```

The run should end with `OK`. For visible examples:

```bash
python examples/manual_walkthrough.py
python examples/fcs_discharge_walkthrough.py
python examples/check_gym.py
```

The manual example writes `outputs/manual_episode.csv` and follows a fixed script; it is not an optimized controller. The export example has a [hand calculation](ENVIRONMENT.md#5-check-a-transition-by-hand). Training and evaluation commands are in [TRAINING.md](../TRAINING.md). Repeated standalone training commands choose numbered output directories automatically.

## What is tested

| Area | Main test files | What the checks cover |
|---|---|---|
| Physical system and customers | `test_simulator.py`, `test_station_visit.py`, `test_choice_intercept.py` | Whole-request service, accessibility and choice probabilities, travel, shared station capacity, battery limits, balanced energy and reward ledgers |
| Gymnasium and scenario loading | `test_gym.py`, `test_scenario_entrypoints.py` | Declared observation/action formats, variable request counts, configuration routing |
| Earlier PPO | `test_training.py`, `test_training_improvements.py`, `test_training_monitoring.py` | Stage-aware returns, likelihood replay, network updates, evaluation, and checkpoint continuation |
| Earlier-PPO experiments | `test_dispatch_activation.py`, `test_dispatch_roles.py`, `test_inventory_shaping.py`, `test_warmstart.py`, `test_training_output_directories.py` | Optional architecture and shaping settings, imitation isolation, repeated-command behavior |
| Original table-based ADP | `test_fitted_value.py` | Bilinear interpolation, sampled backward backups, ordered vehicle states, terminal reward counted once, checkpoint round trips |
| Rollout ADP | `test_rollout_adp.py`, `test_general_adp_study.py` | Different fleet sizes, shared-draw score replay, terminal salvage, negative profits, action deduplication, planning RNG isolation, roundoff guards, price selection, counters and saved evidence |
| Learned pricing for rollout and inventory | `test_adaptive_pricing.py`, `test_priced_rollout.py`, `test_priced_adp_study.py` | Pricing information limits, legal prices, immutable policies, safe checkpoints, policy-search reproduction, constant-price parity, future pricing, simulator score replay, selection, disjoint banks and frozen evidence |
| Joint PPO | `test_paper_ppo.py` | Both GAE trace factors, joint fleet clipping, entropy sum, restricted pricing information, valid discrete actions, four-network updates, RNG isolation, exact continuation |
| Benchmarks and study protocols | `test_benchmarks.py`, `test_compact_benchmarks.py`, `test_b3_*.py`, `test_paper_study.py`, `test_source_provenance.py` | Candidate outcomes versus simulator execution, paired evaluation, selection rules, separate seeds, source/checkpoint guards |

**Likelihood replay** recalculates recorded action probabilities before learning changes them. **Checkpoint continuation** verifies that restored optimizer and random-generator state produce the same next update. Tests of synthetic edge cases, such as two vehicles sharing a constrained station, establish behavior for those cases; they do not estimate their frequency in practice.

The two PPO advantage calculations are intentionally different. At GAE trace coefficient 1, both implementations' targets reduce to the full discounted return for their learning rewards. At coefficient 0, earlier PPO still combines customer choice and dispatch in its pricing target; joint PPO's pricing target uses the old dispatch critic directly. [LEGACY_PPO.md](../methods/LEGACY_PPO.md) and [JOINT_PPO.md](../methods/JOINT_PPO.md) give their respective equations. Passing one method's tests must not be interpreted as establishing that the methods are identical.

Gymnasium may warn about unbounded observations because arrivals have no artificial count cap and one supported price-noise distribution is unbounded. It may also warn that energy actions use kWh rather than normalized fractions. These are interface choices described in [ENVIRONMENT.md](ENVIRONMENT.md). The checker validates the interface; it does not establish that an arbitrary off-the-shelf PPO implementation can handle this two-stage, variable-request problem.

## Evidence behind the learned-pricing comparison

The latest study is stored in `outputs/priced_adp_study_20261025/`. It trains price policies for both rollout and inventory dispatch, with separate environment banks for training, validation and final testing. `protocol.json` freezes the algorithm, budgets, source files, scenario configurations and bank definitions before training. `provenance/` records historical seed-use checks and execution logs; `source_snapshot/` preserves the frozen implementation.

| Evidence | Location within the study |
|---|---|
| Every proposed price vector, sampled return, elite update and saved policy | `training/<case>/<phase>/seed_<seed>/` |
| All 49 inventory settings and their validation outcomes | `rules/<case>/` |
| Fixed, initial learned and refined pricing candidates | `validation/<case>/` |
| Final pricing choices, including any fixed-price fallback | `selection/<case>.json` |
| Frozen input hashes and all independent episode outcomes | `test/` |
| Profit comparisons, price usage and paired intervals | `analysis.json` |
| Search reconstruction, accounting, uncertainty and training replays | `audit.json` |
| Simulator-based rollout score checks and final episode replays | `physics_audit.json` |

The declared training budget is 29,184 candidate episodes across the two networks and three seeds. Candidates share environmental scenarios within each generation: these are 1,344 distinct training days, not 29,184 independent demand days. Validation evaluates 5,248 policy episodes on 32 separate days per network. The final comparison evaluates 6,912 policy episodes on 256 new days per network. Replays verify recorded behavior and do not add independent performance observations.

The independent search audit reconstructs proposal randomness and every elite-distribution update, verifies the final pricing checkpoints, checks that validation determines the selected policies, reconciles business-profit ledgers and recomputes paired bootstrap intervals. A separate physics audit evaluates rollout candidates through actual `ChargingSimulator.dispatch` transitions. It does not use the planner's fast current-outcome helper to check that helper's own answer. It also checks future pricing records and replays complete final episodes.

**Both completed audits pass, with zero invalid actions in all 6,912 final episodes.** The search audit replays 36 recorded training episodes. The physics audit replays six complete final episodes, covering all three selected seeds in both networks, and independently checks the three deduplicated candidates at the first decision of seed 7 in each network. Those candidate checks use six current transitions and 528 future transitions; together with the 72 decisions in complete-episode replays, the physics audit counts 606 transitions. The maximum discounted-profit or candidate-score discrepancy is 2.27e-13, below its 1e-8 tolerance.

Constant-price parity tests verify that wrapping a fixed price in the new pricing interface reproduces the original rollout controller. Information-boundary tests alter hidden requests, choices and congestion fields and verify that pricing remains unchanged. These checks distinguish an implementation error from a policy that is valid but economically weak. See [COMPARISON.md](../COMPARISON.md) for the measured outcome.

## Evidence behind the rollout comparison

The earlier fixed-price study is stored in `outputs/general_adp_study_20261025/`. Its `protocol.json` freezes the source, planning settings, validation/test banks and saved control models. `rule_selection/` and `price_selection/` contain every evaluated setting and the validation choices. `test/` contains individual episodes, exact planning seed tables and candidate scores. `analysis.json` summarizes results; `audit.json` records independent checks.

The audit reconstructs the planner's local random stream, checks candidate argmax decisions and model-query counts, reconciles every episode's profit ledger, recomputes paired intervals, and verifies source/checkpoint hashes. The 5,376 final policy episodes use 256 independent scenarios per network. Planning seeds 7, 19 and 37 are repeated executions of an online controller, not offline training runs.

The supplementary `reference_inventory/` bundle adds 256 evaluations of the original exhaustive half/full inventory rule on the same small-case bank. Its separate audit verifies the fixed historical settings, source hashes, complete episode records and profit ledger; its result is not part of the primary 5,376-episode count.

The separate `outputs/adp_transfer_20261025/` checks two additional known configurations. Its 384 policy episodes have zero invalid actions. The audit also reproduces complete episodes and planning traces exactly. These changed configurations do not test robustness when the planner's model is wrong.

Before the final studies, an independent replay used actual simulator transitions to check 6,912 future transitions and 51 decisions, including a terminal inventory payment. The maximum score discrepancy was 2.27e-13. Its script, detailed records and the full unit-test log are retained under `outputs/general_adp_development_20261025/`. Replay checks correctness; it is not additional independent performance evidence.

## Evidence behind the original joint-PPO comparison

The files listed below are local experiment records under `outputs/`, which is not included in Git. Use the saved study files or [reproduce the experiment](../TRAINING.md#reproduce-the-six-method-study) to inspect them.

The original six-method study trained joint PPO with seeds 7, 19, and 37 for 500 updates each. It reused three saved approximate-DP models and three earlier-PPO models, and selected one rule from each of three heuristic families using separate validation scenarios.

| Evidence | Location in `outputs/paper_ppo_study/` |
|---|---|
| Training settings, independent seed banks, and selection criteria | `protocol.json` |
| Original full test-suite log | `logs/unit_tests.log` |
| Audited results, uncertainty, budgets, and operating measures | `analysis.json` |
| Implementation and algorithm specification | `provenance/executed_training_source/` |
| Verification after installation in this project | `provenance/installed_verification.json` |

Three new PPO runs collected 288,000 training physical periods in total. Rule selection evaluated 82 candidates on 128 validation scenarios, or 10,496 policy episodes. Final testing ran 12 controllers on 512 base scenarios and 256 scenarios in each of two stress cases: 12,288 policy episodes over 1,024 distinct scenarios. Both rule validation and final testing recorded zero invalid actions.

The saved audit checks individual episode profits, paired uncertainty intervals, reward components, operating measures, training budgets, and source/checkpoint fingerprints. The post-installation check replayed 18 recorded episodes across all nine learned models; those replays check reproduction and are not extra independent performance evidence. The [comparison guide](../COMPARISON.md) interprets the scores and explains why they do not prove general superiority or global convergence.

## Reproduce a recorded experiment

Saved protocols identify the implementation and checkpoints used by each study. A historical audit verifies these fingerprints against the saved source snapshot; it reports separately whether the current source still matches. Presentation changes do not change the recorded experiment, and a verified snapshot preserves the original implementation for reproduction.

Starting or continuing a study requires its exact frozen source files. Use a fresh study directory after changing the implementation. The saved-results viewer (`--show`) reads the completed analysis without rerunning policies or modifying evidence.

[The historical validation record](../archive/VALIDATION_HISTORY.md) contains earlier test counts, execution checks, and study evidence.
