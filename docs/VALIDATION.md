# What the checks establish

The project needs two kinds of evidence: the implementation must follow its rules, and a trained controller must earn useful profit on new scenarios. A checkpoint that reloads correctly can still contain a weak policy. This page covers correctness and reproducibility; [COMPARISON.md](COMPARISON.md) covers measured performance.

The current full suite contains **211 tests** covering the simulator, interfaces, all three learners, and study protocols. The completed manuscript-PPO comparison also records 12,288 test policy episodes with zero invalid actions. Those are separate facts: the first concerns implementation checks, the second concerns execution of the saved experiment.

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

The manual example writes `outputs/manual_episode.csv` and follows a fixed script; it is not an optimized controller. The export example has a [hand calculation](ENVIRONMENT.md#5-check-a-transition-by-hand). Short training commands for all three learners are in [TRAINING.md](TRAINING.md). Repeated standalone training commands choose numbered output directories automatically.

## What is tested

| Area | Main test files | What the checks cover |
|---|---|---|
| Physical system and customers | `test_simulator.py`, `test_station_visit.py`, `test_choice_intercept.py` | Whole-request service, accessibility and choice probabilities, travel, shared station capacity, battery limits, balanced energy and reward ledgers |
| Gymnasium and scenario loading | `test_gym.py`, `test_scenario_entrypoints.py` | Declared observation/action formats, variable request counts, configuration routing |
| Earlier PPO | `test_training.py`, `test_training_improvements.py`, `test_training_monitoring.py` | Stage-aware returns, likelihood replay, network updates, evaluation, and checkpoint continuation |
| Earlier-PPO experiments | `test_dispatch_activation.py`, `test_dispatch_roles.py`, `test_inventory_shaping.py`, `test_warmstart.py`, `test_training_output_directories.py` | Optional architecture and shaping settings, imitation isolation, repeated-command behavior |
| Approximate DP | `test_fitted_value.py` | Bilinear interpolation, sampled backward backups, ordered vehicle states, terminal reward counted once, checkpoint round trips |
| Manuscript PPO | `test_paper_ppo.py` | Both GAE trace factors, joint fleet clipping, entropy sum, restricted pricing information, valid discrete actions, four-network updates, RNG isolation, exact continuation |
| Benchmarks and study protocols | `test_benchmarks.py`, `test_compact_benchmarks.py`, `test_b3_*.py`, `test_paper_study.py` | Candidate outcomes versus simulator execution, paired evaluation, selection rules, separate seeds, source/checkpoint guards |

**Likelihood replay** recalculates recorded action probabilities before learning changes them. **Checkpoint continuation** verifies that restored optimizer and random-generator state produce the same next update. Tests of synthetic edge cases, such as two vehicles sharing a constrained station, establish behavior for those cases; they do not estimate their frequency in practice.

The two PPO advantage calculations are intentionally different. At GAE trace coefficient 1, both implementations' targets reduce to the full discounted return for their learning rewards. At coefficient 0, earlier PPO still combines customer choice and dispatch in its pricing target; manuscript PPO's pricing target uses the old dispatch critic directly. [LEGACY_PPO.md](LEGACY_PPO.md) and [PAPER_PPO.md](PAPER_PPO.md) give their respective equations. Passing one method's tests must not be interpreted as establishing that the methods are identical.

Gymnasium may warn about unbounded observations because arrivals have no artificial count cap and one supported price-noise distribution is unbounded. It may also warn that energy actions use kWh rather than normalized fractions. These are interface choices described in [ENVIRONMENT.md](ENVIRONMENT.md). The checker validates the interface; it does not establish that an arbitrary off-the-shelf PPO implementation can handle this two-stage, variable-request problem.

## Evidence behind the current comparison

The latest study trained manuscript PPO with seeds 7, 19, and 37 for 500 updates each. It reused three saved approximate-DP models and three earlier-PPO models, and selected one rule from each of three heuristic families using separate validation scenarios.

| Evidence | Location in `outputs/paper_ppo_study/` |
|---|---|
| Training settings, independent seed banks, and selection criteria | [protocol.json](../outputs/paper_ppo_study/protocol.json) |
| Original full test-suite log | [logs/unit_tests.log](../outputs/paper_ppo_study/logs/unit_tests.log) |
| Audited results, uncertainty, budgets, and operating measures | [analysis.json](../outputs/paper_ppo_study/analysis.json) |
| Original implementation and manuscript | [executed source snapshot](../outputs/paper_ppo_study/provenance/executed_training_source/) |
| Verification after installation in this project | [installed_verification.json](../outputs/paper_ppo_study/provenance/installed_verification.json) |

Three new PPO runs collected 288,000 training physical periods in total. Rule selection evaluated 82 candidates on 128 validation scenarios, or 10,496 policy episodes. Final testing ran 12 controllers on 512 base scenarios and 256 scenarios in each of two stress cases: 12,288 policy episodes over 1,024 distinct scenarios. Both rule validation and final testing recorded zero invalid actions.

The saved audit checks individual episode profits, paired uncertainty intervals, reward components, operating measures, training budgets, and source/checkpoint fingerprints. The post-installation check replayed 18 recorded episodes across all nine learned models; those replays check reproduction and are not extra independent performance evidence. The [comparison guide](COMPARISON.md) interprets the scores and explains why they do not prove general superiority or global convergence.

## Documentation review

The project-wide clarity review on 2026-10-21 checked the guides against the implementation and saved evidence, separated current instructions from historical reports, and made the two manuscript versions and three controllers explicit. The review checked Markdown links and section references, mathematical notation, executable examples, and CLI options. The full suite passed again: **211 tests in 4.161 seconds**. It does not retrain or retune the recorded policies.

The physical-model, learner, and experiment source files stay unchanged in this review. The recorded source hashes therefore remain meaningful, and existing checkpoints and numerical evidence retain their original interpretation. Documentation and VS Code launch labels can change without claiming a new performance result.

[The historical validation record](archive/VALIDATION_HISTORY.md) retains earlier test counts, smoke runs, and study evidence. They describe successive versions of the project rather than additional checks required before each ordinary run.
