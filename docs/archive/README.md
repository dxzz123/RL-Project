# Historical studies

These reports explain how the project developed. For the current question—how approximate DP compares with both PPO algorithms—read [COMPARISON.md](../COMPARISON.md). For commands, use [TRAINING.md](../TRAINING.md).

The reports below used different customer models, test scenarios, rule grids, and trained policies. Compare methods within a report's shared test set. A change in profit between two reports is not, by itself, evidence that an algorithm improved.

| Report | Customer model | What it investigated |
|---|---|---|
| [Initial training analysis](TRAINING_ANALYSIS.md) | `b=0` | Why the short PPO run lost profit, and what longer training changed |
| [First benchmark study](POLICY_BENCHMARKS.md) | `b=0` | PPO versus simple and coordinated dispatch rules |
| [Activation experiment](ACTIVATION_EXPERIMENT.md) | `b=0` | Replacing the dispatch actor's final Tanh with SiLU |
| [First SiLU + b=3 study](SILU_B3_STUDY.md) | `b=3` | Three 500-update PPO models versus retuned rules |
| [Targeted PPO improvements](B3_TRAINING_IMPROVEMENTS.md) | `b=3` | Observation, optimization, exploration, and inventory experiments |
| [Earlier fitted-value study](FITTED_VALUE_STUDY.md) | `b=3` | Approximate DP versus earlier PPO and three rules; creates the controls reused in the latest study |
| [Historical validation record](VALIDATION_HISTORY.md) | Multiple | The checks and evidence recorded at successive development stages |

The [fitted-value guide](../FITTED_VALUE.md) explains the approximate-DP implementation and where to find its earlier five-method comparison. The latest six-method study reevaluates those fitted-value models and the targeted earlier-PPO models on fresh common scenarios, then adds Joint PPO.

## Which saved models are reused?

The study lineage is:

```text
Targeted earlier PPO (b3_improved_study)
             |
             v
Earlier fitted-value study (fitted_value_study)
  - copies the three final earlier-PPO models
  - trains the three approximate-DP models
             |
             v
Joint PPO comparison study (paper_ppo_study)
  - copies those six fixed controls
  - trains three Joint PPO models
  - retunes three rules and tests every method on shared scenarios
```

The last study does not retrain approximate DP or earlier PPO. Its scorecard measures the saved models again on the new scenarios. Training seeds 7, 19, and 37 identify each method's three runs; they do not make different algorithms experience identical training trajectories.

Old executable scripts remain in `experiments/` because they share helper functions and some recorded protocols verify their exact locations and hashes. See [the experiment index](../../experiments/README.md). Old results remain in `outputs/` with their checkpoint and provenance records.

## Model and algorithm specifications

[MDP_RL.pdf](../MDP_RL.pdf) specifies the physical model. [MDP_RL_2.pdf](../MDP_RL_2.pdf) specifies Joint PPO. [proposal.pdf](../proposal.pdf) is the earlier project proposal. The sources have distinct roles; [NOTATION.md](../NOTATION.md) maps their notation to the implemented model and calls out implementation choices.

Each report retains the settings, results, and evidence for its experiment.
