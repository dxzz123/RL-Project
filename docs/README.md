# Reading the project

The three main guides answer different questions. Start with the one that matches your task; there is no need to read every reference page first.

| Stop | What you will understand |
|---|---|
| **1. [Model](MODEL.md)** | One operating period: customer choice, vehicle actions, batteries, and profit |
| **2. [Comparison](COMPARISON.md)** | How ADP and PPO differ, which methods currently earn more, and what the tests establish |
| **3. [Training](TRAINING.md)** | Installation, short training runs, checkpoints, and the full comparison workflow |

## Explore a method

These guides explain the algorithms after you understand the charging model.

| Question | Guide |
|---|---|
| How does ADP learn future value and choose a dispatch action? | [Approximate DP](methods/FITTED_VALUE.md) |
| How does Joint PPO learn connected pricing and fleet decisions? | [Joint PPO](methods/JOINT_PPO.md) |
| How does the earlier PPO implementation work? | [Earlier PPO/MAPPO](methods/LEGACY_PPO.md) |

## Look up a detail

| Question | Reference |
|---|---|
| How do I call the simulator or change a scenario? | [Environment API](reference/ENVIRONMENT.md) |
| What does a symbol mean, and where is it in Python? | [Notation](reference/NOTATION.md) |
| Which source file should I read or change? | [Code map](reference/ARCHITECTURE.md) |
| What do the automated checks and saved audits verify? | [Validation](reference/VALIDATION.md) |
| How does the charging benefit `b` affect customers leaving? | [Outside option](reference/OUTSIDE_OPTION.md) |

The [experiment index](../experiments/README.md) identifies the scripts used by the training workflow. The mathematical specifications are [the charging model](MDP_RL.pdf) and [Joint PPO](MDP_RL_2.pdf).

[Historical studies](archive/README.md) preserve previous experiments and the original proposal. Their results use different settings or test scenarios; use the current comparison above for the main scorecard.
