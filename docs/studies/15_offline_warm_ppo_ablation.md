# Study 15: Warm-started PPO ablation at fleet size 10

## Question

For the fleet-10 baseline problem, how much comes from the offline warm start, how much can PPO learn from random initialization over 2,048 updates, and how well do the two stages work together?

## Ablations

| Variant | Initialization and training |
|---|---|
| Warm-start only | Train the pricing actor/critic and dispatch actor/critic on the mixed offline trajectory bank, then test immediately with no on-policy PPO |
| PPO only | Randomly initialize all four networks and train for 2,048 PPO updates with no offline data |
| Warm-started PPO | Apply the same offline warm start, then train for 2,048 PPO updates |

## Matched protocol

- Fleet size is 10, with initial locations `(1, 3, 5, 7, 9, 15, 17, 19, 21, 23)` and 55 kWh per vehicle.
- All variants use the same network architecture, pricing and dispatch action spaces, PPO hyperparameters, fitted historical market model, reward scale, and seeds 7, 19, and 37.
- The offline stage uses 768 mixed-policy training days and 96 separate validation days. It behavior-clones the logged premium price levels and fleet dispatch operations while fitting both critics to complete discounted final-profit returns.
- Offline fitting runs for 80 epochs and restores the checkpoint with the lowest validation loss.
- The on-policy stage contains 2,048 updates of 16 episodes, or 32,768 training days per seed. There is no decaying pricing or dispatch behavior-cloning constraint during this stage. Pricing and dispatch are jointly optimized from realized business profit using interleaved GAE.
- Every checkpoint is evaluated stochastically on the same frozen 128 baseline test days with the same action-seed convention.
- The three policy seeds are averaged within each test day before the 128 daily values are averaged.

Only the offline initialization changes between the two PPO-trained rows. Warm-start only isolates the offline stage without online optimization.

## Results

Cells report mean discounted profit per day with 95% confidence intervals in brackets. Marginal and paired intervals use 100,000 percentile-bootstrap resamples of the 128 shared test days after averaging the three policy seeds within each day.

| Ablation | Offline days | PPO updates | On-policy PPO days | Mean discounted profit/day [95% CI] |
|---|---:|---:|---:|---:|
| Warm-start only | 768 | 0 | 0 | 274.11 [266.56, 281.67] |
| PPO only | 0 | 2,048 | 32,768 | 439.03 [432.33, 445.70] |
| Warm-started PPO | 768 | 2,048 | 32,768 | **465.86 [459.25, 472.39]** |

Mean differences:

- Warm-started PPO minus Warm-start only: **+191.75 [185.04, 198.34]** per day.
- Warm-started PPO minus PPO only: **+26.83 [23.68, 30.02]** per day.
- PPO only minus Warm-start only: **+164.91 [158.31, 171.39]** per day.

## Findings

- PPO from random initialization reaches a positive 439.03 after 2,048 updates. This contrasts with its negative result at the earlier 512-day budget and shows that the shorter run was not enough to characterize its long-horizon behavior.
- The offline stage alone produces a positive policy with mean profit 274.11.
- Warm-started PPO reaches the highest mean, 465.86. Online training improves the warm-start-only policy by 191.75 per day, while the warm start retains a 26.83-per-day advantage over PPO only at the same online budget. Both paired 95% intervals exclude zero under the fixed-policy test-day analysis.
- The result supports the warm start as a useful initialization at this training budget, but it is specific to this architecture, hyperparameter set, offline bank, simulator, and three seeds. It is not a general claim that warm starts always outperform sufficiently trained PPO.

## Validation and artifacts

The study evaluates 9 policy checkpoints over 128 shared test days, for 1,152 policy-seed test episodes. All completed with zero invalid actions and zero future-information queries. Every PPO-only and Warm-started PPO checkpoint records exactly 2,048 updates and 32,768 on-policy training days.

The confidence intervals quantify test-day variation conditional on the fixed simulator setup, offline bank, and trained checkpoints. They do not include uncertainty from a new offline dataset or from retraining with new seeds.

The machine-readable CI table is [offline_warm_ppo_ablation_2048_ci.csv](../results/offline_warm_ppo_ablation_2048_ci.csv). The detailed summary contains all daily values, policy-seed means, checkpoint metadata, training budgets, validation counts, and mean differences: [offline_warm_ppo_ablation_2048_summary.json](../results/offline_warm_ppo_ablation_2048_summary.json). Curated per-seed evaluation files, PPO-only update-level training metrics, configurations, protocols, and run metadata are available in the [ablation output bundle](../../outputs/ppo_ablation_2048/); the reused Warm-started PPO training evidence is in the [Warm-started PPO output bundle](../../outputs/warm_started_ppo_2048/). The earlier 512-day ablation remains available as historical evidence in [offline_warm_ppo_ablation_fleet10_summary.json](../results/offline_warm_ppo_ablation_fleet10_summary.json).
