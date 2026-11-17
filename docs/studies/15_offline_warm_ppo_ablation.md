# Study 15: Offline-warm PPO ablation at fleet size 10

## Question

For the fleet-10 baseline problem, how much comes from the offline warm start, how much comes from 512 on-policy PPO training days alone, and how well do the two stages work together?

## Ablations

| Variant | Initialization and training |
|---|---|
| Warm-start only | Train the pricing actor/critic and dispatch actor/critic on the mixed offline trajectory bank, then test immediately with no on-policy PPO |
| PPO only | Randomly initialize all four networks and train for 512 on-policy PPO days with no offline data |
| Offline-warm PPO | Apply the same offline warm start, then train for 512 on-policy PPO days |

## Matched protocol

- Fleet size is 10, with initial locations `(1, 3, 5, 7, 9, 15, 17, 19, 21, 23)` and 55 kWh per vehicle.
- All variants use the same network architecture, pricing and dispatch action spaces, PPO hyperparameters, fitted historical market model, reward scale, and seeds 7, 19, and 37.
- The offline stage uses 768 mixed-policy training days and 96 separate validation days. It behavior-clones the logged premium price levels and fleet dispatch operations while fitting both critics to complete discounted final-profit returns.
- Offline fitting runs for 80 epochs and restores the checkpoint with the lowest validation loss.
- The on-policy stage contains 32 updates of 16 episodes, or 512 training days per seed. Pricing and dispatch are jointly optimized from realized business profit using interleaved GAE.
- Every checkpoint is evaluated stochastically on the same frozen 128 baseline test days with the same action-seed convention.
- The three policy seeds are averaged within each test day before the 128 daily values are averaged.

Only the presence of the offline stage and the presence of the on-policy stage change across rows.

## Results

| Ablation | Offline days | On-policy PPO days | Mean discounted profit/day |
|---|---:|---:|---:|
| Warm-start only | 768 | 0 | 274.11 |
| PPO only | 0 | 512 | -201.42 |
| Offline-warm PPO | 768 | 512 | **345.64** |

Mean differences:

- Offline-warm PPO minus Warm-start only: **+71.53** per day.
- Offline-warm PPO minus PPO only: **+547.06** per day.
- Warm-start only minus PPO only: **+475.53** per day.

## Findings

- The offline warm start is essential at this training budget. PPO from random initialization remains deeply negative after 512 on-policy days.
- The offline stage alone already produces a positive policy with mean profit 274.11.
- On-policy PPO adds substantial value after a useful offline initialization, improving the warm-start-only policy by 71.53 per day.
- The combination is stronger than either component alone: the warm start supplies a viable initial pricing and dispatch behavior, while on-policy training improves it using realized profit.
- The negative PPO-only result is specific to this architecture, hyperparameter set, and 512-day budget; it is not a general claim that PPO cannot learn from scratch with more data or different tuning.

## Validation and artifacts

The study evaluates 9 policy checkpoints over 128 shared test days, for 1,152 policy-seed test episodes. All completed with zero invalid actions. Every PPO-only and Offline-warm PPO checkpoint records exactly 512 on-policy training days.

The machine-readable summary contains all daily values, policy-seed means, checkpoint hashes, training budgets, and mean differences: [offline_warm_ppo_ablation_fleet10_summary.json](../results/offline_warm_ppo_ablation_fleet10_summary.json).
