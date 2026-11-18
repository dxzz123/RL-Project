# Study 14: Fleet-size sensitivity of six methods

## Question

How do the three heuristics, Rollout ADP, NN ADP, and Warm-started PPO perform under baseline demand when the MCV fleet contains 5, 10, or 15 vehicles?

## Matched protocol

- The private simulator uses the baseline demand and choice scenario.
- Every result uses the same frozen bank of 128 random test-day seeds. This bank was named `volume_up` in the source candidate protocol, but this study evaluates those seeds with the unscaled baseline arrival rates.
- The fleet-5 initial locations are `(1, 5, 9, 17, 23)`.
- The fleet-10 initial locations are the original `(1, 3, 5, 7, 9, 15, 17, 19, 21, 23)` layout.
- The fleet-15 layout adds one vehicle at each fleet-5 representative location to the original fleet-10 layout.
- Every vehicle starts with 55 kWh.
- The three heuristics, Rollout ADP, and NN ADP use the same frozen learned-pricing coefficient vector and fitted historical market model. Heuristic and Rollout ADP parameters are unchanged from the matched benchmark.
- Rollout ADP uses planning seeds 7, 19, and 37.
- Because the NN ADP feature width includes the number of vehicles, the fleet-5 and fleet-15 networks are trained from scratch. Each uses 768 offline training days, 96 separate validation days, the mixed logging policy, 80 maximum epochs, and policy seeds 7, 19, and 37. Fleet 10 reuses the sealed matched NN ADP checkpoint.
- Warm-started PPO uses the same fleet-specific mixed offline trajectories as NN ADP. Its pricing actor/critic and dispatch actor/critic are warm-started for 80 epochs, selecting by the separate 96-day validation set.
- Each PPO seed then receives 2,048 updates of 16 fresh on-policy days each, or 32,768 on-policy training days. There is no decaying pricing or dispatch behavior-cloning constraint during the online phase. Realized final profit jointly updates pricing and dispatch through interleaved GAE.
- Fleet sizes 5 and 15 resume their validated update-32 checkpoints and continue to a total of 2,048 updates. Fleet size 10 reuses the sealed 2,048-update checkpoint from Study 16.
- Rollout ADP, NN ADP, and Warm-started PPO use seeds 7, 19, and 37. Their three policy seeds are averaged within each test day before the 128 daily values are averaged.
- PPO evaluation is stochastic with the same frozen action-seed convention used by the demand benchmark.

This design changes the number and initial deployment of vehicles together. It does not estimate a causal fleet-size effect independent of deployment geometry. The five non-PPO methods share frozen pricing, while PPO updates its pricing actor during on-policy training, so the PPO comparison reflects joint pricing and dispatch learning. The comparison also does not equalize simulator or optimization compute across methods.

## Results

Values are mean discounted profit per 12-hour test day.

| Method | Fleet size = 5 | Fleet size = 10 | Fleet size = 15 |
|---|---:|---:|---:|
| Service-first heuristic | 347.28 | 470.02 | 562.37 |
| Constant-inventory heuristic | 386.90 | 457.04 | 540.32 |
| Time-varying-inventory heuristic | 431.97 | 502.34 | 567.24 |
| Rollout ADP | 423.20 | 491.88 | **573.41** |
| NN ADP | **436.09** | **502.62** | 555.62 |
| Warm-started PPO (2,048 updates) | 399.98 | 465.86 | 516.54 |

For Warm-started PPO, the mean discounted profit per vehicle is 80.00, 46.59, and 34.44 at fleet sizes 5, 10, and 15, respectively.

## Findings

- NN ADP has the highest mean at fleet sizes 5 and 10, while Rollout ADP has the highest mean at fleet size 15.
- The 2,048-update Warm-started PPO result improves on the earlier 512-day result by 58.07, 120.22, and 147.12 per day at fleet sizes 5, 10, and 15, respectively.
- Warm-started PPO total profit increases from 399.98 at fleet size 5 to 465.86 at fleet size 10 and 516.54 at fleet size 15. Its incremental gain is smaller from 10 to 15 vehicles than from 5 to 10, and its profit per vehicle declines as the fleet grows.
- Warm-started PPO remains below the strongest non-PPO method at every tested fleet size. This is a result for the fixed architecture, hyperparameters, deployment layouts, and training budget used here, not a general ordering of the methods.

## Validation and artifacts

The PPO addition evaluates 9 policy checkpoints over 128 shared test days, for 1,152 PPO policy-seed test episodes. All completed with zero invalid actions and zero future-information queries. Every PPO checkpoint records 2,048 updates and 32,768 on-policy training days.

The earlier five-method fleet-size evidence remains unchanged. The new machine-readable summary records the three PPO fleet sizes, daily values, per-policy-seed means, training budgets, validation counts, and comparisons with the earlier 512-day PPO results: [fleet_size_warm_ppo_2048_summary.json](../results/fleet_size_warm_ppo_2048_summary.json). The earlier six-method, 512-day PPO summary remains available as historical evidence in [fleet_size_sensitivity_summary.json](../results/fleet_size_sensitivity_summary.json).
