# Study 14: Fleet-size sensitivity of six methods

## Question

How do the three heuristics, Rollout ADP, NN ADP, and Offline-warm PPO perform under baseline demand when the MCV fleet contains 5, 10, or 15 vehicles?

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
- Offline-warm PPO uses the same fleet-specific mixed offline trajectories as NN ADP. Its pricing actor/critic and dispatch actor/critic are warm-started for 80 epochs, selecting by the separate 96-day validation set. Each seed then receives 512 fresh on-policy PPO training days. Realized final profit jointly updates pricing and dispatch through interleaved GAE.
- Fleet sizes 5 and 15 use newly trained PPO networks because the observation and centralized multi-head dispatch dimensions depend on the vehicle count. Fleet 10 reuses its existing sealed Offline-warm PPO checkpoints.
- Rollout ADP, NN ADP, and Offline-warm PPO use seeds 7, 19, and 37. Their three policy seeds are averaged within each test day before the 128 daily values are averaged.
- PPO evaluation is stochastic with the same frozen action-seed convention used by the demand benchmark.

This design changes the number and initial deployment of vehicles together. It does not estimate a causal fleet-size effect independent of deployment geometry. The five non-PPO methods share frozen pricing, while PPO updates its pricing actor during on-policy training, so the PPO comparison reflects joint pricing and dispatch learning.

## Results

Values are mean discounted profit per 12-hour test day.

| Method | Fleet size = 5 | Fleet size = 10 | Fleet size = 15 |
|---|---:|---:|---:|
| Service-first heuristic | 347.28 | 470.02 | 562.37 |
| Constant-inventory heuristic | 386.90 | 457.04 | 540.32 |
| Time-varying-inventory heuristic | 431.97 | 502.34 | 567.24 |
| Rollout ADP | 423.20 | 491.88 | **573.41** |
| NN ADP | **436.09** | **502.62** | 555.62 |
| Offline-warm PPO | 341.91 | 345.64 | 369.42 |

## Findings

- NN ADP has the highest mean at fleet sizes 5 and 10.
- Rollout ADP has the highest mean at fleet size 15.
- Time-varying inventory is close to the best method at every tested fleet size.
- Offline-warm PPO is below all five alternatives at every fleet size after the matched offline warm start and 512 on-policy training days per seed.
- More vehicles raise the mean profit for every method, but the gain is not uniform across controllers. PPO rises only modestly from fleet 5 to fleet 15 compared with the non-PPO methods.

## Validation and artifacts

The PPO addition evaluates 9 policy checkpoints over 128 shared test days, for 1,152 PPO policy-seed test episodes. All completed with zero invalid actions. Every PPO checkpoint records 512 on-policy training days. Fleet sizes 5 and 15 use newly generated checkpoints; fleet 10 uses the previously sealed checkpoints at source commit `574ef42b1c3d5a4ce1dc9998a5670e89eb5fc452`.

The earlier five-method fleet-size evidence remains unchanged. The machine-readable summary contains all six methods, daily values, per-policy-seed means, and PPO training-day counts: [fleet_size_sensitivity_summary.json](../results/fleet_size_sensitivity_summary.json).
