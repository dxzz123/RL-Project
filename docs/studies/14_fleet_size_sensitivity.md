# Study 14: Fleet-size sensitivity of the five non-PPO methods

## Question

How do the three heuristics, Rollout ADP, and NN ADP perform under baseline demand when the MCV fleet contains 5, 10, or 15 vehicles?

PPO is intentionally excluded from this study.

## Matched protocol

- The private simulator uses the baseline demand and choice scenario.
- All methods use the same learned-pricing coefficient vector and the same fitted historical market model. The scenario-bound pricing actor is reconstructed for each fleet size without changing its coefficients.
- Every result uses the same frozen bank of 128 random test-day seeds. This bank was named `volume_up` in the source candidate protocol, but this study evaluates those seeds with the unscaled baseline arrival rates.
- The fleet-5 initial locations are `(1, 5, 9, 17, 23)`.
- The fleet-10 initial locations are the original `(1, 3, 5, 7, 9, 15, 17, 19, 21, 23)` layout.
- The fleet-15 layout adds one vehicle at each fleet-5 representative location to the original fleet-10 layout.
- Every vehicle starts with 55 kWh.
- Heuristic and Rollout ADP parameters are unchanged from the matched benchmark.
- Rollout ADP uses planning seeds 7, 19, and 37.
- Because the NN ADP feature width includes the number of vehicles, the fleet-5 and fleet-15 networks are trained from scratch. Each uses 768 fresh offline training days, 96 separate validation days, the mixed logging policy, 80 maximum epochs, and policy seeds 7, 19, and 37. Fleet 10 reuses the sealed matched NN ADP checkpoint.
- For Rollout ADP and NN ADP, the three policy seeds are averaged within each test day before the 128 daily values are averaged.

This design changes the number and initial deployment of vehicles together. It does not estimate a causal fleet-size effect independent of deployment geometry.

## Results

Values are mean discounted profit per 12-hour test day.

| Method | Fleet size = 5 | Fleet size = 10 | Fleet size = 15 |
|---|---:|---:|---:|
| Service-first heuristic | 347.28 | 470.02 | 562.37 |
| Constant-inventory heuristic | 386.90 | 457.04 | 540.32 |
| Time-varying-inventory heuristic | 431.97 | 502.34 | 567.24 |
| Rollout ADP | 423.20 | 491.88 | **573.41** |
| NN ADP | **436.09** | **502.62** | 555.62 |

## Findings

- NN ADP has the highest mean at fleet sizes 5 and 10.
- Rollout ADP has the highest mean at fleet size 15.
- Time-varying inventory is close to the best method at every tested fleet size.
- More vehicles raise the mean profit for every method in this experiment, but the gain is not uniform across controllers.

## Validation and artifacts

The new fleet-5 and fleet-15 evaluation contains 2,304 method-seed test episodes. All 2,304 completed with zero invalid actions. The fleet-10 column reuses the previously frozen evaluation at source commit `574ef42b1c3d5a4ce1dc9998a5670e89eb5fc452`.

The machine-readable summary contains the daily values and per-policy-seed means: [fleet_size_sensitivity_summary.json](../results/fleet_size_sensitivity_summary.json).


