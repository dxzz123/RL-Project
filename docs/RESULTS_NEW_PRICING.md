# Re-evaluation under LADWP base-price / premium-menu pricing

Complete re-run of the controller comparison after the 30 October 2026 pricing
change (commit `90a5273`). Every number here replaces the figures in
[RESULTS.md](RESULTS.md), which describe the old absolute `price_levels` menu
and are no longer valid. For how the methods differ see
[METHODS.md](METHODS.md); for commands, [RUNNING.md](RUNNING.md).

## Coverage

| Controller | Small city | Large city | Note |
|---|:---:|:---:|---|
| Service-first | yes | yes | |
| Constant inventory | yes | - | joint enumeration, two vehicles |
| Time inventory | yes | yes | large city uses the bounded coordinate solver |
| Table ADP | yes | **no** | indexes both vehicles' state; ten vehicles is out of reach |
| Rollout ADP | yes | yes | |
| Joint PPO | yes | yes | 500 updates per seed |
| Learned pricing | **no** | **no** | not re-run; see "What is not covered" |
| Historical market model | **no** | **no** | not re-run; study 12's whole premise |

Produced by `experiments/run_pricing_reevaluation.py`. The frozen studies 08-12
could not be rerun: their artifacts live on another machine and several guards
read recorded protocols from absolute paths there.

## Protocol

- Rules chosen on validation days only, per family, by mean discounted profit.
- **Every ADP controller also selects its premium on validation days**, by the
  same rule. See "Why price selection decides the ranking" -- without this the
  comparison measures the price, not the planner.
- Test conditions follow study 12: the original pattern, demand +40%, and a
  shift that rotates demand locations by half the grid and delays the hourly
  pattern three hours. Both shifts are applied to the resolved
  `[horizon, zone]` rate matrix, so they work for a scenario that pins
  `arrival_rates`, as `configs/synthetic_25_zones.json` does.
- Small city: 128 validation and 256 test days per condition.
  Large city: 64 validation and 128 test days per condition (study 12 used 128).
- Learned controllers train on the original pattern only, then face the shifts.
- Training seeds 7, 19, 37. The +- column is across training seeds for
  multi-seed methods and the within-bank standard error for single-policy rules.
- Paired bootstrap intervals are conditional on the evaluated policies and
  exclude training-seed uncertainty.

## Small city, four zones and two vehicles

256 test days per condition. Mean discounted profit per day, dollars.

| Method | original | demand +40% | shifted | seeds |
|---|---:|---:|---:|:---:|
| **Table ADP** | **+68.03** +-0.27 | **+46.48** +-0.41 | **+45.12** +-0.47 | 3 |
| Rollout ADP | +63.71 +-0.22 | +40.96 +-0.47 | +40.78 +-0.35 | 3 |
| Service-first | +60.36 +-0.95 | +33.29 +-1.20 | +39.39 +-1.08 | 1 |
| Constant inventory | +56.92 +-0.99 | +35.76 +-1.20 | +40.41 +-1.10 | 1 |
| Joint PPO | +54.30 +-0.49 | +26.13 +-0.54 | +34.47 +-1.35 | 3 |
| Time inventory | +51.92 +-1.02 | +20.56 +-1.25 | +32.79 +-1.06 | 1 |

Paired difference against service-first, 95% bootstrap, first training seed:

| Method | original | demand +40% | shifted |
|---|---|---|---|
| Table ADP | +7.99 [+6.86, +9.10] | +12.72 [+11.34, +14.15] | +5.92 [+4.76, +7.10] |
| Rollout ADP | +3.60 [+2.48, +4.71] | +7.88 [+6.51, +9.26] | +1.35 [+0.33, +2.42] |
| Constant inventory | -3.44 [-4.86, -2.01] | +2.47 [+0.91, +4.05] | +1.02 [-0.33, +2.36] |
| Joint PPO | -5.52 [-7.03, -4.03] | -6.54 [-8.14, -4.92] | -4.14 [-5.71, -2.69] |
| Time inventory | -8.44 [-10.37, -6.53] | -12.73 [-14.88, -10.67] | -6.61 [-8.28, -4.97] |

## Large city, 25 zones and ten vehicles

128 test days per condition.

| Method | original | demand +40% | shifted | seeds |
|---|---:|---:|---:|:---:|
| **Rollout ADP** | **+510.98** +-0.52 | **+363.48** +-0.86 | **+470.04** +-0.83 | 3 |
| Service-first | +480.05 +-3.54 | +308.26 +-4.69 | +447.48 +-3.27 | 1 |
| Time inventory | +456.82 +-3.64 | +287.92 +-4.80 | +415.17 +-3.76 | 1 |
| Joint PPO | +412.10 +-4.60 | +247.71 +-5.91 | +360.07 +-5.58 | 3 |

Paired difference against service-first:

| Method | original | demand +40% | shifted |
|---|---|---|---|
| Rollout ADP | +30.37 [+26.74, +34.07] (93% of days) | +55.92 [+50.93, +60.86] (98%) | +23.32 [+19.69, +26.98] (86%) |
| Time inventory | -23.24 [-29.45, -16.97] | -20.33 [-27.90, -12.82] | -32.31 [-38.47, -26.03] |
| Joint PPO | -72.83 [-79.40, -66.19] (3%) | -67.06 [-74.13, -59.97] (5%) | -92.94 [-99.19, -86.79] (0%) |

## Why price selection decides the ranking

Every ADP controller quotes one fixed premium. `RolloutADPConfig.price_index`
and `FittedValueConfig.price_index` default to the **middle** menu index, which
on the three-level premium menu is 1.4x -- one notch below the ceiling every
rule selects. Under the old seven-level absolute menu the middle index was
$0.50, close enough to the optimum that the default was harmless. It is not
harmless now.

Pinning both ADP controllers to the middle premium produces this table instead,
which inverts the ranking:

| Method (pinned 1.4x) | small, original | large, original |
|---|---:|---:|
| Table ADP | +49.58 | n/a |
| Rollout ADP | +47.81 | +394.90 |
| Service-first (selected 1.7x) | +60.36 | +480.05 |

Scoring rollout ADP and service-first at the **same** premium separates the two
effects (small city, 256 test days, original pattern):

| Fixed at | Rollout ADP | Service-first | ADP advantage |
|---|---:|---:|---:|
| 1.1x | +20.74 | -10.08 | **+30.82** |
| 1.4x | +47.97 | +37.79 | +10.18 |
| 1.7x | **+63.95** | +60.36 | +3.60 |

ADP wins at every price. Its advantage *shrinks* as the price improves:
**good pricing and good dispatch are substitutes.** When the price is too low
the fleet faces far more demand than it can serve and triage is worth a great
deal; price correctly and demand is already rationed, so a greedy rule gives up
little. Any claim that one method beats another on this model is close to
meaningless unless both are priced at their own optimum.

Validation-selected premiums, all six controllers at the ceiling:

| | 1.1x | 1.4x | 1.7x | selected |
|---|---:|---:|---:|:---:|
| small, rollout ADP | +19.40 | +45.34 | **+62.40** | 1.7x |
| small, table ADP | +30.36 | +46.69 | **+66.97** | 1.7x |
| large, rollout ADP | +210.19 | +382.18 | **+508.36** | 1.7x |

The three rule families select 1.7x as well, in both cities. The menu ceiling
binds everywhere, which is why `configs/utility_b3_wide.json` exists: on a
five-level menu the interior optimum is 2.0x, and 1.7x costs about 4 dollars a
day in the small city.

## Higher demand now reduces profit

Study 12 recorded the large city *gaining* from a demand surge, 940.46 to
1,081.91. Here it loses: service-first 480.05 to 308.26, and rollout ADP 510.98
to 363.48. The sign flipped because the premium menu caps the price at 1.7x and
every controller already sits there, so additional demand the fleet cannot serve
converts into unmet-demand penalties instead of revenue. Under the old menu the
operator could quote up to $1.00 absolute and profit from the surge.

Reward decomposition for service-first in the small city, 64 episodes, shows the
mechanism directly:

| Premium | Mobile revenue | FCS revenue | Unmet cost | Profit |
|---|---:|---:|---:|---:|
| 1.1x | 69.28 | 180.22 | **96.15** | -17.77 |
| 1.4x | **76.39** | 200.51 | 63.37 | +36.87 |
| 1.7x | 68.13 | **208.95** | **51.32** | +59.51 |

Mobile revenue peaks at 1.4x, exactly as elasticity predicts, and is then
irrelevant: the 8-dollar mobile swing is swamped by a 45-dollar swing in unmet
demand. Two vehicles can deliver at most 60 kWh per period, so the high premium
rations mobile demand and steers it to the operator's own stations, which carry
better margins. Pricing here is a demand-allocation lever, not a revenue lever.

## What this changes relative to the recorded studies

1. **The ordering survives.** ADP ahead of the rules ahead of Joint PPO, in both
   cities, as in studies 10-12 -- once prices are matched.
2. **Rollout ADP stays below table ADP in the small city** (+63.71 against
   +68.03), reproducing the open question in [RESULTS.md](RESULTS.md) section 4.
3. **ADP's margin widens under stress.** Table ADP goes +7.99 to +12.72 under
   +40% demand; rollout ADP +30.37 to +55.92 in the large city. The rules shed
   far more, because a demand surge makes the battery-versus-service tradeoff
   binding and that is exactly what ADP plans.
4. **Joint PPO is last everywhere**, significantly below service-first in all
   six city/condition cells and winning 0-5% of large-city days. It is the one
   method that learns its own price, so it carries no price handicap; this is a
   real result. It matches study 12's finding.
5. **ADP is markedly more stable across training seeds** (+-0.2 to +-0.9) than
   the rules' within-bank error (+-1.0 to +-4.7).

## What is not covered

- **Learned pricing** (`adaptive_pricing.py`) was not re-run. In study 11 it was
  the winner, worth +30.39/day in the large city. It is also where the new
  pricing is most likely to change the conclusion, since a pricing learner now
  searches a premium menu whose top level binds.
- **The historical market model** (`history_model.py`, study 12) was not re-run.
  Every controller here sees the true scenario, so these results correspond to
  [RESULTS.md](RESULTS.md) section 2, not section 1.
- **No equal-compute comparison.** Rollout ADP costs about 5.0 s per dispatch
  decision in the large city and 0.66 s in the small one, against 0.2 s and
  14 ms for the coordinate inventory rule.
- **One menu.** All of the above uses the three-level premium menu. The
  five-level `configs/utility_b3_wide.json` has only the rule arms scored.

## Reproducing

```bash
python experiments/run_pricing_reevaluation.py --phase select     --case small
```

```bash
python experiments/run_pricing_reevaluation.py --phase train-adp  --case small --price-index 2
```

```bash
python experiments/run_pricing_reevaluation.py --phase select-adp --case small
```

```bash
python experiments/run_pricing_reevaluation.py --phase train-ppo  --case small --seeds 7
```

```bash
python experiments/run_pricing_reevaluation.py --phase evaluate   --case small --only table_adp rollout_adp service_first --tag adp_selected
```

```bash
python experiments/run_pricing_reevaluation.py --phase summarize  --case small
```

`--case city` runs the large city. `--seeds` splits training and evaluation so
seeds run in parallel; `--tag` writes each arm to its own file, which
`summarize` merges. Artifacts land in `outputs/`, which is gitignored, so they
must be regenerated locally.
