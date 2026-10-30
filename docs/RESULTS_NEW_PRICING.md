# Results under LADWP base-price / premium-menu pricing

Re-run of the method comparison after the 30 October 2026 pricing change. Every
number on this page replaces the figures in [RESULTS.md](RESULTS.md), which
describe the old absolute `price_levels` menu and are no longer valid. For how
the methods differ see [METHODS.md](METHODS.md); for commands,
[RUNNING.md](RUNNING.md).

These runs used the four-zone `configs/utility_b3.json`. The 25-zone city and
the rollout, learned-pricing and historical-model controllers on this branch
have not been re-run yet, so [RESULTS.md](RESULTS.md) remains the only record
for those and is stale in the same way.

Two scenarios are reported, because the first one turned out to be capped:

| Scenario | `mcv_premium_levels` | Config |
|---|---|---|
| Three-level | 1.1x, 1.4x, 1.7x | `configs/utility_b3.json` |
| Five-level ("wide") | 1.1x, 1.4x, 1.7x, 2.0x, 2.2x | `configs/utility_b3_wide.json` |

## Protocol

Produced by `experiments/run_new_pricing_comparison.py`, which rebuilds the
comparison end to end. The frozen `run_fitted_value_study.py` pipeline could not
be reused: its `DEFAULT_PPO_ROOT` points at a path on another machine, and its
protocol guards pin the old absolute menu.

- Rule settings chosen on **128 validation episodes**; test banks never consulted.
- Scored on **512 base** episodes and **256** per stress bank.
- Stress banks: `high_demand` scales `mean_requests_by_zone` by 1.5;
  `low_inventory` starts every battery at 35 kWh.
- Learned methods train on the base scenario only, then face the shifted banks,
  so stress columns measure generalization, not retuning.
- Training seeds 7, 19, 37. Joint PPO: 500 updates per seed. Table ADP:
  `grid_step=10`, `samples_per_period=8`, price fixed at the **middle** menu index.
- Paired bootstrap intervals are conditional on the fixed evaluated policies and
  exclude training-seed uncertainty, which the +- columns report separately.

## The premium response curve

Service-first, recharge threshold 0.2, validation bank. This is the single most
important result on the page.

| Premium | 1.1x | 1.4x | 1.7x | 2.0x | 2.2x |
|---|---|---|---|---|---|
| Mean discounted profit | -13.50 | +37.02 | +58.90 | **+63.51** | +59.27 |

The optimum is interior at **2.0x**, and the curve is steeply asymmetric:
under-pricing one notch costs about 22 dollars, over-pricing one notch about 4.
Under the three-level menu every winning setting sat at the 1.7x ceiling, so
that scenario measures a cap rather than a choice.

`price_max = 1.00` admits premiums up to 2.296x given a 0.396 peak base and the
10% fee; 2.2x leaves 0.042 of headroom.

### Why high prices win

Reward decomposition for service-first, three-level menu, 64 episodes:

| Premium | Mobile revenue | FCS revenue | Unmet cost | Profit |
|---|---|---|---|---|
| 1.1x | 69.28 | 180.22 | **96.15** | -17.77 |
| 1.4x | **76.39** | 200.51 | 63.37 | +36.87 |
| 1.7x | 68.13 | **208.95** | **51.32** | +59.51 |

Mobile revenue peaks at 1.4x, exactly as price elasticity predicts, and is then
irrelevant: the 8-dollar mobile swing is swamped by a 45-dollar swing in unmet
demand. Two vehicles can deliver at most 60 kWh per period, so cheap pricing
attracts demand the fleet cannot serve and pays 0.30 per unserved kWh. The high
premium rations mobile demand and steers it to the operator's own stations,
which carry better margins and their own 40 kWh capacity. Pricing here is a
**demand-allocation lever**, not a revenue lever.

## Three-level menu

| Method | base | high demand | low inventory | seeds |
|---|---|---|---|---|
| service_first | **+59.22** +-0.67 | +25.51 +-1.28 | +41.98 +-0.94 | 1 |
| constant_inventory | +56.02 +-0.73 | **+27.81** +-1.23 | **+42.10** +-0.96 | 1 |
| joint_ppo | +53.61 +-0.77 | +17.96 +-0.58 | +41.64 +-0.58 | 3 |
| time_inventory | +50.43 +-0.72 | +11.89 +-1.38 | +36.28 +-1.06 | 1 |
| table_adp | +48.78 +-1.49 | +4.35 +-2.79 | +34.05 +-1.06 | 3 |

Every method is significantly below service-first on the base bank. This
inverts the old ranking, where fitted value led on base and low inventory.

**This table is confounded.** Table ADP and time inventory both quote the middle
menu index, which is 1.4x here -- the sharply suboptimal notch. See the
diagnostic below before drawing conclusions about either method.

## Five-level menu

| Method | base | high demand | low inventory | seeds |
|---|---|---|---|---|
| table_adp | **+66.91** +-0.12 | **+38.58** +-0.49 | **+52.35** +-0.29 | 3 |
| time_inventory | +64.98 +-0.69 | +35.02 +-1.29 | +51.52 +-0.95 | 1 |
| service_first | +61.82 +-0.72 | +38.56 +-1.18 | +48.50 +-0.98 | 1 |
| joint_ppo | +59.35 +-1.43 | +25.14 +-2.60 | +45.96 +-1.70 | 3 |
| constant_inventory | +56.35 +-0.74 | +28.25 +-1.23 | +43.77 +-1.05 | 1 |

Table ADP wins all three banks, with the smallest across-seed spread of any
method (+-0.12 on base). Against service-first it is +5.22 [+4.11, +6.35] on
base and +4.18 [+2.50, +5.85] on low inventory; on high demand the two are
statistically indistinguishable, -0.48 [-2.29, +1.32]. Joint PPO is level with
service-first on base, -0.97 [-2.13, +0.14].

## Diagnostic: table ADP's fixed price, not its dispatch

Refitting the three-level value table at each premium and scoring it on the base
test bank, against service-first at the same price:

| ADP fixed at | Table ADP | Service-first | ADP advantage |
|---|---|---|---|
| 1.1x | +30.15 | -13.50 | **+43.65** |
| 1.4x | +47.11 | +37.02 | +10.09 |
| 1.7x | **+67.04** | +58.90 | +8.14 |

(ADP on the 512-episode test bank, service-first on the 128-episode validation
bank; the gaps are far larger than the difference between banks.)

Table ADP beats service-first at **every** price, and its 1.4x and 1.7x scores
reproduce the two tables above (+48.78 across three seeds, and +66.91). So the
entire 18-dollar swing between the two scenarios came from relocating its fixed
price, not from the extra options. Its dispatch was never the weakness.

The same applies to time inventory, which `compact.candidates` also pins to the
middle index: +50.43 to +64.98 across the two scenarios. The methods that choose
their own price moved far less -- service-first +2.60, constant inventory +0.33.

## What this means

1. **Pricing dominates dispatch under the new model.** One notch of premium is
   worth more than the entire spread between dispatch rules. Any comparison that
   fixes price across methods is mostly measuring the fixed price.
2. **Fixing the mobile price is now a real handicap**, where the old absolute
   menu's $0.60 happened to sit near the optimum. Table ADP should learn or at
   least search over price; the rollout ADP on `origin/10_26_version`, with its
   `adaptive_pricing.py`, is the natural next step.
3. **Joint PPO's variance grows with the pricing action space**: +-0.58 on the
   stress banks at three levels, +-2.60 at five. It is the only method that
   learns its own price, and 500 updates buys less as the menu widens.
4. **Use the five-level scenario for future comparisons.** The three-level menu
   caps every method at a corner and compresses the differences between them.
5. Reported profits are now positive throughout, where the old walkthrough lost
   money, because the premium menu prices well above the former fixed $0.60.

## Reproducing

```bash
python experiments/run_new_pricing_comparison.py --phase select  --scenario configs/utility_b3_wide.json --study-dir outputs/wide_pricing_comparison
python experiments/run_new_pricing_comparison.py --phase train-adp --scenario configs/utility_b3_wide.json --study-dir outputs/wide_pricing_comparison
python experiments/run_new_pricing_comparison.py --phase train-ppo --seeds 7 --scenario configs/utility_b3_wide.json --study-dir outputs/wide_pricing_comparison
python experiments/run_new_pricing_comparison.py --phase evaluate --scenario configs/utility_b3_wide.json --study-dir outputs/wide_pricing_comparison
```

`--seeds` splits training so the three seeds can run in parallel. Dropping
`--scenario`/`--study-dir` runs the three-level scenario. Study artifacts land in
`outputs/`, which is gitignored, so they must be regenerated locally.
