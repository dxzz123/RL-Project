# 11. Learned pricing for rollout ADP and the inventory rule

**Date:** 2026-10-25 · **Runner:** `experiments/run_priced_adp_study.py` · **Saved results:** `outputs/priced_adp_study_20261025/`

**Question.** Fixed prices leave a decision unused. If prices can react to time, demand and fleet state, how much profit do they add, and does the gain come from better pricing or from better dispatch?

**Answer.** In the large city, learned pricing makes rollout ADP the best controller tested: 941.91 per day, +30.39 over its fixed-price version and +17.11 over the inventory rule with its own learned prices. All three seeds improve. In the small city the gain is small (+0.70), and the exhaustive inventory rule stays ahead.

## Setup

**Pricing rule.** Twelve coefficients shared by all zones turn twelve bounded features into a price target, which is rounded to the nearest allowed price ([METHODS.md](../METHODS.md#learned-pricing)). Pricing sees time, expected arrival rates, fleet locations and batteries, and current station and grid prices. It does not see current requests. Inside rollout ADP, the same rule also sets prices in every imagined future hour.

**Training (per city and seed: 7, 19, 37).**

1. Initial fit with cheap service-first dispatch: cross-entropy search with 24 candidates per round, 16 shared days per round, 12 rounds (4,608 candidate episodes).
2. Refinement with the intended dispatcher, done separately for rollout ADP and for the inventory rule: 8 candidates, 4 days, 4 rounds (128 candidate episodes each).

Candidates within a round share the same days, so episode counts are not counts of distinct demand days. Across both cities there are 29,184 candidate episodes over 1,344 distinct training days.

**Validation (32 days per city)** chooses, for each dispatcher and seed, one of three options: the best fixed price, the initial learned rule, or the refined rule. The fixed-price baselines were also chosen there:

- Rollout ADP checked all seven prices and picked $0.60 for the small city and $0.80 for the large one.
- The bounded inventory rule checked 49 price and battery-value combinations. It picked $0.80 in both cities, with battery values 0.20/0.00 (small) and 0.20/0.10 (large) dollars per kWh, early/late.

**Test:** 256 new days per city, shared by every controller. 6,912 policy episodes in total.

**Extra reference (small city only).** The exhaustive inventory rule with its historical settings: $0.60, battery values 0.30/0.15. It was fixed before testing and not retuned.

## Results

| Controller | Small city | Large city |
|---|---:|---:|
| Rollout ADP, best fixed price | 130.26 | 911.52 |
| **Rollout ADP, learned pricing** | 130.95 | **941.91** |
| Inventory rule (bounded), best fixed price | 124.76 | 914.50 |
| Inventory rule (bounded), learned pricing | 126.84 | 924.80 |
| Inventory rule (exhaustive), reference | **133.63** | not feasible |

Paired differences with 95% intervals:

| Comparison | Small city | Large city |
|---|---|---|
| Learned-pricing ADP − fixed-price ADP | +0.70 [0.25, 1.13] | **+30.39** [27.68, 33.08] |
| Learned-pricing ADP − learned-pricing inventory | +4.12 [2.63, 5.58] | **+17.11** [15.34, 18.92] |
| Learned-pricing ADP − fixed-price inventory | +6.20 [4.36, 8.02] | +27.41 [24.58, 30.38] |
| Learned-pricing ADP − exhaustive inventory | −2.67 [−3.83, −1.52] | — |
| Inventory: learned − fixed pricing | +2.08 [1.39, 2.78] | +10.30 [8.15, 12.50] |

In the large city every seed has a positive interval for each of the first three comparisons.

### Which option each seed chose on validation

| City and dispatcher | Seed 7 | Seed 19 | Seed 37 |
|---|---|---|---|
| Small, rollout ADP | Fixed | Initial learned | Fixed |
| Small, inventory | Fixed | Refined | Refined |
| Large, rollout ADP | Initial learned | Refined | Initial learned |
| Large, inventory | Initial learned | Refined | Refined |

Refinement does not always help. It searches with noisy returns and a small budget, which is why the initial rule and the fixed price stay on the menu. In the small city the +0.70 comes entirely from seed 19.

### Per seed

| City | Seed | Fixed-price ADP | Learned-pricing ADP | Learned-pricing inventory | ADP − inventory |
|---|---:|---:|---:|---:|---|
| Small | 7 | 130.32 | 130.32 | 124.76 | 5.56 [3.48, 7.60] |
| Small | 19 | 130.17 | 132.27 | 128.69 | 3.58 [1.98, 5.11] |
| Small | 37 | 130.27 | 130.27 | 127.06 | 3.21 [1.56, 4.80] |
| Large | 7 | 911.89 | 939.01 | 930.33 | 8.68 [6.63, 10.68] |
| Large | 19 | 911.02 | 942.75 | 920.93 | 21.81 [18.93, 24.73] |
| Large | 37 | 911.65 | 943.99 | 923.14 | 20.85 [17.74, 24.07] |

### Pricing or dispatch?

To separate the two, the inventory rule was run with **rollout ADP's learned prices** (same price rule, the rule's own battery values). The realized prices can still differ, because different dispatch leads to different fleet states.

- Rollout ADP beats this price-swapped inventory rule by 7.39 [6.15, 8.62] in the small city and 18.44 [16.68, 20.21] in the large one. So dispatch matters even with the same pricing rule.
- The inventory rule's own pricing gain is 2.08 (small) and 10.30 (large). So pricing matters even with the same dispatcher.

### What changed operationally (large city, fleet kWh per day)

| Controller | Mobile delivered | Recharge bought | Unmet mobile | Battery left above reserve |
|---|---:|---:|---:|---:|
| Fixed-price ADP | 380.44 | 182.68 | 11.22 | 33.30 |
| Learned-pricing ADP | 605.53 | 396.60 | 42.49 | 38.31 |
| Fixed-price inventory | 369.28 | 207.85 | 20.21 | 47.70 |
| Learned-pricing inventory | 450.87 | 177.06 | 39.89 | 36.99 |

The learned ADP prices are mostly $0.60 and $0.80 and average $0.7525 across zone-hour offers. Fixed-price ADP charges $0.80. Lower prices pull customers from stations to mobile service. ADP serves 59% more mobile energy and buys more than twice the recharge energy. Discounted mobile revenue rises by 100.62. Most of that is offset by lower station revenue, less export revenue and higher electricity, travel and unmet-demand costs, leaving +30.39. This is an accounting breakdown; it does not identify which feature of the price rule drives each decision.

Against the inventory rule with its own learned prices, ADP earns 79.78 more in mobile revenue, also with more procurement and travel cost and less station revenue. Net: +17.11. Neither minimum recharge nor minimum leftover battery defines the best policy.

## Limits

- **Unequal computation.** Mean large-city dispatch time was 492.60 ms for learned-pricing ADP and 28.08 ms for the learned-pricing inventory rule. A pricing call takes about 0.21 ms. Both dispatchers got the same pricing budget, but rollout ADP also simulates futures.
- **Known model.** Test days come from the same simulator the planners use. Robustness to a wrong demand or choice model is not tested.
- **No transfer.** Each price checkpoint is fitted to its own scenario. Using it on another scenario has not been tested.
- **Not re-run here:** Table ADP and PPO. Their latest comparison with rollout ADP is [study 10](10_rollout_adp.md).

## Correctness checks

309 automated tests pass. Both independent audits pass, and all 6,912 final episodes have zero invalid actions.

- **Search audit.** Reconstructs every proposal and elite update, verifies the checkpoints and the validation choices, reconciles profit ledgers, recomputes the intervals, and replays 36 training episodes.
- **Physics audit.** Re-scores rollout candidates through the real simulator rather than the planner's fast helper, and replays six complete final episodes.

The largest discrepancy found was 2.27e-13. See [reference/VALIDATION.md](../reference/VALIDATION.md#evidence-behind-the-learned-pricing-comparison-study-11).

## Reproduce

Show the saved scorecard:

```bash
python experiments/run_priced_adp_study.py --phase show \
  --study-dir outputs/priced_adp_study_20261025
```

Evaluate one saved price rule (large city, seed 7, which chose the initial rule) with rollout dispatch on new days:

```bash
python -m charging_env.rl.priced_rollout \
  --pricing-checkpoint outputs/priced_adp_study_20261025/training/city/warm/seed_7/checkpoint.npz \
  --seed 7 --rollout-samples 8 --episodes 16 --eval-seed-base 2410000001 \
  --output-dir outputs/priced_rollout
```

Replay the whole study in a new directory. This reuses the original day banks, so it reproduces the result but is not new evidence:

```bash
S=outputs/priced_adp_repeat
python experiments/run_priced_adp_study.py --phase init --replay-of outputs/priced_adp_study_20261025 --study-dir $S
python experiments/run_priced_adp_study.py --phase rules    --workers 3 --study-dir $S
python experiments/run_priced_adp_study.py --phase train    --workers 3 --study-dir $S
python experiments/run_priced_adp_study.py --phase validate --workers 3 --study-dir $S
python experiments/run_priced_adp_study.py --phase select              --study-dir $S
python experiments/run_priced_adp_study.py --phase evaluate --workers 3 --study-dir $S
python experiments/run_priced_adp_study.py --phase show                --study-dir $S
```

Notes on the phases:

- **New study.** Use `--control-root .` instead of `--replay-of`. Initialization then checks that the new day banks don't reuse seeds from earlier studies. It needs the saved `outputs/general_adp_study_20261025/reference_inventory/protocol.json`.
- **Long runs.** Large-city rollout jobs can run quietly for many minutes. Training prints once per search round, and validation and testing print once per completed policy.
- **Re-running a phase.** Completed jobs are reused when a phase is called again. A partly finished training directory is kept and reported, never silently resumed.
- **One job at a time.** `--case city --seed 7 --training-phase rollout` runs a single refinement job.
