# Results

> [!IMPORTANT]
> **Station pricing changed on 30 October 2026.** Station base prices are now
> drawn per period from LADWP-anchored off-peak/normal/peak bands, EV customers
> pay a 10% service fee over that base, and the mobile menu is a premium over
> the nearest station's EV price. Every profit figure on this page predates that
> change and no longer describes the current scenarios. See
> [MODEL.md](MODEL.md#4-station-base-prices-and-the-mobile-price-menu) for the
> mechanism and [RESULTS_NEW_PRICING.md](RESULTS_NEW_PRICING.md) for a completed
> re-run of the comparison under it.

Start with the current comparison in section 1: controllers that estimate demand and customer choice from historical observations. Section 2 preserves the earlier comparison in which those market parameters were known. Section 3 explains the main findings that led to these experiments. Each study has a full report in [studies/](studies/README.md) with settings and reproduction commands.

> **How to read the numbers**
>
> - **Score.** Mean discounted profit per simulated 12-hour day, in dollars, on **test days**: random seeds never used for training or for choosing settings. Higher is better.
> - **Compare within a table only.** Each study draws its own test days, so the same controller scores slightly differently across studies.
> - **Seeds.** Learned controllers are optimized with seeds 7, 19 and 37, then evaluated separately. Their row averages those three policies. In Study 12 they share one historical market-model fit; these are not three independently collected histories.
> - **Intervals.** "+4.12 [2.63, 5.58]" means a mean difference of 4.12 per day, with a paired 95% bootstrap interval over the shared test days. It covers day-to-day randomness for the fixed policies, not uncertainty from retraining or collecting a different history.
> - **Scope.** All observations and test days are synthetic. Study 12 separates the fitted model used by controllers from the private simulator used for testing, but still assumes particular demand and choice families. These results are not evidence about a real city.

Earlier studies use two scenarios; the current historical-model study uses the large city only:

| Name | Config | Zones | Vehicles | Stations | Expected requests per day |
|---|---|---:|---:|---:|---:|
| Small city | `configs/utility_b3.json` | 4 | 2 | 2 | 165 |
| Large city | `configs/synthetic_25_zones.json` | 25 | 10 | 5 | 1,032.5 |

Both generating scenarios use 12 one-hour periods and customer charging benefit `b = 3`. Study 12 estimates that benefit from records instead of giving it to the controllers. See [MODEL.md](MODEL.md) for the model definitions.

## 1. Current comparison: a market model learned from history

**Study 12, 2026-10-27: rollout ADP, Joint PPO and inventory in the large city.** Every controller uses the same demand and customer-choice model fitted from 120 historical days. Another 32 historical days check its predictions. The records include every quote opportunity, including customers who decline all offers; they are simulated complete quote histories, not real sales data.

ADP plans with the fitted simulator, PPO trains in it, and the inventory rule is tuned in it. Both ADP and inventory can use learned prices. The private test environment supplies actual current observations but does not reveal its true demand forecasts or customer-preference parameters to a controller.

PPO completed **500 updates of 16 episodes for each of seeds 7, 19 and 37**, following an inventory-teacher warm start. All three selected PPO checkpoints are the final, update-500 models. Pricing and checkpoint selection used separate fitted-model validation days. The fitted market model and all selected controllers are then frozen, with no refitting or tuning for a test condition.

Each condition has **128 fresh test days**, shared by the compared controllers:

| Test condition | What changes from the historical generating regime? |
|---|---|
| Original pattern | Only the sampled days; this evaluates fitted-model controllers in the historical generating regime |
| Demand +40% | Every arrival rate is multiplied by 1.4 |
| Shifted places and times | Demand locations rotate 180 degrees and the hourly pattern is delayed by three hours, circularly; total expected daily demand is unchanged |

Customer-choice parameters stay unchanged in all three conditions. The controllers are not told which condition is running.

| Controller | Original pattern | Demand +40% | Shifted places and times |
|---|---:|---:|---:|
| **Rollout ADP, selected pricing** | **940.46** | **1,081.91** | **932.46** |
| Joint PPO, selected checkpoint | 718.00 | 771.83 | 717.58 |
| Inventory rule, selected pricing | 925.53 | 1,038.58 | 917.91 |
| Inventory rule, fixed price | 915.67 | 1,060.72 | 908.82 |

The first three rows average the three selected policies. The fixed-price inventory rule is one controller, tuned to price 0.80 USD/kWh and early/late battery values 0.20/0.10 USD/kWh. The completed test contains 3,840 policy episodes.

**Rollout ADP leads in all three conditions.** Its mean advantage per day, with paired 95% bootstrap intervals, is:

| ADP compared with | Original pattern | Demand +40% | Shifted places and times |
|---|---:|---:|---:|
| Joint PPO | +222.46 [216.16, 228.75] | +310.08 [302.76, 317.59] | +214.88 [209.12, 221.14] |
| Inventory, selected pricing | +14.93 [12.50, 17.38] | +43.33 [39.23, 47.50] | +14.55 [12.30, 16.99] |
| Inventory, fixed price | +24.79 [20.97, 28.61] | +21.18 [16.25, 26.14] | +23.64 [19.76, 27.48] |

The advantage is consistent across seeds 7, 19 and 37: each ADP policy has a positive paired interval against the corresponding PPO and learned-price inventory policy, and against fixed-price inventory, in every condition. This does not mean ADP wins on every day. These intervals describe the shared test-day variation, conditional on one historical fit and these three optimization seeds.

**Learned prices do not always help inventory.** They add 9.85 per day in the original pattern and 9.09 after the location/time shift, but lose **22.14 per day** to fixed prices under 40% higher demand. In that higher-demand condition, both inventory versions deliver similar mobile energy (486.9 versus 483.3 kWh/day). The learned-price version collects less discounted mobile revenue (344.39 versus 369.48) and incurs more total unmet-demand cost (115.63 versus 105.60); its lower electricity and travel costs do not offset those losses. Pricing selected for the historical regime therefore fails to retain its advantage under this volume shift.

**PPO's service gap remains.** Under the original pattern, it delivers 449.5 mobile kWh/day against ADP's 601.9, and leaves 414.2 mobile kWh/day unmet against 37.1. These observations locate the shortfall in the complete pricing-and-dispatch policy; they do not isolate a single training defect.

**Cost and scope.** On the original-pattern tests, dispatch averages about 453 ms for ADP, 29 ms for learned-price inventory and 9 ms for PPO. ADP retains its advantage with an estimated market model and these two specified demand shifts, but uses substantially more online computation. This is not an equal-computation comparison or a test of arbitrary model misspecification.

Full report: [studies/12_historical_model.md](studies/12_historical_model.md).

## 2. Historical reference: the market model was known

**Study 11, 2026-10-25: learned pricing with rollout ADP and the inventory rule.** In this earlier experiment, controllers were given the correct simulator parameters. Each dispatcher received its own learned prices, separating better pricing from better dispatch. These are historical results on 256 test days per city; they cannot be compared numerically with Study 12's different test days.

| Controller | Small city | Large city |
|---|---:|---:|
| Rollout ADP, best fixed price | 130.26 | 911.52 |
| **Rollout ADP, learned pricing** | 130.95 | **941.91** |
| Inventory rule (bounded search), best fixed price | 124.76 | 914.50 |
| Inventory rule (bounded search), learned pricing | 126.84 | 924.80 |
| Inventory rule (exhaustive search), fixed $0.60 | **133.63** | not feasible for 10 vehicles |

"Learned pricing" means each seed chose, on validation days, between the best fixed price, the first fitted price rule and its refinement ([METHODS.md](METHODS.md#learned-pricing)).

**Large city.** Learned pricing is a clear win:

- vs. fixed-price rollout ADP: **+30.39** [27.68, 33.08], or 3.3%;
- vs. the inventory rule with its own learned prices: **+17.11** [15.34, 18.92];
- vs. the fixed-price inventory rule: **+27.41** [24.58, 30.38].

All three seeds have positive intervals in all three comparisons. Learned prices average $0.75, mostly $0.60 or $0.80, while the fixed price chosen on validation is $0.80. The lower prices attract more mobile customers, and ADP serves 605.5 kWh per day instead of 380.4 and recharges more to do it. That adds 100.62 in discounted mobile revenue, of which 30.39 survives as profit after lower station revenue and higher electricity, travel and unmet-demand costs.

**Small city.** Learned pricing barely matters: +0.70 [0.25, 1.13], and two of three seeds kept the fixed price. Rollout ADP beats the bounded inventory rule by 4.12 [2.63, 5.58] but stays 2.67 [1.52, 3.83] below the exhaustive inventory rule. Table ADP was not run in this study; on the previous study's test days it led the small city (Finding E).

**Cost.** Rollout ADP spends about 0.5 seconds per dispatch decision in the large city, against about 28 ms for the inventory rule. This is not an equal-computation comparison.

Full report: [studies/11_learned_pricing.md](studies/11_learned_pricing.md).

## 3. How we got here

### Finding A: PPO learned, but lost to a simple inventory rule (studies 01–05, Oct 17–18)

Early PPO (the [earlier PPO](METHODS.md#earlier-ppo) implementation) became profitable with enough training but kept losing to a rule that values stored battery energy:

| Study | Customer model | PPO | Inventory rule |
|---|---|---:|---:|
| [02](studies/02_first_benchmarks.md) First benchmarks | `b = 0` | 16.49 | 22.75 |
| [04](studies/04_silu_b3.md) SiLU activation, `b = 3` | `b = 3` | 116.42 | 125.26 |
| [05](studies/05_ppo_improvements.md) Targeted fixes | `b = 3` | 116.72 | 125.51 |

Diagnostics in study 05 showed what the rule did that PPO didn't. Over 64 days, the rule bought 170 kWh of recharge energy per day and PPO bought 0.24 kWh. PPO also priced higher ($0.83 against $0.60) and often sent both vehicles to the same zone. Fixes to the network inputs, optimization and exploration moved PPO by less than one dollar. Studies [01](studies/01_ppo_training_curve.md) and [03](studies/03_silu_activation.md) cover the training curve and the activation change.

### Finding B: Table ADP is the strongest small-city controller (studies 06–07, Oct 20–21)

[Table ADP](METHODS.md#table-adp) exploits the fact that two vehicles have a small state space. Study 07 also introduced Joint PPO. On 512 base days, plus two stress tests with frozen settings:

| Controller | Base | Demand +50% | Start at 35 kWh |
|---|---:|---:|---:|
| **Table ADP** | **137.76** | 140.56 | **126.67** |
| Time-dependent inventory rule | 133.41 | 133.63 | 121.75 |
| Constant inventory rule | 129.02 | 129.77 | 115.90 |
| Service-first | 123.25 | **144.48** | 113.56 |
| Joint PPO | 123.19 | 131.93 | 106.90 |
| Earlier PPO | 117.42 | 127.09 | 105.70 |

Joint PPO beat earlier PPO by 5.77 [4.95, 6.55] but trailed the inventory rule by 10.22. Service-first wins when demand rises, so no single controller wins everywhere. Reports: [06](studies/06_table_adp.md), [07](studies/07_joint_ppo.md).

### Finding C: PPO stops recharging early, and a warm start fixes that (study 08, Oct 24)

Recharging costs money now and pays off only if later prices attract customers and later dispatch serves them. A random early policy usually pays the cost and wastes the energy, so PPO learns to avoid recharging. Recharge probability fell from 23.5% at initialization to 0.03% after training, although recharging was feasible in 98.6% of decisions. Once recharging is that rare, PPO almost never sees the sequences that would teach it otherwise.

The repair: imitate the inventory rule for 512 days, then train with ordinary PPO. Test set: 512 small-city days.

| Controller | Profit | Recharge (kWh/day) | Mobile delivered (kWh/day) |
|---|---:|---:|---:|
| Original Joint PPO | 123.86 | 0.21 | 69.25 |
| Imitation only | 127.17 | 78.44 | 132.43 |
| **Imitation + 500 PPO updates** | **129.20** | 90.27 | 141.72 |
| Time-dependent inventory rule | 134.70 | 80.79 | 140.26 |
| Table ADP | 138.58 | 109.86 | 159.77 |

The repair gains 5.33 [4.22, 6.42] over original Joint PPO, and every seed improves. PPO training adds 2.03 on top of imitation alone. The rule still wins by 5.50. Report: [08](studies/08_recharge_warm_start.md).

### Finding D: In the large city, PPO falls far behind (study 09, Oct 24)

The 25-zone city was built to test scaling. Table ADP cannot run there because it only supports two vehicles, and the exhaustive inventory search becomes too large, so the inventory rule switches to bounded search. Test set: 256 days.

| Controller | Profit |
|---|---:|
| **Service-first** | **893.04** |
| Inventory rule (bounded) | 863.49 |
| Joint PPO, warm start + 500 updates | 728.36 |
| Imitation only | 625.56 |

PPO improved 102.80 per day over its imitation start but lost to the inventory rule by 135.12, and all three seeds lost. Its vehicles issued about as many service commands as the rule's but delivered less energy per command (7.5 against 11.5 kWh). Report: [09](studies/09_25_zone_scale.md).

### Finding E: Rollout ADP works at any fleet size (study 10, Oct 25)

[Rollout ADP](METHODS.md#rollout-adp) needs no training and handles the 10-vehicle fleet. Fixed price chosen on validation: $0.60 small, $0.80 large. Test set: 256 days per city.

| Controller | Small city | Large city |
|---|---:|---:|
| Table ADP | **136.10** | not applicable |
| Exhaustive inventory rule (reference) | 131.41 | not feasible |
| Rollout ADP | 127.78 | 910.21 |
| Joint PPO (saved, warm-started) | 126.53 | 729.89 |
| Inventory rule (bounded, retuned) | 126.35 | **913.65** |
| Service-first (retuned) | 123.20 | 896.22 |
| Current-profit only (no future value) | 105.64 | 816.39 |

Rollout ADP beat PPO in both cities, by 180.32 per day in the large one. At a fixed price it was roughly tied with the bounded inventory rule: +1.43 small, −3.45 large. It stayed below both small-city specialists. Two extra configurations with no retuning (9 zones and 5 vehicles; 36 zones and 12 vehicles) showed the same pattern: rollout ADP beat service-first and was at or slightly below the inventory rule. Report: [10](studies/10_rollout_adp.md).

Adding learned prices to this controller produced Study 11's known-model scorecard in section 2. Study 12 takes the next step: estimating the market model from historical quotes before training or planning.

## 4. What remains open

- **Small city.** In the known-model studies, rollout ADP with learned prices (130.95) remains below the exhaustive inventory rule (133.63) and Table ADP. Candidate explanations are its approximations: imagined futures follow service-first instead of a stronger policy, only a handful of plans are compared, and its search proposals always recharge the full feasible amount. None has been tested in isolation. Study 12 does not repeat the small-city comparison.
- **PPO in the large city.** Earlier known-model comparisons put warm-started PPO roughly 135–185 per day behind the rules. After three fresh 500-update runs in Study 12, PPO still trails both inventory versions and ADP. Vehicles competing for requests and imitation errors that lead to unfamiliar states remain possible causes; the service and unmet-demand measurements do not isolate them.
- **Equal compute.** No experiment compares methods at equal decision time or equal simulator access.
- **Transfer.** Study 12 tests frozen controllers under two demand shifts in the same city. It does not test learned prices on a new network or fleet, or online adaptation after conditions change.
- **Model error.** Study 12 introduces estimation error in demand and customer choice, then tests specified demand shifts. The assumed distribution families still match the generator. Wrong choice families, changing customer preferences, missing quote records, and errors in travel or operating costs remain untested. One fitted history also leaves uncertainty across different historical datasets unresolved.

## 5. Saved evidence

Study results live under `outputs/`, which is not in Git. With a completed local study folder present, each line prints its saved scorecard without rerunning anything:

| Study | Directory | Show the saved scorecard |
|---|---|---|
| 12 Historical market model | `outputs/history_adp_study_20261027/` | `python experiments/run_history_adp_study.py --phase show --study-dir outputs/history_adp_study_20261027` |
| 11 Learned pricing | `outputs/priced_adp_study_20261025/` | `python experiments/run_priced_adp_study.py --phase show --study-dir outputs/priced_adp_study_20261025` |
| 10 Rollout ADP | `outputs/general_adp_study_20261025/` | `python experiments/run_general_adp_study.py --phase show --study-dir outputs/general_adp_study_20261025` |
| 10 Extra configurations | `outputs/adp_transfer_20261025/` | `python experiments/run_adp_transfer.py --show --output-dir outputs/adp_transfer_20261025` |
| 09 Large city PPO | `outputs/scale_25_study_20261024/` | `python experiments/run_scale_study.py --phase show --study-dir outputs/scale_25_study_20261024` |
| 08 Recharge warm start | `outputs/recharge_study_20261024/` | `python experiments/run_recharge_study.py --phase show --study-dir outputs/recharge_study_20261024` |
| 07 Joint PPO, six methods | `outputs/paper_ppo_study/` | `python experiments/summarize_paper_ppo_study.py --show` |

Frozen protocols, checkpoints and every test episode are saved locally. The study reports link the independent audit evidence, and [reference/VALIDATION.md](reference/VALIDATION.md) explains what those checks cover.
