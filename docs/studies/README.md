# Study reports

Each report covers one experiment: the question it asked, its settings, its results and how to reproduce it. [RESULTS.md](../RESULTS.md) summarizes them; read a report when you need the full detail.

Numbers in different reports come from different test days, and sometimes from different customer models or code versions. Compare controllers within a report, not across reports.

| # | Date | Report | Question | Customer model |
|---:|---|---|---|---|
| 12 | Oct 27 | [Historical model](12_historical_model.md) | How do ADP, PPO and inventory perform with estimated demand and choice, including private demand shifts? **Current scorecard.** | Parameters estimated from historical quotes; private generating model has `b = 3` |
| 11 | Oct 25 | [Learned pricing](11_learned_pricing.md) | How much do learned prices add to rollout ADP and the inventory rule with a known simulator? | `b = 3` |
| 10 | Oct 25 | [Rollout ADP](10_rollout_adp.md) | Can a decision-time planner handle any fleet size? | `b = 3` |
| 09 | Oct 24 | [25-zone city](09_25_zone_scale.md) | Does warm-started PPO keep up in a city five times larger? | `b = 3` |
| 08 | Oct 24 | [Recharge warm start](08_recharge_warm_start.md) | Why did PPO stop recharging, and can imitation fix it? | `b = 3` |
| 07 | Oct 21 | [Joint PPO](07_joint_ppo.md) | Does Joint PPO beat earlier PPO, and how far is it from Table ADP? | `b = 3` |
| 06 | Oct 20 | [Table ADP](06_table_adp.md) | How strong is a learned value table for two vehicles? | `b = 3` |
| 05 | Oct 18 | [PPO improvements](05_ppo_improvements.md) | Why did earlier PPO lose to the inventory rule? Do targeted fixes help? | `b = 3` |
| 04 | Oct 17 | [SiLU with b = 3](04_silu_b3.md) | Three 500-update PPO models against retuned rules | `b = 3` |
| 03 | Oct 17 | [SiLU activation](03_silu_activation.md) | Does replacing the last Tanh with SiLU make dispatch more responsive? | `b = 0` |
| 02 | Oct 17 | [First benchmarks](02_first_benchmarks.md) | How does trained PPO compare with simple and coordinated rules? | `b = 0` |
| 01 | Oct 17 | [Training curve](01_ppo_training_curve.md) | Why did early training look worse, and what did longer training change? | `b = 0` |

[validation_history.md](validation_history.md) records which checks and test counts existed at each stage.

## Which saved models each study reuses

Later studies reuse earlier trained models as fixed controls instead of retraining them:

```text
05 PPO improvements ──► trains earlier-PPO models
        │
        ▼
06 Table ADP ──► reuses those earlier-PPO models, trains Table ADP
        │
        ▼
07 Joint PPO ──► reuses both, trains Joint PPO
        │
        ▼
08 Recharge warm start ──► reuses 07's Joint PPO and Table ADP, trains warm-started PPO (small city)

09 25-zone city ──► self-contained: trains warm-started PPO for the large city

10 Rollout ADP ──► reuses PPO from 08 and 09, and Table ADP from 07
        │
        ▼
11 Learned pricing ──► reuses the exhaustive-inventory reference from 10
```

## Scripts and saved results

Runners for studies 06–11 are in `experiments/`; runners for studies 01–05 are in `experiments/archive/`. Each study's results directory under `outputs/` holds a frozen protocol, a snapshot of the exact source code that ran, checkpoints, every test episode and an audit.

The protocol records a hash of every source file. Studies 09, 10 and 11 still match the current code exactly, so they can be replayed as they are. For older studies the code has changed since, so exact reproduction uses the source snapshot stored with the results.

The original project proposal is [../proposal.pdf](../proposal.pdf).
