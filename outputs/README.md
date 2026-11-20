# Curated experiment outputs

This directory contains the compact, auditable outputs behind the current 2,048-update fleet-size sensitivity and Warm-started PPO ablation studies.

| Bundle | Contents |
|---|---|
| `fleet_size_ppo_2048/` | Fleet-size 5, 10, and 15 evaluation results; fleet-5 and fleet-15 training metrics, configurations, protocols, scenarios, and run metadata |
| `ppo_ablation_2048/` | Warm-start-only, PPO-only, and Warm-started PPO evaluation results; PPO-only training metrics, configurations, protocols, scenarios, and run metadata |
| `warm_started_ppo_2048/` | Training evidence for the fleet-10 Warm-started PPO checkpoints reused by both studies |

Each evaluation JSON contains the frozen 128-day test results for one policy seed. The `training_metrics.csv.gz` files retain the update-level learning metrics in gzip-compressed CSV format. Every trained policy directory also contains its final-update `checkpoint.pt` and best-validation `best_checkpoint.pt`. Run metadata record the training budget and SHA-256 hashes of the final checkpoints and protocols; [`checkpoint_manifest.csv`](checkpoint_manifest.csv) records the role, size, and SHA-256 digest of all checkpoint files.

To keep the repository reviewable, this curated bundle excludes offline `.pkl` datasets, Python caches, duplicate console logs, and the very large `evaluations.json` and `validation_history.json` traces. Those large traces repeat information needed for checkpoint selection but are not required to reproduce the published tables and confidence intervals from the retained daily results.

The publication-ready summaries and confidence intervals remain under [`docs/results/`](../docs/results/).
