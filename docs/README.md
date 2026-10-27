# Documentation

Start with the four main guides, in order. Everything else is for when you need more depth.

| | Guide | Read it to learn |
|---|---|---|
| 1 | [MODEL.md](MODEL.md) | The charging system: customers, prices, vehicles, batteries, profit |
| 2 | [METHODS.md](METHODS.md) | How each controller decides, and how they differ |
| 3 | [RESULTS.md](RESULTS.md) | What the experiments show, with the current scorecard |
| 4 | [RUNNING.md](RUNNING.md) | Commands to run, train, evaluate and reproduce |

## Deeper material

| Folder | Contents |
|---|---|
| [methods/](methods/) | One deep dive per controller: [ROLLOUT_ADP](methods/ROLLOUT_ADP.md) (with learned pricing), [TABLE_ADP](methods/TABLE_ADP.md), [JOINT_PPO](methods/JOINT_PPO.md), [EARLIER_PPO](methods/EARLIER_PPO.md) |
| [reference/](reference/) | Lookups: [ENVIRONMENT](reference/ENVIRONMENT.md) (simulator API), [NOTATION](reference/NOTATION.md) (symbols ↔ Python names), [CODE_MAP](reference/CODE_MAP.md) (which file does what), [VALIDATION](reference/VALIDATION.md) (what the tests and audits check), [OUTSIDE_OPTION](reference/OUTSIDE_OPTION.md) (why customers leave) |
| [studies/](studies/README.md) | One report per experiment, numbered 01–11 in date order |

## Source documents

- [MDP_RL.pdf](MDP_RL.pdf): paper draft with the model and the original SA-PPO–MAPPO design.
- [MDP_RL_2.pdf](MDP_RL_2.pdf): Joint PPO algorithm specification.
- [proposal.pdf](proposal.pdf): the original project proposal.

Keep `MDP_RL_2.pdf` at this path: the study 07 runner hashes `docs/MDP_RL_2.pdf` into its frozen protocol.
