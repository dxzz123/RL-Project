# Scenario files

Each JSON file sets every field of `charging_env.config.Scenario`: the city layout, fleet, demand, prices, costs and customer model. Load one with `Scenario.from_json(path)` or pass it to a command with `--scenario`.

| File | Zones | Vehicles | Stations | Customer benefit `b` | Use |
|---|---:|---:|---:|---:|---|
| `utility_b3.json` | 4 | 2 | 2 | 3 | **Small city.** The main scenario since study 04 |
| `synthetic_25_zones.json` | 25 | 10 | 5 | 3 | **Large city.** A synthetic scaling test with residential and commercial demand peaks (studies 09–11) |
| `toy_city.json` | 4 | 2 | 2 | 0 | The original customer model (studies 01–03). Same as `Scenario()` with no file |
| `utility_b2.json`, `utility_b4.json` | 4 | 2 | 2 | 2, 4 | Sensitivity checks on `b` |

All five use 12 one-hour periods and the price menu $0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00. So price index 4 means $0.60 and index 5 means $0.80.

`b` is the benefit customers get from charging at all, relative to leaving. A higher `b` means fewer customers leave. It changes customer behavior, not the operator's revenue; see [docs/reference/OUTSIDE_OPTION.md](../docs/reference/OUTSIDE_OPTION.md).

Field meanings and units are in [docs/reference/ENVIRONMENT.md](../docs/reference/ENVIRONMENT.md) and [docs/reference/NOTATION.md](../docs/reference/NOTATION.md).

**Don't edit `utility_b3.json` or `synthetic_25_zones.json` in place.** Their hashes are frozen in the protocols of studies 09–11, and a changed file stops those studies from replaying. To try a variation, copy the file under a new name.
