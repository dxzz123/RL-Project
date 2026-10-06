# Validation of version 0.1.1

Validated locally on 2026-10-05 with Python 3.12.4, NumPy 2.5.3, and Gymnasium 1.4.0, using the project's `.venv` interpreter.

Commands run from the project folder:

```bash
python -m unittest discover -s tests -v
python examples/manual_walkthrough.py
python examples/fcs_discharge_walkthrough.py
python examples/check_gym.py
```

All 32 unit tests pass. The manual walkthrough completes all 12 periods, and
Gymnasium's official environment checker passes. The checker recommends
normalizing continuous action spaces for future learning; the current interface
uses dollars and kWh so that the physical meaning is visible.

The tests include independent hand-calculated cases for customer service,
energy losses, travel, shared fleet exchange capacity, procurement, and profit.
Diagonal-route regressions verify Manhattan distance, travel energy and cost,
remaining operating time, service limits, and customer-option availability.
The tests also verify choice normalization and price response, reproducible seeds,
frozen post-price choices, invalid-action handling, and episode timing.

Version 0.1.1 separates the MCV initial positions (zones 1 and 2) from FCS
positions (zones 0 and 3). The controlled FCS discharge walkthrough starts
with no EV requests and verifies both vehicles can make axis-aligned 4 km trips to a station,
consume 1.4 kWh for travel, and export 10 kWh through each station. Each final
battery is 43.073684 kWh; total export revenue is $2, travel costs $2, and
operating costs $1, for a period profit of -$1. Regression tests also verify
discharge cannot occur at a non-station zone and that travel reduces the time
available for export. The two per-FCS ledger vectors reconcile with per-MCV
charging and export quantities.

These checks validate the implemented simulator against its stated model.
They do not calibrate the synthetic parameters, demonstrate an algorithm's
performance, or establish physical realism beyond the assumptions in MODEL.md.

The simulator and example checks above were run on this computer.
The model documentation was checked on 2026-10-06 with VS Code's bundled
KaTeX renderer: 18 display equations and 73 inline expressions
passed with no math syntax errors.
Follow README.md to select the project's Python interpreter in VS Code.
