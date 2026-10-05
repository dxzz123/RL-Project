"""Hand-written commands to exercise the simulator. No learned/optimized policy."""
import csv
from pathlib import Path

import numpy as np

from charging_env import ChargingSimulator, DispatchAction, Mode, Scenario


def main():
    root = Path(__file__).resolve().parents[1]
    scenario = Scenario.from_json(root / "configs" / "toy_city.json")
    sim = ChargingSimulator(scenario)
    obs, _ = sim.reset(seed=7)
    print("FCS zones:", list(scenario.fcs_zones))
    print("Start:", obs["locations"], "battery:", obs["energy_kwh"], "kWh")
    print("These are manual simulator commands, not an algorithm or benchmark.\n")
    rows = []
    while sim.stage != 2:
        # Pricing is supplied first. No dispatch action has been selected yet.
        prices = np.array([0.50]*scenario.n_mcv)
        obs, reward, terminated, truncated, choice_info = sim.quote_prices(prices)
        print(f"Period {sim.period}: requests={obs['requests_by_zone'].tolist()}")
        print("  Mobile choices (MCV x zone):", obs["mobile_demand"].tolist())
        print("  Fixed choices:", obs["fixed_demand"].tolist(),
              "outside:", int(obs["outside_requests"][0]))
        # Service locations and fixed stations are separate destinations.
        # Fixed script: service, service, station recharge, export, return, wait.
        # Replace this block with your manually chosen actions to explore the physics.
        phase = sim.period % 6
        actions = []
        for m in range(scenario.n_mcv):
            service_zone = scenario.initial_locations[m]
            station_zone = scenario.fcs_zones[m % scenario.n_fcs]
            if phase in (0, 1):
                action = DispatchAction(Mode.SERVE, service_zone)
            elif phase == 2:
                action = DispatchAction(Mode.RECHARGE, station_zone, 20.)
            elif phase == 3:
                action = DispatchAction(Mode.DISCHARGE, station_zone, 10.)
            elif phase == 4:
                action = DispatchAction(Mode.REPOSITION, service_zone)
            else:
                action = DispatchAction(Mode.WAIT, int(obs["locations"][m]))
            actions.append(action)
        obs, reward, terminated, truncated, info = sim.dispatch(actions)
        terms = info["reward_terms"]
        ledger = info["energy_ledger"]
        print(f"  Profit=${reward:.2f}; battery={obs['energy_kwh'].round(2).tolist()} kWh")
        print("  Locations:", obs["locations"].tolist(),
              "travel kWh:", ledger["travel_kwh"].round(2).tolist())
        print("  Grid export through each FCS (kWh):",
              ledger["fcs_export_grid_kwh"].round(2).tolist())
        if info["invalid_actions"]:
            print("  Converted to WAIT:", info["invalid_actions"])
        rows.append({"period": info["completed_period"], "profit": reward,
                     "outside_requests": info["outside_requests"],
                     "unmet_mobile_requests": info["unmet_mobile_requests"],
                     "unmet_fixed_requests": info["unmet_fixed_requests"],
                     "travel_kwh": float(ledger["travel_kwh"].sum()),
                     "charge_grid_kwh": float(ledger["charge_grid_kwh"].sum()),
                     "export_grid_kwh": float(ledger["export_grid_kwh"].sum()), **terms})
    output = root / "outputs"
    output.mkdir(exist_ok=True)
    path = output / "manual_episode.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nEpisode complete. Total profit: ${sim.cumulative_profit:.2f}")
    print(f"Discounted total: ${sim.cumulative_discounted_profit:.2f}")
    print(f"Accounting log: {path}")


if __name__ == "__main__":
    main()
