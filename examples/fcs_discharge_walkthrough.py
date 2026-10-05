"""A controlled one-period trip to FCSs and grid export; no learning algorithm."""
from pathlib import Path

from charging_env import ChargingSimulator, DispatchAction, Mode, Scenario


def main():
    root = Path(__file__).resolve().parents[1]
    scenario = Scenario.from_json(root / "configs" / "toy_city.json")
    sim = ChargingSimulator(scenario)
    # Suppress EV requests in this period to isolate travel and discharge accounting.
    obs, _ = sim.reset(seed=7, options={"requests": [0]*scenario.n_zones})
    starts = obs["locations"].copy()
    print("Initial MCV zones:", starts.tolist())
    print("FCS zones:", list(scenario.fcs_zones))
    print("Initial battery (kWh):", obs["energy_kwh"].tolist())
    obs, _, _, _, _ = sim.quote_prices([0.50]*scenario.n_mcv)
    actions = [DispatchAction(Mode.DISCHARGE, scenario.fcs_zones[m % scenario.n_fcs], 10.)
               for m in range(scenario.n_mcv)]
    for m, action in enumerate(actions):
        distance = scenario.distances[starts[m], action.target_zone]
        print(f"MCV {m}: zone {starts[m]} -> FCS in zone {action.target_zone}; "
              f"{distance:.2f} km, {distance/scenario.speed_kmph:.2f} h travel; "
              f"request {action.energy_kwh:.2f} kWh grid export")
    obs, reward, _, _, info = sim.dispatch(actions)
    ledger = info["energy_ledger"]
    print("Final MCV zones:", obs["locations"].tolist())
    print("Travel energy (kWh):", ledger["travel_kwh"].tolist())
    print("Actual export per MCV (kWh):", ledger["export_grid_kwh"].tolist())
    print("Export through each FCS (kWh):", ledger["fcs_export_grid_kwh"].tolist())
    print("Final battery (kWh):", obs["energy_kwh"].round(6).tolist())
    for name, value in info["reward_terms"].items():
        print(f"  {name}: ${value:.2f}")
    print(f"Period profit: ${reward:.2f}")
    if info["invalid_actions"]:
        print("Converted to WAIT:", info["invalid_actions"])


if __name__ == "__main__":
    main()
