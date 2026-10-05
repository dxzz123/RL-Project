"""Regressions for initially separated vehicles visiting fixed stations."""
import unittest

import numpy as np

from charging_env import ChargingSimulator, DispatchAction, Mode, Scenario


class StationVisitTests(unittest.TestCase):
    def setup_no_requests(self):
        sim = ChargingSimulator(Scenario())
        sim.reset(seed=7, options={"requests": [0, 0, 0, 0]})
        sim.quote_prices([0.50, 0.50])
        return sim

    def test_separated_vehicles_visit_stations_and_export(self):
        sim = self.setup_no_requests()
        self.assertTrue(set(sim.locations).isdisjoint(sim.config.fcs_zones))
        obs, reward, done, truncated, info = sim.dispatch([
            DispatchAction(Mode.DISCHARGE, 0, 10.),
            DispatchAction(Mode.DISCHARGE, 3, 10.),
        ])
        self.assertFalse(done or truncated)
        self.assertEqual(info["invalid_actions"], [])
        np.testing.assert_array_equal(obs["locations"], [0, 3])
        ledger = info["energy_ledger"]
        np.testing.assert_allclose(ledger["travel_kwh"], [1.4, 1.4])
        np.testing.assert_allclose(ledger["export_grid_kwh"], [10., 10.])
        np.testing.assert_allclose(ledger["fcs_export_grid_kwh"], [10., 10.])
        np.testing.assert_allclose(ledger["fcs_charge_grid_kwh"], [0., 0.])
        np.testing.assert_allclose(obs["energy_kwh"], 55.-1.4-10./0.95)
        terms = info["reward_terms"]
        self.assertAlmostEqual(terms["discharge_revenue"], 2.)
        self.assertAlmostEqual(terms["travel_cost"], 2.)
        self.assertAlmostEqual(terms["operating_cost"], 1.)
        self.assertEqual(terms["fixed_service_revenue"], 0.)
        self.assertAlmostEqual(reward, -1.)

    def test_discharge_at_non_station_does_not_move_or_export(self):
        sim = self.setup_no_requests()
        obs, _, _, _, info = sim.dispatch([
            DispatchAction(Mode.DISCHARGE, 1, 10.),
            DispatchAction(Mode.DISCHARGE, 2, 10.),
        ])
        self.assertEqual(len(info["invalid_actions"]), 2)
        np.testing.assert_array_equal(obs["locations"], [1, 2])
        np.testing.assert_allclose(obs["energy_kwh"], [55., 55.])
        np.testing.assert_allclose(info["energy_ledger"]["fcs_export_grid_kwh"], 0.)

    def test_station_trip_leaves_less_time_for_discharge(self):
        sim = self.setup_no_requests()
        obs, _, _, _, info = sim.dispatch([
            DispatchAction(Mode.DISCHARGE, 0, 100.),
            DispatchAction(Mode.DISCHARGE, 3, 100.),
        ])
        # Four km at 20 km/h leaves .8 h; .8 h * 20 kW = 16 grid-side kWh.
        ledger = info["energy_ledger"]
        np.testing.assert_allclose(ledger["fcs_export_grid_kwh"], [16., 16.])
        np.testing.assert_allclose(obs["energy_kwh"], 55.-1.4-16./0.95)
        self.assertAlmostEqual(info["reward_terms"]["discharge_revenue"], 3.2)


if __name__ == "__main__":
    unittest.main()
