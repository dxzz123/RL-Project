"""Deterministic checks of the research model, with no training or policy code."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from charging_env.config import Scenario
from charging_env.simulator import ChargingSimulator, DispatchAction, Mode


def small_scenario(**overrides):
    """A two-period, two-MCV, two-zone scenario with easy-to-check units."""
    values = dict(
        horizon=2,
        zone_xy_km=((0., 0.), (4., 0.)),
        fcs_zones=(0,),
        initial_locations=(0, 0),
        initial_energy_kwh=(40., 40.),
        mean_requests_by_zone=(0., 0.),
        arrival_profile=(1., 1.),
        grid_buy_per_kwh=(0.2, 0.3),
        grid_sell_per_kwh=(0.1, 0.2),
        fcs_retail_per_kwh=(0.5,),
        fcs_ev_capacity_kwh=(40.,),
        fcs_fleet_exchange_capacity_kwh=(20.,),
        charge_efficiency=0.8,
        mobile_service_efficiency=0.8,
        discharge_efficiency=0.8,
        fixed_service_efficiency=0.8,
        discount_per_period=0.9,
    )
    values.update(overrides)
    return replace(Scenario(), **values).validate()


def wait_actions(sim):
    return [DispatchAction(Mode.WAIT, int(z)) for z in sim.locations]


def set_post_choice_demand(sim, mobile, fixed=(0,), outside=0, prices=(0.6, 0.8)):
    """Test-only injection after real pricing; isolates physical/accounting checks.

    The public model always samples choices through quote_prices(). These tests
    inject a known choice realization into its fields so that rewards can be
    verified exactly rather than conditioned on a random draw. Fixed/outside
    requests are assigned to zone 0 solely to keep total request conservation.
    """
    sim.reset(seed=7)
    sim.quote_prices(prices)
    sim.mobile_demand = np.asarray(mobile, dtype=np.int64).copy()
    sim.fixed_demand = np.asarray(fixed, dtype=np.int64).copy()
    sim.outside = int(outside)
    sim.requests = sim.mobile_demand.sum(axis=0)
    sim.requests[0] += int(sim.fixed_demand.sum()) + sim.outside


class ModelTests(unittest.TestCase):
    def assert_observations_equal(self, left, right):
        self.assertEqual(set(left), set(right))
        for key in left:
            with self.subTest(field=key):
                np.testing.assert_array_equal(left[key], right[key])

    def assert_unchanged(self, sim, observation, rng_state, profit):
        self.assert_observations_equal(observation, sim.observe())
        self.assertEqual(rng_state, sim.rng.bit_generator.state)
        self.assertEqual(profit, (sim.cumulative_profit, sim.cumulative_discounted_profit))

    def test_reset_and_two_stage_order(self):
        sim = ChargingSimulator(small_scenario())
        with self.assertRaises(RuntimeError):
            sim.observe()
        with self.assertRaises(RuntimeError):
            sim.quote_prices([0.6, 0.8])
        obs, info = sim.reset(seed=1)
        self.assertEqual(obs["stage"], 0)
        self.assertEqual(info["stage"], "pricing")
        self.assertFalse(obs["action_mask"].any())
        with self.assertRaises(RuntimeError):
            sim.dispatch(wait_actions(sim))
        obs, reward, terminated, truncated, info = sim.quote_prices([0.6, 0.8])
        self.assertEqual(obs["stage"], 1)
        self.assertEqual(obs["period"][0], 0)
        self.assertEqual(reward, 0.)
        self.assertFalse(terminated or truncated)
        self.assertEqual(info["discount"], 1.)
        with self.assertRaises(RuntimeError):
            sim.quote_prices([0.6, 0.8])
        obs, _, terminated, truncated, _ = sim.dispatch(wait_actions(sim))
        self.assertEqual(obs["stage"], 0)
        self.assertEqual(obs["period"][0], 1)
        self.assertFalse(terminated or truncated)

    def test_choice_probabilities_normalize_and_respond_to_own_price(self):
        sim = ChargingSimulator(small_scenario())
        sim.reset(seed=1)
        low = sim.choice_probabilities([0.2, 0.6])
        high = sim.choice_probabilities([1.0, 0.6])
        self.assertEqual(low.shape, (2, 4))
        np.testing.assert_allclose(low.sum(axis=1), 1.)
        np.testing.assert_allclose(high.sum(axis=1), 1.)
        self.assertTrue(np.isfinite(low).all())
        self.assertTrue((high[:, 0] < low[:, 0]).all())
        self.assertTrue((high[:, 1:] > low[:, 1:]).all())

    def test_unavailable_choices_have_zero_probability(self):
        sim = ChargingSimulator(small_scenario(
            mobile_service_kw=5., max_fcs_access_km=1.))
        sim.reset(seed=1)
        probabilities = sim.choice_probabilities([0.6, 0.8])
        np.testing.assert_array_equal(probabilities[:, :2], 0.)
        self.assertEqual(probabilities[1, 2], 0.)
        self.assertEqual(probabilities[1, 3], 1.)

    def test_choices_conserve_requests_and_are_frozen_until_dispatch(self):
        sim = ChargingSimulator(small_scenario())
        original_requests = [12, 11]
        sim.reset(seed=5, options={"requests": original_requests})
        original_requests[0] = 0
        obs, _, _, _, info = sim.quote_prices([0.6, 0.8])
        np.testing.assert_array_equal(info["choice_counts_by_zone"].sum(axis=1), [12, 11])
        self.assertEqual(int(obs["mobile_demand"].sum() + obs["fixed_demand"].sum()
                             + obs["outside_requests"][0]), 23)
        frozen = sim.observe()
        sim.feasible_actions()
        sim.local_observation(0)
        self.assert_observations_equal(frozen, sim.observe())
        _, _, _, _, completed = sim.dispatch(wait_actions(sim))
        self.assertEqual(completed["total_requests"], 23)
        self.assertEqual(completed["unmet_mobile_requests"], int(obs["mobile_demand"].sum()))
        self.assertEqual(completed["outside_requests"], int(obs["outside_requests"][0]))

    def test_seed_reproduces_arrivals_and_choices_across_periods(self):
        scenario = small_scenario(mean_requests_by_zone=(4., 3.))
        left, right = ChargingSimulator(scenario), ChargingSimulator(scenario)
        self.assert_observations_equal(left.reset(seed=314)[0], right.reset(seed=314)[0])
        for _ in range(scenario.horizon):
            a = left.quote_prices([0.6, 0.8])
            b = right.quote_prices([0.6, 0.8])
            self.assert_observations_equal(a[0], b[0])
            np.testing.assert_array_equal(a[4]["choice_counts_by_zone"], b[4]["choice_counts_by_zone"])
            a = left.dispatch(wait_actions(left))
            b = right.dispatch(wait_actions(right))
            self.assert_observations_equal(a[0], b[0])
            self.assertEqual(a[1:4], b[1:4])
            self.assertEqual(a[4]["reward_terms"], b[4]["reward_terms"])

    def test_price_array_is_copied(self):
        sim = ChargingSimulator(small_scenario())
        sim.reset(seed=1)
        prices = np.array([0.6, 0.8])
        sim.quote_prices(prices)
        prices[:] = 1.
        np.testing.assert_array_equal(sim.mobile_prices, [0.6, 0.8])

    def test_outside_choices_do_not_incur_unmet_penalty(self):
        sim = ChargingSimulator(small_scenario())
        set_post_choice_demand(sim, [[0, 0], [0, 0]], outside=9)
        _, reward, _, _, info = sim.dispatch(wait_actions(sim))
        self.assertEqual(info["outside_requests"], 9)
        self.assertEqual(info["unmet_mobile_requests"], 0)
        self.assertEqual(info["unmet_fixed_requests"], 0)
        self.assertEqual(info["reward_terms"]["unmet_demand_cost"], 0.)
        self.assertAlmostEqual(reward, -1.)

    def test_service_travel_efficiency_and_all_profit_terms(self):
        scenario = small_scenario(speed_kmph=8., fcs_ev_capacity_kwh=(15.,))
        sim = ChargingSimulator(scenario)
        set_post_choice_demand(sim, [[0, 3], [0, 0]], fixed=(3,), outside=4)
        self.assertEqual(sim.service_limit(0, 1), 1)  # Half an hour remains after travel.
        actions = [DispatchAction(Mode.SERVE, 1), DispatchAction(Mode.WAIT, 0)]
        _, reward, _, _, info = sim.dispatch(actions)
        np.testing.assert_array_equal(info["served_mobile"], [[0, 1], [0, 0]])
        np.testing.assert_array_equal(info["served_fixed"], [1])
        self.assertEqual(info["unmet_mobile_requests"], 2)
        self.assertEqual(info["unmet_fixed_requests"], 2)
        self.assertEqual(info["outside_requests"], 4)
        ledger = info["energy_ledger"]
        np.testing.assert_allclose(ledger["travel_kwh"], [1.4, 0.])
        np.testing.assert_allclose(ledger["mobile_delivered_kwh"], [10., 0.])
        np.testing.assert_allclose(ledger["fixed_grid_kwh"], [12.5])
        np.testing.assert_allclose(ledger["after_kwh"], [26.1, 40.])
        np.testing.assert_allclose(ledger["balance_residual_kwh"], 0., atol=1e-12)
        expected = dict(mobile_service_revenue=6., fixed_service_revenue=5.,
                        discharge_revenue=0., procurement_cost=2.5, travel_cost=1.,
                        operating_cost=1., unmet_demand_cost=12., terminal_inventory_value=0.)
        for name, value in expected.items():
            with self.subTest(term=name):
                self.assertAlmostEqual(info["reward_terms"][name], value)
        self.assertAlmostEqual(reward, -5.5)
        np.testing.assert_array_equal(sim.locations, [1, 0])

    def test_mcv_serves_only_its_own_users_in_one_zone(self):
        sim = ChargingSimulator(small_scenario())
        set_post_choice_demand(sim, [[1, 2], [3, 0]])
        _, _, _, _, info = sim.dispatch([
            DispatchAction(Mode.SERVE, 1), DispatchAction(Mode.WAIT, 0)])
        np.testing.assert_array_equal(info["served_mobile"], [[0, 2], [0, 0]])
        self.assertEqual(info["unmet_mobile_requests"], 4)

    def test_service_limit_respects_battery_after_travel(self):
        sim = ChargingSimulator(small_scenario(initial_energy_kwh=(15., 40.)))
        set_post_choice_demand(sim, [[1, 1], [0, 0]])
        self.assertEqual(sim.service_limit(0, 0), 0)  # (15-5)*0.8 < one 10-kWh request.
        self.assertEqual(sim.service_limit(0, 1), 0)
        self.assertFalse(sim.feasible_actions()[0, Mode.SERVE].any())

    def test_infeasible_actions_execute_wait_without_travel(self):
        sim = ChargingSimulator(small_scenario(speed_kmph=2.))
        set_post_choice_demand(sim, [[0, 1], [0, 0]])
        _, _, _, _, info = sim.dispatch([
            DispatchAction(Mode.SERVE, 1), DispatchAction(Mode.WAIT, 1)])
        self.assertEqual(len(info["invalid_actions"]), 2)
        np.testing.assert_array_equal(info["executed_modes"], [Mode.WAIT, Mode.WAIT])
        np.testing.assert_array_equal(sim.locations, [0, 0])
        np.testing.assert_allclose(sim.energy, [40., 40.])
        np.testing.assert_allclose(info["energy_ledger"]["travel_kwh"], 0.)
        self.assertEqual(info["reward_terms"]["travel_cost"], 0.)

    def test_reposition_travel_is_allowed_only_above_reserve(self):
        sim = ChargingSimulator(small_scenario(initial_energy_kwh=(5., 40.)))
        set_post_choice_demand(sim, [[0, 0], [0, 0]])
        self.assertFalse(sim.feasible_actions()[0, Mode.REPOSITION, 1])
        self.assertTrue(sim.feasible_actions()[1, Mode.REPOSITION, 1])
        _, _, _, _, info = sim.dispatch([
            DispatchAction(Mode.REPOSITION, 1), DispatchAction(Mode.REPOSITION, 1)])
        self.assertEqual(len(info["invalid_actions"]), 1)
        np.testing.assert_array_equal(sim.locations, [0, 1])
        np.testing.assert_allclose(sim.energy, [5., 38.6])

    def test_charge_and_discharge_share_station_capacity_proportionally(self):
        sim = ChargingSimulator(small_scenario())
        set_post_choice_demand(sim, [[0, 0], [0, 0]])
        _, reward, _, _, info = sim.dispatch([
            DispatchAction(Mode.RECHARGE, 0, 20.), DispatchAction(Mode.DISCHARGE, 0, 20.)])
        ledger = info["energy_ledger"]
        np.testing.assert_allclose(ledger["charge_grid_kwh"], [10., 0.])
        np.testing.assert_allclose(ledger["export_grid_kwh"], [0., 10.])
        np.testing.assert_allclose(sim.energy, [48., 27.5])
        self.assertAlmostEqual(info["reward_terms"]["procurement_cost"], 2.)
        self.assertAlmostEqual(info["reward_terms"]["discharge_revenue"], 1.)
        self.assertEqual(info["reward_terms"]["fixed_service_revenue"], 0.)
        self.assertEqual(info["reward_terms"]["mobile_service_revenue"], 0.)
        self.assertAlmostEqual(reward, -2.)

    def test_exchange_clips_to_battery_capacity_and_reserve(self):
        sim = ChargingSimulator(small_scenario(
            mobile_charge_kw=100., mobile_discharge_kw=100.,
            fcs_fleet_exchange_capacity_kwh=(200.,)))
        set_post_choice_demand(sim, [[0, 0], [0, 0]])
        _, _, _, _, info = sim.dispatch([
            DispatchAction(Mode.RECHARGE, 0, 1_000.),
            DispatchAction(Mode.DISCHARGE, 0, 1_000.)])
        np.testing.assert_allclose(info["energy_ledger"]["charge_grid_kwh"], [50., 0.])
        np.testing.assert_allclose(info["energy_ledger"]["export_grid_kwh"], [0., 28.])
        np.testing.assert_allclose(sim.energy, [80., 5.])

    def test_exchange_clips_to_power_times_available_hours(self):
        sim = ChargingSimulator(small_scenario(
            initial_locations=(1, 1), speed_kmph=8., mobile_charge_kw=10.,
            mobile_discharge_kw=10., fcs_fleet_exchange_capacity_kwh=(100.,)))
        set_post_choice_demand(sim, [[0, 0], [0, 0]])
        _, _, _, _, info = sim.dispatch([
            DispatchAction(Mode.RECHARGE, 0, 100.), DispatchAction(Mode.DISCHARGE, 0, 100.)])
        np.testing.assert_allclose(info["energy_ledger"]["charge_grid_kwh"], [5., 0.])
        np.testing.assert_allclose(info["energy_ledger"]["export_grid_kwh"], [0., 5.])
        np.testing.assert_allclose(sim.energy, [42.6, 32.35])

    def test_zero_station_exchange_capacity_blocks_charge_and_discharge(self):
        sim = ChargingSimulator(small_scenario(fcs_fleet_exchange_capacity_kwh=(0.,)))
        set_post_choice_demand(sim, [[0, 0], [0, 0]])
        self.assertFalse(sim.feasible_actions()[:, Mode.RECHARGE].any())
        self.assertFalse(sim.feasible_actions()[:, Mode.DISCHARGE].any())
        _, _, _, _, info = sim.dispatch([
            DispatchAction(Mode.RECHARGE, 0, 10.), DispatchAction(Mode.DISCHARGE, 0, 10.)])
        self.assertEqual(len(info["invalid_actions"]), 2)
        np.testing.assert_allclose(sim.energy, [40., 40.])

    def test_terminal_value_is_paid_once_and_discounting_counts_periods(self):
        sim = ChargingSimulator(small_scenario(terminal_value_per_stored_kwh=0.1))
        sim.reset(seed=1)
        sim.quote_prices([0.6, 0.8])
        _, first_reward, first_done, _, first_info = sim.dispatch(wait_actions(sim))
        self.assertFalse(first_done)
        self.assertEqual(first_info["reward_terms"]["terminal_inventory_value"], 0.)
        self.assertAlmostEqual(first_reward, -1.)
        self.assertEqual(first_info["discount"], 0.9)
        sim.quote_prices([0.6, 0.8])
        obs, final_reward, final_done, truncated, final_info = sim.dispatch(wait_actions(sim))
        self.assertTrue(final_done)
        self.assertFalse(truncated)
        self.assertEqual(obs["stage"], 2)
        self.assertEqual(obs["period"][0], 2)
        self.assertFalse(obs["action_mask"].any())
        self.assertEqual(final_info["reward_terms"]["terminal_inventory_value"], 8.)
        self.assertAlmostEqual(final_reward, 7.)
        self.assertAlmostEqual(final_info["cumulative_profit"], 6.)
        self.assertAlmostEqual(final_info["cumulative_discounted_profit"], 5.3)
        self.assertEqual(final_info["discount"], 0.)
        with self.assertRaises(RuntimeError):
            sim.dispatch(wait_actions(sim))
        with self.assertRaises(RuntimeError):
            sim.quote_prices([0.6, 0.8])
        self.assertEqual(sim.reset(seed=1)[0]["period"][0], 0)
        self.assertEqual(sim.cumulative_profit, 0.)

    def test_malformed_prices_do_not_mutate_state_or_rng(self):
        sim = ChargingSimulator(small_scenario())
        sim.reset(seed=19, options={"requests": [4, 5]})
        malformed = ([0.6], [0.6, np.nan], [0.6, np.inf], [0.1, 0.8], [0.6, 1.1],
                     [[0.6, 0.8]], ["oops", 0.8])
        for prices in malformed:
            with self.subTest(prices=prices):
                obs, rng = sim.observe(), deepcopy(sim.rng.bit_generator.state)
                profit = (sim.cumulative_profit, sim.cumulative_discounted_profit)
                with self.assertRaises(ValueError):
                    sim.quote_prices(prices)
                self.assert_unchanged(sim, obs, rng, profit)

    def test_malformed_joint_actions_fail_before_any_mutation(self):
        sim = ChargingSimulator(small_scenario())
        set_post_choice_demand(sim, [[0, 1], [0, 0]])
        first = DispatchAction(Mode.REPOSITION, 1)
        malformed = (
            [first],
            [first, object()],
            [first, DispatchAction(99, 0)],
            [first, DispatchAction(1.0, 0)],
            [first, DispatchAction(True, 0)],
            [first, DispatchAction(Mode.WAIT, True)],
            [first, DispatchAction(Mode.WAIT, 2)],
            [first, DispatchAction(Mode.RECHARGE, 0, -1.)],
            [first, DispatchAction(Mode.RECHARGE, 0, np.nan)],
            [first, DispatchAction(Mode.RECHARGE, 0, np.inf)],
        )
        for actions in malformed:
            with self.subTest(actions=actions):
                obs, rng = sim.observe(), deepcopy(sim.rng.bit_generator.state)
                profit = (sim.cumulative_profit, sim.cumulative_discounted_profit)
                with self.assertRaises(ValueError):
                    sim.dispatch(actions)
                self.assert_unchanged(sim, obs, rng, profit)

    def test_observations_and_info_are_independent_copies(self):
        sim = ChargingSimulator(small_scenario())
        set_post_choice_demand(sim, [[1, 0], [0, 0]])
        before = sim.observe()
        for observation in (sim.observe(), sim.local_observation(0)):
            for value in observation.values():
                if isinstance(value, np.ndarray):
                    value.fill(0)
        self.assert_observations_equal(before, sim.observe())
        _, _, _, _, info = sim.dispatch([
            DispatchAction(Mode.SERVE, 0), DispatchAction(Mode.WAIT, 0)])
        info["energy_ledger"]["after_kwh"].fill(0)
        info["served_mobile"].fill(0)
        np.testing.assert_allclose(sim.energy, [27.5, 40.])

    def test_request_override_rejects_bad_shape_type_and_range(self):
        sim = ChargingSimulator(small_scenario())
        for requests in ([1], [1., 2.], [-1, 0], [13, 0], [[1, 2]]):
            with self.subTest(requests=requests), self.assertRaises(ValueError):
                sim.reset(seed=1, options={"requests": requests})
        with self.assertRaises(ValueError):
            sim.reset(options={"unknown": 0})


class ScenarioTests(unittest.TestCase):
    def test_default_scenario_is_valid_and_distances_are_in_km(self):
        scenario = Scenario().validate()
        np.testing.assert_allclose(scenario.distances[0], [0., 4., 4., np.sqrt(32.)])
        np.testing.assert_allclose(scenario.distances, scenario.distances.T)

    def test_invalid_physical_behavioral_and_dimension_settings_are_rejected(self):
        malformed = (
            {"horizon": 0}, {"horizon": True}, {"period_hours": 0.},
            {"zone_xy_km": ((0.,), (1.,))}, {"zone_xy_km": ((np.nan, 0.), (4., 0.))},
            {"initial_locations": (0, 2)}, {"initial_locations": (False, 0)},
            {"initial_energy_kwh": (4., 40.)}, {"initial_energy_kwh": (81., 40.)},
            {"arrival_profile": (1.,)}, {"mean_requests_by_zone": (13., 0.)},
            {"mean_requests_by_zone": (-1., 0.)}, {"grid_sell_per_kwh": (0.3, 0.4)},
            {"charge_efficiency": 0.}, {"discharge_efficiency": 1.1},
            {"fixed_service_efficiency": np.nan}, {"mobile_service_kw": 0.},
            {"reserve_kwh": 81.}, {"price_min": 1.1}, {"price_max": np.inf},
            {"mobile_utility_intercept": np.nan}, {"unmet_penalty_per_kwh": -1.},
            {"fcs_ev_capacity_kwh": (40., 40.)}, {"fcs_zones": (2,)},
        )
        for settings in malformed:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                small_scenario(**settings)

    def test_json_round_trip_and_unknown_keys(self):
        scenario = small_scenario()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scenario.json"
            scenario.to_json(path)
            self.assertEqual(scenario, Scenario.from_json(path))
            raw = json.loads(path.read_text())
            raw["invented_setting"] = 1
            path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, "Unknown scenario"):
                Scenario.from_json(path)


if __name__ == "__main__":
    unittest.main()
