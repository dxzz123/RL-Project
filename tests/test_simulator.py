"""Analytic regressions for the request-level, two-stage model."""
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
    values = dict(
        horizon=2, zone_xy_km=((0., 0.), (4., 0.)), fcs_zones=(0,),
        initial_locations=(0, 0), initial_energy_kwh=(40., 40.),
        mean_requests_by_zone=(0., 0.), arrival_profile=(1., 1.),
        grid_buy_per_kwh=(0.2, 0.3), grid_sell_per_kwh=(0.1, 0.2),
        fcs_retail_per_kwh=(0.5,), fcs_ev_capacity_kwh=(40.,),
        fcs_fleet_exchange_capacity_kwh=(20.,), price_noise_distribution="none",
        charge_efficiency=0.8, mobile_service_efficiency=0.8,
        discharge_efficiency=0.8, fixed_service_efficiency=0.8,
        terminal_value_per_stored_kwh=0., discount_per_period=0.9,
        # Physics and legacy-pricing regressions: keep the fixed absolute menu
        # and the profile/shock station prices. See tests/test_station_pricing.py
        # for the LADWP base-price and premium-menu behaviour.
        fcs_pricing_mode="legacy", mcv_pricing_mode="absolute",
    )
    values.update(overrides)
    return replace(Scenario(), **values).validate()


def wait_actions(sim):
    return [DispatchAction(Mode.WAIT, int(z)) for z in sim.locations]


def set_choices(sim, counts=(0, 0), energies=None, choices=None, prices=(0.6, 0.8)):
    """Inject a known realization after genuine quoting, only to isolate physics."""
    options = {"requests": list(counts)}
    if energies is not None:
        options["request_energy_kwh"] = list(energies)
    sim.reset(seed=7, options=options)
    sim.quote_prices(prices)
    if choices is not None:
        sim.request_choices = np.asarray(choices, dtype=np.int64).copy()
        sim._refresh_demand()
    return sim


class ModelTests(unittest.TestCase):
    def assert_observations_equal(self, left, right):
        self.assertEqual(set(left), set(right))
        for key in left:
            with self.subTest(field=key):
                np.testing.assert_array_equal(left[key], right[key])

    def test_reset_and_two_stage_bellman_timing(self):
        sim = ChargingSimulator(small_scenario())
        with self.assertRaises(RuntimeError):
            sim.observe()
        with self.assertRaises(RuntimeError):
            sim.quote_prices([0.6, 0.8])
        obs, info = sim.reset(seed=1)
        self.assertEqual(obs["stage"], 0)
        self.assertEqual(info["discount"], 1.)
        self.assertFalse(obs["action_mask"].any())
        with self.assertRaises(RuntimeError):
            sim.dispatch(wait_actions(sim))
        obs, reward, terminated, truncated, info = sim.quote_prices([0.6, 0.8])
        self.assertEqual((obs["stage"], obs["period"][0], reward), (1, 0, 0.))
        self.assertEqual(info["discount"], 1.)
        self.assertFalse(terminated or truncated)
        with self.assertRaises(RuntimeError):
            sim.quote_prices([0.6, 0.8])
        obs, reward, done, _, info = sim.dispatch(wait_actions(sim))
        self.assertEqual((obs["stage"], obs["period"][0]), (0, 1))
        self.assertEqual(info["discount"], 0.9)
        self.assertFalse(done)
        sim.quote_prices([0.6, 0.8])
        obs, _, done, _, info = sim.dispatch(wait_actions(sim))
        self.assertEqual((obs["stage"], obs["period"][0]), (2, 2))
        self.assertTrue(done)
        self.assertEqual(info["discount"], 0.)
        self.assertFalse(obs["action_mask"].any())
        with self.assertRaises(RuntimeError):
            sim.dispatch(wait_actions(sim))

    def test_request_specific_mnl_matches_cost_equation_and_zero_outside_utility(self):
        sim = ChargingSimulator(small_scenario())
        sim.reset(seed=3, options={"requests": [2, 0],
            "request_energy_kwh": [5., 15.], "accessible_fcs": [[True], [False]]})
        actual = sim.choice_probabilities([0.6, 0.8])
        # Both requests originate beside the vehicles and station: no travel/queue cost.
        first = np.exp([-0.7 * 0.6 * 5., -0.7 * 0.5 * 5., 0.])
        second = np.array([np.exp(-0.7 * 0.6 * 15.), 0., 1.])
        expected = np.stack([first / first.sum(), second / second.sum()])
        self.assertEqual(actual.shape, (2, 3))  # M, station 0, outside; no named MCVs.
        np.testing.assert_allclose(actual, expected)
        self.assertGreater(actual[0, 0], actual[1, 0])

    def test_increasing_node_price_affects_only_its_requests(self):
        sim = ChargingSimulator(small_scenario())
        sim.reset(seed=1, options={"requests": [2, 2]})
        low = sim.choice_probabilities([0.2, 0.6])
        high = sim.choice_probabilities([1.0, 0.6])
        self.assertTrue((high[:2, 0] < low[:2, 0]).all())
        self.assertTrue((high[:2, 1:] > low[:2, 1:]).all())
        np.testing.assert_array_equal(high[2:], low[2:])
        np.testing.assert_allclose(low.sum(axis=1), 1.)

    def test_average_wait_uses_all_and_only_eligible_mcvs(self):
        scenario = small_scenario(initial_locations=(0, 1), initial_energy_kwh=(40., 40.))
        sim = ChargingSimulator(scenario)
        sim.reset(seed=1, options={"requests": [1, 0]})
        np.testing.assert_allclose(sim.mobile_wait_hours, [0.1, 0.1])
        # With no energy above reserve, vehicle 1 no longer contributes its delay.
        sim = ChargingSimulator(replace(scenario, initial_energy_kwh=(40., 5.)))
        sim.reset(seed=1, options={"requests": [1, 0]})
        np.testing.assert_allclose(sim.mobile_wait_hours, [0., 0.2])
        p = sim.choice_probabilities([0.6, 0.8])[0]
        weights = np.exp([-4.2, -3.5, 0.])
        np.testing.assert_allclose(p, weights / weights.sum())

    def test_inaccessible_options_are_zero_and_outside_is_always_available(self):
        sim = ChargingSimulator(small_scenario(mobile_service_kw=4., max_fcs_access_km=1.))
        sim.reset(seed=1, options={"requests": [1, 1]})
        p = sim.choice_probabilities([0.6, 0.8])
        np.testing.assert_array_equal(p[:, 0], 0.)
        np.testing.assert_array_equal(p[1], [0., 0., 1.])
        self.assertGreater(p[0, 1], 0.)

    def test_prechoice_fcs_queue_depends_on_potential_customer_load(self):
        scenario = small_scenario(fcs_ev_capacity_kwh=(10.,))
        sim = ChargingSimulator(scenario)
        sim.reset(seed=1, options={"requests": [3, 0]})
        np.testing.assert_allclose(sim.fcs_wait_hours, [1.])  # (30/10 - 1)/2 hours.
        before = sim.fcs_wait_hours.copy()
        sim.quote_prices([0.6, 0.8])
        np.testing.assert_array_equal(sim.fcs_wait_hours, before)
        # Dedicated fleet facilities cannot change the customer queue estimate.
        other = ChargingSimulator(replace(scenario, fcs_fleet_exchange_capacity_kwh=(0.,)))
        other.reset(seed=1, options={"requests": [3, 0]})
        np.testing.assert_array_equal(other.fcs_wait_hours, before)

    def test_prices_are_discrete_and_indexed_by_nodes(self):
        sim = ChargingSimulator(Scenario())
        obs, _ = sim.reset(seed=1, options={"requests": [0, 0, 0, 0]})
        # Under premium pricing the menu is per node and per period, so the
        # admissible set comes from the observation rather than a fixed tuple.
        menu = obs["mobile_price_menu"]
        self.assertEqual(menu.shape, (4, sim.config.n_price_levels))
        with self.assertRaises(ValueError):
            sim.quote_prices(menu[:2, 0])  # Old per-MCV shape.
        with self.assertRaises(ValueError):
            sim.quote_prices([menu[0, 0] + 0.01] * 4)  # Between two menu entries.
        sim.quote_price_levels([0, 1, 2, 0])
        np.testing.assert_allclose(sim.mobile_prices,
                                   [menu[0, 0], menu[1, 1], menu[2, 2], menu[3, 0]])
        sim.reset(seed=1)
        for indices in ([0, 1], [0., 1., 2., 3.], [0, 1, 2, 99], [True] * 4):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                sim.quote_price_levels(indices)

    def test_arrivals_are_unbounded_poisson_and_energy_is_heterogeneous(self):
        sim = ChargingSimulator(small_scenario(mean_requests_by_zone=(1000., 1000.)))
        sim.reset(seed=8)
        self.assertTrue((sim.requests > sim.config.max_requests_per_zone).all())
        q = sim.request_energy_kwh
        self.assertEqual(q.size, int(sim.requests.sum()))
        self.assertTrue((q >= 5.).all() and (q <= 15.).all())
        self.assertGreater(np.unique(q).size, 1)
        # Explicit period/node lambda profiles supersede legacy separable factors.
        sim = ChargingSimulator(small_scenario(arrival_rates=((0., 1000.), (1000., 0.))))
        sim.reset(seed=8)
        self.assertEqual(sim.requests[0], 0)
        sim.quote_prices([0.6, 0.8])
        sim.dispatch(wait_actions(sim))
        self.assertEqual(sim.requests[1], 0)
        self.assertGreater(sim.requests[0], 12)

    def test_choices_preserve_each_request_and_freeze_until_dispatch(self):
        sim = ChargingSimulator(small_scenario())
        sim.reset(seed=5, options={"requests": [12, 11]})
        obs, _, _, _, info = sim.quote_prices([0.6, 0.8])
        np.testing.assert_array_equal(info["choice_counts_by_zone"].sum(axis=1), [12, 11])
        self.assertEqual(int(sim.mobile_demand.sum() + sim.fixed_demand.sum() + sim.outside), 23)
        self.assertEqual(sim.request_choices.shape, (23,))
        frozen = sim.observe()
        sim.feasible_actions()
        sim.local_observation(0)
        self.assert_observations_equal(frozen, sim.observe())
        _, _, _, _, info = sim.dispatch(wait_actions(sim))
        self.assertEqual(info["total_requests"], 23)
        self.assertEqual(info["unmet_mobile_requests"], int(obs["mobile_demand"].sum()))

    def test_node_pool_assigns_after_dispatch_without_duplicates_or_lost_skips(self):
        sim = set_choices(ChargingSimulator(small_scenario(initial_energy_kwh=(15., 40.))),
                          counts=(3, 0), energies=(10., 5., 15.), choices=(0, 0, 0))
        # Vehicle 0 can deliver 8 kWh: skip first request, serve 5; vehicle 1 serves 10+15.
        np.testing.assert_array_equal(sim.mobile_demand, [3, 0])
        _, _, _, _, info = sim.dispatch([DispatchAction(Mode.SERVE, 0)] * 2)
        np.testing.assert_array_equal(info["served_mobile"], [[1, 0], [2, 0]])
        np.testing.assert_allclose(info["delivered_mobile_kwh"], [[5., 0.], [25., 0.]])
        np.testing.assert_array_equal(info["assigned_mcv"], [1, 0, 1])
        np.testing.assert_array_equal(info["served_request_mask"], True)
        np.testing.assert_allclose(sim.energy, [8.75, 8.75])
        self.assertEqual(info["unmet_mobile_requests"], 0)

    def test_serving_one_node_cannot_claim_other_nodes_requests(self):
        sim = set_choices(ChargingSimulator(small_scenario()), counts=(1, 2), choices=(0, 0, 0))
        _, _, _, _, info = sim.dispatch([DispatchAction(Mode.SERVE, 1), DispatchAction(Mode.WAIT, 0)])
        np.testing.assert_array_equal(info["served_mobile"], [[0, 2], [0, 0]])
        self.assertEqual(info["unmet_mobile_requests"], 1)
        self.assertEqual(info["unmet_mobile_kwh"], 10.)

    def test_request_is_not_partially_served(self):
        sim = set_choices(ChargingSimulator(small_scenario(mobile_service_kw=12.)),
                          counts=(1, 0), energies=(15.,), choices=(0,))
        self.assertAlmostEqual(sim.service_capacity_kwh(0, 0), 12.)
        _, _, _, _, info = sim.dispatch([DispatchAction(Mode.SERVE, 0), DispatchAction(Mode.WAIT, 0)])
        self.assertEqual(info["served_mobile"].sum(), 0)
        self.assertEqual(info["unmet_mobile_kwh"], 15.)
        np.testing.assert_allclose(sim.energy, [40., 40.])

    def test_heterogeneous_service_energy_and_all_profit_terms(self):
        sim = set_choices(ChargingSimulator(small_scenario(speed_kmph=8., fcs_ev_capacity_kwh=(15.,))),
            counts=(3, 3), energies=(5., 15., 10., 5., 10., 15.), choices=(1, 1, 2, 0, 0, 0))
        _, reward, _, _, info = sim.dispatch([DispatchAction(Mode.SERVE, 1), DispatchAction(Mode.WAIT, 0)])
        np.testing.assert_array_equal(info["served_mobile"], [[0, 2], [0, 0]])
        np.testing.assert_array_equal(info["served_fixed"], [1])
        self.assertEqual(info["unmet_mobile_kwh"], 15.)
        self.assertEqual(info["unmet_fixed_kwh"], 15.)
        self.assertEqual(info["outside_requests"], 1)
        ledger = info["energy_ledger"]
        np.testing.assert_allclose(ledger["mobile_delivered_kwh"], [15., 0.])
        np.testing.assert_allclose(ledger["fixed_grid_kwh"], [6.25])
        np.testing.assert_allclose(ledger["after_kwh"], [19.85, 40.])
        expected = dict(mobile_service_revenue=12., fixed_service_revenue=2.5,
            discharge_revenue=0., procurement_cost=1.25, travel_cost=1., operating_cost=1.,
            unmet_demand_cost=9., terminal_inventory_value=0.)
        for key, value in expected.items():
            self.assertAlmostEqual(info["reward_terms"][key], value)
        self.assertAlmostEqual(reward, 2.25)
        np.testing.assert_allclose(ledger["balance_residual_kwh"], 0., atol=1e-12)

    def test_outside_only_does_not_incur_unmet_penalty(self):
        sim = set_choices(ChargingSimulator(small_scenario()), counts=(2, 0),
                          energies=(5., 15.), choices=(2, 2))
        _, reward, _, _, info = sim.dispatch(wait_actions(sim))
        self.assertEqual(info["outside_requests"], 2)
        self.assertEqual(info["unmet_mobile_requests"] + info["unmet_fixed_requests"], 0)
        self.assertEqual(info["reward_terms"]["unmet_demand_cost"], 0.)
        self.assertAlmostEqual(reward, -1.)

    def test_manhattan_travel_and_posttravel_power_limit(self):
        sim = set_choices(ChargingSimulator(small_scenario(zone_xy_km=((0., 0.), (3., 4.)),
            initial_locations=(0, 1), speed_kmph=10., mobile_service_kw=40., mobile_charge_kw=10.)),
            counts=(0, 3), choices=(0, 0, 0))
        self.assertAlmostEqual(sim.service_capacity_kwh(0, 1), 12.)
        _, _, _, _, info = sim.dispatch([DispatchAction(Mode.SERVE, 1), DispatchAction(Mode.RECHARGE, 0, 100.)])
        np.testing.assert_array_equal(info["served_mobile"], [[0, 1], [0, 0]])
        ledger = info["energy_ledger"]
        np.testing.assert_allclose(ledger["travel_kwh"], [2.45, 2.45])
        np.testing.assert_allclose(ledger["charge_grid_kwh"], [0., 3.])
        np.testing.assert_allclose(sim.energy, [25.05, 39.95])
        self.assertAlmostEqual(info["reward_terms"]["travel_cost"], 3.5)

    def test_infeasible_actions_execute_wait_without_travel(self):
        sim = set_choices(ChargingSimulator(small_scenario(speed_kmph=2.)),
                          counts=(0, 1), choices=(0,))
        _, _, _, _, info = sim.dispatch([DispatchAction(Mode.SERVE, 1), DispatchAction(Mode.WAIT, 1)])
        self.assertEqual(len(info["invalid_actions"]), 2)
        np.testing.assert_array_equal(info["executed_modes"], [Mode.WAIT, Mode.WAIT])
        np.testing.assert_array_equal(sim.locations, [0, 0])
        np.testing.assert_allclose(sim.energy, [40., 40.])
        self.assertEqual(info["reward_terms"]["travel_cost"], 0.)

    def test_charge_and_discharge_share_only_fleet_budget_proportionally(self):
        sim = set_choices(ChargingSimulator(small_scenario()))
        _, reward, _, _, info = sim.dispatch([
            DispatchAction(Mode.RECHARGE, 0, 20.), DispatchAction(Mode.DISCHARGE, 0, 20.)])
        ledger = info["energy_ledger"]
        np.testing.assert_allclose(ledger["charge_grid_kwh"], [10., 0.])
        np.testing.assert_allclose(ledger["export_grid_kwh"], [0., 10.])
        np.testing.assert_allclose(sim.energy, [48., 27.5])
        self.assertAlmostEqual(info["reward_terms"]["procurement_cost"], 2.)
        self.assertAlmostEqual(info["reward_terms"]["discharge_revenue"], 1.)
        self.assertEqual(info["reward_terms"]["fixed_service_revenue"], 0.)
        self.assertAlmostEqual(reward, -2.)

    def test_station_exchange_battery_reserve_and_remaining_time_limits(self):
        sim = set_choices(ChargingSimulator(small_scenario(mobile_charge_kw=100.,
            mobile_discharge_kw=100., fcs_fleet_exchange_capacity_kwh=(200.,))))
        _, _, _, _, info = sim.dispatch([
            DispatchAction(Mode.RECHARGE, 0, 1000.), DispatchAction(Mode.DISCHARGE, 0, 1000.)])
        np.testing.assert_allclose(info["energy_ledger"]["charge_grid_kwh"], [50., 0.])
        np.testing.assert_allclose(info["energy_ledger"]["export_grid_kwh"], [0., 28.])
        np.testing.assert_allclose(sim.energy, [80., 5.])
        sim = set_choices(ChargingSimulator(small_scenario(initial_locations=(1, 1), speed_kmph=8.,
            mobile_charge_kw=10., mobile_discharge_kw=10., fcs_fleet_exchange_capacity_kwh=(100.,))))
        _, _, _, _, info = sim.dispatch([
            DispatchAction(Mode.RECHARGE, 0, 100.), DispatchAction(Mode.DISCHARGE, 0, 100.)])
        np.testing.assert_allclose(info["energy_ledger"]["charge_grid_kwh"], [5., 0.])
        np.testing.assert_allclose(info["energy_ledger"]["export_grid_kwh"], [0., 5.])

    def test_zero_station_budget_and_low_battery_mask_actions(self):
        sim = set_choices(ChargingSimulator(small_scenario(initial_energy_kwh=(5., 40.),
            fcs_fleet_exchange_capacity_kwh=(0.,))))
        mask = sim.feasible_actions()
        self.assertFalse(mask[:, Mode.RECHARGE].any())
        self.assertFalse(mask[:, Mode.DISCHARGE].any())
        self.assertFalse(mask[0, Mode.REPOSITION, 1])
        self.assertTrue(mask[1, Mode.REPOSITION, 1])

    def test_realized_station_tariffs_determine_retail_and_export_revenue(self):
        scenario = small_scenario(price_noise_distribution="uniform", price_noise_scale=0.2,
            fcs_charge_price_profile=((0.6,), (0.8,)),
            fcs_discharge_price_profile=((0.15,), (0.25,)))
        sim = set_choices(ChargingSimulator(scenario), counts=(1, 0), choices=(1,))
        obs = sim.observe()
        retail = float(obs["fcs_retail_prices"][0])
        discharge = float(obs["fcs_discharge_prices"][0])
        self.assertTrue(0.48 <= retail <= 0.72)
        self.assertAlmostEqual(retail / 0.6, discharge / 0.15)
        _, _, _, _, info = sim.dispatch([DispatchAction(Mode.DISCHARGE, 0, 10.), DispatchAction(Mode.WAIT, 0)])
        self.assertAlmostEqual(info["reward_terms"]["fixed_service_revenue"], 10. * retail)
        self.assertAlmostEqual(info["reward_terms"]["discharge_revenue"], 10. * discharge)
        self.assertAlmostEqual(info["reward_terms"]["procurement_cost"], 2.5)

    def test_terminal_inventory_and_discounted_profit(self):
        sim = ChargingSimulator(small_scenario(terminal_value_per_stored_kwh=0.1))
        sim.reset(seed=1)
        sim.quote_prices([0.6, 0.8])
        _, reward, _, _, info = sim.dispatch(wait_actions(sim))
        self.assertEqual(reward, -1.)
        self.assertEqual(info["reward_terms"]["terminal_inventory_value"], 0.)
        sim.quote_prices([0.6, 0.8])
        _, reward, done, _, info = sim.dispatch(wait_actions(sim))
        self.assertTrue(done)
        self.assertAlmostEqual(reward, 7.)
        self.assertAlmostEqual(info["cumulative_profit"], 6.)
        self.assertAlmostEqual(info["cumulative_discounted_profit"], 5.3)
        with self.assertRaises(RuntimeError):
            sim.quote_prices([0.6, 0.8])

    def test_same_seed_reproduces_requests_prices_choices_and_transitions(self):
        scenario = small_scenario(mean_requests_by_zone=(4., 3.), price_noise_distribution="uniform")
        left, right = ChargingSimulator(scenario), ChargingSimulator(scenario)
        self.assert_observations_equal(left.reset(seed=314)[0], right.reset(seed=314)[0])
        for _ in range(scenario.horizon):
            a, b = left.quote_prices([0.6, 0.8]), right.quote_prices([0.6, 0.8])
            self.assert_observations_equal(a[0], b[0])
            np.testing.assert_array_equal(left.request_choices, right.request_choices)
            a, b = left.dispatch(wait_actions(left)), right.dispatch(wait_actions(right))
            self.assert_observations_equal(a[0], b[0])
            self.assertEqual(a[1:4], b[1:4])
            self.assertEqual(a[4]["reward_terms"], b[4]["reward_terms"])

    def test_bad_prices_and_actions_do_not_mutate_state_or_rng(self):
        sim = ChargingSimulator(small_scenario())
        sim.reset(seed=19, options={"requests": [4, 5]})
        for prices in ([0.6], [0.6, np.nan], [0.6, np.inf], [0.1, 0.8], [0.6, 0.7], [[0.6, 0.8]]):
            obs, rng = sim.observe(), deepcopy(sim.rng.bit_generator.state)
            with self.subTest(prices=prices), self.assertRaises(ValueError):
                sim.quote_prices(prices)
            self.assert_observations_equal(obs, sim.observe())
            self.assertEqual(rng, sim.rng.bit_generator.state)
        sim.quote_prices([0.6, 0.8])
        first = DispatchAction(Mode.REPOSITION, 1)
        bad = ([first], [first, object()], [first, DispatchAction(99, 0)],
            [first, DispatchAction(1.0, 0)], [first, DispatchAction(True, 0)],
            [first, DispatchAction(Mode.WAIT, True)], [first, DispatchAction(Mode.WAIT, 2)],
            [first, DispatchAction(Mode.RECHARGE, 0, -1.)],
            [first, DispatchAction(Mode.RECHARGE, 0, np.nan)])
        for actions in bad:
            obs, rng = sim.observe(), deepcopy(sim.rng.bit_generator.state)
            with self.subTest(actions=actions), self.assertRaises(ValueError):
                sim.dispatch(actions)
            self.assert_observations_equal(obs, sim.observe())
            self.assertEqual(rng, sim.rng.bit_generator.state)

    def test_observations_and_info_are_independent_copies(self):
        sim = set_choices(ChargingSimulator(small_scenario()), counts=(1, 0), choices=(0,))
        before = sim.observe()
        for obs in (sim.observe(), sim.local_observation(0)):
            for value in obs.values():
                if isinstance(value, np.ndarray):
                    value.fill(0)
        self.assert_observations_equal(before, sim.observe())
        _, _, _, _, info = sim.dispatch([DispatchAction(Mode.SERVE, 0), DispatchAction(Mode.WAIT, 0)])
        info["energy_ledger"]["after_kwh"].fill(0)
        info["served_mobile"].fill(0)
        np.testing.assert_allclose(sim.energy, [27.5, 40.])

    def test_local_observation_exposes_peer_fleet_state(self):
        sim = set_choices(ChargingSimulator(small_scenario(
            initial_locations=(0, 1), initial_energy_kwh=(30., 40.))))
        first, second = sim.local_observation(0), sim.local_observation(1)
        np.testing.assert_array_equal(first["locations"], [0])
        np.testing.assert_array_equal(first["energy_kwh"], [30.])
        np.testing.assert_array_equal(first["other_locations"], [1])
        np.testing.assert_array_equal(first["other_energy_kwh"], [40.])
        np.testing.assert_array_equal(second["other_locations"], [0])
        np.testing.assert_array_equal(second["other_energy_kwh"], [30.])
        self.assertEqual(first["action_mask"].shape, (1, len(Mode), 2))
        self.assertEqual(second["action_mask"].shape, first["action_mask"].shape)

    def test_invalid_reset_does_not_replace_an_active_episode(self):
        sim = set_choices(ChargingSimulator(small_scenario()), counts=(1, 0), choices=(0,))
        obs, rng = sim.observe(), deepcopy(sim.rng.bit_generator.state)
        with self.assertRaises(ValueError):
            sim.reset(seed=123, options={"requests": [1, 0], "request_energy_kwh": [100.]})
        self.assert_observations_equal(obs, sim.observe())
        self.assertEqual(rng, sim.rng.bit_generator.state)

    def test_request_override_validation_and_copying(self):
        sim = ChargingSimulator(small_scenario())
        bad = ({"requests": [1]}, {"requests": [1., 2.]}, {"requests": [-1, 0]},
            {"requests": [1, 0], "request_energy_kwh": [4.]},
            {"requests": [1, 0], "request_energy_kwh": [16.]},
            {"requests": [1, 0], "request_energy_kwh": []},
            {"requests": [1, 0], "accessible_fcs": [[True, False]]}, {"unknown": 0})
        for options in bad:
            with self.subTest(options=options), self.assertRaises(ValueError):
                sim.reset(seed=1, options=options)
        counts, energies, accessible = [1, 0], [7.], [[True]]
        sim.reset(seed=1, options={"requests": counts, "request_energy_kwh": energies, "accessible_fcs": accessible})
        counts[0], energies[0], accessible[0][0] = 0, 14., False
        np.testing.assert_array_equal(sim.requests, [1, 0])
        np.testing.assert_array_equal(sim.request_energy_kwh, [7.])
        np.testing.assert_array_equal(sim.accessible_fcs, [[True]])


class ScenarioTests(unittest.TestCase):
    def test_default_manhattan_distances(self):
        c = Scenario().validate()
        np.testing.assert_allclose(c.distances[0], [0., 4., 4., 8.])
        np.testing.assert_allclose(c.distances, c.distances.T)
        np.testing.assert_array_equal(np.diag(c.distances), 0.)

    def test_invalid_scenario_settings_rejected(self):
        malformed = ({"horizon": 0}, {"horizon": True}, {"period_hours": 0.},
            {"zone_xy_km": ((0.,), (1.,))}, {"initial_locations": (0, 2)},
            {"initial_energy_kwh": (4., 40.)}, {"initial_energy_kwh": (81., 40.)},
            {"arrival_profile": (1.,)}, {"mean_requests_by_zone": (-1., 0.)},
            {"charge_efficiency": 0.}, {"discharge_efficiency": 1.1},
            {"mobile_service_kw": 0.}, {"reserve_kwh": 81.},
            {"price_sensitivity_per_dollar": 0.}, {"unmet_penalty_per_kwh": -1.},
            {"fcs_ev_capacity_kwh": (40., 40.)}, {"fcs_zones": (2,)},
            {"price_levels": (0.2, 0.2)}, {"price_levels": (0.1, 0.5)},
            {"request_min_kwh": 15.}, {"request_kwh": 16.},
            {"price_noise_distribution": "bogus"}, {"price_noise_scale": -0.1},
            {"price_noise_distribution": "uniform", "price_noise_scale": 1.}, {"discount_per_period": 0.},
            {"arrival_rates": ((1.,), (1.,))}, {"fcs_charge_price_profile": ((0.5,),)})
        for settings in malformed:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                small_scenario(**settings)

    def test_json_roundtrip_nested_profiles_and_unknown_keys(self):
        scenario = small_scenario(arrival_rates=((0., 2.), (3., 0.)),
                                 fcs_charge_price_profile=((0.5,), (0.7,)))
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
