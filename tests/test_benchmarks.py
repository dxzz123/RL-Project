"""Independent checks of observable-state benchmarks against simulator physics."""
from copy import deepcopy
from dataclasses import replace
from itertools import product
from pathlib import Path
import tempfile
import unittest

import numpy as np

from charging_env.benchmarks.policies import (
    adaptive_prices, coordinated_dispatch, dispatch_outcome, fixed_prices,
    random_dispatch, random_prices, service_first_dispatch,
)
from charging_env.benchmarks.evaluation import evaluate_policy, paired_comparison
from charging_env.config import Scenario
from charging_env.simulator import ChargingSimulator, DispatchAction, Mode

try:
    import torch
except ImportError:
    torch = None


def benchmark_scenario(**overrides):
    values = dict(
        horizon=3, zone_xy_km=((0., 0.), (4., 0.)), fcs_zones=(0,),
        initial_locations=(0, 1), initial_energy_kwh=(40., 45.),
        mean_requests_by_zone=(5., 5.), arrival_profile=(1., 1., 1.),
        grid_buy_per_kwh=(.2, .3, .4), grid_sell_per_kwh=(.1, .2, .3),
        fcs_retail_per_kwh=(.5,), fcs_ev_capacity_kwh=(20.,),
        fcs_fleet_exchange_capacity_kwh=(10.,),
        charge_efficiency=.8, discharge_efficiency=.8,
        mobile_service_efficiency=.8, fixed_service_efficiency=.8,
        price_noise_distribution="uniform", price_noise_scale=.1,
        price_sensitivity_per_dollar=.1,
        # Physics and legacy-pricing regressions: keep the fixed absolute menu
        # and the profile/shock station prices. See tests/test_station_pricing.py
        # for the LADWP base-price and premium-menu behaviour.
        fcs_pricing_mode="legacy", mcv_pricing_mode="absolute",
    )
    values.update(overrides)
    return replace(Scenario(), **values).validate()


def known_choices(scenario, energies, choices):
    sim = ChargingSimulator(scenario)
    sim.reset(seed=21, options={
        "requests": [len(energies), 0], "request_energy_kwh": energies,
    })
    sim.quote_prices([.6, .8])
    # Test fixtures fix realized choices to isolate physical allocation from MNL.
    sim.request_choices = np.asarray(choices, dtype=np.int64)
    sim._refresh_demand()
    return sim


class BenchmarkPolicyTests(unittest.TestCase):
    def assert_observations_equal(self, actual, expected):
        self.assertEqual(set(actual), set(expected))
        for field in actual:
            with self.subTest(field=field):
                np.testing.assert_array_equal(actual[field], expected[field])

    def assert_matches_simulator(self, sim, actions):
        obs = sim.observe()
        previous = deepcopy(obs)
        rng_state = deepcopy(sim.rng.bit_generator.state)
        predicted = dispatch_outcome(obs, sim.config, actions)
        self.assert_observations_equal(obs, previous)
        self.assertEqual(rng_state, sim.rng.bit_generator.state)
        actual_obs, reward, _, _, info = sim.dispatch(actions)
        self.assertAlmostEqual(predicted["reward"], reward, places=10)
        self.assertEqual(set(predicted["reward_terms"]), set(info["reward_terms"]))
        for term, value in info["reward_terms"].items():
            with self.subTest(term=term):
                self.assertAlmostEqual(predicted["reward_terms"][term], value, places=10)
        for field, expected in (
            ("next_energy_kwh", info["energy_ledger"]["after_kwh"]),
            ("next_locations", actual_obs["locations"]),
            ("served_request_mask", info["served_request_mask"]),
            ("assigned_mcv", info["assigned_mcv"]),
            ("delivered_mobile_kwh", info["delivered_mobile_kwh"]),
            ("charge_grid_kwh", info["energy_ledger"]["charge_grid_kwh"]),
            ("export_grid_kwh", info["energy_ledger"]["export_grid_kwh"]),
        ):
            with self.subTest(field=field):
                np.testing.assert_allclose(predicted[field], expected, atol=1e-10, rtol=0)
        return predicted, info

    def assert_legal(self, obs, actions):
        self.assertEqual(len(actions), len(obs["locations"]))
        for m, action in enumerate(actions):
            self.assertIsInstance(action, DispatchAction)
            self.assertTrue(obs["action_mask"][m, action.mode, action.target_zone])
            self.assertTrue(np.isfinite(action.energy_kwh))
            self.assertGreaterEqual(action.energy_kwh, 0)

    def test_forward_model_matches_random_joint_actions_over_whole_episodes(self):
        scenario = benchmark_scenario(terminal_value_per_stored_kwh=.17)
        rng = np.random.default_rng(871)
        for seed in range(40):
            sim = ChargingSimulator(scenario)
            obs, _ = sim.reset(seed=seed)
            for period in range(scenario.horizon):
                sim.quote_price_levels(random_prices(obs, scenario, rng))
                obs = sim.observe()
                actions = []
                for mask in obs["action_mask"]:
                    feasible = np.argwhere(mask)
                    mode, zone = feasible[rng.integers(len(feasible))]
                    # Include amounts beyond individual limits to test physical clipping.
                    energy = float(rng.uniform(0., 100.)) if mode in (3, 4) else 0.
                    actions.append(DispatchAction(Mode(int(mode)), int(zone), energy))
                with self.subTest(seed=seed, period=period):
                    self.assert_matches_simulator(sim, actions)
                obs = sim.observe()

    def test_first_fit_requests_are_served_once_under_competing_vehicles(self):
        scenario = benchmark_scenario(
            initial_locations=(0, 0), initial_energy_kwh=(25., 25.),
            fcs_ev_capacity_kwh=(15.,), price_noise_distribution="none",
        )
        sim = known_choices(scenario, [15., 5., 5., 15., 10., 10., 5.],
                            [0, 0, 0, 0, 1, 1, 1])
        predicted, info = self.assert_matches_simulator(sim, [
            DispatchAction(Mode.SERVE, 0), DispatchAction(Mode.SERVE, 0),
        ])
        np.testing.assert_array_equal(predicted["assigned_mcv"], [0, 1, 1, -1, -1, -1, -1])
        np.testing.assert_array_equal(predicted["served_request_mask"], [1, 1, 1, 0, 1, 0, 1])
        np.testing.assert_allclose(predicted["next_energy_kwh"], [6.25, 12.5])
        self.assertAlmostEqual(predicted["reward"], 10.25)
        self.assertEqual(info["unmet_mobile_requests"], 1)
        self.assertEqual(info["unmet_fixed_requests"], 1)

    def test_opposite_exchange_directions_share_station_capacity(self):
        scenario = benchmark_scenario(initial_locations=(0, 0), initial_energy_kwh=(40., 40.))
        sim = known_choices(scenario, [], [])
        predicted, _ = self.assert_matches_simulator(sim, [
            DispatchAction(Mode.RECHARGE, 0, 10.),
            DispatchAction(Mode.DISCHARGE, 0, 10.),
        ])
        np.testing.assert_allclose(predicted["charge_grid_kwh"], [5., 0.])
        np.testing.assert_allclose(predicted["export_grid_kwh"], [0., 5.])
        np.testing.assert_allclose(predicted["next_energy_kwh"], [44., 33.75])

    def test_wait_retains_operating_cost_and_terminal_inventory_once(self):
        scenario = benchmark_scenario(
            horizon=1, arrival_profile=(1.,), grid_buy_per_kwh=(.2,),
            grid_sell_per_kwh=(.1,), initial_energy_kwh=(40., 45.),
            terminal_value_per_stored_kwh=.25,
        )
        sim = known_choices(scenario, [], [])
        actions = [DispatchAction(Mode.WAIT, int(z)) for z in sim.locations]
        predicted, _ = self.assert_matches_simulator(sim, actions)
        self.assertAlmostEqual(predicted["reward"], .25 * 85. - 1.)
        self.assertAlmostEqual(predicted["reward_terms"]["terminal_inventory_value"], 21.25)

    def test_terminal_coordinator_ignores_nonterminal_shadow_value(self):
        scenario = benchmark_scenario(
            horizon=1, arrival_profile=(1.,), grid_buy_per_kwh=(.2,),
            grid_sell_per_kwh=(.1,), terminal_value_per_stored_kwh=.07,
        )
        sim = known_choices(scenario, [8., 12.], [0, 0])
        obs = sim.observe()
        self.assertEqual(coordinated_dispatch(obs, scenario, shadow_price=0.),
                         coordinated_dispatch(obs, scenario, shadow_price=1000.))

    def test_all_rule_actions_are_legal_at_reserve_full_battery_and_no_requests(self):
        rng = np.random.default_rng(853)
        for energy in ((5., 5.), (80., 80.), (5., 80.), (40., 45.)):
            scenario = benchmark_scenario(initial_energy_kwh=energy)
            sim = known_choices(scenario, [], [])
            obs = sim.observe()
            rules = (
                random_dispatch(obs, scenario, rng),
                service_first_dispatch(obs, scenario),
                coordinated_dispatch(obs, scenario, shadow_price=.2),
            )
            for actions in rules:
                with self.subTest(energy=energy, actions=actions):
                    self.assert_legal(obs, actions)
                    predicted = dispatch_outcome(obs, scenario, actions)
                    self.assertFalse(predicted["invalid_actions"])
                    self.assertGreaterEqual(min(predicted["next_energy_kwh"]), scenario.reserve_kwh - 1e-9)
                    self.assertLessEqual(max(predicted["next_energy_kwh"]), scenario.battery_capacity_kwh + 1e-9)

    def test_policy_helpers_do_not_mutate_observations(self):
        scenario = benchmark_scenario()
        sim = ChargingSimulator(scenario)
        obs, _ = sim.reset(seed=81)
        expected = deepcopy(obs)
        rng = np.random.default_rng(63)
        for levels in (fixed_prices(obs, scenario, 3), adaptive_prices(obs, scenario),
                       random_prices(obs, scenario, rng)):
            self.assertEqual(levels.shape, (scenario.n_zones,))
            self.assertTrue(np.issubdtype(levels.dtype, np.integer))
            self.assertTrue(((levels >= 0) & (levels < len(scenario.price_levels))).all())
            np.testing.assert_array_equal(levels[~obs["mobile_available"].astype(bool)], 0)
        self.assert_observations_equal(obs, expected)
        sim.quote_price_levels(adaptive_prices(obs, scenario))
        obs = sim.observe()
        expected = deepcopy(obs)
        for actions in (random_dispatch(obs, scenario, rng),
                        service_first_dispatch(obs, scenario),
                        coordinated_dispatch(obs, scenario, shadow_price=.2)):
            self.assert_legal(obs, actions)
        self.assert_observations_equal(obs, expected)

    def test_policies_do_not_use_future_price_or_arrival_profiles(self):
        scenario = benchmark_scenario()
        changed_future = replace(scenario,
            grid_buy_per_kwh=(.2, 9., 9.), grid_sell_per_kwh=(.1, 8., 8.),
            arrival_profile=(1., 0., 100.),
            fcs_charge_price_profile=((.5,), (7.,), (7.,)),
            fcs_discharge_price_profile=((.1,), (6.,), (6.,)),
        ).validate()
        sim = ChargingSimulator(scenario)
        obs, _ = sim.reset(seed=73)
        np.testing.assert_array_equal(adaptive_prices(obs, scenario), adaptive_prices(obs, changed_future))
        sim.quote_price_levels(fixed_prices(obs, scenario, 1))
        obs = sim.observe()
        self.assertEqual(service_first_dispatch(obs, scenario), service_first_dispatch(obs, changed_future))
        self.assertEqual(coordinated_dispatch(obs, scenario, shadow_price=.2),
                         coordinated_dispatch(obs, changed_future, shadow_price=.2))

    def test_myopic_coordinator_never_loses_to_wait_for_same_quoted_demand(self):
        scenario = benchmark_scenario()
        for seed in range(12):
            sim = ChargingSimulator(scenario)
            obs, _ = sim.reset(seed=seed + 151)
            sim.quote_price_levels(adaptive_prices(obs, scenario))
            obs = sim.observe()
            wait = [DispatchAction(Mode.WAIT, int(z)) for z in obs["locations"]]
            chosen = coordinated_dispatch(obs, scenario, shadow_price=0.)
            self.assert_legal(obs, chosen)
            self.assertGreaterEqual(dispatch_outcome(obs, scenario, chosen)["reward"] + 1e-9,
                                    dispatch_outcome(obs, scenario, wait)["reward"])

    def test_joint_search_matches_individual_simulator_trials(self):
        scenario = benchmark_scenario(
            initial_locations=(0, 0), initial_energy_kwh=(30., 60.),
            fcs_fleet_exchange_capacity_kwh=(12.,),
        )
        sim = known_choices(scenario, [7., 11., 15.], [0, 0, 0])
        obs = sim.observe()
        options = []
        for m in range(scenario.n_mcv):
            choices = []
            for mode, z in np.argwhere(obs["action_mask"][m]):
                distance = scenario.distances[obs["locations"][m], z]
                time = scenario.period_hours - distance / scenario.speed_kmph
                stock = obs["energy_kwh"][m] - distance * scenario.travel_kwh_per_km
                if mode == Mode.RECHARGE:
                    maximum = min(time * scenario.mobile_charge_kw,
                                  (scenario.battery_capacity_kwh - stock) / scenario.charge_efficiency)
                elif mode == Mode.DISCHARGE:
                    maximum = min(time * scenario.mobile_discharge_kw,
                                  (stock - scenario.reserve_kwh) * scenario.discharge_efficiency)
                else:
                    choices.append(DispatchAction(Mode(int(mode)), int(z)))
                    continue
                choices.extend(DispatchAction(Mode(int(mode)), int(z), fraction * maximum)
                               for fraction in (.5, 1.))
            options.append(choices)
        scores = []
        shadow = .2
        for joint in product(*options):
            trial = deepcopy(sim)
            _, reward, _, _, info = trial.dispatch(list(joint))
            scores.append(reward + shadow * (
                info["energy_ledger"]["after_kwh"].sum() - obs["energy_kwh"].sum()))
        chosen = coordinated_dispatch(obs, scenario, shadow_price=shadow)
        _, reward, _, _, info = sim.dispatch(chosen)
        score = reward + shadow * (info["energy_ledger"]["after_kwh"].sum() - obs["energy_kwh"].sum())
        self.assertAlmostEqual(score, max(scores), places=10)


class BenchmarkEvaluationTests(unittest.TestCase):
    @staticmethod
    def without_timing(result):
        result = deepcopy(result)
        result.pop("mean_policy_decision_ms", None)
        for episode in result.get("episodes", []):
            episode.pop("mean_policy_decision_ms", None)
        return result

    def test_random_policy_is_repeatable_order_independent_and_preserves_global_rng(self):
        scenario = benchmark_scenario()
        pricing = lambda obs, rng: random_prices(obs, scenario, rng)
        dispatch = lambda obs, rng: random_dispatch(obs, scenario, rng)
        state = np.random.get_state()
        first = evaluate_policy(scenario, [601, 603, 605], pricing, dispatch, action_seed=117)
        after = np.random.get_state()
        self.assertEqual(state[0], after[0])
        np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])
        repeated = evaluate_policy(scenario, [601, 603, 605], pricing, dispatch, action_seed=117)
        self.assertEqual(self.without_timing(first), self.without_timing(repeated))
        reverse = evaluate_policy(scenario, [605, 603, 601], pricing, dispatch, action_seed=117)
        episodes = self.without_timing(first)["episodes"]
        self.assertEqual(episodes, list(reversed(self.without_timing(reverse)["episodes"])))

    def test_policy_rng_draws_cannot_change_environment_samples(self):
        scenario = benchmark_scenario()
        def pricing(obs, rng):
            return fixed_prices(obs, scenario, 1)
        def noisy_pricing(obs, rng):
            rng.random(1000)
            return pricing(obs, rng)
        def wait(obs, rng):
            return [DispatchAction(Mode.WAIT, int(z)) for z in obs["locations"]]
        first = evaluate_policy(scenario, [611, 613], pricing, wait)
        extra_draws = evaluate_policy(scenario, [611, 613], noisy_pricing, wait)
        self.assertEqual(self.without_timing(first), self.without_timing(extra_draws))

    def test_empty_demand_fulfillment_and_single_episode_ci_are_undefined(self):
        scenario = benchmark_scenario(mean_requests_by_zone=(0., 0.))
        result = evaluate_policy(scenario, [621],
            lambda obs, rng: fixed_prices(obs, scenario, 0),
            lambda obs, rng: [DispatchAction(Mode.WAIT, int(z)) for z in obs["locations"]])
        self.assertIsNone(result["confidence_interval_95"])
        self.assertIsNone(result["standard_error_discounted_profit"])
        self.assertIsNone(result["pooled_selected_service_fulfillment"])
        self.assertIsNone(result["mean_selected_service_fulfillment"])
        self.assertEqual(result["action_counts"]["WAIT"], scenario.horizon * scenario.n_mcv)
        expected = -scenario.operating_cost_per_mcv_period * scenario.n_mcv * sum(
            scenario.discount_per_period**t for t in range(scenario.horizon))
        self.assertAlmostEqual(result["mean_discounted_profit"], expected)
        self.assertEqual(result["loss_probability"], 1.)

    def test_paired_bootstrap_aligns_seeds_and_preserves_rng(self):
        candidate = {"episodes": [
            {"seed": 101, "discounted_profit": 12.},
            {"seed": 103, "discounted_profit": 1.},
            {"seed": 105, "discounted_profit": 4.},
        ]}
        baseline = {"episodes": [
            {"seed": 105, "discounted_profit": 2.},
            {"seed": 101, "discounted_profit": 10.},
            {"seed": 103, "discounted_profit": -1.},
        ]}
        before = np.random.get_state()
        actual = paired_comparison(candidate, baseline, bootstrap_samples=1000, seed=63)
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        self.assertEqual(actual["mean_difference"], 2.)
        self.assertEqual(actual["paired_bootstrap_confidence_interval_95"], [2., 2.])
        self.assertEqual(actual["candidate_win_fraction"], 1.)
        self.assertEqual(actual, paired_comparison(candidate, baseline, bootstrap_samples=1000, seed=63))

    def test_pairing_rejects_different_duplicate_or_nonfinite_observations(self):
        valid = {"episodes": [{"seed": 21, "discounted_profit": 2.}]}
        for bad in (
            {"episodes": [{"seed": 22, "discounted_profit": 1.}]},
            {"episodes": [{"seed": 21, "discounted_profit": 1.},
                          {"seed": 21, "discounted_profit": 1.}]},
            {"episodes": [{"seed": 21, "discounted_profit": float("nan")}]},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                paired_comparison(valid, bad)
        self.assertIsNone(paired_comparison(valid, valid)["paired_bootstrap_confidence_interval_95"])


@unittest.skipIf(torch is None, "Install the train extra to test frozen learned policies")
class LearnedBenchmarkAdapterTests(unittest.TestCase):
    def test_public_localization_matches_simulator_and_copies_arrays(self):
        from charging_env.benchmarks.learned import localize_observation
        sim = known_choices(benchmark_scenario(), [7., 12.], [0, 0])
        obs = sim.observe()
        before = deepcopy(obs)
        for m in range(sim.config.n_mcv):
            actual = localize_observation(obs, m, sim.config.n_mcv)
            expected = sim.local_observation(m)
            self.assertEqual(set(actual), set(expected))
            for key in actual:
                np.testing.assert_array_equal(actual[key], expected[key])
                if isinstance(actual[key], np.ndarray):
                    actual[key].fill(0)
        for key in obs:
            np.testing.assert_array_equal(obs[key], before[key])

    def test_common_harness_reproduces_trainer_evaluation_and_restores_torch_rng(self):
        from charging_env.benchmarks.learned import evaluate_checkpoint
        from charging_env.rl.trainer import Trainer, TrainingConfig
        scenario = benchmark_scenario()
        trainer = Trainer(scenario, TrainingConfig(seed=27, hidden_size=16, torch_threads=1))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "policy.pt"
            trainer.save_checkpoint(checkpoint)
            for deterministic in (False, True):
                expected = trainer.evaluate([631, 633, 635], deterministic=deterministic, action_seed=791)
                rng = torch.get_rng_state().clone()
                actual = evaluate_checkpoint(scenario, [631, 633, 635], checkpoint,
                    deterministic=deterministic, action_seed=791)
                torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
                self.assertEqual(actual["mean_discounted_profit"], expected["mean_discounted_profit"])
                self.assertEqual(actual["action_counts"], expected["action_counts"])
                self.assertEqual(actual["mean_reward_terms"], expected["mean_reward_terms"])
                for observed, reference in zip(actual["episodes"], expected["episodes"]):
                    for key in ("seed", "profit", "discounted_profit", "reward_terms",
                                "discounted_reward_terms", "served_requests", "total_requests"):
                        self.assertEqual(observed[key], reference[key])


if __name__ == "__main__":
    unittest.main()
