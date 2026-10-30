"""LADWP-anchored station base prices and the mobile premium menu.

Covers the three pricing rules directly: a typed base price per period and
station, an EV service fee on top of it, and a mobile menu quoted as a premium
over the nearest station's EV price. Legacy profile/shock pricing keeps its
coverage in the physics regressions.
"""
from dataclasses import replace
import unittest

import numpy as np

from charging_env.config import PERIOD_TYPES, Scenario, ladwp_period_type
from charging_env.simulator import ChargingSimulator, DispatchAction, Mode


def pricing_scenario(**changes):
    return replace(Scenario(), **changes).validate()


def base_price_draws(scenario, period, seeds=range(200)):
    """Realized station base prices at one period, across independent seeds."""
    sim = ChargingSimulator(scenario)
    draws = []
    for seed in seeds:
        sim.reset(seed=int(seed))
        for _ in range(period):
            sim.quote_price_levels(np.zeros(scenario.n_zones, dtype=int))
            sim.dispatch([DispatchAction(Mode.WAIT, int(z)) for z in sim.locations])
        draws.append(sim.fcs_base_prices.copy())
    return np.asarray(draws)


class PeriodTypeTests(unittest.TestCase):
    def test_ladwp_windows_classify_boundary_hours(self):
        expected = {16: "peak", 20: "peak", 21: "normal", 22: "normal",
                    7: "normal", 15: "normal", 23: "off_peak", 0: "off_peak", 6: "off_peak"}
        for hour, name in expected.items():
            with self.subTest(hour=hour):
                self.assertEqual(PERIOD_TYPES[ladwp_period_type(hour)], name)
        # Hours wrap, so a 12-period day may run past midnight.
        self.assertEqual(ladwp_period_type(24), ladwp_period_type(0))
        self.assertEqual(ladwp_period_type(25), ladwp_period_type(1))

    def test_default_day_covers_all_three_types(self):
        c = pricing_scenario()
        names = [PERIOD_TYPES[j] for j in c.resolved_period_types]
        self.assertEqual(len(names), c.horizon)
        self.assertEqual(set(names), set(PERIOD_TYPES))
        # Starting at 14:00, the LADWP peak window lands on periods 2-6.
        self.assertEqual([i for i, n in enumerate(names) if n == "peak"], [2, 3, 4, 5, 6])

    def test_explicit_period_types_override_the_clock(self):
        names = ("peak",) * 12
        c = pricing_scenario(period_types=names)
        np.testing.assert_array_equal(c.resolved_period_types,
                                      np.full(12, PERIOD_TYPES.index("peak")))
        with self.assertRaises(ValueError):
            pricing_scenario(period_types=("bogus",) * 12)
        with self.assertRaises(ValueError):
            pricing_scenario(period_types=("peak",) * 11)


class BasePriceTests(unittest.TestCase):
    def test_draws_stay_inside_the_period_band(self):
        c = pricing_scenario()
        for period in range(c.horizon):
            low, high = c.resolved_base_price_bounds[period]
            draws = base_price_draws(c, period, seeds=range(40))
            with self.subTest(period=period, type=PERIOD_TYPES[c.resolved_period_types[period]]):
                self.assertTrue((draws >= low - 1e-12).all())
                self.assertTrue((draws <= high + 1e-12).all())

    def test_sample_mean_matches_the_posted_ladwp_rate(self):
        c = pricing_scenario()
        for name, mean in zip(PERIOD_TYPES, c.fcs_base_price_mean):
            period = int(np.flatnonzero(c.resolved_period_types == PERIOD_TYPES.index(name))[0])
            draws = base_price_draws(c, period)
            with self.subTest(type=name):
                self.assertAlmostEqual(float(draws.mean()), mean, delta=0.004)

    def test_peak_exceeds_normal_exceeds_off_peak_for_every_draw(self):
        c = pricing_scenario()
        by_type = {}
        for name in PERIOD_TYPES:
            period = int(np.flatnonzero(c.resolved_period_types == PERIOD_TYPES.index(name))[0])
            by_type[name] = base_price_draws(c, period, seeds=range(60))
        self.assertGreater(by_type["normal"].min(), by_type["off_peak"].max())
        self.assertGreater(by_type["peak"].min(), by_type["normal"].max())
        # Ranges widen with the period type, as the LADWP spread does.
        widths = [np.ptp(by_type[name]) for name in PERIOD_TYPES]
        self.assertEqual(widths, sorted(widths))

    def test_stations_draw_independently(self):
        c = pricing_scenario()
        draws = base_price_draws(c, 0, seeds=range(80))
        self.assertEqual(draws.shape[1], c.n_fcs)
        self.assertGreater(np.abs(draws[:, 0] - draws[:, 1]).max(), 1e-6)

    def test_ev_pays_the_service_fee_and_the_fleet_exchanges_at_base(self):
        for alpha in (0., 0.1, 0.25):
            c = pricing_scenario(fcs_service_fee_rate=alpha)
            sim = ChargingSimulator(c)
            obs, _ = sim.reset(seed=3)
            with self.subTest(alpha=alpha):
                np.testing.assert_allclose(obs["fcs_retail_prices"],
                                           obs["fcs_base_prices"] * (1. + alpha))
                np.testing.assert_allclose(obs["fcs_discharge_prices"], obs["fcs_base_prices"])

    def test_fleet_recharge_is_bought_at_the_station_base_price(self):
        c = pricing_scenario(mean_requests_by_zone=(0.,) * 4, initial_locations=(0, 0),
                             initial_energy_kwh=(20., 20.))
        sim = ChargingSimulator(c)
        sim.reset(seed=5)
        post, *_ = sim.quote_price_levels(np.zeros(c.n_zones, dtype=int))
        base = sim.fcs_base_prices.copy()
        _, _, _, _, info = sim.dispatch([DispatchAction(Mode.RECHARGE, 0, 10.)] * 2)
        terms = info["reward_terms"]
        # Station 0 is the only one visited, so procurement is base[0] * kWh drawn.
        drawn = (sim.energy - 20.).sum() / c.charge_efficiency
        self.assertGreater(drawn, 0.)
        self.assertAlmostEqual(terms["procurement_cost"], float(base[0] * drawn), places=8)


class PremiumMenuTests(unittest.TestCase):
    def test_nearest_station_anchors_each_zone(self):
        c = pricing_scenario()
        # Stations sit in zones 0 and 3; zones 1 and 2 are equidistant, so ties
        # resolve to the lower station index.
        np.testing.assert_array_equal(c.nearest_fcs, [0, 0, 0, 1])
        sim = ChargingSimulator(c)
        obs, _ = sim.reset(seed=9)
        premiums = np.asarray(c.mcv_premium_levels)
        for z in range(c.n_zones):
            anchor = obs["fcs_retail_prices"][c.nearest_fcs[z]]
            with self.subTest(zone=z):
                np.testing.assert_allclose(obs["mobile_price_menu"][z], anchor * premiums)

    def test_premium_reference_can_drop_the_service_fee(self):
        c = pricing_scenario(mcv_premium_reference="base_price")
        sim = ChargingSimulator(c)
        obs, _ = sim.reset(seed=9)
        premiums = np.asarray(c.mcv_premium_levels)
        for z in range(c.n_zones):
            anchor = obs["fcs_base_prices"][c.nearest_fcs[z]]
            with self.subTest(zone=z):
                np.testing.assert_allclose(obs["mobile_price_menu"][z], anchor * premiums)

    def test_quoting_a_level_charges_that_premium(self):
        c = pricing_scenario()
        sim = ChargingSimulator(c)
        obs, _ = sim.reset(seed=11)
        menu = obs["mobile_price_menu"].copy()
        levels = np.array([2, 0, 1, 2])
        post, *_ = sim.quote_price_levels(levels)
        np.testing.assert_allclose(post["mobile_prices"],
                                   menu[np.arange(c.n_zones), levels])
        self.assertEqual(c.n_price_levels, 3)

    def test_level_index_must_address_the_premium_menu(self):
        c = pricing_scenario()
        sim = ChargingSimulator(c)
        sim.reset(seed=11)
        for levels in ([0, 1, 2, 3], [3, 3, 3, 3], [-1, 0, 0, 0]):
            with self.subTest(levels=levels), self.assertRaises(ValueError):
                sim.quote_price_levels(np.asarray(levels))

    def test_menu_tracks_the_base_price_across_periods(self):
        c = pricing_scenario()
        sim = ChargingSimulator(c)
        obs, _ = sim.reset(seed=13)
        seen = []
        for _ in range(c.horizon):
            seen.append((obs["fcs_base_prices"].copy(), obs["mobile_price_menu"].copy()))
            post, *_ = sim.quote_price_levels(np.ones(c.n_zones, dtype=int))
            obs, _, done, _, _ = sim.dispatch(
                [DispatchAction(Mode.WAIT, int(z)) for z in post["locations"]])
            if done:
                break
        premiums = np.asarray(c.mcv_premium_levels)
        for base, menu in seen:
            anchor = base[c.nearest_fcs] * (1. + c.fcs_service_fee_rate)
            np.testing.assert_allclose(menu, anchor[:, None] * premiums[None, :])
        # Peak periods really do quote more than off-peak ones.
        peak = seen[2][1].max()
        off_peak = seen[-1][1].max()
        self.assertGreater(peak, off_peak)

    def test_absolute_mode_keeps_the_fixed_shared_menu(self):
        c = pricing_scenario(fcs_pricing_mode="legacy", mcv_pricing_mode="absolute")
        sim = ChargingSimulator(c)
        obs, _ = sim.reset(seed=2)
        self.assertEqual(c.n_price_levels, len(c.price_levels))
        for z in range(c.n_zones):
            np.testing.assert_allclose(obs["mobile_price_menu"][z], c.price_levels)


class PricingValidationTests(unittest.TestCase):
    def test_modes_and_fee_are_checked(self):
        for settings in ({"fcs_pricing_mode": "bogus"}, {"mcv_pricing_mode": "bogus"},
                         {"mcv_premium_reference": "bogus"},
                         {"fcs_service_fee_rate": -0.1}, {"fcs_service_fee_rate": float("nan")},
                         {"first_period_hour_of_day": 24}, {"first_period_hour_of_day": -1},
                         {"first_period_hour_of_day": True}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                pricing_scenario(**settings)

    def test_base_price_bands_must_be_disjoint_and_widening(self):
        malformed = (
            # Overlapping normal and peak supports.
            {"fcs_base_price_halfwidth": (0.01218, 0.06, 0.02933)},
            # Non-widening ranges break the "peak range > normal range" ordering.
            {"fcs_base_price_halfwidth": (0.03, 0.02, 0.01)},
            # A half-width at or above the mean admits nonpositive prices.
            {"fcs_base_price_mean": (0.01, 0.27739, 0.36667),
             "fcs_base_price_halfwidth": (0.01218, 0.01387, 0.02933)},
            # Decreasing means invert the required ordering.
            {"fcs_base_price_mean": (0.36667, 0.27739, 0.24368)},
            {"fcs_base_price_mean": (0.2, 0.3)},
            {"fcs_base_price_halfwidth": (-0.01, 0.01387, 0.02933)},
        )
        for settings in malformed:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                pricing_scenario(**settings)

    def test_premium_levels_must_be_increasing_and_positive(self):
        for levels in ((1.4, 1.1), (1.1, 1.1), (0., 1.4), (-1., 1.4), ()):
            with self.subTest(levels=levels), self.assertRaises(ValueError):
                pricing_scenario(mcv_premium_levels=levels)

    def test_premium_prices_must_fit_the_declared_price_bounds(self):
        with self.assertRaises(ValueError):
            pricing_scenario(mcv_premium_levels=(1.1, 1.4, 4.0))
        with self.assertRaises(ValueError):
            pricing_scenario(price_min=0.5)

    def test_base_mode_rejects_explicit_station_profiles(self):
        for settings in ({"fcs_charge_price_profile": tuple((0.4, 0.4) for _ in range(12))},
                         {"fcs_discharge_price_profile": tuple((0.3, 0.3) for _ in range(12))}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                pricing_scenario(**settings)

    def test_premium_pricing_requires_base_price_stations(self):
        with self.assertRaises(ValueError):
            pricing_scenario(fcs_pricing_mode="legacy", mcv_pricing_mode="premium")

    def test_json_roundtrip_preserves_the_pricing_settings(self):
        import json
        import tempfile
        from pathlib import Path
        c = pricing_scenario(period_types=("peak",) * 12, fcs_service_fee_rate=0.15,
                             mcv_premium_levels=(1.05, 1.2, 1.35))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scenario.json"
            c.to_json(path)
            data = json.loads(path.read_text())
            self.assertEqual(data["fcs_service_fee_rate"], 0.15)
            self.assertEqual(data["mcv_premium_levels"], [1.05, 1.2, 1.35])
            self.assertEqual(data["period_types"], ["peak"] * 12)
            self.assertEqual(Scenario.from_json(path), c)


if __name__ == "__main__":
    unittest.main()
