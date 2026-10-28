"""Statistical and information-boundary checks for historical model fitting."""
from contextlib import redirect_stdout
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np

from charging_env.config import Scenario
from charging_env.choice import charging_choice_probabilities
from charging_env.rl.history_model import (PUBLIC_FIELDS, HistoricalLog,
    HistoricalModel, HistoryFitConfig, estimate_arrival_rates, fit_history_model,
    public_model_config, public_scenario)


def scenario():
    return replace(Scenario(), horizon=3, arrival_profile=(1.,)*3,
        grid_buy_per_kwh=(.15,)*3, grid_sell_per_kwh=(.1,)*3).validate()


def synthetic_log(seed=7, days=120, rate=24., coefficients=(2.4, .45, .8, .6, .15)):
    """Independently draw known choice responses, without estimator internals."""
    rng = np.random.default_rng(seed)
    c = scenario(); f = c.n_fcs
    counts = rng.poisson(rate, size=(days, c.horizon, c.n_zones))
    cells = np.repeat(np.arange(counts.size), counts.ravel())
    day, period, zone = np.unravel_index(cells, counts.shape)
    n = len(cells); energy = rng.uniform(5., 15., n)
    payment = np.zeros((n, f+2)); payment[:, :-1] = rng.uniform(.2, 1., size=(n, f+1))*energy[:, None]
    mobile_wait = rng.uniform(0., 3., n); fixed_wait = rng.uniform(0., 5., (n, f))
    travel = rng.uniform(0., 10., (n, f))
    available = rng.random((n, f+2)) > .15; available[:, -1] = True
    b, eta, wm, wf, distance = coefficients
    utilities = np.zeros((n, f+2))
    utilities[:, 0] = b-eta*payment[:, 0]-wm*mobile_wait
    utilities[:, 1:-1] = b-eta*payment[:, 1:-1]-wf*fixed_wait-distance*travel
    utilities[~available] = -np.inf
    weights = np.exp(utilities-utilities.max(axis=1, keepdims=True))
    cumulative = np.cumsum(weights/weights.sum(axis=1, keepdims=True), axis=1)
    cumulative[:, -1] = 1.
    chosen = (rng.random(n)[:, None] >= cumulative).sum(axis=1)
    return HistoricalLog(public_model_config(c), counts, day, period, zone, energy,
        payment, mobile_wait, fixed_wait, travel, available, chosen,
        np.arange(days, dtype=np.int64))


def arguments(log):
    names = ('counts', 'request_day', 'request_period', 'request_zone',
        'request_energy_kwh', 'alternative_payment', 'mobile_wait_hours',
        'fixed_wait_hours', 'fixed_travel_km', 'available', 'chosen', 'day_ids')
    return {name: getattr(log, name).copy() for name in names}


class HistoricalModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.log = synthetic_log()
        cls.model, cls.report = fit_history_model(cls.log)

    def test_choice_parameter_recovery_and_heldout_likelihood(self):
        expected = np.array([2.4, .45, .8, .6, .15])
        error = np.abs(self.model.choice_coefficients-expected)
        self.assertTrue(np.all(error < [.2, .035, .1, .1, .04]), str(error))
        self.assertTrue(self.report['choice']['converged'])
        self.assertFalse(self.report['choice']['weakly_identified'])
        self.assertEqual(self.report['choice']['fisher_rank'], 5)
        heldout = synthetic_log(seed=191, days=32)
        fitted = self.model.choice_log_likelihood(heldout)
        truth = HistoricalModel(heldout.public_config, self.model.scenario.resolved_arrival_rates,
            (5., 15.), expected, {})
        uninformed = HistoricalModel(heldout.public_config, self.model.scenario.resolved_arrival_rates,
            (5., 15.), (0., 1., 0., 0., 0.), {})
        self.assertLess(fitted['mean_negative_log_likelihood'], uninformed.choice_log_likelihood(heldout)['mean_negative_log_likelihood']-.1)
        self.assertLess(abs(fitted['mean_negative_log_likelihood']-truth.choice_log_likelihood(heldout)['mean_negative_log_likelihood']), .005)

    def test_demand_means_include_zero_days_and_documented_smoothing(self):
        counts = np.array([[[0, 0], [0, 0]], [[4, 0], [8, 0]]], dtype=int)
        np.testing.assert_array_equal(estimate_arrival_rates(counts, 0.), [[2., 0.], [4., 0.]])
        np.testing.assert_allclose(estimate_arrival_rates(counts, 1.), [[7/3, 0.], [11/3, 0.]])
        np.testing.assert_array_equal(estimate_arrival_rates(np.zeros((3, 2, 4), dtype=int)), np.zeros((2, 4)))
        self.assertLess(np.abs(self.model.scenario.resolved_arrival_rates-24.).mean(), .6)
        self.assertLess(abs(self.model.scenario.request_min_kwh-5.), .02)
        self.assertLess(abs(self.model.scenario.request_max_kwh-15.), .02)

    def test_hidden_truth_poisoning_cannot_change_public_fields_or_fit(self):
        c = scenario()
        poisoned = replace(c, charging_utility_intercept=-999., price_sensitivity_per_dollar=17.,
            mobile_wait_cost_per_hour=99., fixed_wait_cost_per_hour=77., fixed_travel_cost_per_km=66.,
            request_min_kwh=100., request_max_kwh=300., request_kwh=200.,
            mean_requests_by_zone=(400.,)*4, arrival_profile=(88.,)*3,
            arrival_rates=((999.,)*4,)*3, max_requests_per_zone=9999)
        self.assertEqual(public_model_config(c), public_model_config(poisoned))
        self.assertFalse(set(PUBLIC_FIELDS) & {'arrival_rates', 'mean_requests_by_zone',
            'charging_utility_intercept', 'request_min_kwh', 'max_requests_per_zone'})
        small = synthetic_log(days=3, rate=4.)
        same = HistoricalLog(public_model_config(poisoned), **arguments(small))
        a, _ = fit_history_model(small); b, _ = fit_history_model(same)
        np.testing.assert_array_equal(a.choice_coefficients, b.choice_coefficients)
        self.assertEqual(a.scenario, b.scenario)
        self.assertEqual(small.provenance(), same.provenance())
        changed = same.public_config; changed['unmet_penalty_per_kwh'] = 999.
        self.assertNotEqual(changed, same.public_config)

    def test_mapped_scenario_reproduces_fitted_choice_likelihood(self):
        log = self.log; c = self.model.scenario
        mobile_cost = log.alternative_payment[:, 0]+c.mobile_wait_cost_per_hour*log.mobile_wait_hours
        fixed_cost = (log.alternative_payment[:, 1:-1]+c.fixed_wait_cost_per_hour*log.fixed_wait_hours
            +c.fixed_travel_cost_per_km*log.fixed_travel_km)
        probabilities = charging_choice_probabilities(c, mobile_cost, fixed_cost,
            log.available[:, 0], log.available[:, 1:-1])
        expected = float(np.log(probabilities[np.arange(log.n_requests), log.chosen]).mean())
        self.assertAlmostEqual(expected, self.model.choice_log_likelihood(log)['mean_log_likelihood'], places=13)

    def test_logs_checkpoint_roundtrip_and_safe_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory)/'log.npz'; self.log.save(log_path)
            restored = HistoricalLog.load(log_path)
            self.assertEqual(self.log.provenance(), restored.provenance())
            checkpoint = Path(directory)/'model.npz'; self.model.save_checkpoint(checkpoint)
            model = HistoricalModel.load_checkpoint(checkpoint)
            self.assertEqual(model.scenario, self.model.scenario)
            self.assertEqual(model.provenance()['model_sha256'], self.model.provenance()['model_sha256'])
            self.assertIsNotNone(model.provenance()['checkpoint_sha256'])
            with self.assertRaises(FileExistsError): self.model.save_checkpoint(checkpoint)
            with np.load(checkpoint, allow_pickle=False) as data:
                arrays = {name: data[name] for name in data.files}
            arrays['choice_coefficients'] = arrays['choice_coefficients']+.1
            corrupt = Path(directory)/'changed.npz'; np.savez(corrupt, **arrays)
            with self.assertRaises(ValueError): HistoricalModel.load_checkpoint(corrupt)
            exposed = model.choice_coefficients; exposed.setflags(write=True); exposed[:] = 999.
            self.assertFalse((model.choice_coefficients == 999.).all())
            provenance = model.provenance(); provenance['fit']['fit_days'] = 999
            self.assertNotEqual(model.provenance()['fit']['fit_days'], 999)

    def test_rejects_malformed_or_censored_histories_and_extra_truth_fields(self):
        log = synthetic_log(days=2, rate=1.)
        kwargs = arguments(log); kwargs['counts'][0, 0, 0] += 1
        with self.assertRaises(ValueError): HistoricalLog(log.public_config, **kwargs)
        kwargs = arguments(log); kwargs['available'][:, -1] = False
        with self.assertRaises(ValueError): HistoricalLog(log.public_config, **kwargs)
        kwargs = arguments(log); kwargs['request_energy_kwh'][0] = np.nan
        with self.assertRaises(ValueError): HistoricalLog(log.public_config, **kwargs)
        kwargs = arguments(log); kwargs['day_ids'][:] = 0
        with self.assertRaises(ValueError): HistoricalLog(log.public_config, **kwargs)
        public = log.public_config; public['charging_utility_intercept'] = 3.
        with self.assertRaises(ValueError): HistoricalLog(public, **arguments(log))
        with self.assertRaises(ValueError): HistoryFitConfig(choice_ridge=-1.).validate()
        with self.assertRaises(ValueError): HistoryFitConfig(max_iterations=True).validate()

    def test_weakly_identified_data_is_flagged_without_true_parameter_fallback(self):
        log = synthetic_log(days=3, rate=3.)
        kwargs = arguments(log)
        kwargs['mobile_wait_hours'][:] = 0.; kwargs['fixed_wait_hours'][:] = 0.
        kwargs['fixed_travel_km'][:] = 0.
        constant_attributes = HistoricalLog(log.public_config, **kwargs)
        model, report = fit_history_model(constant_attributes)
        self.assertTrue(report['choice']['weakly_identified'])
        self.assertLess(report['choice']['fisher_rank'], 5)
        self.assertIsNone(report['choice']['approximate_coefficient_standard_errors'])
        np.testing.assert_array_equal(model.choice_coefficients[2:], np.zeros(3))

    def test_loader_rejects_hostile_shape_before_allocating(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'hostile.npz'
            buffer = io.BytesIO()
            np.lib.format.write_array_header_1_0(buffer,
                dict(descr='<f8', fortran_order=False, shape=(10**12,)))
            with zipfile.ZipFile(path, 'w') as archive:
                for name in ('arrival_rates', 'request_bounds', 'choice_coefficients', 'metadata'):
                    archive.writestr(name+'.npy', buffer.getvalue())
            with self.assertRaises(ValueError): HistoricalModel.load_checkpoint(path)

    def test_public_adapter_has_only_neutral_behavior_and_fit_uses_no_rng(self):
        c = public_scenario(public_model_config(scenario()))
        self.assertEqual(c.charging_utility_intercept, 0.)
        np.testing.assert_array_equal(c.resolved_arrival_rates, np.zeros((3, 4)))
        np.random.seed(17); before = np.random.get_state()
        fit_history_model(synthetic_log(days=2, rate=1.))
        after = np.random.get_state()
        for a, b in zip(before, after): np.testing.assert_array_equal(a, b)

    def test_observable_holdout_diagnostics_include_zero_counts_and_probabilities(self):
        holdout = synthetic_log(seed=305, days=8)
        report = self.model.diagnostics(holdout)
        self.assertFalse(report['uses_behavioral_truth'])
        self.assertGreater(report['choice']['mean_multiclass_brier_score'], 0.)
        self.assertLess(report['choice']['mean_multiclass_brier_score'], 2.)
        self.assertAlmostEqual(sum(report['choice']['predicted_alternative_shares']), 1.)
        self.assertAlmostEqual(sum(report['choice']['observed_alternative_shares']), 1.)
        self.assertGreater(report['request_energy']['fraction_in_fitted_range'], .99)
        self.assertTrue(np.isfinite(report['demand']['mean_poisson_deviance']))
        zero = HistoricalModel(holdout.public_config, np.zeros((3, 4)), (5., 15.),
            self.model.choice_coefficients, {})
        missed = zero.diagnostics(holdout)
        self.assertGreater(missed['demand']['positive_counts_at_zero_predicted_rate'], 0)
        self.assertTrue(np.isfinite(missed['demand']['mean_poisson_deviance']))
        self.assertEqual(missed['poisson_rate_floor'], 1e-12)

    def test_real_history_fit_trains_ppo_and_reloads_restricted_checkpoints(self):
        from charging_env.rl.history_control import collect_historical_log
        from charging_env.rl.paper_ppo import PaperPPOConfig, PaperPPOTrainer
        truth = replace(scenario(), charging_utility_intercept=3.)
        log = collect_historical_log(truth, [4500000000, 4500000002], start_day=0)
        fitted, _ = fit_history_model(log)
        c = fitted.scenario
        self.assertTrue(all(type(value) is float for row in c.arrival_rates for value in row))
        self.assertTrue(all(type(value) is float for value in c.mean_requests_by_zone))
        def native(value):
            self.assertNotIsInstance(value, np.generic)
            if isinstance(value, dict):
                for child in value.values(): native(child)
            elif isinstance(value, (list, tuple)):
                for child in value: native(child)
        native(asdict(c))
        config = PaperPPOConfig(seed=7, updates=1, episodes_per_update=1, epochs=1,
            minibatch_size=8, hidden_size=8, torch_threads=1, eval_episodes=1,
            eval_interval=1, eval_seed_base=4500001001, training_seed_base=2250002000,
            warmstart_seed_base=2250003000, warmstart_episodes=1, warmstart_epochs=1,
            warmstart_minibatch_size=8, warmstart_dispatch_solver='coordinate').validate()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'model.npz'; fitted.save_checkpoint(path)
            reloaded_model = HistoricalModel.load_checkpoint(path)
            self.assertEqual(reloaded_model.scenario, c)
            self.assertEqual(reloaded_model.provenance()['model_sha256'], fitted.provenance()['model_sha256'])
            output = Path(temporary)/'ppo'
            with redirect_stdout(io.StringIO()):
                summary = PaperPPOTrainer(reloaded_model.scenario, config).train(output)
            self.assertTrue(summary['checkpoint_reload_matches'])
            self.assertEqual(summary['update_count'], 1)
            for filename in ('checkpoint.pt', 'best_checkpoint.pt', 'warmstart_checkpoint.pt'):
                # load_checkpoint uses torch.load(weights_only=True), with no
                # NumPy safe-global allowlist or unrestricted pickle fallback.
                checkpoint = PaperPPOTrainer.load_checkpoint(output/filename)
                self.assertEqual(checkpoint.scenario, c)
                self.assertEqual(checkpoint.config, config)
                native(asdict(checkpoint.scenario))


if __name__ == '__main__': unittest.main()
