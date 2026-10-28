"""Fitted-model control, real current observations and complete quote logging."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from charging_env.benchmarks.evaluation import evaluate_policy
from charging_env.benchmarks.policies import service_first_dispatch
from charging_env.config import Scenario
from charging_env.rl.adaptive_pricing import LearnedPricingPolicy, PARAMETER_COUNT
from charging_env.rl.history_control import (
    DISPATCH_FIELDS, collect_historical_log, collect_historical_logs,
    evaluate_history_policy, history_dispatch_observation, history_pricing_observation,
    model_scenario_sha256,
)
from charging_env.rl.history_model import (
    HistoricalLog, HistoricalModel, fit_history_model, public_model_config, public_scenario,
)
from charging_env.rl.priced_rollout import PricedRolloutADPPolicy
from charging_env.rl.rollout_adp import RolloutADPConfig
from charging_env.simulator import ChargingSimulator


PRICE_KEYS = {'stage','period','locations','energy_kwh','arrival_rates',
    'fcs_retail_prices','fcs_discharge_prices','grid_buy_price','pricing_mask'}


def scenario(**changes):
    base = dict(horizon=3, zone_xy_km=((0.,0.),(2.,0.),(4.,0.)), fcs_zones=(0,),
        initial_locations=(0,2), initial_energy_kwh=(30.,25.),
        mean_requests_by_zone=(2.,3.,0.), arrival_profile=(1.,1.,1.),
        fcs_retail_per_kwh=(.45,), grid_buy_per_kwh=(.15,.16,.17),
        grid_sell_per_kwh=(.1,.11,.12), fcs_ev_capacity_kwh=(30.,),
        fcs_fleet_exchange_capacity_kwh=(30.,), charging_utility_intercept=3.,
        terminal_value_per_stored_kwh=.05)
    return replace(Scenario(), **(base | changes)).validate()


def fitted(truth):
    return replace(truth, arrival_rates=((.3,.4,.2),)*truth.horizon,
        request_min_kwh=4.5, request_max_kwh=14.5, charging_utility_intercept=.2,
        price_sensitivity_per_dollar=1.4, mobile_wait_cost_per_hour=.6,
        fixed_wait_cost_per_hour=.7, fixed_travel_cost_per_km=.1).validate()


def without_timing(value):
    if isinstance(value, dict):
        return {key:without_timing(item) for key,item in value.items()
                if key not in ('mean_policy_decision_ms','seconds','history_boundary')}
    if isinstance(value, list):
        return [without_timing(item) for item in value]
    return value


class Rule:
    def __init__(self, model):
        self.scenario = model
        self.seen_prices, self.seen_dispatch, self.resets = [],[],[]

    def reset_episode(self, seed, action_seed):
        self.resets.append((seed,action_seed))

    def prices(self, obs, rng):
        self.seen_prices.append(deepcopy(obs))
        return np.full(self.scenario.n_zones,4,dtype=np.int64)

    def dispatch(self, obs, rng):
        self.seen_dispatch.append(deepcopy(obs))
        return service_first_dispatch(obs,self.scenario,.2)


class HistoryControlTests(unittest.TestCase):
    def test_pricing_forecast_poison_and_latent_fields_are_ignored(self):
        truth = scenario(); model = fitted(truth)
        raw = ChargingSimulator(truth).reset(seed=1001)[0]
        expected = history_pricing_observation(raw,model)
        poisoned = deepcopy(raw)
        poisoned.update(arrival_rates=np.array([np.nan]), request_details='future secret',
            requests_by_zone=np.full(3,999), potential_energy_kwh=np.full(3,999.),
            fcs_wait_hours=np.array([999.]), pricing_mask=np.ones((1,1)),
            mobile_available=np.zeros(3), eligible_mcvs='poison', future_realization='secret')
        actual = history_pricing_observation(poisoned,model)
        self.assertEqual(set(actual),PRICE_KEYS)
        for name in PRICE_KEYS:
            np.testing.assert_array_equal(actual[name],expected[name])
        np.testing.assert_array_equal(actual['arrival_rates'],model.resolved_arrival_rates[0])
        actual['energy_kwh'][0] = -1.
        self.assertGreater(raw['energy_kwh'][0],0.)

    def test_dispatch_preserves_current_observations_only(self):
        truth=scenario(); model=fitted(truth); sim=ChargingSimulator(truth)
        sim.reset(seed=1003); raw=sim.quote_price_levels(np.full(3,4))[0]
        raw['future_realization']='must not leak'; raw['arrival_rates']=np.array([np.nan])
        observed=history_dispatch_observation(raw,model)
        self.assertEqual(set(observed),set(DISPATCH_FIELDS))
        for key in DISPATCH_FIELDS:
            if key != 'arrival_rates': np.testing.assert_array_equal(observed[key],raw[key])
        np.testing.assert_array_equal(observed['arrival_rates'],model.resolved_arrival_rates[0])
        observed['action_mask'].fill(0)
        self.assertTrue(raw['action_mask'].any())
        with self.assertRaises(ValueError): history_pricing_observation(raw,model)

    def test_private_truth_full_reward_ledger_parity(self):
        truth=scenario(); model=fitted(truth); controllers=[]; hooks=[]
        def factory(value):
            self.assertIsNot(value,truth); self.assertIsNot(value,model)
            self.assertEqual(value,model)
            controllers.append(Rule(value)); return controllers[-1]
        result=evaluate_history_policy(truth,model,[1011,1013],factory,
            action_seed=99,label='boundary',on_decision=hooks.append)
        direct=Rule(model)
        reference=evaluate_policy(truth,[1011,1013],direct.prices,direct.dispatch,
            action_seed=99,label='boundary',on_episode_start=direct.reset_episode)
        self.assertEqual(without_timing(result),without_timing(reference))
        self.assertEqual(controllers[0].resets,[(1011,99),(1013,99)])
        self.assertEqual(len(hooks),6); self.assertEqual(result['history_boundary']['model_queries'],0)
        for obs in controllers[0].seen_prices:
            self.assertEqual(set(obs),PRICE_KEYS)
        for obs in controllers[0].seen_dispatch:
            np.testing.assert_array_equal(obs['arrival_rates'],model.resolved_arrival_rates[int(obs['period'][0])])
        self.assertEqual(result['invalid_actions'],0)

    def test_true_forecast_never_reaches_callbacks_even_when_invalid(self):
        truth=scenario(); model=fitted(truth); original=ChargingSimulator.observe
        def poison(sim):
            obs=original(sim); obs['arrival_rates']=np.full(3,np.nan)
            obs['hidden_next_arrivals']='forbidden'
            return obs
        with patch.object(ChargingSimulator,'observe',poison):
            result=evaluate_history_policy(truth,model,[1021],Rule)
        self.assertEqual(result['invalid_actions'],0)

    def test_rollout_samples_only_fitted_model_not_true_future(self):
        truth=scenario(); model=fitted(truth); controllers=[]; future_models=[]; live=[]
        class Live(ChargingSimulator):
            def __init__(self,c):
                self.steps=0; super().__init__(c); live.append(self)
            def dispatch(self,actions):
                self.steps+=1; return super().dispatch(actions)
        def future(c):
            self.assertEqual(c,model); self.assertNotEqual(c,truth)
            future_models.append(c); return ChargingSimulator(c)
        def factory(c):
            coefficients=np.zeros(PARAMETER_COUNT); coefficients[0]=.6
            policy=PricedRolloutADPPolicy(c,LearnedPricingPolicy(c,coefficients),
                RolloutADPConfig(seed=7,rollout_samples=2))
            controllers.append(policy); return policy
        with patch('charging_env.benchmarks.evaluation.ChargingSimulator',Live), \
             patch('charging_env.rl.priced_rollout.ChargingSimulator',future):
            result=evaluate_history_policy(truth,model,[1031],factory)
        self.assertEqual(len(live),1); self.assertIs(live[0].config,truth)
        self.assertEqual(live[0].steps,truth.horizon)
        self.assertGreater(len(future_models),0)
        self.assertEqual(result['history_boundary']['model_queries'],len(future_models))
        self.assertEqual(controllers[0].statistics['future_model_queries'],len(future_models))
        for row in result['history_boundary']['decisions']:
            self.assertEqual(row['model_scenario_sha256'],model_scenario_sha256(model))
        for row in controllers[0].decisions:
            seeds=np.asarray(row['future_draw_seeds'],dtype=np.int64)
            self.assertTrue((seeds % 2 == 0).all()); self.assertFalse((seeds==1031).any())

    def test_mismatched_models_and_mutation_fail_closed(self):
        truth=scenario(); model=fitted(truth)
        for factory in (lambda c:Rule(truth),
                        lambda c:self._with_training_scenario(Rule(c),truth),
                        lambda c:self._with_pricer(Rule(c),Rule(truth))):
            with self.assertRaisesRegex(ValueError,'fitted historical model'):
                evaluate_history_policy(truth,model,[1041],factory)
        class Mutating(Rule):
            def dispatch(self,obs,rng):
                result=super().dispatch(obs,rng); self.scenario=truth; return result
        with self.assertRaisesRegex(ValueError,'fitted historical model'):
            evaluate_history_policy(truth,model,[1041],Mutating)
        with self.assertRaisesRegex(ValueError,'public physics'):
            evaluate_history_policy(replace(truth,mobile_service_kw=12.),model,[1041],Rule)

    @staticmethod
    def _with_training_scenario(policy,truth):
        policy.training_scenario=truth; return policy

    @staticmethod
    def _with_pricer(policy,pricer):
        policy.pricing_policy=pricer; return policy

    def test_ppo_loaded_from_fitted_model_keeps_its_encoders(self):
        from charging_env.rl.paper_ppo import PaperPPOConfig,PaperPPOTrainer
        truth=scenario(); model=fitted(truth)
        trainer=PaperPPOTrainer(model,PaperPPOConfig(updates=1,episodes_per_update=1,
            epochs=1,minibatch_size=4,hidden_size=8,eval_episodes=1,warmstart_episodes=0))
        # Exercise a real fitted-model training update, then checkpoint loading.
        trainer.update(trainer.collect_rollout())
        with tempfile.TemporaryDirectory() as folder:
            checkpoint=Path(folder)/'policy.pt'; trainer.save_checkpoint(checkpoint)
            loaded=PaperPPOTrainer.load_checkpoint(checkpoint)
            loaded.set_evaluation_mode(True)
            result=evaluate_history_policy(truth,model,[1051],lambda c:loaded)
            self.assertEqual(result['invalid_actions'],0)
            self.assertEqual(loaded.training_scenario,model)
            self.assertEqual(loaded.pricing_actor.encoder.scenario,model)
            self.assertEqual(loaded.dispatch_actor.encoder.scenario,model)
            loaded.pricing_actor.encoder.scenario=truth
            with self.assertRaisesRegex(ValueError,'encoder.scenario'):
                evaluate_history_policy(truth,model,[1051],lambda c:loaded)

    def test_logger_uses_only_public_adapter_and_retains_outside_and_zeros(self):
        truth=scenario(charging_utility_intercept=-10.)
        public=public_model_config(truth); placeholder=public_scenario(public); calls=[]
        original=service_first_dispatch
        def spy(obs,c,threshold):
            self.assertEqual(c,placeholder); self.assertNotEqual(c,truth)
            np.testing.assert_array_equal(obs['arrival_rates'],np.zeros(3))
            calls.append(deepcopy(obs)); return original(obs,c,threshold)
        with patch('charging_env.rl.history_control.service_first_dispatch',spy):
            log=collect_historical_log(truth,[1061,1063],logging_seed=89,start_day=120)
        self.assertEqual(log.public_config,public)
        np.testing.assert_array_equal(log.day_ids,[120,121])
        self.assertEqual(log.counts.shape,(2,3,3)); self.assertTrue((log.counts[:,:,2]==0).all())
        self.assertEqual(int(log.counts.sum()),len(log.chosen))
        self.assertGreater(int((log.chosen==truth.n_fcs+1).sum()),0)
        self.assertEqual(len(calls),6)
        self.assertTrue(log.available[:,-1].all()); self.assertTrue((log.alternative_payment[:,-1]==0).all())
        self.assertNotIn('arrival_rates',log.public_config)
        self.assertNotIn('price_sensitivity_per_dollar',log.public_config)
        self.assertNotIn('request_min_kwh',log.public_config)
        # Each generated quote/payment/choice is reproduced through the original simulator.
        neutral=public_scenario(log.public_config)
        for day,seed in enumerate((1061,1063)):
            sim=ChargingSimulator(truth); sim.reset(seed=seed)
            rng=np.random.default_rng(np.random.SeedSequence([89,120+day,0x48495354]))
            for period in range(truth.horizon):
                levels=rng.integers(0,len(truth.price_levels),size=truth.n_zones)
                obs=sim.quote_price_levels(levels)[0]; select=(log.request_day==day)&(log.request_period==period)
                details=obs['request_details']; origins=details[:,0].astype(int)
                np.testing.assert_array_equal(log.chosen[select],details[:,2].astype(int))
                np.testing.assert_array_equal(log.request_energy_kwh[select],details[:,1])
                np.testing.assert_array_equal(log.alternative_payment[select,0],details[:,1]*obs['mobile_prices'][origins])
                sim.dispatch(original(history_dispatch_observation(obs,neutral),neutral,.2))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'quotes.npz'; log.save(path); restored=HistoricalLog.load(path)
            self.assertEqual(restored.provenance(),log.provenance())

    def test_empty_quote_days_and_chronology_validation(self):
        truth=scenario(arrival_rates=((0.,0.,0.),)*3)
        log=collect_historical_log(truth,[1071],start_day=5)
        self.assertEqual(log.counts.sum(),0); self.assertEqual(log.alternative_payment.shape,(0,3))
        for ids in ([1,1],[2,1],[-1,0],[1.5,2.5]):
            with self.assertRaises(ValueError):
                collect_historical_logs(truth,[1071,1073],day_ids=ids)

    def test_collected_history_fit_reload_and_private_truth_control(self):
        truth=scenario()
        log=collect_historical_log(truth,[1081,1083,1085,1087],logging_seed=17)
        model,report=fit_history_model(log)
        self.assertEqual(public_model_config(model.scenario),log.public_config)
        self.assertNotEqual(model.scenario,truth)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model.npz'; model.save_checkpoint(path)
            restored=HistoricalModel.load_checkpoint(path)
            self.assertEqual(restored.scenario,model.scenario)
            controllers=[]
            def factory(c):
                coefficients=np.zeros(PARAMETER_COUNT); coefficients[0]=.6
                controller=PricedRolloutADPPolicy(c,LearnedPricingPolicy(c,coefficients),
                    RolloutADPConfig(seed=19,rollout_samples=2))
                controllers.append(controller); return controller
            result=evaluate_history_policy(truth,restored.scenario,[1091],factory)
            self.assertEqual(result['invalid_actions'],0)
            self.assertEqual(controllers[0].scenario,restored.scenario)
            self.assertEqual(controllers[0].pricing_policy.scenario,restored.scenario)
            self.assertEqual(result['history_boundary']['model_scenario_sha256'],
                             model_scenario_sha256(restored.scenario))


if __name__=='__main__': unittest.main()
