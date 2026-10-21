"""Reject configuration drift and incomplete evidence before study execution."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from run_b3_improved_training import config_for, verify_training_inputs, verify_pilot_summary
from charging_env.config import Scenario


class StudyRunnerGuards(unittest.TestCase):
    def test_scenario_drift_is_rejected_for_pilot_and_full_runs(self):
        scenario=replace(Scenario(),charging_utility_intercept=3.)
        protocol={'scenario':json.loads(json.dumps(asdict(scenario))),
                  'configuration_by_seed':{'7':asdict(config_for(7))}}
        for pilot in (False,True):
            settings=config_for(7,pilot=pilot)
            verify_training_inputs(scenario,settings,protocol,7,pilot=pilot)
            with self.assertRaisesRegex(ValueError,'Scenario differs'):
                verify_training_inputs(replace(scenario,charging_utility_intercept=4.),
                                       settings,protocol,7,pilot=pilot)

    def test_config_drift_is_rejected(self):
        scenario=replace(Scenario(),charging_utility_intercept=3.)
        protocol={'scenario':json.loads(json.dumps(asdict(scenario))),
                  'configuration_by_seed':{'19':asdict(config_for(19))}}
        with self.assertRaisesRegex(ValueError,'configuration differs'):
            verify_training_inputs(scenario,replace(config_for(19),inventory_potential=0.),protocol,19)

    def test_incomplete_or_unhealthy_pilots_cannot_be_selected(self):
        summary={'total_updates':150,'training_episodes':2400,'training_physical_periods':28800,
                 'checkpoint_reload_matches':True,'all_metrics_finite':True,'invalid_evaluation_actions':0}
        verify_pilot_summary(summary,12)
        for key,value in [('total_updates',149),('training_episodes',2399),
                          ('training_physical_periods',28799),('checkpoint_reload_matches',False),
                          ('all_metrics_finite',False),('invalid_evaluation_actions',1)]:
            with self.subTest(key=key),self.assertRaises(ValueError):
                verify_pilot_summary(dict(summary,**{key:value}),12)


if __name__=='__main__':
    unittest.main()
