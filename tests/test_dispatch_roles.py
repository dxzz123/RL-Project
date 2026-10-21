"""Local dispatch-role encoding, with unchanged legacy policy layouts."""
from copy import deepcopy
from dataclasses import replace
import unittest

import numpy as np

from charging_env.config import Scenario
from charging_env.simulator import ChargingSimulator

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Install the train extra to test dispatch roles')
class DispatchRoleTests(unittest.TestCase):
    def setUp(self):
        from charging_env.rl.networks import DispatchActor, _ObservationEncoder
        self.Actor = DispatchActor
        self.Encoder = _ObservationEncoder
        torch.set_num_threads(1)
        torch.manual_seed(113)
        self.scenario = replace(Scenario(), initial_locations=(0, 0),
                                initial_energy_kwh=(40., 40.),
                                charging_utility_intercept=3.).validate()

    def observations(self, counts=(3, 2, 3, 2)):
        simulator = ChargingSimulator(self.scenario)
        simulator.reset(seed=83, options={'requests': list(counts)})
        post = simulator.quote_price_levels([4] * self.scenario.n_zones)[0]
        return post, [simulator.local_observation(m) for m in range(self.scenario.n_mcv)]

    def test_collocated_vehicles_only_gain_their_supplied_role_identity(self):
        _, observations = self.observations()
        legacy = self.Actor(self.scenario, hidden_size=16)
        roles = self.Actor(self.scenario, hidden_size=16, include_agent_id=True)
        original = deepcopy(observations)
        legacy_dense = np.stack([legacy.encoder._dense(obs) for obs in observations])
        role_dense = np.stack([roles.encoder._dense(obs) for obs in observations])
        np.testing.assert_array_equal(legacy_dense[0], legacy_dense[1])
        np.testing.assert_array_equal(role_dense[:, :-self.scenario.n_mcv], legacy_dense)
        np.testing.assert_array_equal(role_dense[:, -self.scenario.n_mcv:], np.eye(2))
        self.assertFalse(np.array_equal(role_dense[0], role_dense[1]))
        # The public local snapshots remain untouched; no simulator is retained.
        for before, after in zip(original, observations):
            for key in before:
                np.testing.assert_array_equal(before[key], after[key])

    def test_role_feature_is_local_only_and_requires_the_public_identity(self):
        with self.assertRaisesRegex(ValueError, 'only supported for local'):
            self.Encoder(self.scenario, 16, local=False, include_agent_id=True)
        post, local = self.observations()
        actor = self.Actor(self.scenario, hidden_size=16, include_agent_id=True)
        with self.assertRaisesRegex(ValueError, 'local_observation'):
            actor.sample([post])
        missing = deepcopy(local[0])
        missing.pop('agent_id')
        with self.assertRaises(KeyError):
            actor.sample([missing])
        # Adding a role does not require global masks or a full-fleet energy vector.
        self.assertEqual(local[0]['locations'].shape, (1,))
        self.assertEqual(local[0]['energy_kwh'].shape, (1,))
        self.assertEqual(local[0]['action_mask'].shape[0], 1)
        result = actor.sample(local, deterministic=True)
        self.assertTrue(torch.isfinite(result['log_prob']).all())

    def test_identity_flag_rejects_nonbool_values(self):
        for invalid in (0, 1, 'true', None, np.bool_(True)):
            with self.subTest(value=invalid):
                with self.assertRaisesRegex(ValueError, 'include_agent_id must be a bool'):
                    self.Actor(self.scenario, hidden_size=16, include_agent_id=invalid)
                with self.assertRaisesRegex(ValueError, 'include_agent_id must be a bool'):
                    self.Encoder(self.scenario, 16, local=True, include_agent_id=invalid)

    def test_default_policy_preserves_exact_disabled_state_dict_and_outputs(self):
        _, local = self.observations()
        with torch.random.fork_rng():
            torch.manual_seed(223)
            default = self.Actor(self.scenario, hidden_size=16, final_activation='silu')
            torch.manual_seed(223)
            disabled = self.Actor(self.scenario, hidden_size=16, final_activation='silu',
                                  include_agent_id=False)
        self.assertEqual(list(default.state_dict()), list(disabled.state_dict()))
        for name, tensor in default.state_dict().items():
            torch.testing.assert_close(tensor, disabled.state_dict()[name], rtol=0, atol=0)
        # An old-shaped state dict still loads strictly with no new tensor keys.
        clone = self.Actor(self.scenario, hidden_size=16, final_activation='silu')
        clone.load_state_dict(default.state_dict(), strict=True)
        with torch.no_grad():
            for actual in (disabled(local), clone(local)):
                for expected_tensor, actual_tensor in zip(default(local), actual):
                    torch.testing.assert_close(expected_tensor, actual_tensor, rtol=0, atol=0)

    def test_enabled_policy_keeps_masks_and_variable_request_padding_valid(self):
        _, empty = self.observations(counts=(0, 0, 0, 0))
        _, large = self.observations(counts=(129, 127, 0, 0))
        actor = self.Actor(self.scenario, hidden_size=16, final_activation='silu',
                           include_agent_id=True)
        observations = [empty[0], large[1]]
        with torch.no_grad():
            logits, concentrations = actor(observations)
            mask = torch.as_tensor(np.stack([obs['action_mask'] for obs in observations]),
                                   dtype=torch.bool).reshape(2, -1)
            self.assertTrue(torch.isfinite(logits[mask]).all())
            self.assertTrue(torch.isneginf(logits[~mask]).all())
            self.assertTrue(torch.isfinite(concentrations).all())
            sample = actor.sample(observations, deterministic=True)
            individual = actor.sample([empty[0]], deterministic=True)
            self.assertTrue(torch.isfinite(sample['log_prob']).all())
            self.assertTrue(torch.isfinite(sample['entropy']).all())
            self.assertTrue(mask[torch.arange(2), sample['actions']].all())
            torch.testing.assert_close(sample['actions'][:1], individual['actions'])
            torch.testing.assert_close(sample['log_prob'][:1], individual['log_prob'],
                                       rtol=1e-5, atol=1e-6)
            replay = actor.evaluate_actions(observations, sample['actions'], sample['energy_fraction'])
            torch.testing.assert_close(replay['log_prob'], sample['log_prob'])
            torch.testing.assert_close(replay['entropy'], sample['entropy'])


if __name__ == '__main__':
    unittest.main()
