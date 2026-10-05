"""Verify the Gymnasium interface; no training or learning."""
from gymnasium.utils.env_checker import check_env

from charging_env.gym_env import ChargingEnv


if __name__ == "__main__":
    env = ChargingEnv(render_mode="ansi")
    check_env(env, skip_render_check=True)
    env.reset(seed=7)
    print("Gymnasium API checks passed.")
    print(env.render())
    env.close()

