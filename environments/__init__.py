__all__ = ["g1"]

import gym


gym.envs.register(
    id="G1Dist-v1",
    entry_point="environments.g1:G1DistanceEnv",
    max_episode_steps=10000,
    kwargs={},
)

gym.envs.register(
    id="G1ControlEnv-v1",
    entry_point="environments.g1:G1ControlEnv",
    max_episode_steps=10000,
    kwargs={},
)
