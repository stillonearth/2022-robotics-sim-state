__all__ = ["g1_distance", "g1_direction"]

import gym


gym.envs.register(
     id='G1Dist-v0',
     entry_point='environments.g1_distance:G1DistanceEnv',
     max_episode_steps=10000,
     kwargs={},
)

gym.envs.register(
     id='G1Dir-v0',
     entry_point='environments.g1_direction:G1DirectionEnv',
     max_episode_steps=10000,
     kwargs={},
)