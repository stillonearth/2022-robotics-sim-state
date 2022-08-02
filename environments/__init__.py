__all__ = ["g1_distance"]

import gym
import envpool


gym.envs.register(
     id='G1Dist-v0',
     entry_point='environments.g1_distance:G1DistanceEnv',
     max_episode_steps=10000,
     kwargs={},
)
