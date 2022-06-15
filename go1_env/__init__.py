import gym

from gym import utils
from gym.envs.mujoco import mujoco_env

import numpy as np

class G1Env(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, xml_path=None):
        xml_path = "/mnt/c/Users/Sergei/git/walking-robot-neural-control/2022-robotics-sim-state/go1_env/xml/go1.xml"
        mujoco_env.MujocoEnv.__init__(
            self, xml_path, 1)
        utils.EzPickle.__init__(self)

    def step(self, a):
        xposbefore = self.get_body_com("trunk")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("trunk")[0]
        self.render()

        forward_reward = (xposafter - xposbefore) / self.dt
        ctrl_cost = 0.5 * np.square(a).sum()
        contact_cost = (
            0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        )
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return (
            ob,
            reward,
            done,
            dict(
                reward_forward=forward_reward,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
            ),
        )

    def _get_obs(self):
        return np.concatenate(
            [
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat,
                np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            ]
        )

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

gym.envs.register(
     id='G1-v0',
     entry_point='go1_env:G1Env',
     max_episode_steps=10000,
     kwargs={},
)
