import numpy as np

from gym import utils
from gym.envs.mujoco import MujocoEnv
from gym.spaces import Box, Dict

import os

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


class G1DistanceEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "single_rgb_array",
            "single_depth_array",
        ],
        "render_fps": 100,
    }

    def __init__(
        self,
        ctrl_cost_weight=0.5,
        use_contact_forces=True,
        contact_cost_weight=5e-4,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.1, 1.0),
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.05,
        exclude_current_positions_from_observation=True,
        **kwargs
    ):
        utils.EzPickle.__init__(**locals())

        xml_path = os.path.abspath("./mujoco_menagerie/unitree_a1/scene.xml")

        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._contact_force_range = contact_force_range
        self._reset_noise_scale = reset_noise_scale
        self._use_contact_forces = use_contact_forces

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        obs_shape = 35
        if not exclude_current_positions_from_observation:
            obs_shape += 2
        if use_contact_forces:
            obs_shape += 84

        observation_space = Box(low=-np.inf, high=np.inf,
                                shape=(obs_shape,), dtype=np.float64)
        self.desired_goal = np.ones((obs_shape,))

        MujocoEnv.__init__(
            self, xml_path, 5, observation_space=observation_space, **kwargs
        )

    @property
    def healthy_reward(self):
        return (
            float(self.is_healthy or self._terminate_when_unhealthy)
            * self._healthy_reward
        )

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def contact_forces(self):
        raw_contact_forces = self.data.cfrc_ext
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    @property
    def contact_cost(self):
        contact_cost = self._contact_cost_weight * np.sum(
            np.square(self.contact_forces)
        )
        return contact_cost

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = np.isfinite(state).all() and min_z <= state[2] <= max_z
        return is_healthy

    @property
    def terminated(self):
        terminated = not self.is_healthy if self._terminate_when_unhealthy else False
        return terminated

    def step(self, action):
        xy_position_before = self.get_body_com("trunk")[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("trunk")[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        forward_reward = x_velocity * 10.0
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward

        costs = ctrl_cost = self.control_cost(action)

        terminated = self.terminated
        observation = self._get_obs()
        info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_survive": healthy_reward,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
        }
        if self._use_contact_forces:
            contact_cost = self.contact_cost
            costs += contact_cost
            info["reward_ctrl"] = -contact_cost

        reward = rewards - costs

        self.renderer.render_step()
        return observation, reward, terminated, False, info

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        observation = None
        if self._use_contact_forces:
            contact_force = self.contact_forces.flat.copy()
            observation = np.concatenate((position, velocity, contact_force))
        else:
            observation = np.concatenate((position, velocity))

        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + np.random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale *
            np.random.standard_normal(self.model.nv)
        )
        self.set_state(qpos, qvel)
        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        assert self.viewer is not None
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)


class G1GoalDistanceEnv(G1DistanceEnv):

    def __init__(
        self,
        ctrl_cost_weight=0.5,
        use_contact_forces=True,
        contact_cost_weight=5e-4,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.1, 1.0),
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.05,
        exclude_current_positions_from_observation=True,
        **kwargs
    ):
        super().__init__(
            ctrl_cost_weight,
            use_contact_forces,
            contact_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            contact_force_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            ** kwargs,
        )

        obs_shape = 35
        if not exclude_current_positions_from_observation:
            obs_shape += 2
        if use_contact_forces:
            obs_shape += 84

        observation_space = Dict({
            "observation": Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float64),
            "achieved_goal": Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float64),
            "desired_goal": Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float64),
        })

        self.desired_goal = np.ones((obs_shape,))

        xml_path = os.path.abspath("./mujoco_menagerie/unitree_a1/scene.xml")
        MujocoEnv.__init__(
            self, xml_path, 5, observation_space=observation_space, **kwargs
        )

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        observation = None
        if self._use_contact_forces:
            contact_force = self.contact_forces.flat.copy()
            observation = np.concatenate((position, velocity, contact_force))
        else:
            observation = np.concatenate((position, velocity))

        return {
            "observation": observation.copy(),
            "achieved_goal": observation.copy(),
            "desired_goal": self.desired_goal,
        }

    def compute_reward(self, achieved_goal, desired_goal, _info):
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return -(distance > 0).astype(np.float32)

    def reset(self):
        # Enforce that each G1DistanceEnv uses a Goal-compatible observation space.
        if not isinstance(self.observation_space, Dict):
            raise error.Error(
                'G1DistanceEnv requires an observation space of type gym.spaces.Dict')
        result = super(G1DistanceEnv, self).reset()
        for key in ['observation', 'achieved_goal', 'desired_goal']:
            if key not in result:
                raise error.Error(
                    'G1DistanceEnv requires the "{}" key to be part of the observation dictionary.'.format(key))
        return result


class G1GoalDistanceEnv(G1DistanceEnv):
    """G1 environment with target direction, as described in [1]. The 
    code is adapted from
    https://github.com/cbfinn/maml_rl/blob/9c8e2ebd741cb0c7b8bf2d040c4caeeb8e06cc95/rllab/envs/mujoco/ant_env_rand_direc.py

    The ant follows the dynamics from MuJoCo [2], and receives at each 
    time step a reward composed of a control cost, a contact cost, a survival 
    reward, and a reward equal to its velocity in the target direction. The 
    tasks are generated by sampling the target directions from a Bernoulli 
    distribution on {-1, 1} with parameter 0.5 (-1: backward, +1: forward).

    [1] Chelsea Finn, Pieter Abbeel, Sergey Levine, "Model-Agnostic 
        Meta-Learning for Fast Adaptation of Deep Networks", 2017 
        (https://arxiv.org/abs/1703.03400)
    [2] Emanuel Todorov, Tom Erez, Yuval Tassa, "MuJoCo: A physics engine for 
        model-based control", 2012 
        (https://homes.cs.washington.edu/~todorov/papers/TodorovIROS12.pdf)
    """

    def __init__(
        self,
            ctrl_cost_weight=0.5,
            use_contact_forces=True,
            contact_cost_weight=5e-4,
            healthy_reward=1.0,
            terminate_when_unhealthy=True,
            healthy_z_range=(0.1, 1.0),
            contact_force_range=(-1.0, 1.0),
            reset_noise_scale=0.05,
            exclude_current_positions_from_observation=True,
            task={},
            **kwargs
    ):
        self._task = task
        self._goal_dir = task.get('direction', 1)
        self._goal_orientation = task.get('orienation', 1)
        self._action_scaling = None

        super().__init__(
            ctrl_cost_weight,
            use_contact_forces,
            contact_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            contact_force_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            ** kwargs,
        )

    def step(self, action):
        xy_position_before = self.get_body_com("trunk")[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("trunk")[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity
        abs_velocity = np.linalg.norm(xy_velocity)

        goal = np.array([np.cos(self._goal_dir), np.sin(self._goal_dir)])
        projection = np.dot(xy_velocity, goal *
                            abs_velocity) / np.linalg.norm(goal)
        projection_norm = projection.norm()
        colinear = int((goal - projection_norm).norm() == 0)

        forward_reward = projection.norm() * (-1**(1 - colinear))
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward

        costs = ctrl_cost = self.control_cost(action)

        terminated = self.terminated
        observation = self._get_obs()
        info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_survive": healthy_reward,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
        }
        if self._use_contact_forces:
            contact_cost = self.contact_cost
            costs += contact_cost
            info["reward_ctrl"] = -contact_cost

        reward = rewards - costs

        self.renderer.render_step()
        return observation, reward, terminated, False, info

    def sample_tasks(self, num_tasks):
        directions = np.random.uniform(0, 2*np.pi, size=(num_tasks,))
        return [{
            'direction': direction,
            'orienation': 0,
        } for direction in directions]

    def meta_params(self):
        return np.array([self._goal_dir, self._goal_orientation])

    def reset_task(self, task):
        self._task = task
        self._goal_dir = task['direction']
        self._goal_orientation = task['orienation']
