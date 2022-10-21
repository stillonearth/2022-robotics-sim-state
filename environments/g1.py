import numpy as np

from gym import utils
from gym.envs.mujoco import MujocoEnv
from gym.spaces import Box, Dict

import os
import mujoco
import math

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
        healthy_z_range=(0.12, 1.0),
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.05,
        exclude_current_positions_from_observation=True,
        max_steps=500,
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
        self.max_steps = max_steps
        self.n_step = 0

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        obs_shape = 35
        if not exclude_current_positions_from_observation:
            obs_shape += 2
        if use_contact_forces:
            obs_shape += 84

        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float64
        )

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
        return terminated or (self.n_step > self.max_steps)

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
        self.n_step += 1
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
        qvel = self.init_qvel + self._reset_noise_scale * np.random.standard_normal(
            self.model.nv
        )
        self.set_state(qpos, qvel)
        observation = self._get_obs()
        self.n_step = 0
        return observation

    def viewer_setup(self):
        assert self.viewer is not None
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)


def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians


class G1ControlEnv(G1DistanceEnv):
    def __init__(
        self,
        ctrl_cost_weight=0.1,
        use_contact_forces=True,
        contact_cost_weight=5e-3,
        healthy_reward=0.2,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.12, 1.0),
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.05,
        exclude_current_positions_from_observation=True,
        mode="direction",
        max_steps=500,
        task={},
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
            **kwargs,
        )

        self._task = task
        self._goal_dir = task.get("direction", 1)
        self._goal_orientation = task.get("orientation", 1)
        self._action_scaling = None
        self.world_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.base_vec_z = np.array([0.0, 0.0, 1.0])
        self.base_vec_x = np.array([1.0, 1.0, 0.0])
        self.mode = mode

        obs_shape = 35
        if not exclude_current_positions_from_observation:
            obs_shape += 2
        if use_contact_forces:
            obs_shape += 84

        obs_shape += 2

        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float64
        )

        xml_path = os.path.abspath("./mujoco_menagerie/unitree_a1/scene.xml")
        MujocoEnv.__init__(
            self, xml_path, 5, observation_space=observation_space, **kwargs
        )

    def rotation_angles(self, matrix, order):
        """
        input
            matrix = 3x3 rotation matrix (numpy array)
            oreder(str) = rotation order of x, y, z : e.g, rotation XZY -- 'xzy'
        output
            theta1, theta2, theta3 = rotation angles in rotation order
        """
        r11, r12, r13 = matrix[0]
        r21, r22, r23 = matrix[1]
        r31, r32, r33 = matrix[2]

        if order == "xzx":
            theta1 = np.arctan(r31 / r21)
            theta2 = np.arctan(r21 / (r11 * np.cos(theta1)))
            theta3 = np.arctan(-r13 / r12)

        elif order == "xyx":
            theta1 = np.arctan(-r21 / r31)
            theta2 = np.arctan(-r31 / (r11 * np.cos(theta1)))
            theta3 = np.arctan(r12 / r13)

        elif order == "yxy":
            theta1 = np.arctan(r12 / r32)
            theta2 = np.arctan(r32 / (r22 * np.cos(theta1)))
            theta3 = np.arctan(-r21 / r23)

        elif order == "yzy":
            theta1 = np.arctan(-r32 / r12)
            theta2 = np.arctan(-r12 / (r22 * np.cos(theta1)))
            theta3 = np.arctan(r23 / r21)

        elif order == "zyz":
            theta1 = np.arctan(r23 / r13)
            theta2 = np.arctan(r13 / (r33 * np.cos(theta1)))
            theta3 = np.arctan(-r32 / r31)

        elif order == "zxz":
            theta1 = np.arctan(-r13 / r23)
            theta2 = np.arctan(-r23 / (r33 * np.cos(theta1)))
            theta3 = np.arctan(r31 / r32)

        elif order == "xzy":
            theta1 = np.arctan(r32 / r22)
            theta2 = np.arctan(-r12 * np.cos(theta1) / r22)
            theta3 = np.arctan(r13 / r11)

        elif order == "xyz":
            theta1 = np.arctan(-r23 / r33)
            theta2 = np.arctan(r13 * np.cos(theta1) / r33)
            theta3 = np.arctan(-r12 / r11)

        elif order == "yxz":
            theta1 = np.arctan(r13 / r33)
            theta2 = np.arctan(-r23 * np.cos(theta1) / r33)
            theta3 = np.arctan(r21 / r22)

        elif order == "yzx":
            theta1 = np.arctan(-r31 / r11)
            theta2 = np.arctan(r21 * np.cos(theta1) / r11)
            theta3 = np.arctan(-r23 / r22)

        elif order == "zyx":
            theta1 = np.arctan(r21 / r11)
            theta2 = np.arctan(-r31 * np.cos(theta1) / r11)
            theta3 = np.arctan(r32 / r33)

        elif order == "zxy":
            theta1 = np.arctan(-r12 / r22)
            theta2 = np.arctan(r32 * np.cos(theta1) / r22)
            theta3 = np.arctan(-r31 / r33)

        return (theta1, theta2, theta3)

    def get_euler_angles(self, name="trunk"):
        now_pos_mat = np.array(self.data.body(name).xmat).reshape((3, 3))
        return self.rotation_angles(now_pos_mat, "xyz")

    def step(self, action):
        xy_position_before = self.get_body_com("trunk")[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("trunk")[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity
        abs_velocity = np.linalg.norm(xy_velocity)

        goal_direction = np.array([np.cos(self._goal_dir), np.sin(self._goal_dir)])
        projected_speed = 10 * np.dot(xy_velocity, goal_direction)

        rot_x, rot_y, rot_z = self.get_euler_angles()

        goal_orientation = np.array([np.cos(self._goal_orientation), np.sin(self._goal_orientation)])
        orientation = np.array([np.cos(rot_z), np.sin(rot_z)])
        projected_orientation = 10 * np.dot(orientation, goal_orientation)
        trunk_orientation_reward = (np.cos(rot_x) / 2 + np.cos(rot_y) / 2) / 5

        healthy_reward = self.healthy_reward

        if self.mode == "direction":
            rewards = projected_speed
        elif self.mode == "orientation":
            rewards = projected_orientation
        elif self.mode == "direction+orientation":
            rewards = projected_speed + projected_orientation + abs_velocity

        rewards += trunk_orientation_reward + healthy_reward

        costs = ctrl_cost = self.control_cost(action)

        terminated = self.terminated
        observation = self._get_obs()
        info = {
            "reward_orientation": projected_orientation,
            "reward_speed": projected_speed,
            "reward_ctrl": -ctrl_cost,
            "reward_survive": healthy_reward,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
        }
        if self._use_contact_forces:
            contact_cost = self.contact_cost
            costs += contact_cost
            info["reward_ctrl"] = -contact_cost

        reward = rewards - costs

        self.renderer.render_step()
        self.n_step += 1
        return observation, reward, terminated, False, info

    def reset_model(self):
        task = self.sample_tasks(1)[0]
        self.reset_task(task)
        self.n_step = 0
        return super().reset_model()

    def _get_obs(self):
        model_obs = super()._get_obs()
        return np.concatenate(
            [
                model_obs,
                [
                    self._goal_dir,
                    self._goal_orientation,
                ],
            ]
        )

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        rot_x, rot_y, rot_z = self.get_euler_angles()
        is_healthy = (
            np.isfinite(state).all()
            and min_z <= state[2] <= max_z
            and rot_x <= np.pi / 2
            and rot_y <= np.pi / 2
        )
        return is_healthy

    @property
    def terminated(self):
        terminated = super().terminated or (self.n_step > self.max_steps)
        return terminated

    def sample_tasks(self, num_tasks):
        directions = np.random.uniform(0, 2 * np.pi, size=(num_tasks,))
        orientations = np.random.uniform(0, 2 * np.pi, size=(num_tasks,))
        velocities = np.random.uniform(0, 2.0, size=(num_tasks,))

        if self.mode == "direction":
            orientations *= 0
        elif self.mode == "orientation":
            directions *= 0

        return [
            {
                "direction": d,
                "orientation": o,
                "velocity": v,
            }
            for (d, o, v) in zip(directions, orientations, velocities)
        ]

    def reset_task(self, task):
        self._task = task
        self._goal_dir = task["direction"]
        self._goal_orientation = task["orientation"]
        self._goal_velocity = task["velocity"]
