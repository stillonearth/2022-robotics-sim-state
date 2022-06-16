[image1]: https://www.unitree.com/uploads/531_(3)_e17eb66c60.png "Unitree G1"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"
[image3]: https://raw.githubusercontent.com/cwiz/DRLND-Project-Continuous_Control/master/images/vpg-gradient.svg?sanitize=true "VPG-Loss"
[image4]: https://raw.githubusercontent.com/cwiz/DRLND-Project-Continuous_Control/master/images/vpg-algorithm.svg?sanitize=true "VPG-Algorithm"
[image5]: https://github.com/cwiz/DRLND-Project-Continuous_Control/blob/master/images/rewards-reacher.png?raw=true "Crawler-Rewards"
[image6]: https://raw.githubusercontent.com/cwiz/DRLND-Project-Continuous_Control/master/images/sac-algorithm.svg?sanitize=true "SAC-Algorithm"
[image7]: https://github.com/cwiz/DRLND-Project-Continuous_Control/blob/master/images/rewards-crawler.png?raw=true "Crawler-Crawler"

# Meta-Trained Continuous Control System for Robotic Dog Unitree G1

**Sergei Surovtsev**

## Project Description
This project involves applying policy optimization algorithms to continuous control task MuJoCo simulator and meta-learning 4-DoF control system for a robotic dog. 

![Unitree G1][image1]

### Ant-G1 Forward Movement Environment

**Goal**

* Move straight and avoid falling

**Observation**

* 119 variables corresponding to position, rotation, velocity, and angular velocities of each limb plus the acceleration and angular acceleration of the body

**Actions**

* Vector Action space: (Continuous) Size of 12, corresponding to target rotations for joints.

**Rewards**

* **speed**: forward speed
* **survive**: 1 for keeping body frame from contacting surface
* **control_cost**: ``0.5 * np.square(a).sum()``
* **contact_cost**: ``0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))``
* **REWARD = speed + survive - control_cost - contact_cost**

### Ant-G1 Directional Movement Environment

**Goal**

* Move to designated directional and avoid falling

**Observation**

*same as in previous task*

**Actions**

*same as in previous task*

**Rewards**

* **speed**: 1st derivative in desired direction
* **survive**: 0.05 for keeping body frame from contacting surface
* **control_cost**: ``0.5 * np.square(a).sum()``
* **contact_cost**: ``0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))``
* **REWARD = speed + survive - control_cost - contact_cost**
 
## Project Goals

* Evaluate state of robotics simulators (June 2022)
* Apply robotic simulator to consumer robot
* Investigate feasibility for synthesizing neural control with meta-learning approach

## Technical Formulation of Problem 

* Set up [MuJoCo Environment](https://mujoco.org/)
* Set up [mujoco_py](https://github.com/openai/mujoco-py)
* Complete Soft Actor Critic - G1 - Forward.ipynb
* Adapt [3] from Ant to G1 Environment

## Mathematical Models

* **Soft Actor-Critic Method** as described in [2] with minor changes in implementation around PyTorch 2.0 inplace gradients.
* **Trust-Region Policy Optimization** [4]
* **Model-Agnostic Meta-Learning**  [1] with **TRPO** [4] with modified implementation from [3]

## Neural Networks

### SAC — Forward

Neural Networks used in this project are straight from previous work [1].

### MMAL TRPO — Direction

64x64 fully-connected with tanh activation

## Results

### G1 — Forward Task with SAC

#### Hyperparameters

```python
NET_SIZE = 512
LR = 3e-4
GAMMA = 0.99
BATCH_SIZE = 256
BUFFER_SIZE = int(1e6)
ALPHA = 0.2
TAU = 0.005
TARGET_UPDATE_INTERVAL = 1
GRADIENT_STEPS = 1
```

#### Rewards plot

*ommited*

### G1 — MAML Direction with TRPO

Adopted **MAML-TRPO** code: https://github.com/stillonearth/pytorch-maml-rl

#### Hyperparameters

`pytorch-maml` [config file](https://github.com/stillonearth/pytorch-maml-rl/blob/master/configs/maml/g1-dir.yaml)

#### Rewards plot

*ommited*

## Notes

* This work is primarily adaptation of previous solutions and evaluation of Mid-2022 state of robotic simulators.
* MAML-TRPO as of now doesn't works only on CPU set up, CUDA implementation is bugged somewhere in mp code
* In layman terms: SAC proves that control problem can be solved in principle, MAML solution shows feasibility of synthesized control system.

## References

[1] Chelsea Finn, Pieter Abbeel, Sergey Levine, Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks, 2017, https://arxiv.org/abs/1703.03400
[2] Sergei Surovtsev, Using Policy Optimization (PO) Algorithms Soft Actor-Critic (SAC) and Proximal Policy Optimization (PPO) for Solving Unity ML Reacher and Crawler Continous Control Environments, 2019, https://github.com/cwiz/DRLND-Project-Continuous_Control/blob/master/WRITEUP.md
[3] Tristan Deleu, Reinforcement Learning with Model-Agnostic Meta-Learning (MAML), https://github.com/tristandeleu/pytorch-maml-rl

