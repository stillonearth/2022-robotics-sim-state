[image1]: https://www.unitree.com/uploads/531_(3)_e17eb66c60.png "Unitree G1"
[image2]: https://github.com/stillonearth/2022-robotics-sim-state/blob/master/images/rewards-g1.png?raw=true "Rewards - G1"

# Meta-Trained Continuous Control System for Robotic Dog Unitree G1

**Sergei Surovtsev**

## Project Description
This project involves applying policy optimization algorithms to continuous control task for Unitree G1 robot in MuJoCo simulator and meta-learning control system for it. 

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

*same as in previous task* + direction and orientation model

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

Oracle policy: meta-training step includes task parameters as inputs.  [3]

## Results

### G1 — Forward Task with SAC

#### Rewards plot

![Rewards][image2]

### G1 — MAML Direction with TRPO

Adopted **MAML-TRPO** code: https://github.com/stillonearth/pytorch-maml-rl

```bash
git clone https://github.com/stillonearth/pytorch-maml-rl && cd pytorch-maml-rl
python3 train.py --config configs/maml/g1-dir.yaml --output-folder maml-g1-dir --seed 1 --num-workers 1
python3 test.py --config maml-g1-dir/config.json --policy maml-g1-dir/policy.th --output maml-g1-dir/results.npz --meta-batch-size 40 --num-batches 10  --num-workers 1
```

#### How to use oracle meta-policy

```
# get environment obervations
observations = self.envs.step(action) 

# append task meta-parameters
new_observations = []
for i, env in enumerate(self.envs.envs):
    [np.append(observations[i], env.unwrapped.meta_params()))
observations = np.array(new_observations)

# invoke policy
with torch.no_grad():
    while not self.envs.dones.all():
        observations_tensor = torch.from_numpy(observations).to(torch.float32)
        pi = self.policy(observations_tensor, params=params)
```

#### Rewards plot

*ommited*

## Notes

* This work is primarily adaptation of previous solutions and evaluation of Mid-2022 state of robotic simulators
* MAML-TRPO as of now works only on CPU setup, CUDA implementation is bugged somewhere in mp code
* In layman terms SAC proves that control problem can be solved in principle, MAML solution shows feasibility of synthesized control system
* Environment for distance control: https://github.com/stillonearth/2022-robotics-sim-state/blob/master/environments/g1_distance.py
* Environment for meta-directional control: https://github.com/stillonearth/pytorch-maml-rl/blob/master/maml_rl/envs/mujoco/g1.py

## References

[1] Chelsea Finn, Pieter Abbeel, Sergey Levine, Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks, 2017, https://arxiv.org/abs/1703.03400

[2] Sergei Surovtsev, Using Policy Optimization (PO) Algorithms Soft Actor-Critic (SAC) and Proximal Policy Optimization (PPO) for Solving Unity ML Reacher and Crawler Continous Control Environments, 2019, https://github.com/cwiz/DRLND-Project-Continuous_Control/blob/master/WRITEUP.md

[3] Tristan Deleu, Reinforcement Learning with Model-Agnostic Meta-Learning (MAML), https://github.com/tristandeleu/pytorch-maml-rl

