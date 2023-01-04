# Continous Control with Deep Reinforcement Learning for Unitree A1 Quadruped Robot

The goal of this project is to evaluate training methods for synthesising a control system for quadruped robot.

State-of-the-art approaches focus on Model-Predictive-Control (MPC). There are two main approaches to MPC: Model-Based and Model-Free. Model-Based MPC uses a model of the system to predict the future states of the system. Model-Free MPC uses a neural network to predict the future states of the system. The model-free approach is more flexible and can be used for systems that are not fully known. However, it is more difficult to train and requires a large amount of data.

There has been a lot of research on model-free synthesized control. This projects starts with established method Soft-Actor-Critic and tries to make an efficent reward function for control signals.

## Usage

Pull repository with submodules:

```bash
git clone https://github.com/stillonearth/continuous_control-unitree-a1.git --depth 1
cd continuous_control-unitree-a1
git submodule update --init --recursive
pip install -r requirements.txt
```

## Soft-Actor-Critic

[Soft Actor-Critic](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html) is model-free direct policy-optimization algorithm. It means it can be used in environment where no a-priori world and transition models are known, such as real world. Algorithm is sample-efficient because it accumulates (s,a,r,s') pairs in experience replay buffer. SAC implementation from stable_baselines3 was used in this work.

![image](https://user-images.githubusercontent.com/97428129/199162597-e0de3c74-11d9-4b0a-86fa-6bbdc500361e.png)

## Training with control tasks

This differentiates from a previous goal as now control singal is supplied to a neural network. At each episode in training a random control task is sampled. This makes this algorithm similar to a meta-training approach such as [REPTILE](https://d4mucfpksywv.cloudfront.net/research-covers/reptile/reptile_update.pdf) This is done by using the gradient of the each new task to update the single model.

## Environments

### 1. Unitree Go-1 — Forward

This environment closely follows [Ant-v4](https://github.com/openai/gym/blob/master/gym/envs/mujoco/ant_v4.py) from OpenAI Gym. The robot is rewarded for moving forward and keeping it's body within a certain range of z-positions.

### 2. Unitree Go-1 — Control

This environment also adds 2 control signals to robot's observations: velocity direction, and body orientation. The robot is rewarded for following control signals.

## Results

Logs are included in `logs/sac` folder.

<img width="888" alt="image" src="https://user-images.githubusercontent.com/97428129/199161786-b1b1ac5c-5fc4-4e92-bc51-252ecf8c83b4.png">

**Policies after 2M episodes**

### Control — Direction

![image](https://raw.githubusercontent.com/stillonearth/continuous_control-unitree-g1/master/renders/g1-control-direction_2.avif)

### Control — Orientation

![image](https://raw.githubusercontent.com/stillonearth/continuous_control-unitree-g1/master/renders/g1-control-orientation_1.avif)

### Control — Orientation + Direction

Trained for 25M  episodes:

![image](https://user-images.githubusercontent.com/97428129/199981691-0cc7f82f-19f3-47a4-8e2b-f23e61ef8ce1.png)

![image](https://raw.githubusercontent.com/stillonearth/continuous_control-unitree-g1/master/renders/g1-control-direction%2Borientation_2.avif)

These are not state-of-the art results but can be used for futher iterating on performance of model.
