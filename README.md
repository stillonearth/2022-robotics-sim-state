# Continous Control with Deep Reinforcement Learning for Unitree Go-1 Quadruped Robot

The goal of this project is to evaluate training methods for synthesising a control system for quadruped robot.

State-of-the-art approaches focus on Model-Predictive-Control (MPC). There are two main approaches to MPC: Model-Based and Model-Free. Model-Based MPC uses a model of the system to predict the future states of the system. Model-Free MPC uses a neural network to predict the future states of the system. The model-free approach is more flexible and can be used for systems that are not fully known. However, it is more difficult to train and requires a large amount of data.

There has been a lot of research on model-free synthesized control. This projects starts with established method Soft-Actor-Critic and tries to make an efficent reward function for control signals.

## Usage

Pull repository with submodules:

```bash
git clone https://github.com/stillonearth/continuous_control-unitree-g1.git
cd continuous_control-unitree-g1
git submodule update --init --recursive
pip install -r requirements.txt
```

## Soft-Actor-Critic

[SAC](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html) implementation from stable_baselines3 was chosen to train a set of models.

## Training with control tasks

[REPTILE](https://d4mucfpksywv.cloudfront.net/research-covers/reptile/reptile_update.pdf) established a framework for meta-training an agent. The idea is to train a set of models on a set of tasks and then use the models to train a new model on a new task. The new model is trained by using the models trained on the old tasks as a starting point. This is done by using the gradient of the new task to update the old models. The new model is then trained by using the updated old models as a starting point. This process is repeated until the new model converges.

In this project each new rollout samples a new task with control signal is included into observation space and tries synthesize a single model that generalizes to all tasks.

## Environments

### 1. Unitree Go-1 — Forward

This environment closely follows [Ant-v4](https://github.com/openai/gym/blob/master/gym/envs/mujoco/ant_v4.py) from OpenAI Gym. The robot is rewarder for moving forward and keeping it's body within a certain range of z-positions.

### 2. Unitree Go-1 — Control

This environment also adds 2 control signals to robot's observations: velocity direction, and body orientation. The robot is rewarded for following control signals.

## Results

Logs are included in `logs/sac` folder.

<img width="888" alt="image" src="https://user-images.githubusercontent.com/97428129/199161786-b1b1ac5c-5fc4-4e92-bc51-252ecf8c83b4.png">

**Policies after 2M episodes**

### Control — Direction

<img src="https://raw.githubusercontent.com/stillonearth/continuous_control-unitree-g1/master/renders/g1-control-direction_2.avif">

### Control — Orientation

<img src="https://raw.githubusercontent.com/stillonearth/continuous_control-unitree-g1/master/renders/g1-control-orientation_1.avif">

### Control — Orientation + Direction

<img src="https://raw.githubusercontent.com/stillonearth/continuous_control-unitree-g1/master/renders/g1-control-direction%2Borientation_2.avif">

