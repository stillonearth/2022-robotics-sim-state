[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"
[image3]: https://raw.githubusercontent.com/cwiz/DRLND-Project-Continuous_Control/ok/images/vpg-gradient.svg "VPG-Loss"
[image4]: https://raw.githubusercontent.com/cwiz/DRLND-Project-Continuous_Control/ok/images/vpg-algorithm.svg "VPG-Algorithm"

# Project 1: Navigation

**Sergei Surovtsev**
<br/>
Udacity Deep Reinforcement Learning Nanodegree
<br/>
Class of May 2019

## Project Description
This project is about applying policy optimization algorithms to continous control tasks in Unity ML agent simulator. First task is called Reacher simulates two-joint robotic arm in reaching objective and second is Crawler which simulates four-legged two-join robot that has to learn locomotion.

### Reacher

![Reacher][image1]

**Goal**

* Position end-effector within green sphere
* Achieve average score over 30 over 100 episodes with 1000 frames each

**Observation**

* Vector of 33 that consists of position, rotation, velocity, and angular velocities of the arm.

**Actions**

* Continous vector space for rotations in 2 planes for arm's joints (size of 4)

**Rewards**

* Agent is getting rewards if end-effector is within green sphere

This environment is not MDP because it's next state doesn't depend on previous state and action pair. Though difficulty of this environment with sparse initial rewards, and some algorithms that rely on Markovian dynamics models and instant feedback may fail to achieve convergence.

### Crawler [TODO: implement]

![Crawler][image2]
 
## Project Goals

* Introduction to Policy Gradient Methods
* Study of applicability of Policy Gradient Methods to Robotic Manipulation and Locomotion tasks
* Implementation of some of state-of-the-art algorithms

## Technical Formulation of Problem 

* Set up Environemnt as described in [Project Repository](https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control)
* Complete Crawler.ipynb 
* [Optional] Complete Reacher.ipynb 

## Mathematical Models

**Policy Optimization (PO)** algorithms optimize stochastic Policy function. main motivation behind them is that they can deal with stochastisitiy and support continous control tasks (unlike DQN which only supports discretized control).

In such setting input of Policy is state and output is probability distribution of actions. A Normal distribution is usually chosen and then output of policy is distribution parameters: mean and standard deviation.

PO algorithm then would need to push probilities of actions yielding high reward up and decrease probabilities of bad actions.

Gradient of return of policy is denoted as:

![VPG-Loss][image3]

PO algorithm then updates policy using gradient ascend. A is training signal and is typically a Reward or advantage signal. 

### Vanilla Policy Gradient

The basic PO algorithm is **Vanilla Policy Gradient (VPG)**

![VPG][image4]

VPG is on-policy algorithm meaning that it doesn't use sample from past. It suffers from bias error and also of high-variance. [1]

### Soft Actor Critic

More recent algorithm in PO family is **Soft-Actor Critic (SAC)** [2]. SAC is off-policy algorithm and fixes some of drawbacks of VPG with sample-inneficiency, high variance, exploration and biasing. 

SAC uses following techniques:

* Clipped double-Q trick
* Entropy regularization

## Results

### Reacher

#### Neural Networks

#### Hyperparameters

```python
LR = 3e-3
GAMMA = 0.99
BATCH_SIZE = 256
BUFFER_SIZE = int(1e6)
ALPHA = 0.01
TAU = 0.05
TARGET_UPDATE_INTERVAL = 1
GRADIENT_STEPS = 2
```

#### Rewards-per-Episode Plots

### Crawler

#### Neural Networks

#### Hyperparameters

#### Rewards-per-Episode Plots

## Acknowledgements

## References

[1] https://spinningup.openai.com/en/latest/algorithms/vpg.html
[2] https://spinningup.openai.com/en/latest/algorithms/sac.html