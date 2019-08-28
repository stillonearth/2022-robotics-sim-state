[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"
[image3]: https://raw.githubusercontent.com/cwiz/DRLND-Project-Continuous_Control/master/images/vpg-gradient.svg?sanitize=true "VPG-Loss"
[image4]: https://raw.githubusercontent.com/cwiz/DRLND-Project-Continuous_Control/master/images/vpg-algorithm.svg?sanitize=true "VPG-Algorithm"
[image5]: https://github.com/cwiz/DRLND-Project-Continuous_Control/blob/master/images/rewards-reacher.png?raw=true "Crawler-Rewards"
[image6]: https://raw.githubusercontent.com/cwiz/DRLND-Project-Continuous_Control/master/images/sac-algorithm.svg?sanitize=true "SAC-Algorithm"
[image7]: https://github.com/cwiz/DRLND-Project-Continuous_Control/blob/master/images/rewards-crawler.png?raw=true "Crawler-Crawler"

# Project 2: Continuous Control 

**Sergei Surovtsev**
<br/>
Udacity Deep Reinforcement Learning Nanodegree
<br/>
Class of May 2019

## Project Description
This project is about applying policy optimization algorithms to continuous control tasks in Unity ML agent simulator. 

* **Reacher** simulates two-joint robotic arm that has to learn to reach an object
* **Crawler** simulates four-legged two-join robot that has to learn locomotion

### Reacher

![Reacher][image1]

**Goal**

* Position end-effector within green sphere
* Achieve average score over 30 over 100 episodes with 1000 frames each

**Observation**

* Vector of 33 that consists of position, rotation, velocity, and angular velocities of the arm.

**Actions**

* Continuous vector space for rotations in 2 planes for arm's joints (size of 4)

**Rewards**

* Agent is getting rewards if end-effector is within radius of green sphere

Reacher environment provides sparse initial rewards. Because of that simple algorithms may fail to achieve initial convergence.

### Crawler

![Crawler][image2]

**Goal**

* Move straight and avoid falling
* Achieve average score over 30 over 100 episodes with 1000 frames each

**Observation**

* 117 variables corresponding to position, rotation, velocity, and angular velocities of each limb plus the acceleration and angular acceleration of the body

**Actions**

* Vector Action space: (Continuous) Size of 20, corresponding to target rotations for joints.

**Rewards**

* +0.03 times body velocity in the goal direction
* +0.01 times body direction alignment with goal direction
 
## Project Goals

* Introduction to Policy Gradient Methods
* Study of applicability of Policy Gradient Methods to Robotic Manipulation and Locomotion tasks
* Implementation of some of state-of-the-art algorithms

## Technical Formulation of Problem 

* Set up Environment as described in [Project Repository](https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control)
* Complete Crawler.ipynb 
* Complete Reacher.ipynb 

## Mathematical Models

**Policy Optimization (PO)** algorithms optimize stochastic Policy function. PO algorithms can deal with stochasticity and support continuous control tasks.

Input of continuous Policy is state and output is probability distribution of actions. A Normal distribution is usually chosen and output of policy is distribution parameters: mean and standard deviation. PO algorithm then would need to push probabilities of actions yielding high reward up and decrease probabilities of suboptimal ones.

Gradient of return of policy is defined as [1]:

![VPG-Loss][image3]

PO algorithm updates policy using gradient ascend. **A** signal is usually a Rewards (normalized / rewards-to-go) or Advantage (usually generalized advantage estimate). 

### Vanilla Policy Gradient

The basic PO algorithm is **Vanilla Policy Gradient (VPG)**

![VPG][image4]

VPG is on-policy algorithm meaning that it doesn't use sample from experience. It is easy to implement and it works for basic environments, but it also suffers from biasiang and high variance. [1]

### Soft Actor Critic

More recent algorithm in PO family is **Soft-Actor Critic (SAC)** [2]. SAC is off-policy algorithm that fixes some of drawbacks of earlier PO algorithms it terms of sample-inefficiency, variance, biasing and exploration/exploitation.

SAC uses following techniques:

#### Clipped double-Q trick

Double Q-trick deals with drawback of DDPG (previous off-policy PO algorithm) where Q-network overestimates Q-values and then P-network starts to exploit Q-network which leads to breaking. Here we use two Q networks and then use smallest to train P-network on each iteration (which are initialized randomly and trained in same way). Intuition to this trick is that single Q-network can develop sharp peaks at some of values, but having two of them smoothes this effect. [5]

#### Entropy regularization

In entropy-regularized setting agent gets a bonus each time step proportional to entropy of policy. We regulate entropy by introducing a hyperparameter alpha. Then we re-formularize loss to include entropy. This approach enables us to regulate how much exploitation we want to get from a policy optimizer.

![SAC][image6]

## Neural Networks

Both environments used identical Neural Network architectures. Here are PyTorch classes describing them:

```python
class Policy(nn.Module):
    
    def __init__(self, state_size, action_size=1, n_agents=1, fc1_size=NET_SIZE, fc2_size=NET_SIZE):
        super(Policy, self).__init__()
        
        self.bn0 = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, fc1_size)
        self.bn1 = nn.BatchNorm1d(fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.bn2 = nn.BatchNorm1d(fc1_size)
        self.fc3_mu = nn.Linear(fc2_size, action_size)
        self.fc3_std = nn.Linear(fc2_size, action_size)

    def forward(self, state, log_std_min=-20, log_std_max=2):
        x = self.bn0(state)
        x = torch.relu(self.bn1(self.fc1(state)))
        x = torch.relu(self.bn2(self.fc2(x)))

        mean = self.fc3_mu(x)
        std = self.fc3_std(x)
        std = torch.clamp(std, log_std_min, log_std_max).exp()

        return mean, std
    
class Value(nn.Module):
    
    def __init__(self, state_size, action_size=1, n_agents=1, fc1_size=NET_SIZE, fc2_size=NET_SIZE):
        
        super(Value, self).__init__()
        
        self.bn0 = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, 1)
        
    def forward(self, x):
        x = self.bn0(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
class Q(nn.Module):
    
    def __init__(self, state_size, action_size, n_agents=1, fc1_size=NET_SIZE, fc2_size=NET_SIZE):
        
        super(Q, self).__init__()
        
        self.bn0 = nn.BatchNorm1d(state_size+action_size)
        self.fc1 = nn.Linear(state_size + action_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, 1)
        
    def forward(self, s, a):
        x = torch.cat([s, a], 1)
        x = self.bn0(x)
        x = torch.relu(self.fc1(x)) 
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
```

## Results

### Reacher

Prior choosing Soft-Actor Critic I have tested **Proximal-Policy Optimization (PPO)** and **Deep-Q-Prop** algorithms. They are implemented and included to ```algorithms``` directory but none were able to perform as well as SAC. 

I would like to include some bits of experience obtained from implementing algorithms:

* In sparse-initial reward setting major improvement was regulating number of optimization epochs per on-policy step
* Batch-normalization helped to maintain monotonic learning curve in sparse-reward setting
* Whether policy training takes of lot of time use of tensorboard is advices to visualize losses and rewards
* It seems getting model a memory such as LSRM or GRU can improve it's performance in MDP settings (experiments omitted here)

#### Hyperparameters

```python
NET_SIZE = 128
LR = 3e-3
GAMMA = 0.99
BATCH_SIZE = 256
BUFFER_SIZE = int(1e6)
ALPHA = 0.01
TAU = 0.05
TARGET_UPDATE_INTERVAL = 1
GRADIENT_STEPS = 2
```

#### Rewards plot

![Rewards][image5]

### Crawler

#### Hyperparameters

```python
NET_SIZE = 256
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

![Rewards][image7]

## Acknowledgements

github.com/luigifaticoso for implementation of reparameterization trick.

## Future Work

Chosen method was sufficient to solve two environments. It is sample efficient and has good convergence properties. On the other hand pure on-policy methods such as PPO allow to include recurrent cells to the model,which allows model to reason in temporal terms. Adding memory to off-policy methods is not straightforward. Another improvement can be made with Prioritized Experience Replay to improve model's convergence and training time.

## References

[1] OpenAI, Spinning Up DeepRL, Vanilla Policy Gradient, 2018, https://spinningup.openai.com/en/latest/algorithms/vpg.html

[2] OpenAI, Spinning Up DeepRL, Soft Actor-Critic, 2018, https://spinningup.openai.com/en/latest/algorithms/sac.html

[3] Luigi Faticoso, Soft actor-critic implemented using pytorch on Lunar Lander Continuos, https://github.com/luigifaticoso/Soft-Actor-Critic-with-lunar-lander-continuos-v2

[4] Unity Agents, Example Learning Environments, https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#crawler

[5] OpenAI, Spinning Up DeepRL, Twin Delayed DDPG, 2018, https://spinningup.openai.com/en/latest/algorithms/td3.html
