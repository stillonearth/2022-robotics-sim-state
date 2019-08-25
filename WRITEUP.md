[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"
[image3]: https://raw.githubusercontent.com/cwiz/DRLND-Project-Continuous_Control/ok/images/vpg-gradient.svg?sanitize=true "VPG-Loss"
[image4]: https://raw.githubusercontent.com/cwiz/DRLND-Project-Continuous_Control/ok/images/vpg-algorithm.svg?sanitize=true "VPG-Algorithm"
[image5]: https://raw.githubusercontent.com/cwiz/DRLND-Project-Continuous_Control/ok/images/rewards.png "VPG-Algorithm"
[image6]: https://raw.githubusercontent.com/cwiz/DRLND-Project-Continuous_Control/ok/images/sac-algorithm.svg?sanitize=true "SAC-Algorithm"

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

![SAC][image6]

## Results

### Reacher

#### Neural Networks

```
NET_SIZE = 128

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

#### Rewards plot

![Rewards][image4]

### Crawler

#### Neural Networks

#### Hyperparameters

#### Rewards plot

## Acknowledgements

## References

[1] https://spinningup.openai.com/en/latest/algorithms/vpg.html
[2] https://spinningup.openai.com/en/latest/algorithms/sac.html