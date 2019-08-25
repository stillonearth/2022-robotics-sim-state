import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from torch.autograd import Variable

from utils import ReplayBuffer, rewards_to_go

LR = 5e-4
N_POLICY_ITERATIONS = 30

class Agent():
    
    def __init__(
        self, state_size, action_size, 
        q_network, value_network, policy_network, 
        n_agents, device, gamma=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.device = device
        
        self.q_network = q_network
        self.q_network_optimizer = optim.Adam(self.q_network.parameters(), lr=LR)

        self.value_network = value_network
        self.value_network_optimizer = optim.Adam(self.value_network.parameters(), lr=LR)

        self.policy_network = policy_network
        self.policy_network_optimizer = optim.Adam(self.policy_network.parameters(), lr=LR)
    
        self.replay_buffer = ReplayBuffer(
            action_size=action_size, 
            buffer_size=int(1e5), 
            batch_size=512, 
            seed=0,
            device=device,
        )

        self.n_agents = n_agents
        self.n_policy_iterations = N_POLICY_ITERATIONS

    def fit_q_network(self, trajectory):
        for _ in range(trajectory['n_episodes']*trajectory['n_agents']):
            states, actions, rewards, next_states, _ = self.replay_buffer.sample()

            q_targets = rewards + self.value_network(next_states).mean()
            q_expected = self.q_network(states, actions)

            loss = F.mse_loss(q_expected, q_targets)
            self.q_network_optimizer.zero_grad()
            loss.backward()
            self.q_network_optimizer.step()
        
    def fit_value_network(self, trajectory):
        states = torch.from_numpy(trajectory['states']).float().to(self.device)
        rewards = torch.from_numpy(rewards_to_go(trajectory['rewards'])).float().to(self.device)
        target = self.value_network(states).reshape(self.n_agents, -1)
        objective = rewards.reshape(self.n_agents, -1)
        loss = F.mse_loss(target, objective)
        self.value_network_optimizer.zero_grad()
        loss.backward()
        self.value_network_optimizer.step()

    def fit_policy_network(self, trajectory):
        A_hat = self.generalized_advantage_estimate(trajectory)
        A_dash, a_grad, e_a = self.q_prop_advantage(trajectory)
        A_dash = A_dash.detach().cpu().numpy()
        a_grad = a_grad.detach()
        e_a = e_a

        etta = np.sign(np.stack([np.correlate(A_hat[i], A_dash[i], "same") for i in range(self.n_agents)]))

        signal = A_hat - etta * A_dash
        for _ in range(self.n_policy_iterations):
            policy_loss = -self.ppo_clip_objective(
                trajectory['states'], 
                trajectory['actions'], 
                signal, 
                trajectory['log_probs'], 
                etta,
                a_grad,
                e_a,
            )
            self.policy_network_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_network_optimizer.step()

    def advantage_estimate(self, states, rewards, next_states, value_network, n_agents, t_max, gamma=0.995):
        current_values = value_network.estimate(states).reshape(n_agents, t_max).detach().cpu().numpy()
        next_values = value_network.estimate(next_states).reshape(n_agents, t_max).detach().cpu().numpy()
        rewards = rewards.reshape(n_agents, t_max)
        
        advantages = rewards + gamma * np.nan_to_num(next_values) - current_values
        return advantages
    
    def generalized_advantage_estimate(self, trajectory):

        states    = trajectory['states']
        rewards   = rewards_to_go(trajectory['rewards'])
        log_probs = trajectory['log_probs']
        
        next_states = np.roll(states, -1, axis=2)
        next_states[:,-1,:] = np.nan

        return self.advantage_estimate(
            states, 
            rewards, 
            next_states, 
            self.value_network, 
            trajectory['n_agents'], 
            trajectory['n_episodes'],
        )

    def q_prop_advantage(self, trajectory):
        actions = torch.from_numpy(trajectory['actions']).float().to(self.device)
        expected_actions = self.expected_action(actions)
        
        s = Variable(torch.from_numpy(trajectory['states']).float().to(self.device), requires_grad=True)
        a = Variable(actions, requires_grad=True)

        q_value = self.q_network(s, a)
        q_value.backward(torch.ones(q_value.size()).to(self.device))

        A = (a.grad * (actions-expected_actions))
        return A.mean(2), a.grad, expected_actions

    def expected_action(self, actions):
        e_a = actions.mean(1).mean(0)
        return e_a

    def ppo_clip_objective(self, states, actions, signal, old_log_probs, etta, a_grad, e_a, epsilon=0.1):
        _, new_log_probs = self.policy_network.act(states)
        ratio = new_log_probs / old_log_probs.detach()
        signal = torch.from_numpy(signal).float().to(self.device)[:,:,None]
        
        mean_actions = torch.from_numpy(actions).float().to(self.device).mean(1)
        states = torch.from_numpy(states).float().to(self.device)
        etta = torch.from_numpy(etta).float().to(self.device)

        return torch.min(
            ratio*signal, 
            torch.clamp(ratio, 1-epsilon, 1+epsilon)*signal
        ).mean(1).mean(0).mean() + (etta.unsqueeze(2) * a_grad).mean(1).mean(0).mean()
        
    def step(self, trajectory):
        self.replay_buffer.add(trajectory)
        
        self.fit_q_network(trajectory)
        self.fit_value_network(trajectory)
        self.fit_policy_network(trajectory)
