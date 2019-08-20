import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from utils import ReplayBuffer

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()


LR = 3e-4
GAMMA = 0.99
LAMBDA = 0.96
TRAIN_V_ITERS = 40
TRAIN_P_ITERS = 40
EPSILON = 0.1
BETA = 0.01

class Agent():

    def __init__(self, state_size, action_size, policy_network, value_network, n_agents, device, use_gae=True):
        
        self.state_size = state_size
        self.action_size = action_size
        self.n_agents = n_agents
        self.device = device

        self.policy_network = policy_network(state_size=state_size, action_size=action_size).to(device)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=LR)

        self.value_network = value_network(state_size=state_size, action_size=1).to(device)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=LR)
        self.epsilon = EPSILON
        self.beta = BETA

        self.reset_memory()
        self.buffer = ReplayBuffer(int(128), 64)
        self.use_gae = use_gae

    def reset_memory(self):
        self.rnn_memory = None

    def policy_loss(self, old_log_probs, states, actions, rewards, epsilon=EPSILON, beta=BETA):

        distribution, _ = self.policy_network(states, None)
        new_log_prob = distribution.log_prob(actions)
        new_probs = torch.exp(new_log_prob)
        ratio = torch.exp(new_log_prob - old_log_probs)

        # clipped function
        clip = torch.clamp(ratio, 1-epsilon, 1+epsilon)
        rewards = rewards.reshape(self.n_agents, clip.shape[1], -1)
        clipped_surrogate = torch.min(ratio*rewards, clip*rewards)
        entropy = -(new_probs*old_log_probs + (1.0-new_probs)*old_log_probs)
        loss = (clipped_surrogate + beta*entropy).mean()

        return loss

    def value_loss(self, states, rewards):
        estimated_value = self.value_network(states).reshape(self.n_agents, -1)
        return (estimated_value - rewards).pow(2).mean(1).mean()
    
    def act(self, state):
        state = torch.from_numpy(state).float().to(self.device).unsqueeze(1)
        self.policy_network.eval()
        with torch.no_grad():
            action_distribution, self.rnn_memory = self.policy_network(state, self.rnn_memory)
        self.policy_network.train()

        action = action_distribution.sample()
        return action.detach().cpu().numpy()

    def action_probs(self, states, actions):
        self.policy_network.eval()
        log_probs = None
        with torch.no_grad():
            distribution, _ = self.policy_network(states, None)
            log_probs = distribution.log_prob(actions).detach()
        self.policy_network.train()
        return log_probs

    def learn(self, trajectory):
        states = torch.from_numpy(trajectory['states']).float().to(self.device)
        actions = torch.from_numpy(trajectory['actions']).float().to(self.device)
        rewards = rewards_to_go(trajectory['rewards'], self.n_agents, self.device)
        next_states = torch.from_numpy(trajectory['next_states']).float().to(self.device)
        dones = torch.from_numpy(trajectory['dones']).float().to(self.device)
        log_probs = self.action_probs(states, actions)

        policy_signal = None
        if self.use_gae:
            self.buffer.add(states, rewards)
            policy_signal = generalized_advantage_estimate(states, rewards, next_states, dones, self.value_network).detach()
        else:
            policy_signal = rewards

        # print(policy_signal.shape)
        # policy_signal = (policy_signal - policy_signal.mean()) / (policy_signal.std() + 1e-10)

        # Optimize Policy
        for _ in range(TRAIN_P_ITERS):
            self.policy_optimizer.zero_grad()
            pl = self.policy_loss(log_probs, states, actions, policy_signal, self.epsilon, self.beta)
            writer.add_scalar('loss/policy', pl.cpu().detach().numpy())
            pl.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1)
            self.policy_optimizer.step()
            del pl
        
        if self.use_gae:
            # Optimize Value Function
            for _ in range(TRAIN_V_ITERS):
                self.value_optimizer.zero_grad()
                s_, r_ = self.buffer.sample()
                all_rewards = torch.stack(r_)
                r_mean = all_rewards.mean()
                r_std = all_rewards.std() + 1e-10
                losses = []
                for s, r in zip(s_, r_):
                    losses.append(self.value_loss(s, r).mean())
                loss = torch.stack(losses).mean()
                writer.add_scalar('loss/value', loss.cpu().detach().numpy())
                loss.backward()
                self.value_optimizer.step()
                del loss

        self.epsilon *= .999
        self.beta *= .995

        self.reset_memory()
        
def advantage_estimate(states, rewards, next_states, dones, n_agents, t_max, value_network, gamma=GAMMA):
    ve = None
    value_network.eval()
    with torch.no_grad():
        current_values = value_network(states).reshape(n_agents, t_max)
        next_values = value_network(next_states).reshape(n_agents, t_max)
        ones = torch.ones_like(dones)

        ve = rewards + gamma*next_values*(ones-dones) - current_values
    value_network.train()
    return ve

def generalized_advantage_estimate(states, rewards, next_states, dones, value_network, gamma=GAMMA, lambda_=LAMBDA):

    n_agents = states.shape[0]
    n_episodes = states.shape[1]

    r_mean = rewards.mean()
    r_std = rewards.std() + 1e-10

    advantage = advantage_estimate(states, (rewards-r_mean)/r_std, next_states, dones, n_agents, n_episodes, value_network, gamma)
    
    shifted_advantages = [advantage]
    coeffs = [1.0]
    for i in range(1, advantage.shape[1]):
        sa = shifted_advantages[-1].roll(-1)
        sa[:, -1] = 0
        shifted_advantages.append(sa)
        coeffs.append((lambda_*gamma)**i)

    coeffs = np.array(coeffs)
    coeffs = torch.from_numpy(coeffs).float().to(advantage.device)
    shifted_advantages = torch.cat(shifted_advantages).reshape(n_agents, n_episodes, -1)

    return (coeffs*shifted_advantages).sum(axis=2).reshape(n_agents, -1)

def rewards_to_go(rewards, n_agents, device):
    """
        Computes rewards-to-go

        Parameters
        ----------
        rewards: np.Array
    # """
    rewards = np.flip(np.flip(rewards, axis=1).cumsum(1), axis=1).copy()
    return torch.from_numpy(rewards).float().to(device)