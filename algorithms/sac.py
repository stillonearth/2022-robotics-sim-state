import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

from collections import namedtuple, deque

from torch.utils.tensorboard import SummaryWriter


LR = 3e-4
GAMMA = 0.99
BATCH_SIZE = 256
BUFFER_SIZE = int(1e6)
ALPHA = 0.2
TAU = 0.005
TARGET_UPDATE_INTERVAL = 1
GRADIENT_STEPS = 1

class Agent():

    def __init__(self, state_size, action_size, policy_network, value_network, q_network, n_agents, device):
        
        self.state_size = state_size
        self.action_size = action_size
        self.n_agents = n_agents
        self.device = device

        # Initialize Policy Network
        self.policy_network = policy_network(state_size=state_size, action_size=action_size).to(device)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=LR)

        # Initialize Value Network
        self.value_network_local = value_network(state_size=state_size, action_size=action_size).to(device)
        self.value_network_target = value_network(state_size=state_size, action_size=action_size).to(device)
        self.value_optimizer = optim.Adam(self.value_network_local.parameters(), lr=LR)

        # Initialize Q Network
        self.q_network_1 = q_network(state_size=state_size, action_size=action_size).to(device)
        self.q_network_2 = q_network(state_size=state_size, action_size=action_size).to(device)
        self.q_optimizer_1 = optim.Adam(self.q_network_1.parameters(), lr=LR)
        self.q_optimizer_2 = optim.Adam(self.q_network_2.parameters(), lr=LR)

        # Initialize Replay Memory
        self.memory = ReplayBuffer(self.action_size, BUFFER_SIZE, BATCH_SIZE, 0)

        # Step counter
        self.step_counter = 0
        self.writer = SummaryWriter()

    def value_q(self, states, next_states, rewards, dones, network, gamma=GAMMA):
        return (rewards + gamma * (1 - dones) * network(next_states))
    
    def value_v(self, states, alpha=ALPHA):
        actions, log_probs = self.sample_action(states)

        q_target_1 = self.q_network_1(states, actions.detach()) 
        q_target_2 = self.q_network_2(states, actions.detach())

        return torch.min(q_target_1, q_target_2) - alpha * log_probs

    def write_loss_to_log(self, loss, name):
        self.writer.add_scalar(name, loss.detach().cpu().numpy())

    def learn(self):

        for _ in range(GRADIENT_STEPS):

            states, actions, rewards, next_states, dones = self.memory.sample(self.device)

            # Calculate V and Q targets
            y_q = self.value_q(states, next_states, rewards, dones, self.value_network_target).detach()
            y_v = self.value_v(states)

            
            # Update V-function
            v_loss = (self.value_network_local(states)-y_v).pow(2).mean()
            torch.nn.utils.clip_grad_norm_(self.value_network_local.parameters(), 1)
            self.value_optimizer.zero_grad()
            v_loss.backward()
            self.value_optimizer.step()

            # Update Policy-function
            p_actions, p_log_probs = self.sample_action(states)
            
            p_loss = -(self.q_network_1(states, p_actions) - ALPHA * p_log_probs).mean()
            torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1)
            self.policy_optimizer.zero_grad()
            p_loss.backward()
            self.policy_optimizer.step()

            # Update Q-functions
            q_loss_1 = (self.q_network_1(states, actions) - y_q).pow(2).mean()
            self.q_optimizer_1.zero_grad()
            q_loss_1.backward() 
            self.q_optimizer_1.step()
            
            q_loss_2 = (self.q_network_2(states, actions) - y_q).pow(2).mean()
            self.q_optimizer_2.zero_grad()
            q_loss_2.backward()
            self.q_optimizer_2.step()

            # self.q_optimizer_1.step()
            # self.q_optimizer_2.step()
            # self.value_optimizer.step()
            # self.policy_optimizer.step()

            # update Value network
            self.soft_update(self.value_network_local, self.value_network_target)

            # Write losses to tensorbord log
            self.write_loss_to_log(q_loss_1, 'rewards/q_loss_1')
            self.write_loss_to_log(q_loss_2, 'rewards/q_loss_2')
            self.write_loss_to_log(p_loss, 'rewards/p_loss')
            self.write_loss_to_log(v_loss, 'rewards/v_loss')

    def optimize_loss(self, loss, optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def soft_update(self, local_model, target_model, tau=TAU):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                tau*local_param.data + (1.0-tau)*target_param.data)


    def step(self, states, actions, rewards, next_states, dones):
        for i in range(states.shape[0]):
            self.memory.add(states[i], actions[i], rewards[i], next_states[i], dones[i])
            self.step_counter += 1

        if self.step_counter >= TARGET_UPDATE_INTERVAL and len(self.memory.memory) > BATCH_SIZE:
            self.learn()
            self.step_counter = 0

    def sample_action(self, state, epsilon=1e-6):
        (mean, stddev) = self.policy_network(state)
        sigma = torch.distributions.Normal(0, 1).sample()
        action = mean + stddev * sigma
        log_prob = torch.distributions.Normal(mean, stddev).log_prob(action)

        return torch.tanh(action), log_prob

    def act(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        self.policy_network.eval()
        with torch.no_grad():
            action, _ = self.policy_network(state)
        self.policy_network.train()

        return action.detach().cpu().numpy()

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=[
                                     "state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, device):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack(
            [e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack(
            [e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)