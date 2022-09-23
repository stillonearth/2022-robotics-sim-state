import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

from collections import namedtuple, deque

from torch.utils.tensorboard import SummaryWriter


LR = 1e-4
GAMMA = 0.99
BATCH_SIZE = 1024
BUFFER_SIZE = int(1e6)
ALPHA = 0.2
TAU = 0.005
TARGET_UPDATE_INTERVAL = 1
GRADIENT_STEPS = 2


class Agent():

    def __init__(
        self,
        state_size,
        action_size,
        actor,
        critic,
        n_agents,
        device,
    ):

        self.state_size = state_size
        self.action_size = action_size
        self.n_agents = n_agents
        self.device = device

        # Initialize Policy Network
        self.actor = actor(
            state_size=state_size, action_size=action_size).to(device)
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=LR)

        # Initialize Q Network
        self.critic_1_target = critic(
            state_size=state_size, action_size=action_size).to(device)
        self.critic_2_target = critic(
            state_size=state_size, action_size=action_size).to(device)
        self.critic_1_local = critic(
            state_size=state_size, action_size=action_size).to(device)
        self.critic_2_local = critic(
            state_size=state_size, action_size=action_size).to(device)
        self.q_optimizer_1 = optim.Adam(
            self.critic_1_local.parameters(), lr=LR)
        self.q_optimizer_2 = optim.Adam(
            self.critic_2_local.parameters(), lr=LR)

        # Initialize Replay Memory
        self.memory = ReplayBuffer(
            self.action_size, BUFFER_SIZE, BATCH_SIZE, 0)

        # Step counter
        self.step_counter = 0
        self.writer = SummaryWriter()

    def learn(self, alpha=ALPHA, gamma=GAMMA):

        for _ in range(GRADIENT_STEPS):

            states, actions, rewards, next_states, dones = self.memory.sample(
                self.device)

            # Compute targets for Q
            p_actions, p_log_probs = self.sample_action(next_states)
            min_q = torch.min(
                self.critic_1_target(next_states, p_actions.detach()),
                self.critic_2_target(next_states, p_actions.detach()),
            )
            y = (rewards + gamma * (1 - dones) *
                 (min_q - alpha * p_log_probs.detach()))

            # Update Q-functions
            q_loss_1 = (self.critic_1_local(
                states, actions) - y.detach()).pow(2).mean()
            torch.nn.utils.clip_grad_norm_(self.critic_1_local.parameters(), 1)
            self.optimize_loss(q_loss_1, self.q_optimizer_1)

            q_loss_2 = (self.critic_2_local(
                states, actions) - y.detach()).pow(2).mean()
            torch.nn.utils.clip_grad_norm_(self.critic_2_local.parameters(), 1)
            self.optimize_loss(q_loss_2, self.q_optimizer_2)

            # Update Policy-function
            p_actions, p_log_probs = self.sample_action(states)
            min_q = torch.min(
                self.critic_1_local(states, p_actions),
                self.critic_2_local(states, p_actions),
            )
            p_loss = -(min_q - alpha * p_log_probs).mean()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
            self.optimize_loss(p_loss, self.actor_optimizer)

            # Update target networks
            self.soft_update(self.critic_1_local, self.critic_1_target)
            self.soft_update(self.critic_2_local, self.critic_2_target)

            # Write losses to tensorbord log
            self.write_loss_to_log(q_loss_1, 'rewards/q_loss_1')
            self.write_loss_to_log(q_loss_2, 'rewards/q_loss_2')
            self.write_loss_to_log(p_loss, 'rewards/p_loss')

    def sample_action(self, state):
        (mean, stddev) = self.actor(state)
        sigma = torch.distributions.Normal(0, 1).sample()
        action = mean + stddev * sigma
        log_prob = torch.distributions.Normal(mean, stddev).log_prob(action)

        return torch.tanh(action), log_prob

    def step(self, states, actions, rewards, next_states, dones):
        for i in range(states.shape[0]):
            self.memory.add(states[i], actions[i],
                            rewards[i], next_states[i], dones[i])
            self.step_counter += 1

        if self.step_counter >= TARGET_UPDATE_INTERVAL and len(self.memory.memory) > BATCH_SIZE:
            self.learn()
            self.step_counter = 0

    def act(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action, _ = self.actor(state)
        self.actor.train()

        return action.detach().cpu().numpy()

    def optimize_loss(self, loss, optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def write_loss_to_log(self, loss, name):
        self.writer.add_scalar(name, loss.detach().cpu().numpy())

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
