import numpy as np
import torch
import random

from collections import namedtuple, deque


def unroll_trajectory(fn_interact, fn_reset, policy_network, n_agents, t_max=1000):
    """
        Unroll a trajectory and return 
        list of action probabilities, states, actions and rewards

        Parameters
        ----------
        fn_interact: lambda(action) -> (state, reward, done)
        Function that executes action in environment and return next state and reward
        fn_reset: lambda() -> state
        Function to reset environment
        policy_network: torch.nn.Module
        A neural network mapping states to action probabilities 
        The action to take at a given state
        n_agents: int
        Number of agents executing a policy
        t_max: int
        Maximum number of episodes in trajectory

        Returns
        -------
        trajectory: dict{
            "states": np.Array
            "actions": np.Array
            "log_probs": torch.Tensor
            "rewards": np.Array
            "num_episodes": int
        }
    """
    
    n_episodes = 0
    state_list,  next_states_list, dones_list, reward_list, prob_list, action_list = [], [], [], [], [], []
    
    states = np.array(fn_reset())
    
    for _ in range(t_max):
        n_episodes += 1

        actions, log_probs = policy_network.act(states)
        actions = actions.cpu().detach().numpy()
        
        next_states, rewards, dones = fn_interact(actions)
        
        next_states = np.array(next_states).reshape(n_agents, -1)
        rewards     = np.array(rewards).reshape(n_agents)
        dones       = np.array(dones).reshape(n_agents)
        
        prob_list.append(log_probs)
        state_list.append(states)
        next_states_list.append(next_states)
        dones_list.append(dones)
        action_list.append(actions)
        reward_list.append(rewards)
        
        if dones.any():
            break

        states = next_states

    return {
        "log_probs"    : torch.cat(prob_list).reshape(n_agents, n_episodes, -1),
        "states"       : np.array(state_list).reshape(n_agents, n_episodes, -1),
        "next_states"  : np.array(next_states_list).reshape(n_agents, n_episodes, -1),
        "actions"      : np.array(action_list).reshape(n_agents, n_episodes, -1),
        "rewards"      : np.array(reward_list).reshape(n_agents, n_episodes),
        "dones"        : np.array(dones_list).reshape(n_agents, n_episodes),
        "n_episodes" : n_episodes,
        "n_agents"   : n_agents,
    }

def rewards_to_go(rewards):
    """
        Computes rewards-to-go
        
        Parameters
        ----------
        rewards: np.Array
    """

    return np.flip(np.flip(rewards).cumsum(1)).copy()
        
def advantage_estimate(states, rewards, next_states, value_network, n_agents, t_max, gamma=0.995):
    """
        Computes Advantage for each episode in trajectory
        
        Parameters
        ----------
        states: np.Array
        rewards: np.Array
        next_states: np.Array
        value_network: torch.Module
        Neural network estimating Value function
        n_agents: int
        Number of agents
        t_max: int
        Number of episodes in trajectory
        gamma: float
        Discount value
    """

    current_values = value_network.estimate(states).reshape(n_agents, t_max).detach().cpu().numpy()
    next_values = value_network.estimate(next_states).reshape(n_agents, t_max).detach().cpu().numpy()
    rewards = rewards.reshape(n_agents, t_max)
    
    advantages = rewards + gamma * np.nan_to_num(next_values) - current_values
    return advantages


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, device, seed):
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
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device

    def add(self, trajectory):
        """Add a new experience to memory."""
        for n in range(trajectory['n_agents']):
            for i in range(trajectory['n_episodes']):
                experience = self.experience(
                    trajectory['states'][n,i], 
                    trajectory['actions'][n,i], 
                    trajectory['rewards'][n,i], 
                    trajectory['next_states'][n,i], 
                    trajectory['dones'][n,i],
                )
                self.memory.append(experience)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states      = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(self.device)
        actions     = torch.from_numpy(np.vstack([e.action for e in experiences])).float().to(self.device)
        rewards     = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(self.device)
        dones       = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)