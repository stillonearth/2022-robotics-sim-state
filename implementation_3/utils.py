import numpy as np
import torch
import random

from collections import namedtuple, deque


def unroll_trajectory(fn_interact, fn_reset, agent, n_agents, t_max=1000, n_trajectories=20):
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
  
    states = np.array(fn_reset())
    state_list, next_states_list, dones_list, reward_list, action_list = [], [], [], [], []
    for _ in range(t_max):
        actions = agent.act(states)
        next_states, rewards, dones = fn_interact(actions)
        
        next_states = np.array(next_states).reshape(n_agents, -1)
        rewards     = np.array(rewards).reshape(n_agents)
        dones       = np.array(dones).reshape(n_agents).astype(int)

        if len(dones_list) > 0:
            rewards = rewards * (1-dones_list[-1])
        
        state_list.append(states)
        next_states_list.append(next_states)
        dones_list.append(dones)
        action_list.append(actions)
        reward_list.append(rewards)
        
        if dones.all():
            break

        states = next_states

    n_episodes = len(state_list)
    return {
        "states"      : np.array(state_list).reshape(n_agents, n_episodes, -1),        
        "actions"     : np.array(action_list).reshape(n_agents, n_episodes, -1),
        "rewards"     : np.array(reward_list).reshape(n_agents, n_episodes),
        "next_states" : np.array(next_states_list).reshape(n_agents, n_episodes, -1),
        "dones"       : np.array(dones_list).reshape(n_agents, n_episodes),
        "n_episodes" : n_episodes,
        "n_agents"   : n_agents,
    }



class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.

        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["states", "rewards"])

    def add(self, states, rewards):
        """Add a new experience to memory."""
        e = self.experience(states, rewards)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        k = np.min([self.batch_size, len(self.memory)])
        experiences = random.sample(self.memory, k=k)

        states = [e.states for e in experiences]
        rewards = [e.rewards for e in experiences]

        return states, rewards

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
