import numpy as np
import torch
import random

from collections import namedtuple, deque


def unroll_trajectory(fn_interact, fn_reset, agent, n_agents, t_max=1000):
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
    state_list,  next_states_list, dones_list, reward_list, action_list = [], [], [], [], []
    
    states = np.array(fn_reset())
    
    for _ in range(t_max):
        n_episodes += 1

        actions = agent.act(states)
        
        next_states, rewards, dones = fn_interact(actions)
        
        next_states = np.array(next_states).reshape(n_agents, -1)
        rewards     = np.array(rewards).reshape(n_agents)
        dones       = np.array(dones).reshape(n_agents)
        
        state_list.append(states)
        next_states_list.append(next_states)
        dones_list.append(dones)
        action_list.append(actions)
        reward_list.append(rewards)
        
        if dones.any():
            break

        states = next_states

    return {
        "states"       : np.array(state_list).reshape(n_agents, n_episodes, -1),
        "next_states"  : np.array(next_states_list).reshape(n_agents, n_episodes, -1),
        "actions"      : np.array(action_list).reshape(n_agents, n_episodes, -1),
        "rewards"      : np.array(reward_list).reshape(n_agents, n_episodes),
        "dones"        : np.array(dones_list).astype(int).reshape(n_agents, n_episodes),
        "n_episodes" : n_episodes,
        "n_agents"   : n_agents,
    }
