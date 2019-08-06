import numpy as np
import torch


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
    
    num_episodes = 0
    state_list, reward_list, prob_list, action_list = [], [], [], []
    
    states = np.array(fn_reset())
    
    for _ in range(t_max):
        num_episodes += 1

        actions, log_probs = policy_network.act(states)
        actions = actions.cpu().detach().numpy()
        
        next_states, rewards, done = fn_interact(actions)
        
        next_states = np.array(next_states).reshape(n_agents, -1)
        rewards     = np.array(rewards).reshape(n_agents)
        done        = np.array(done).reshape(n_agents)
        
        prob_list.append(log_probs)
        state_list.append(states)
        action_list.append(actions)
        reward_list.append(rewards)
        
        if done.any():
            break

        states = next_states

    return {
        "log_probs"    : torch.cat(prob_list).reshape(n_agents, num_episodes, -1),
        "states"       : np.array(state_list).reshape(n_agents, num_episodes, -1), 
        "actions"      : np.array(action_list).reshape(n_agents, num_episodes, -1),
        "rewards"      : np.array(reward_list).reshape(n_agents, num_episodes),
        "num_episodes" : num_episodes,
    }

def rewards_to_go(rewards):
    """
        Computes rewards-to-go
        
        Parameters
        ----------
        rewards: np.Array
    """

    return np.flip(np.flip(rewards).cumsum(1)).copy()
        