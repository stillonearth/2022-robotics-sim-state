import numpy as np


def collect_trajectories(env, brain_name, policy, t_max=200):
    """
        Unroll a trajectory and return 
        list of action probabilities, states, actions and rewards

        Parameters
        ----------
        env: unityagents.UnityEnvironment
        Environment to play on.
        brain_name: String
        Name of brain used for UnityEnvironment
        policy: torch.nn.Module
        A neural network mapping states to action probabilities 
        The action to take at a given state
        t_max: int
        Maximum number of episodes in trajectory
    """
    
    env_info = env.reset(train_mode=True)[brain_name]
    # num_agents = len(env.ps)

    state_list  = []
    reward_list = []
    prob_list   = []
    action_list = []

    states = env_info.vector_observations
    
    for _ in range(t_max):

        action_probabilities = policy(states).squeeze().cpu().detach().numpy()
        actions = np.argmax(action_probabilities)
        env_info = env.step(actions)[brain_name]
        next_states = env_info.vector_observations
        
        prob_list.append(action_probabilities)
        state_list.append(states)
        action_list.append(actions)
        reward_list.append(env_info.rewards)

        if np.any(env_info.local_done):
            break

        states = next_states

    return prob_list, state_list, action_list, reward_list