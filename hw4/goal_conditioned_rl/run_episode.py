"""Run the Q-network on the environment for fixed steps.

Complete the code marked TODO."""
import numpy as np # pylint: disable=unused-import
import torch # pylint: disable=unused-import


def run_episode(
    env,
    q_net, # pylint: disable=unused-argument
    steps_per_episode,
):
    """Runs the current policy on the given environment.

    Args:
        env (gym): environment to generate the state transition
        q_net (QNetwork): Q-Network used for computing the next action
        steps_per_episode (int): number of steps to run the policy for

    Returns:
        episode_experience (list): list containing the transitions
                        (state, action, reward, next_state, goal_state)
        episodic_return (float): reward collected during the episode
        succeeded (bool): DQN succeeded to reach the goal state or not
    """

    # list for recording what happened in the episode
    episode_experience = []
    succeeded = False
    episodic_return = 0.0

    # reset the environment to get the initial state
    state, goal_state = env.reset() # pylint: disable=unused-variable

    for _ in range(steps_per_episode):

        # ======================== TODO modify code ========================

        # append goal state to input, and prepare for feeding to the q-network
        input_state = np.concatenate((state, goal_state), axis=0)
        input_state = np.expand_dims(input_state, axis=0)

        # forward pass to find action
        input_tensor = torch.from_numpy(input_state).float()
        with torch.no_grad():
            q_values = q_net.forward(input_tensor)
        action = np.argmax(q_values.detach().numpy())

        # take action, use env.step
        next_state, reward, done, info = env.step(action)

        # add transition to episode_experience as a tuple of
        # (state, action, reward, next_state, goal)
        episode_experience.append((state, action, reward, next_state, goal_state))

        # update episodic return
        episodic_return += reward

        # update state
        state = next_state

        # update succeeded bool from the info returned by env.step
        succeeded = info.get('successful_this_state', False)

        # break the episode if done=True
        if done:
            break

        # ========================      END TODO       ========================

    return episode_experience, episodic_return, succeeded
