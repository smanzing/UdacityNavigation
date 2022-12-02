import torch
import numpy as np
from collections import deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def dqn(env, brain_name, agent, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995,
        min_avg_score=13.0, continue_learning=False):
    """
    Deep Q-Learning.

    @param env: unity environment
    @param brain_name: name of the brain that we control from Python
    @param agent: the RL agent that we train
    @param n_episodes: maximum number of training episodes
    @param max_t: maximum number of timesteps per episode
    @param eps_start: starting value of epsilon, for epsilon-greedy action selection
    @param eps_end: minimum value of epsilon
    @param eps_decay: multiplicative factor (per episode) for decreasing epsilon
    @param min_avg_score: minimum average score over 100 episodes that the agent must achieve to consider the task fulfilled
    @param continue_learning: if true, the agent continues to learn after reaching min_avg_score until reaching n_episodes
    @return:
    """

    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    min_score_achieved = False
    for i_episode in range(1, n_episodes + 1):
        # reset the environment
        env_info = env.reset(train_mode=True)[brain_name]
        # get current state
        state = env_info.vector_observations[0]
        # initialize score
        score = 0
        for t in range(max_t):
            # select action
            action = agent.act(state, eps)
            # simulate
            env_info = env.step(action)[brain_name]
            # get next state, reward, and check if episode has finished
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]

            agent.step(state, action, reward, next_state, done)

            # update state and reward
            state = next_state
            score += reward

            if done:
                break

        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= min_avg_score and not min_score_achieved:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint_min_score_achieved.pth')
            min_score_achieved = True
            if not continue_learning:
                break
    torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
    return scores
