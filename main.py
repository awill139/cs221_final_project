import gym
import numpy as np
import matplotlib.pyplot as plt
from d3qn import Agent

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    n_actions = env.action_space.n
    obs_shape = list(env.observation_space.shape)
    agent = Agent(gamma = 0.99, n_actions = n_actions, epsilon = 1.0, batch_size = 64, input_dims = obs_shape)

    n_games = 1000
    scores = []
    log_scores = []
    eps_history = []

    #prepopulate memory
    # while agent.memory.mem_cntr < 50000:
    #     done = False
    #     score = 0
    #     obs = env.reset()
    #     while not done:
    #         action = agent.choose_action(obs)
    #         obs_, reward, done, info = env.step(action)

    #         agent.store_transition(obs, action, reward, obs_, int(done))
    #         obs = obs_

    for i in range(n_games):
        done = False
        score = 0
        obs = env.reset()
        # env.render()
        while not done:
            # env.render()
            action = agent.choose_action(obs)
            obs_, reward, done, info = env.step(action)

            agent.store_transition(obs, action, reward, obs_, int(done))
            score += reward
            obs = obs_
            agent.learn()
        eps_history.append(agent.epsilon)
        scores.append(score)
        
        avg_score = np.mean(scores[-100:])
        if i % 25 == 0:
            log_scores.append(score)
            print('episode: {}\t curr_score: {}\t avg score: {}'.format(i, score, avg_score))

    linspace = np.linspace(0, n_games, len(log_scores))
    plt.plot(linspace, log_scores)
    plt.ylabel('Scores')
    plt.xlabel('Games')
    plt.savefig('scores_learned_td_1000.png')
    plt.savefig('scores_learned_td_1000_bu.png')
    # plt.savefig('scores_weighted_1000.png')
    # plt.savefig('scores_weighted_1000_bu.png')
    # plt.savefig('scores_learned_1000_q_0.png')
    # plt.savefig('scores_learned_1000_q_0_bu.png')