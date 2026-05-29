
"""
Unlike to SARSA, it's a T.D off policy learning algorithm
It always prefers future optimal behaviour
It chooses the shortest approach although risk associated with it

"""

import numpy as np


class QLearning:
    def __init__(self, env, gamma, alpha, epsilon, n_actions=4, num_episodes=500, seed=0):
        """
        env: environment for monte carlo simulation
        gamma: discounting factor
        alpha: learning rate
        epsilon: epsilon for epsilon-greedy policy
        n_actions: number of actions
        num_episodes: no of episodes
        """
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.num_episodes = num_episodes
        self.seed = seed
        self.Q = np.zeros((self.env.ncol * self.env. nrow, n_actions))
        self.return_list = []  # store the return for each episode

    # Using epsilon-greedy exploration strategy
    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            # only take the first best action when tie happens
            action = np.argmax(self.Q[state])
        return action

    # For best action
    def best_action(self, state):
        Q_max = np.max(self.Q[state])
        a = [0 for _ in range(self.n_actions)]
        for i in range(self.n_actions):
            if self.Q[state][i] == Q_max:
                a[i] = 1
        return a

    # Updating Q(s,a)

    def update(self, state, action, reward, next_state):
        next_action = np.argmax(self.Q[next_state])
        td_error = reward + self.gamma * \
            self.Q[next_state][next_action] - self.Q[state][action]
        self.Q[state][action] += self.alpha*td_error

    def train(self):
        np.random.seed(self.seed)
        for i in range(self.num_episodes):
            episode_reward = 0
            state = self.env.reset()
            done = False

            while not done:
                action = self.take_action(state)
                next_state, reward, done = self.env.step(action)
                episode_reward += reward
                self.update(state, action, reward, next_state)
                state = next_state
            self.return_list.append(episode_reward)
            if (i + 1) % 10 == 0:
                print("Average reward for the last 10 episodes with "
                      "from {} to {} is: {}".format(i - 9, i + 1, np.mean(self.return_list[-10:])))
