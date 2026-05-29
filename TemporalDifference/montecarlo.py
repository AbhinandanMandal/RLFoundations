
import numpy as np
from collections import defaultdict


class MonteCarlo:
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
        # Q table formation, initally all values are zero
        self.Q = np.zeros((self.env.ncol * self.env.nrow, n_actions))
        # return sum for each (state, action) pair
        self.returns_sum = defaultdict(float)
        # number of times when (state, action) pair is similar
        self.returns_count = defaultdict(float)
        # Monte Carlo compites Q(s,a) = Total Returns / Visits
        self.return_list = []  # stores total reward per episodes, it is used to monitor learning

    # This implements epsilon-greedy approach for exploration and exploitation

    def take_action(self, state):
        """
        state: current state
        return: action to take
        """
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            action = np.argmax(self.Q[state])
        return action  # balances exploration and exploitation action

    # For best optimal action
    def best_action(self, state):
        """
        state: the current state
        return: the best actions
        """
        Q_max = np.max(self.Q[state])
        a = [0 for _ in range(self.n_actions)]  # let say a = [0, 0, 0, 0]
        for i in range(self.n_actions):
            if self.Q[state][i] == Q_max:
                a[i] = 1  # and later a = [0, 1, 1, 0], so 1, 2 actions are best
        return a

    # Monte Carlo update
    def update(self, episode_trajectory):
        """
        It update Q values.
        After update we've like
        [
        .....
        (state, action, reward),
        (state, action, reward),
        .....
        ]
        e.g.
        [
        (state, action, pair)
        (0,1,0),
        (1,2,0),
        (5,3,10)
        ]
        episode_trajectory: a list of (state, action, reward)
        """
        sa_in_episode = set([(x[0], x[1]) for x in episode_trajectory]
                            # unique and first occurance (state, action) pair
                            )
        for state, action in sa_in_episode:
            sa_pair = (state, action)  # a key formation
            first_occurrence_idx = next(i for i, x in enumerate(
                # first occurance finding
                episode_trajectory) if x[0] == state and x[1] == action)
            # Returns calculation from the next time step t+1 of rewards, x[2] is reward
            G = sum(x[2]*(self.gamma**i)
                    for i, x in enumerate(episode_trajectory[first_occurrence_idx:]))
            self.returns_sum[sa_pair] += G
            self.returns_count[sa_pair] += 1.0
            self.Q[state][action] = self.returns_sum[sa_pair] / \
                self.returns_count[sa_pair]  # Q(s,a) value estimate findings

    # Training the learning process
    def train(self):
        np.random.seed(self.seed)
        for i in range(self.num_episodes):
            state = self.env.reset()
            episode = []  # list of (state, action, reward)
            episode_reward = 0
            while True:
                action = self.take_action(state=state)
                # finds the next_state, reward and it's done or not
                next_state, reward, done = self.env.step(action)
                episode.append((state, action, reward))
                self.update(episode)  # updating
                state = next_state
                episode_reward += reward
                if done:
                    break  # If the episode is complete then break it
                self.return_list.append(episode_reward)
                if (i+1) % 10 == 0:
                    print("Average reward for the last 10 episodes with "
                          "from {} to {} is: {}".format(i - 9, i + 1, np.mean(self.return_list[-10:])))
