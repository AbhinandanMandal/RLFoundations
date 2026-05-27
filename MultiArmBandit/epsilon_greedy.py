
"""
RL from scratch github repository: https://github.com/Bingzw/reinforcement_learning_from_scratch
Video Lecture to understand Epsilon-Greedy and UCB: https://youtu.be/p13V-93aG1c?si=IYQWMG11r2M43kBw
"""

import numpy as np


# Base solver class of all Multi-Arm Bandit algorithms
class BaseSolver:
    def __init__(self, bandit):
        self.bandit = bandit  # Total number of bandits
        # No of times each arm has been pulled
        self.counts = np.zeros(self.bandit.K)

        # cumulative reget
        # Answers how much reward did we lose by not choosing the best arm
        self.regret = 0.0
        self.actions = []  # Total number of actions
        self.regrets = []  # Total number of regrets

    def update_regret(self, k):
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)

    def run_one_step(self):
        # return the index of the arm to pull
        raise NotImplementedError

    def run(self, num_steps):
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)


class EpsilonGreedy(BaseSolver):
    """
    EpsilonGreedy algorithm follows greedy approach of exploitation with probability epsilon
    probability epsilon helps it to choose a random arm 10% of time.
    Over 90%, it chooses the greedy approached arm
    """

    def __init__(self, bandit, epsilon, init_prob=1.0):
        """
        bandit: Multi-Arm Bandit environment
        epsilon: probability of choosing a random arm
        init_prob: initial reward of each arm
        """
        super(EpsilonGreedy, self).__init__(bandit)
        self.epsilon = epsilon
        # Initial estimate value of each arm
        self.estimates = init_prob*np.ones(self.bandit.K)
        """
        Let say, total bandit = 5
        self.estimates = [1.0, 1.0, 1.0, 1.0, 1.0]
        """

    def run_one_step(self):
        """
        If the random numbers probability is less than epsilon then we explore random arm
        Else, we choose greedy approach
        """
        if np.random.rand() < self.epsilon:
            k = np.random.choice(self.bandit.K)
        else:
            k = np.argmax(self.estimates)
        reward = self.bandit.pull(k)  # Expected reward of the arm
        # Updating the expected reward for specific arm
        self.estimates[k] += 1.0/(self.counts[k]+1) * \
            (reward - self.estimates[k])
        return k # arm 
    

