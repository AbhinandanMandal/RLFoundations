
"""
Thomson Sampling is a baysian appraoch of solving Multi-Arm bandit problem
Instead of choosing the highest estimated reward directly,
It samples from probability distribution. 

TS balances exploration and exploitation naturally
"""

from MultiArmBandit.epsilon_greedy import BaseSolver
import numpy as np


class ThomsonSampling(BaseSolver):
    def __init__(self, bandit):
        super(ThomsonSampling, self).__init__(bandit)
        self.alpha = np.ones(self.bandit.K)
        self.beta = np.ones(self.bandit.K)

    def run_one_step(self):
        samples = np.random.beta(self.alpha, self.beta)
        k = np.argmax(samples)
        r = self.bandit.pull(k)
        self.alpha[k] += 1
        self.beta[k] += 1-r
        return k
