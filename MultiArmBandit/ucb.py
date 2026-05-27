
"""
UCB: Upper Confidence Bound
The fundamental approach of UCB is: Instead of choosing arm with highest current value
It chooses the arm with highest potential value

Fundamentally UCB is a deterministic algorithm, not a random one.
"""
import numpy as np
from MultiArmBandit.epsilon_greedy import BaseSolver


class UCB(BaseSolver):
    def __init__(self, bandit, coef, init_prob=1.0):
        """
        bandit: multi-armed bandit environment
        coef: cofficient controlling weight of upper confidence bound
        init_prob: initial expected reward of each arm
        """
        super(UCB, self).__init__(bandit)
        self.coef = coef
        self.total_count = 0
        self.estimates = init_prob*np.ones(self.bandit.K)

    def run_one_step(self):
        self.total_count += 1
        ucb = self.estimates + self.coef * \
            np.sqrt(np.log(self.total_count) / (2*(self.counts+1)))
        k = np.argmax(ucb)
        r = self.bandit.pull(k)
        self.estimates[k] += 1.0/(self.counts[k]+1) * (r - self.estimates[k])
        return k
