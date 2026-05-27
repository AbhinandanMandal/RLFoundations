
"""
In Decay EPsilon-Greedy, Epsilon decreases over span of time t
"""
from MultiArmBandit.epsilon_greedy import EpsilonGreedy


class DecayEpsilonGreedy(EpsilonGreedy):
    def __init__(self, bandit, init_epsilon, decay_rate, init_prob=1.0):
        super(DecayEpsilonGreedy, self).__init__(
            bandit, init_epsilon, init_prob)
        self.decay_rate = decay_rate

    def run_one_step(self):
        self.epsilon *= self.decay_rate
        return super(DecayEpsilonGreedy, self).run_one_step()
