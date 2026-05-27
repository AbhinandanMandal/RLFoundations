
import numpy as np

"""
The agent solvers like Epsilon-Greedy, Decay Epsilon-Greedy, UCB, Thomson sampling 
They are choosing an arm and this MABEnv. class returns reward associated with the arm

"""



class MABEnv:
    def __init__(self, K, seed = 100):
        """
        K: number of arms
        k: the index of arm to pull
        return: the reward of the arm, the reward is 1 with probability probs[k], else 0
        """
        np.random.seed(seed)
        self.probs  = np.random.rand(K) # Initializing probabilities of each arm
        self.best_idx = np.argmax(self.probs) # The best arm
        self.K = K # Total number of arms
        self.best_prob = self.probs[self.best_idx] # The probability of the best arm

    def pull(self,k):
        return np.random.binomial(1, self.probs[k])
    