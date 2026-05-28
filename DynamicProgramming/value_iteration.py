
"""
Dynamic Programming: Valued based approach, applicable for known environment.
1. value iteration: aims to find the optimal value function by iteratively updating the bellman value function.
2. policy iteration: aims to find the optimal policy by iteratively updating the bellman Q equation
"""


class ValueIteration:
    def __init__(self, env, theta, gamma, n_actions=4):
        """
        env: Agent Environment
        theta: Threshold for stopping iteration
        gamma: Disocunting facotr
        n_actions: No of possible actions that agent can take

        Here we're considering grid world for agent. So, agent can take
        only 4 actions, UP, DOWN, LEFT, RIGHT
        """
        self.env = env
        self.theta = theta
        self.gamma = gamma
        # Initially V[s] = 0, cause we don't know which states are good or bad
        self.V = [0]*self.env.ncol * self.env.nrow
        # Initially all policy are None for same reason
        self.policy = [None]*self.env.ncol * self.env.nrow
        self.n_actions = n_actions

    def value_iteration(self):
        cnt = 0
        while True:  # We repeatedly improve values untill convergence
            max_diff = 0
            # We compute updated value seperately, it avoids overwriting old values during iteration
            new_V = [0]*self.env.ncol * self.env.nrow
            for s in range(self.env.ncol * self.env.nrow):  # for each state in environment
                qsa_list = []  # quality pairs (state, action)
                for action_idx in range(self.n_actions):
                    qsa = 0
                    """
                    self.env.P[s][action_idx] contains all possible transitions
                    Each transition has (p, next_state, reward, done)
                    Here, 
                    p = probability of action taken
                    next_state = next state
                    reward = immediate reward
                    done = terminal state ? 

                    Example
                    [
                    (0.8, 5, -1, False),
                    (0.2, 6, -1, False)
                    ]
                    Means 80% chance of state 5, reward -1, not a terminal state
                          20% chance of state 6, reward -1, not a terminal state
                    """
                    for p, next_state, reward, done in self.env.P[s][action_idx]:
                        # Bellman update
                        qsa += p*(reward + self.gamma *
                                  self.V[next_state] * (1 - done))
                    qsa_list.append(qsa)
                new_V[s] = max(qsa_list)
                max_diff = max(max_diff, abs(new_V[s] - self.V[s]))
            self.V = new_V
            cnt += 1  # Increase counting by 1
            if max_diff < self.theta:
                break
        print("Run value iterations for {} tines".format(cnt))
        self.get_policy()  # After value converges, we'll compute which action gives maximul value

    def get_policy(self):
        for s in range(self.env.nrow * self.env.ncol):
            qsa_list = []
            for action_idx in range(self.n_actions):
                qsa = 0
                for p, next_state, reward, done in self.env.P[s][action_idx]:
                    # qsa += reward + self.gamma*p*self.V[next_state]*(1-done)
                    qsa += p*(reward + self.gamma *
                              self.V[next_state]*(1 - done))
                qsa_list.append(qsa)
            max_q = max(qsa_list)
            cnt_q = qsa_list.count(max_q)
            self.policy[s] = [1/cnt_q if q == max_q else 0 for q in qsa_list]
        return self.policy


"""
Function get_policy()
After the value converges, it'll give which actions give maximum value
Insife get_policy(), we'll again compute qsa pair and will find max_q to visualize
if multiple simulatanious actions are optimal.

For example, 
qsa_list = [10, 10, 5, 2]
then max_q = 10, cnt = 2
So, probabilistic policy will be [0.5, 0.5, 0, 0], (up, down are equally optimal)

Time complexity of the following value iteration method is O(SAT)
S = States
A = Actions
T = Transitions
"""
