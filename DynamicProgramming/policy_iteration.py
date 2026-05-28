
import copy


class PolicyIteration:
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
        self.n_actions = n_actions
        self.V = [0]*self.env.ncol * self.env.nrow
        self.policy = [[0.25]*4]*self.env.ncol * \
            self.env.nrow  # Uniform policy beginning

    def policy_evaluation(self):
        cnt = 1
        while True:
            max_diff = 0
            new_V = [0] * self.env.ncol * self.env.nrow
            for s in range(self.env.ncol * self.env.nrow):
                qsa_list = []
                for action_idx in range(self.n_actions):
                    qsa = 0
                    for p, next_state, reward, done in self.env.P[s][action_idx]:
                        qsa += p * (reward + self.gamma *
                                    self.V[next_state] * (1 - done))
                    qsa = qsa * self.policy[s][action_idx]
                    qsa_list.append(qsa)
                new_V[s] = sum(qsa_list)
                max_diff = max(max_diff, abs(new_V[s] - self.V[s]))
            self.V = new_V
            if max_diff < self.theta:
                break
            cnt += 1
        print("Ran policy evaluation for {} times".format(cnt))

    def policy_improvement(self):
        for s in range(self.env.nrow * self.env.ncol):
            qsa_list = []
            for a in range(self.n_actions):
                qsa = 0
                for p, next_state, reward, done in self.env.P[s][a]:
                    qsa += p * (reward + self.gamma *
                                self.V[next_state] * (1 - done))
                qsa_list.append(qsa)
            max_q = max(qsa_list)
            cnt_q = qsa_list.count(max_q)
            self.policy[s] = [1 / cnt_q if q == max_q else 0 for q in qsa_list]
            return self.policy

    def policy_iteration(self):
        cnt = 1
        while True:
            self.policy_evaluation()
            old_policy = copy.deepcopy(self.policy)
            new_policy = self.policy_improvement()
            if new_policy == old_policy:
                break
            self.policy = new_policy
            cnt += 1
        print("Ran policy evaluation for {} times".format(cnt))
