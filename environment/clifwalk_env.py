
class ClifWalkingWithPEnv:
    def __init__(self, ncol=12, nrow=4):
        self.nrow = nrow
        self.ncol = ncol
        # Specifed actions are up, down, right, left
        self.action = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        self.P = self.createP()

    def createP(self):  # Creation of possible transitions
        P = [[[] for j in range(4)] for i in range(self.nrow * self.ncol)]
        # P consist of p = probability, next_state, reward, done given current state and action
        for i in range(self.nrow):
            for j in range(self.ncol):
                for action_idx in range(len(self.action)):
                    cur_state = i*self.ncol+j
                    if i == self.nrow-1 and j > 0:
                        # reached the clif
                        next_state = i*self.ncol+j
                        reward = 0
                        done = True
                        P[cur_state][action_idx] = [
                            (1.0, next_state, reward, done)]
                    else:
                        next_i = min(
                            self.ncol-1, max(0, j+self.action[action_idx][0]))
                        next_j = min(
                            self.nrow-1, max(0, i+self.action[action_idx][1]))
                        next_state = next_j*self.ncol + next_i
                        reward = -1
                        done = False
                        if next_j == self.nrow-1 and next_i > 0:
                            done = True
                            if next_i != self.ncol-1:
                                reward = -100
                        P[cur_state][action_idx] = [
                            (1.0, next_state, reward, done)]
        return P


class CliffWalkingWithoutPEnv:
    def __init__(self, ncol=12, nrow=4):
        self.nrow = nrow
        self.ncol = ncol
        self.x = 0
        self.y = self.nrow - 1

    def step(self, action):
        action_space = [[0, -1], [0, 1], [-1, 0],
                        [1, 0]]  # up, down, left, right
        self.x = min(max(self.x + action_space[action][0], 0), self.ncol - 1)
        self.y = min(max(self.y + action_space[action][1], 0), self.nrow - 1)
        next_state = self.y * self.ncol + self.x
        reward = -1
        done = False
        if self.y == self.nrow - 1 and self.x > 0:  # reach to the cliff or the goal
            done = True
            if self.x != self.ncol - 1:
                reward = -100
        return next_state, reward, done

    def reset(self):
        self.x = 0
        self.y = self.nrow - 1
        return self.y * self.ncol + self.x
