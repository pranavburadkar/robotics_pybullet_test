import numpy as np

class RLAgent:
    def __init__(self, action_space=4):
        self.q_table = np.zeros(action_space)
        self.epsilon = 0.2

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(len(self.q_table))
        else:
            return np.argmax(self.q_table)

    def update(self, action, reward):
        alpha, gamma = 0.4, 0.9
        best_q = np.max(self.q_table)
        self.q_table[action] += alpha * (reward + gamma * best_q - self.q_table[action])
