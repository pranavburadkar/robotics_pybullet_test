import numpy as np

class RLAgent:
    def __init__(self, state_space_size, action_space=4):
        self.q_table = np.zeros((state_space_size, action_space))
        self.epsilon = 0.2 # Exploration-exploitation trade-off
        self.alpha = 0.4   # Learning rate
        self.gamma = 0.9   # Discount factor
        self.state_space_size = state_space_size

    def _discretize_state(self, num_unknown, map_total_cells):
        # Discretize the percentage of unknown cells into a few bins
        # Example: 0-25%, 25-50%, 50-75%, 75-100%
        percentage_unknown = num_unknown / map_total_cells
        if percentage_unknown > 0.75:
            return 0  # Mostly unknown
        elif percentage_unknown > 0.5:
            return 1  # Moderately unknown
        elif percentage_unknown > 0.25:
            return 2  # Partially known
        else:
            return 3  # Mostly known

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.q_table.shape[1]) # Random action
        else:
            return np.argmax(self.q_table[state, :]) # Best action from Q-table

    def update(self, state, action, reward, next_state):
        best_next_q = np.max(self.q_table[next_state, :])
        self.q_table[state, action] += self.alpha * (reward + self.gamma * best_next_q - self.q_table[state, action])
