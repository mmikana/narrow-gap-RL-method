import numpy as np
class ReplayMemory:
    def __init__(self, memo_capacity, state_dim, action_dim):
        self.memo_size = memo_capacity
        self.state_memo = np.zeros((self.memo_size, state_dim))
        self.next_state_memo = np.zeros((self.memo_size, state_dim))
        self.action_memo = np.zeros((self.memo_size, action_dim))
        self.reward_memo = np.zeros(self.memo_size)
        self.done_memo = np.zeros(self.memo_size)
        self.memo_counter = 0

    def add_memory(self, state, action, reward, next_state, done):
        index = self.memo_counter % self.memo_size
        self.state_memo[index] = state
        self.next_state_memo[index] = next_state
        self.action_memo[index] = action
        self.reward_memo[index] = reward
        self.done_memo[index] = done

        self.memo_counter += 1

    def sample_memory(self, batch_size):
        current_memo_size = min(self.memo_counter, self.memo_size)
        batch = np.random.choice(current_memo_size, batch_size, replace=False)
        batch_state = self.state_memo[batch]
        batch_action = self.action_memo[batch]
        batch_reward = self.reward_memo[batch]
        batch_next_state = self.next_state_memo[batch]
        batch_done = self.done_memo[batch]

        return batch_state, batch_action, batch_reward, batch_next_state, batch_done




