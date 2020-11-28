import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, state_dim, action_dim, max_timesteps,
                 start_timesteps, max_size=int(1e6),
                 alpha=1.0, beta=0.0, beta_anneal=True):
        super().__init__(state_dim, action_dim, max_size)
        self.priority = np.zeros(max_size)
        self.adjustment = 0
        self.train_timesteps = max_timesteps - start_timesteps - self.adjustment

        self.alpha = float(alpha)
        self.beta = beta
        self.beta_anneal = beta_anneal

    def add(self, state, action, next_state, reward, done):
        self.priority[self.ptr] = max(np.max(self.priority), 1.0)

        super().add(state, action, next_state, reward, done)

    def sample(self, batch_size):
        # import pdb; pdb.set_trace()
        scaled_priorities = np.power(self.priority, self.alpha)[:self.size]
        self.prob = scaled_priorities / np.sum(scaled_priorities)
        self.ind = np.random.choice(
            self.size, p=self.prob, size=batch_size, replace=True)
        self.weights = self.compute_weights()

        if self.beta_anneal:
            self.beta = min(self.beta + (1.0 / self.train_timesteps), 1.0)

        return (
            torch.FloatTensor(self.state[self.ind]).to(self.device),
            torch.FloatTensor(self.action[self.ind]).to(self.device),
            torch.FloatTensor(self.next_state[self.ind]).to(self.device),
            torch.FloatTensor(self.reward[self.ind]).to(self.device),
            torch.FloatTensor(self.not_done[self.ind]).to(self.device)
        )

    def update_priority(self, td_error):
        self.priority[self.ind] = np.abs(td_error.detach().numpy())

    def compute_weights(self):
        weights = ((1.0 / self.size) * (1.0 / np.take(self.prob, self.ind)))
        beta_weights = np.power(weights, self.beta)
        return beta_weights / np.max(beta_weights)
