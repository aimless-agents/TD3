import numpy as np
import torch
from numba import jit
import matplotlib.pyplot as plt
from ray import tune


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

    def sample(self, batch_size, use_rank=False):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )
    
    def add_hindsight(self, trajectory, goal, env, k=4, fetch_reach=False):
        p_hindsight = 1.0 - (1.0 / (1.0 + k))
        hindsight_er = np.random.uniform(size=(len(trajectory) - 1)) < p_hindsight
        for i in range(len(trajectory) - 1):
            state, action, next_state, gc_reward, done_bool = trajectory[i]
            if hindsight_er[i]:
                idx = np.random.choice(np.arange(i + 1, len(trajectory)))
                future_state, _, _, _, _ = trajectory[idx]
                x, next_x, gc_reward = self.updated_hindsight_experience(state, next_state, future_state, env, fetch_reach)
            else:
                x = np.concatenate([np.array(state["observation"]) if fetch_reach else np.array(state), goal])
                next_x = np.concatenate([np.array(next_state["observation"]) if fetch_reach else np.array(next_state), goal])
            self.add(x, action, next_x, gc_reward, done_bool)

    def updated_hindsight_experience(self, state, next_state, future_state, env, fetch_reach): 
        if fetch_reach:
            new_goal = future_state["desired_goal"]
            x = np.concatenate([np.array(state["observation"]), new_goal])
            next_x = np.concatenate([np.array(next_state["observation"]), new_goal])
            gc_reward = env.compute_reward(next_state["achieved_goal"], new_goal, {})
        else:
            new_goal = np.array([future_state[0] + future_state[2], future_state[1] + future_state[3]])
            x = np.concatenate([state, new_goal])
            next_x = np.concatenate([next_state, new_goal])
            gc_reward = env.goal_cond_reward(next_state, new_goal) 
        return x, next_x, gc_reward  


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, state_dim, action_dim, max_timesteps,
                 start_timesteps, max_size=int(1e6),
                 alpha=1.0, beta=0.0):
        super().__init__(state_dim, action_dim, max_size)
        self.priority = np.zeros(max_size)
        self.adjustment = 0
        self.start_timesteps = start_timesteps
        self.max_timesteps = max_timesteps
        self.train_timesteps = max_timesteps - start_timesteps - self.adjustment
        self.rank_ctr = 0
        self.norm_list = np.zeros(max_size)
        self.batched_ranklist = np.zeros(max_size)

        self.alpha = float(alpha)
        self.beta = beta

    def add(self, state, action, next_state, reward, done):
        self.priority[self.ptr] = max(np.max(self.priority), 1.0)

        super().add(state, action, next_state, reward, done)

    def rank_probs(self):
        if self.rank_ctr % 256 == 0:
            problist = list(enumerate(self.priority[:self.size]))
            problist.sort(key=lambda priority : priority[1])
            ranklist = [(len(problist) - new_idx, old_idx) for (new_idx, (old_idx, _)) in enumerate(problist)]
            batched_ranklist = [(1.0/rank, i) for rank, i in ranklist]
            '''
            each segment is of size self.size/batch_size S
            sample 1
            '''
            self.batched_ranklist = batched_ranklist.copy()
            self.batched_ranklist.sort(key=lambda rankidx : rankidx[0], reverse=True)
            
            batched_ranklist.sort(key=lambda rankidx : rankidx[1])
            new_list = [score for score, idx in batched_ranklist]
            norm_list = new_list / np.sum(new_list)
            self.norm_list[:self.size] = norm_list


    def sample(self, batch_size, use_rank=False):
        if use_rank:
            self.rank_probs()
            # self.prob = self.norm_list[:self.size]
            # self.prob /= np.sum(self.prob)
            # self.ind = np.random.choice(self.size, p=self.prob, size=batch_size, replace=True)
            self.ind = np.zeros(batch_size)
            self.prob = np.zeros(batch_size)

            # prelim_p = np.zeros(batch_size)
            # p = np.zeros(batch_size)
            for i in range(256):
                S = (len(self.batched_ranklist)//batch_size)
                if i==255:
                    segment = self.batched_ranklist[i * S:]
                else:
                    segment = self.batched_ranklist[i * S:(i * S) + S]
                p = np.array([rank for (rank, _) in segment])
                p /= np.sum(p)
                rand_choice = np.random.choice(len(segment), p=p)
                self.ind[i] = segment[rand_choice][1]
                self.prob[i] = segment[rand_choice][0]
            self.ind = self.ind.astype(int)
            self.rank_ctr += 1

        else:
            scaled_priorities = np.power(self.priority, self.alpha)[:self.size]
            self.prob = scaled_priorities
            self.prob /= np.sum(self.prob)
            self.ind = np.random.choice(self.size, p=self.prob, size=batch_size, replace=True)
        self.weights = self.compute_weights(use_rank)

        self.beta = min(self.beta + (1.0 / (self.train_timesteps - self.start_timesteps)), 1.0)

        return (
            torch.FloatTensor(self.state[self.ind]).to(self.device),
            torch.FloatTensor(self.action[self.ind]).to(self.device),
            torch.FloatTensor(self.next_state[self.ind]).to(self.device),
            torch.FloatTensor(self.reward[self.ind]).to(self.device),
            torch.FloatTensor(self.not_done[self.ind]).to(self.device)
        )

    def plot(self, list_to_plot):
        plt.plot(np.arange(len(list_to_plot)), list_to_plot)
        plt.show()

    def update_priority(self, td_error):
        self.priority[self.ind] = np.abs(td_error.detach().numpy())

    def compute_weights(self, use_rank):
        if use_rank:
            weights = ((1.0 / self.size) * (1.0 / self.prob))
        else:
            weights = ((1.0 / self.size) * (1.0 / np.take(self.prob, self.ind)))
        beta_weights = np.power(weights, self.beta)
        return beta_weights / np.max(beta_weights)

# Calculate the epsilon range given upper+lower bounds, maximum timesteps, decay type.
# Possible decay types:
    # - linear
    # - exponential
# Assumes that the env is custom reacher environment, 
#   where episodes are always 150 timesteps (I think??)
def epsilon_calc(eps_upper, eps_lower, max_timesteps, 
                    decay='linear'):
    num_episodes = int(np.ceil(max_timesteps / 150))     # 150 for custom reacher specifically
    x = np.arange(num_episodes)
    if eps_upper == eps_lower:
        return np.full(num_episodes, eps_upper)
    if decay == 'linear':
        epsilon_step = (eps_upper - eps_lower) / num_episodes
        return np.arange(eps_upper, eps_lower, -epsilon_step)
    if decay == 'exp':
        return eps_upper * (1 - eps_lower) ** x

def get_train_configuration(args):
    config = {}
    if args.tune_run:
        if args.prioritized_replay:
            config = {
                "beta": tune.grid_search([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
                "alpha": tune.grid_search([0.3, 0.4, 0.5, 0.6]),
                "discount": tune.grid_search([0.995, 0.996, 0.997, 0.998, 0.999]),
                "tau": tune.grid_search([1e-5, 5e-4, 1e-4])
            }
        elif args.use_hindsight:
            eps_ranges = [
                [7e-3, 1e-3],
                [7e-3, 1e-4],
                [7e-3, 5e-5],
                [7e-3, 1e-5],
                [5e-3, 1e-3],
                [5e-3, 1e-4],
                [5e-3, 5e-5],
                [5e-3, 1e-5],
                [1e-3, 1e-4],
                [1e-3, 5e-5],
                [1e-3, 1e-5]
            ]
            config = {
                "epsilons": tune.grid_search(eps_ranges),
                "seed": tune.grid_search([0, 2, 4, 8, 16])
            }
        else: 
            config = {
                "discount": tune.grid_search([0.995, 0.996, 0.997, 0.998, 0.999]),
                "tau": tune.grid_search([1e-5, 5e-4, 1e-4])
            }
    return config

class GeneralUtils():
    def __init__(self, args):
        self.args = args
        self.fetch_reach = 'FetchReach' in args.env

    def compute_x_goal(self, state, env, sigma=1e-3):
        goal = None
        if self.args.use_hindsight:
            if self.fetch_reach:
                goal = state["desired_goal"]
                x = np.concatenate([np.array(state["observation"]), goal])
            else:
                goal = env.sample_goal_state(sigma=sigma)
                x = np.concatenate([np.array(state), goal])
        elif self.fetch_reach:
            x = np.array(state["observation"])
        else:
            x = np.array(state)
        return x, goal