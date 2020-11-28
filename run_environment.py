import os
import sys
import copy
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from TD3 import Actor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    env_name = "Hopper-v3"
    filename = os.path.join(sys.path[0], "models", "TD3_" + env_name + "_0_actor")

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    policy = Actor(state_dim, action_dim, max_action)

    # load actor
    policy.load_state_dict(torch.load(filename))

    state, done = env.reset(), False

    while not done:
        env.render()
        raw_state = torch.FloatTensor(np.array(state).reshape(1, -1)).to(device)
        action = policy(raw_state).cpu().data.numpy().flatten()
        next_state, reward, done, _ = env.step(action)
        state = next_state 
    env.close()



    