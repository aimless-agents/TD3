import os
import sys
import numpy as np
from TD3 import *
import gym
import pybulletgym

def select_action(actor, state):
    state = torch.FloatTensor(state.reshape(1, -1))
    return actor(state).cpu().data.numpy().flatten()

gym.envs.register(
                            id='OurReacher-v0',
                            entry_point='our_reacher_env:OurReacherEnv',
                            max_episode_steps=150,
                            reward_threshold=100.0,
                            )
# import pdb; pdb.set_trace()
epsilon = 5e-2
env = gym.make('OurReacher-v0', epsilon=epsilon, render=True)
env = gym.make("FetchReach-v1")


state_dim = env.observation_space["observation"].shape[0] + 3
action_dim = env.action_space.shape[0] 
max_action = float(env.action_space.high[0])

filename = "models/TD3_FetchReach-v1_k1_202101261131_actor"
actor_to_load = os.path.join(os.path.dirname(os.path.realpath(__file__)), filename)

policy = Actor(state_dim, action_dim, max_action).to(device)
policy.load_state_dict(torch.load(actor_to_load))

state, done = env.reset(), False
for i in range(10000):
    # goal = env.sample_goal_state(sigma=0)
    goal = state["desired_goal"]
    env.render()
    state = np.concatenate([np.array(state["observation"]), goal])
    action = select_action(policy, np.array(state))
    # observation, reward, done, info = env.step(env.action_space.sample())
    state, reward, done, _ = env.step(action) # take a random action
    if done:
        state, done = env.reset(), False
env.close()