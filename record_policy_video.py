import os
import sys
import numpy as np
from TD3 import *
import argparse
import gym
import pybulletgym
from glob import glob
from gym import wrappers

def select_action(actor, state):
    state = torch.FloatTensor(state.reshape(1, -1))
    return actor(state).cpu().data.numpy().flatten()

parser = argparse.ArgumentParser()
parser.add_argument("--policy", default="TD3")  
# Policy name (TD3, DDPG or OurDDPG)
parser.add_argument("--env", default="HalfCheetahMuJoCoEnv-v0") # OpenAI gym environment name
parser.add_argument("--prioritized_replay", default=False, action='store_true')		# Include this flag to use prioritized replay buffer
parser.add_argument("--use_rank", default=False, action="store_true")               # Include this flag to use rank-based probabilities
parser.add_argument("--use_hindsight", default=False, action="store_true")          # Include this flag to use HER
parser.add_argument("--custom_env", default=False, action="store_true")             # our custom environment name
# annealing reacher epsilon: default is a linear 5e-4 -> 5e-4 (aka constant at 5e-4)
parser.add_argument("--reacher_epsilon_bounds", default=[5e-4, 5e-4], nargs=2, type=float, help="upper and lower epsilon bounds")
parser.add_argument("--decay_type", default="linear", help="'linear' or 'exp' epsilon decay")
parser.add_argument("--k", default=1, type=int)                             # k number of augmentations for HER
args, unknown = parser.parse_known_args()
if unknown:
    print("WARNING: unknown arguments:", unknown)

eps_bounds = args.reacher_epsilon_bounds
fetch_reach = "FetchReach" in args.env
exp_descriptors = [
    args.policy, 'CustomReacher' if args.custom_env else args.env,
    f"{'rank' if args.use_rank else 'proportional'}PER" if args.prioritized_replay else '', 
    'HER' if args.use_hindsight else '',
    f"{args.decay_type}decay-eps{f'{eps_bounds[0]}-{eps_bounds[1]}' if eps_bounds[0] != eps_bounds[1] else f'{eps_bounds[0]}'}" if args.custom_env else "",
    f"k{args.k}",
]
exp_descriptors = [x for x in exp_descriptors if len(x) > 0]
file_name = "_".join(exp_descriptors)       # file name root (minus timestamp)

if args.custom_env:
    gym.envs.register(
        id='OurReacher-v0',
        entry_point='our_reacher_env:OurReacherEnv',
        max_episode_steps=150,
        reward_threshold=100.0,
    )
    epsilon = eps_bounds[1]     # use latest epsilon
    env = gym.make('OurReacher-v0', epsilon=float(epsilon), render=True)
else:
    env = gym.make(args.env)

if fetch_reach:
    if args.use_hindsight:
        state_dim = env.observation_space["observation"].shape[0] + 3
    else:
        state_dim = env.observation_space["observation"].shape[0]
else:
    state_dim = env.observation_space.shape[0]
if args.custom_env:
    state_dim += 2

action_dim = env.action_space.shape[0] 
max_action = float(env.action_space.high[0])

files = []
for f in glob(f"./models/{file_name}*"):
    if f.endswith("actor"):
        files.append(f)
files.sort(reverse=True)        # most recent -> least recent

if len(files) == 0:
    print("WARNING: no files found matching", file_name)
    quit()
actor_filename = files[0]         # just use the most recent one; we can add options later
# outfile_stem = f"./plots/{Path(file_to_load).stem}"      # output file stem, with timestamp

actor_to_load = os.path.join(os.path.dirname(os.path.realpath(__file__)), actor_filename)

policy = Actor(state_dim, action_dim, max_action).to(device)
policy.load_state_dict(torch.load(actor_to_load))

env = wrappers.Monitor(env, f'./recordings/{file_name}/', force=True, video_callable=lambda episode_id: True)
# env.render()
state, done = env.reset(), False
for _ in range(1000):
    # env.render()
    if args.custom_env:
        goal = env.sample_goal_state(sigma=0)
        state = np.concatenate([np.array(state), goal])
    elif fetch_reach:
        if args.use_hindsight:
            goal = state["desired_goal"]
        else:
            goal = np.array([])
        state = np.concatenate([np.array(state["observation"]), goal])
    else:
        state = np.array(state)
    action = select_action(policy, state)
    state, reward, done, _ = env.step(action)

    if done:
        state, done = env.reset(), False
env.close()

print(f"Copy paste this command into terminal to create the output mp4: ./concat_vids.sh \"{file_name}\"")