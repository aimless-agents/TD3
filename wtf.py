import numpy as np
import torch
import gym
import argparse
import os
import sys
import pybulletgym

import utils
import TD3
import OurDDPG
import DDPG
import warnings

import ray 
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.schedulers import ASHAScheduler
from ray.tune.registry import register_env

def train(
    config,
    start_timesteps,
    max_timesteps,
    policy_noise, 
    expl_noise,
    noise_clip, 
    policy_freq,
    batch_size,
    seed,
    policy,
    prioritized_replay,
    env_name,
    eval_freq,
    discount,
    tau
):
    if prioritized_replay:
        alpha = float(config["alpha"])
        beta = float(config["beta"])
    else:
        discount = float(config["discount"])
        tau = float(config["tau"])
    import pybulletgym
    warnings.filterwarnings("ignore")
    env = gym.make('AntPyBulletEnv-v0')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="HalfCheetah-v2")          # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=75e4, type=int)   # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)     # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)         # Target network update rate
    parser.add_argument("--beta_step", default=0.008)               # Beta annealing step-size (should be 1/max_timesteps)
    parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name

    parser.add_argument("--prioritized_replay", default=False, action='store_true')		# Include this flag to use prioritized replay buffer
    parser.add_argument("--smoke_test", default=False, action='store_true')             # Include this flag to run a smoke test
    args = parser.parse_args()

    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if args.prioritized_replay:
        config = {
            "beta": tune.grid_search([0.3, 0.4, 0.5, 0.6]),
            "alpha": tune.grid_search([0.4, 0.5, 0.6, 0.7])
        }
    else: 
        config = {
            "discount": tune.grid_search([0.995, 0.996, 0.997, 0.998, 0.999]),
            "tau": tune.grid_search([1e-5, 5e-4, 1e-4])
        }

    kwargs = {}

    if not args.smoke_test:
        kwargs["start_timesteps"] = args.start_timesteps
        kwargs["max_timesteps"] = args.max_timesteps
        kwargs["eval_freq"] = args.eval_freq
    else:
        kwargs["start_timesteps"] = 25
        kwargs["max_timesteps"] = 75
        kwargs["eval_freq"] = 5
        
    kwargs["policy_noise"] = args.policy_noise
    kwargs["expl_noise"] = args.expl_noise
    kwargs["noise_clip"] = args.noise_clip
    kwargs["batch_size"] = args.batch_size
    kwargs["policy_freq"] = args.policy_freq
    kwargs["seed"] = args.seed
    kwargs["policy"] = args.policy
    kwargs["prioritized_replay"] = args.prioritized_replay
    kwargs["env_name"] = args.env
    kwargs["discount"] = args.discount
    kwargs["tau"] = args.tau

    result = tune.run(
        tune.with_parameters(train, **kwargs),
        local_dir=os.path.join(os.getcwd(), "results", "tune_results"),
        num_samples=1,
        scheduler=ASHAScheduler(metric="episode_reward_mean", mode="max"),
        config=config
    )
    gym.make('AntPyBulletEnv-v0')