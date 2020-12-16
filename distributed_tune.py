import numpy as np
import torch
import gym
import argparse
import os
import sys

import utils
import TD3
import OurDDPG
import DDPG

import ray 
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.schedulers import ASHAScheduler


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward

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

    env = gym.make(env_name)

    # Set seeds
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": discount,
        "tau": tau,
    }

    # Initialize policy
    if policy == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = policy_noise * max_action
        kwargs["noise_clip"] = noise_clip * max_action
        kwargs["policy_freq"] = policy_freq
        kwargs["prioritized_replay"] = prioritized_replay
        policy = TD3.TD3(**kwargs)
    elif policy == "OurDDPG":
        policy = OurDDPG.DDPG(**kwargs)
    elif policy == "DDPG":
        policy = DDPG.DDPG(**kwargs)

    if prioritized_replay:
        replay_buffer = utils.PrioritizedReplayBuffer(state_dim, action_dim, max_timesteps, start_timesteps, alpha=alpha, beta=beta)
    else:
        replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    
    # Evaluate untrained policy
    evaluations = [eval_policy(policy, env_name, seed)]

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(int(max_timesteps)):
        
        episode_timesteps += 1
        # Select action randomly or according to policy
        if t < start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                policy.select_action(np.array(state))
                + np.random.normal(0, max_action * expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done, _ = env.step(action) 
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= start_timesteps:
            policy.train(replay_buffer, batch_size)

        if done: 
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1 

        # Evaluate episode
        if (t + 1) % eval_freq == 0:
            avg_reward = eval_policy(policy, env_name, seed)
            tune.report(episode_reward_mean=avg_reward)
            evaluations.append(avg_reward)

if __name__ == "__main__":

    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.environ["PYTHONPATH"] = parent_dir + ":" + os.environ.get("PYTHONPATH", "")
    ray.init(address='auto', _redis_password='5241590000000000')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="HalfCheetah-v2")          # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=75e4, type=int)   # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)                 # Discount factor
    parser.add_argument("--tau", default=0.005)                     # Target network update rate
    parser.add_argument("--beta_step", default=0.008)               # Beta annealing step-size (should be 1/max_timesteps)
    parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name

    parser.add_argument("--prioritized_replay", default=True, action='store_false')		# Include this flag to use prioritized replay buffer
    parser.add_argument("--smoke_test", default=True, action='store_false')             # Include this flag to run a smoke test
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

    best_trial = result.get_best_trial("episode_reward_mean", "max", "last")
    print("best trial: ", best_trial.config)
    print("best trial last result: ", best_trial.last_result)