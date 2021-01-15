import numpy as np
import torch
import gym
import argparse
import os

import utils
import TD3
import OurDDPG
import DDPG
from our_reacher_env import OurReacherEnv

import pybulletgym

# Runs policy for X episodes and returns reward average and std
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10, 
        custom_env=False):
    if custom_env:
        # eval_env = OurReacherEnv()
        eval_env = gym.make('OurReacher-v0')
    else:
        eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    rewards = np.zeros(eval_episodes)
    original_rewards = np.zeros(eval_episodes)

    for i in range(eval_episodes):
        returns = 0.0
        original_returns = 0.0
        state, done = eval_env.reset(), False
        if custom_env and policy.use_hindsight:
            goal = eval_env.sample_goal_state(sigma=0)
        while not done:
            if policy.use_hindsight:
                x = np.concatenate([np.array(state), goal])
            else:
                x = np.array(state)
            action = policy.select_action(x)
            if policy.use_hindsight:
                eval_env.set_goal(goal)
            state, reward, done, _ = eval_env.step(action)
            returns += reward
            if policy.use_hindsight:
                original_returns += eval_env.original_rewards

        rewards[i] = returns
        original_rewards[i] = original_returns

    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    avg_original_reward = np.mean(original_rewards)

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f} with original reward as {avg_original_reward:.3f}")
    print("---------------------------------------")
    return [avg_reward, std_reward]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--policy", default="TD3")
    # OpenAI gym environment name
    parser.add_argument("--env", default="HalfCheetah-v2")
    # our custom environment name
    parser.add_argument("--custom_env", default=False, action="store_true")
    # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--seed", default=0, type=int)
    # Time steps initial random policy is used
    parser.add_argument("--start_timesteps", default=25e3, type=int)
    # How often (time steps) we evaluate
    parser.add_argument("--eval_freq", default=5e3, type=int)
    # Max time steps to run environment
    parser.add_argument("--max_timesteps", default=1e6, type=int)
    # Std of Gaussian exploration noise
    parser.add_argument("--expl_noise", default=0.1)
    # Batch size for both actor and critic
    parser.add_argument("--batch_size", default=256, type=int)
    # Discount factor
    parser.add_argument("--discount", default=0.99, type=float)
    # Target network update rate
    parser.add_argument("--tau", default=0.005, type=float)
    # Noise added to target policy during critic update
    parser.add_argument("--policy_noise", default=0.2)
    # Range to clip target policy noise
    parser.add_argument("--noise_clip", default=0.5)
    # Frequency of delayed policy updates
    parser.add_argument("--policy_freq", default=2, type=int)
    # Save model and optimizer parameters
    parser.add_argument("--save_model", action="store_true")
    # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--load_model", default="")

    # Whether or not to use prioritized replay buffer
    parser.add_argument("--prioritized_replay", default=False, action='store_true')		# Include this flag to use prioritized replay buffer
    parser.add_argument("--use_rank", default=False, action="store_true")               # Include this flag to use rank-based probabilities
    parser.add_argument("--use_hindsight", default=False, action="store_true")               # Include this flag to use HER
        # to use HER, the environment must implement `goal_cond_reward`
    # initial alpha value for PER
    parser.add_argument("--alpha", default=1.0)
    args = parser.parse_args()

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    if args.custom_env:
        # env = OurReacherEnv()
        gym.envs.register(
                            id='OurReacher-v0',
                            entry_point='our_reacher_env:OurReacherEnv',
                            max_episode_steps=150,
                            reward_threshold=100.0,
                            )
        env = gym.make('OurReacher-v0')

    else:
        env = gym.make(args.env)

    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    if args.use_hindsight:          # include both current state and goal state
        if args.custom_env:
            state_dim += 2      # reacher nonsense; goal = (x, y)
        else:
            state_dim *= 2

    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }

    # Initialize policy
    if args.policy == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        kwargs["prioritized_replay"] = args.prioritized_replay
        kwargs["use_rank"] = args.use_rank
        kwargs["use_hindsight"] = args.use_hindsight
        
        policy = TD3.TD3(**kwargs)
    elif args.policy == "OurDDPG":
        policy = OurDDPG.DDPG(**kwargs)
    elif args.policy == "DDPG":
        policy = DDPG.DDPG(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    if args.prioritized_replay:
        replay_buffer = utils.PrioritizedReplayBuffer(state_dim, action_dim,
                                                      args.max_timesteps, args.start_timesteps,
                                                      alpha=args.alpha)
    else:
        replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

    # Evaluate untrained policy
    evaluations = [eval_policy(policy, args.env, args.seed, custom_env=args.custom_env)]

    state, done = env.reset(), False

    original_episode_reward = 0
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    trajectory = []

    for t in range(int(args.max_timesteps)):

        episode_timesteps += 1
        
        if args.use_hindsight:
            goal = env.sample_goal_state()

        x = np.concatenate([np.array(state), goal]) if args.use_hindsight else np.array(state)
        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                policy.select_action(x) 
                + np.random.normal(0, max_action *
                                   args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action
        if args.use_hindsight:
            env.set_goal(goal)
        next_state, reward, done, _ = env.step(action)
        done_bool = float(
            done) if episode_timesteps < env._max_episode_steps else 0

        if args.use_hindsight:
            next_x = np.concatenate([np.array(next_state), goal])
            # reward = env.goal_cond_reward(next_state, goal)     # store the goal-conditioned reward in buffer
        else:
            next_x = np.array(next_state)
            
        # Store data in replay buffer
        replay_buffer.add(x, action, next_x, reward, done_bool)
        trajectory.append((state, action, np.array(next_state), reward, done_bool))

        state = next_state
        episode_reward += reward
        if args.use_hindsight:
            original_episode_reward += env.original_rewards

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        if done:
            if args.use_hindsight:
                for i in range(len(trajectory) - 1):
                    old_state, old_action, old_next_state, _, old_done_bool = trajectory[i]
                    idx = np.random.choice(np.arange(i+1, len(trajectory)))
                    ng, _, _, _, _ = trajectory[idx]
                    new_goal = np.array([ng[0] + ng[2], ng[1] + ng[3]])
                    x = np.concatenate([old_state, new_goal])
                    next_x = np.concatenate([old_next_state, new_goal])
                    replay_buffer.add(x, old_action, next_x, env.goal_cond_reward(old_next_state, new_goal), old_done_bool)

            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(
                f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f} Original Reward: {original_episode_reward:.3f}")
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            original_episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

            trajectory = []

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            evaluations.append(eval_policy(policy, args.env, args.seed, custom_env=args.custom_env))
            np.save(f"./results/{file_name}_beta", evaluations)
            if args.save_model:
                policy.save(f"./models/{file_name}")
