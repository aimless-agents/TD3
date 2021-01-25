import numpy as np
import torch
import gym
import pybulletgym
import argparse
import os
import sys
import warnings
from datetime import datetime

import utils
import TD3
import OurDDPG
import DDPG

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
# custom_env_params: None if not using custom environment, or 
#   dictionary containing 'epsilon' to denote epsilon used during training
def eval_policy(policy, env_name, seed, eval_episodes=10, 
        custom_env_params=None):
    if custom_env_params:
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
        if custom_env_params:
            goal = eval_env.sample_goal_state(sigma=0)
        while not done:
            if policy.use_hindsight:
                x = np.concatenate([np.array(state), goal])
            else:
                x = np.array(state)
            action = policy.select_action(x)
            # TODO: I don't think we need this, right? -Claire
            # if custom_env_params:
            #   eval_env.set_goal(goal)
            state, reward, done, _ = eval_env.step(action)
            returns += reward
            if custom_env_params:
              original_returns += eval_env.original_rewards

        rewards[i] = returns
        original_rewards[i] = original_returns

    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    avg_original_reward = np.mean(original_rewards)
    std_original_reward = np.std(original_rewards)

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f} with original reward as {avg_original_reward:.3f}")
    print("---------------------------------------")
    if custom_env_params:
        return [avg_reward, std_reward, avg_original_reward, std_original_reward, custom_env_params['epsilon']]
    return [avg_reward, std_reward]

def train(config, args):
    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")
 
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {'CustomReacher' if args.custom_env else args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    import pybulletgym
    warnings.filterwarnings("ignore")
    eps_bounds = args.reacher_epsilon_bounds      # just aliasing with shorter variable name
    if args.custom_env:
        gym.envs.register(
            id='OurReacher-v0',
            entry_point='our_reacher_env:OurReacherEnv',
            max_episode_steps=150,
            reward_threshold=100.0,
        )

        # retrieve epsilon range
        if eps_bounds:
            [a, b] = eps_bounds
        else:
            epsilon = float(config['epsilon']) if args.tune_run else args.reacher_epsilon
            a, b = epsilon, epsilon
        epsilons = utils.epsilon_calc(a, b, args.max_timesteps, args.decay_type)
        env = gym.make('OurReacher-v0', epsilon=epsilons[0], render=False)
    else:
        env = gym.make(args.env)

    if args.tune_run:
        if args.prioritized_replay:
            args.alpha = float(config["alpha"])
            args.beta = float(config["beta"])
        else:
            args.discount = float(config.get("discount", args.discount))
            args.tau = float(config.get("tau", args.tau))

    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    state_dim = env.observation_space.shape[0]
    if args.use_hindsight:          # include both current state and goal state
        if args.custom_env:
            state_dim += 2          # reacher nonsense; goal = (x, y)
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

    exp_descriptors = [
        args.policy, 'CustomReacher' if args.custom_env else args.env,
        f"{'rank' if args.use_rank else 'proportional'}PER" if args.prioritized_replay else '', 
        'HER' if args.use_hindsight else '',
        f"eps{f'{eps_bounds[0]}-{eps_bounds[1]}' if eps_bounds else args.reacher_epsilon}" if args.custom_env else "",
        f"k{args.k}",
        datetime.now().strftime('%Y%m%d%H%M')
    ]
    exp_descriptors = [x for x in exp_descriptors if len(x) > 0]
    file_name = "_".join(exp_descriptors)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    if args.prioritized_replay:
        replay_buffer = utils.PrioritizedReplayBuffer(state_dim, action_dim,
                                                      args.max_timesteps, args.start_timesteps,
                                                      alpha=args.alpha, beta=args.beta)
    else:
        replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    
    # Evaluate untrained policy
    custom_env_params = {'epsilon': env.epsilon} if args.custom_env else None
    evaluations = [eval_policy(policy, args.env, args.seed, custom_env_params=custom_env_params)]

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
                policy.select_action(np.array(x))
                + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action
        if args.use_hindsight:
            env.set_goal(goal)
        next_state, reward, done, _ = env.step(action)
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        if args.use_hindsight:
            next_x = np.concatenate([np.array(next_state), goal])
        else:
            next_x = np.array(next_state)

        # Store data in replay buffer
        replay_buffer.add(x, action, next_x, reward, done_bool)

        trajectory.append((state, action, np.array(next_state), reward, done_bool))

        state = next_state
        episode_reward += reward
        if args.custom_env:
          original_episode_reward += env.original_rewards

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        if done:
            if args.use_hindsight:
                for _ in range(args.k): # factor to increase buffer by
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
            if args.custom_env:
                epsilon = epsilons[episode_num]
                env.set_epsilon(epsilon)
                custom_env_params['epsilon'] = epsilon

            trajectory = []

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            evaled_policy = eval_policy(policy, args.env, args.seed, custom_env_params=custom_env_params)
            evaluations.append(evaled_policy)
            np.save(f"./results/{file_name}", evaluations)
            if args.save_model:
                policy.save(f"./models/{file_name}")
            if args.tune_run:
                tune.report(episode_reward_mean=evaled_policy[0])

if __name__ == "__main__":

    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.environ["PYTHONPATH"] = parent_dir + ":" + os.environ.get("PYTHONPATH", "")

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="HalfCheetahMuJoCoEnv-v0") # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)     # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)         # Target network update rate
    parser.add_argument("--beta", default=0.0, type=float)          # Beta annealing step-size (should be 1/max_timesteps) for PER
    parser.add_argument("--alpha", default=1.0, type=float)         # alpha to use for PER
    parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name

    parser.add_argument("--prioritized_replay", default=False, action='store_true')		# Include this flag to use prioritized replay buffer
    parser.add_argument("--use_rank", default=False, action="store_true")               # Include this flag to use rank-based probabilities
    parser.add_argument("--use_hindsight", default=False, action="store_true")          # Include this flag to use HER
    parser.add_argument("--smoke_test", default=False, action='store_true')             # Include this flag to run a smoke test
    
    parser.add_argument("--custom_env", default=False, action="store_true")             # our custom environment name
    parser.add_argument("--tune_run", default=False, action='store_true')               # Include this flag when trying to tune
    parser.add_argument("--run_type", default="local", help="local or cluster")         # either local or cluster
    parser.add_argument("--reacher_epsilon", default=2e-2, type=float)                  # reacher epsilon

    # annealing reacher epsilon: default is a linear 2e-2 -> 2e-2 (aka constant at 2e-2)
    parser.add_argument("--reacher_epsilon_bounds", default=[2e-2, 2e-2], nargs=2, type=float, help="upper and lower epsilon bounds")
    parser.add_argument("--decay_type", default="linear", help="linear or exp epsilon decay")
    parser.add_argument("--k", default=1, type=int)                                     # k number of augmentations for HER
    args = parser.parse_args()
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {'CustomReacher' if args.custom_env else args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if args.tune_run:
        import ray 
        from ray import tune
        from ray.tune.schedulers import ASHAScheduler

    if "cluster" in args.run_type:
        ray.init(address='auto', _redis_password='5241590000000000', log_to_driver=False)
    
    config = {}
    if args.tune_run:
        if args.prioritized_replay:
            config = {
                "beta": tune.grid_search([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
                "alpha": tune.grid_search([0.3, 0.4, 0.5, 0.6])
            }
        else: 
            config = {
                "discount": tune.grid_search([0.995, 0.996, 0.997, 0.998, 0.999]),
                "tau": tune.grid_search([1e-5, 5e-4, 1e-4])
            }

        if args.use_hindsight:
            config = {
                "epsilon": tune.grid_search(list(np.arange(2e-2, 5.5e-2, 5e-3)))
            }

    kwargs = {}

    if args.smoke_test:
        args.start_timesteps = 25
        args.max_timesteps = 75
        args.eval_freq = 5

    kwargs["args"] = args
    
    if args.tune_run:
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
    else:
        train(config, **kwargs)