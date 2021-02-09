import numpy as np
import torch
import gym
import pybulletgym
import os
import sys
import warnings
from datetime import datetime
import mujoco_py

import ray 
from ray import tune
from ray.tune.schedulers import ASHAScheduler

import utils
import TD3
import OurDDPG
import DDPG
import plotter
from parser import parse_our_args

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10, 
        utils_object=None):
    if utils_object.args.custom_env:
        eval_env = gym.make('OurReacher-v0')
    else:
        eval_env = gym.make(env_name)
    if utils_object.fetch_reach and utils_object.args.fetch_reach_dense:
        eval_env.env.reward_type = "dense"
    eval_env.seed(int(seed) + 100)

    rewards = np.zeros(eval_episodes)
    original_rewards = np.zeros(eval_episodes)

    for i in range(eval_episodes):
        returns = 0.0
        original_returns = 0.0
        state, done = eval_env.reset(), False
        
        while not done:
            x, goal = utils_object.compute_x_goal(state, eval_env, sigma=0)
            action = policy.select_action(x)
            if utils_object.args.custom_env:
              eval_env.set_goal(goal)
            state, reward, done, _ = eval_env.step(action)
            returns += reward
            if utils_object.args.custom_env:
              original_returns += eval_env.original_rewards

        rewards[i] = returns
        original_rewards[i] = original_returns

    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    avg_original_reward = np.mean(original_rewards)
    std_original_reward = np.std(original_rewards)

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f} from {rewards} with original reward as {avg_original_reward:.3f}")
    print("---------------------------------------")
    if utils_object.args.custom_env:
        return [avg_reward, std_reward, avg_original_reward, std_original_reward]
    return [avg_reward, std_reward]

def train(config, args):
    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    import pybulletgym
    warnings.filterwarnings("ignore")
    eps_bounds = args.reacher_epsilon_bounds      # just aliasing with shorter variable name
    utils_object = utils.GeneralUtils(args)

    if args.tune_run:
        if args.prioritized_replay:
            args.alpha = float(config["alpha"])
            args.beta = float(config["beta"])
            args.discount = float(config.get("discount", args.discount))
            args.tau = float(config.get("tau", args.tau))
        elif args.custom_env and args.use_hindsight:
            eps_bounds = [float(config["epsilons"][0]), float(config["epsilons"][1])]
            args.seed = int(config["seed"])
        else:
            args.discount = float(config.get("discount", args.discount))
            args.tau = float(config.get("tau", args.tau))
    
    if args.custom_env:
        gym.envs.register(
            id='OurReacher-v0',
            entry_point='our_reacher_env:OurReacherEnv',
            max_episode_steps=50,
            reward_threshold=100.0,
        )

        # this is assuming we only use epsilon for custom env or fetch reach, where episode tsteps is 50 !!!!
        max_episode_steps = 50

        # retrieve epsilon range
        [a, b] = eps_bounds
        epsilons = utils_object.epsilon_calc(a, b, max_episode_steps)
        env = gym.make('OurReacher-v0', epsilon=epsilons[0], render=False)
    else:
        env = gym.make(args.env)

    if utils_object.fetch_reach and utils_object.args.fetch_reach_dense:
        env.env.reward_type = "dense"

    # Set seeds
    env.seed(int(args.seed))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if utils_object.fetch_reach:
        state_dim = env.reset()["observation"].shape[0]
    else:
        state_dim = env.observation_space.shape[0]
    if args.use_hindsight:          # include both current state and goal state
        if args.custom_env:
            state_dim += 2          # reacher nonsense; goal = (x, y)
        elif utils_object.fetch_reach:
            state_dim += 3          # include fetchreach goal state (x,y,z position)
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
        f"{args.decay_type}decay-eps{f'{eps_bounds[0]}-{eps_bounds[1]}' if eps_bounds[0] != eps_bounds[1] else f'{eps_bounds[0]}'}" if args.custom_env else "",
        f"k{args.k}",
        datetime.now().strftime('%Y%m%d%H%M')
    ]
    if args.tune_run:
        # fudgy: assumes tune_run for non-HER experiments
        exp_descriptors = [
            args.policy, 'CustomReacher' if args.custom_env else args.env,
            f"{'rank' if args.use_rank else 'proportional'}PER" if args.prioritized_replay else '', 
            f"tau{args.tau}", f"discount{args.discount}",
            f"alpha{args.alpha}" if args.prioritized_replay else '',
            f"beta{args.beta}" if args.prioritized_replay else '',
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
    evaluations = [eval_policy(policy, args.env, args.seed, utils_object=utils_object)]
 
    state, done = env.reset(), False

    original_episode_reward = 0
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    trajectory = []

    for t in range(int(args.max_timesteps)):
        
        episode_timesteps += 1
        x, goal = utils_object.compute_x_goal(state, env)
        
        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                policy.select_action(np.array(x))
                + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done, _ = env.step(action)
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        if args.use_hindsight:
            if utils_object.fetch_reach:
                goal = state["desired_goal"]
                next_x = np.concatenate([np.array(next_state["observation"]), goal])
            else:
                # env.set_goal(goal)
                next_x = np.concatenate([np.array(next_state), goal])
        elif utils_object.fetch_reach:
            next_x = np.array(next_state["observation"])
        else:
            next_x = next_state

        # Store data in replay buffer
        if not args.use_hindsight:
            replay_buffer.add(x, action, next_x, reward, done_bool)

        trajectory.append((state, action, next_state, reward, done_bool))

        state = next_state
        episode_reward += reward
        if args.custom_env:
          original_episode_reward += env.original_rewards

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        if done:
            if args.use_hindsight:
                replay_buffer.add_hindsight(trajectory, goal, env, k=args.k, fetch_reach=utils_object.fetch_reach)
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

            trajectory = []

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            evaled_policy = eval_policy(policy, args.env, args.seed, utils_object=utils_object)
            evaluations.append(evaled_policy)
            np.save(f"./results/{file_name}", evaluations)
            if args.save_model:
                policy.save(f"./models/{file_name}")
            if args.plot:
                plotter.plot(file_name, args.custom_env)
            if args.tune_run:
                tune.report(episode_reward_mean=evaled_policy[0])

if __name__ == "__main__":

    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.environ["PYTHONPATH"] = parent_dir + ":" + os.environ.get("PYTHONPATH", "")

    args = parse_our_args()

    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {'CustomReacher' if args.custom_env else args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if "cluster" in args.run_type:
        ray.init(address='auto', _redis_password='5241590000000000', log_to_driver=False)
    
    config = utils.get_train_configuration(args)
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