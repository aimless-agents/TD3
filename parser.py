import argparse

def parse_our_args():
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

    # annealing reacher epsilon: default is a linear 5e-4 -> 5e-4 (aka constant at 5e-4)
    parser.add_argument("--reacher_epsilon_bounds", default=[5e-3, 5e-3], nargs=2, type=float, help="upper and lower epsilon bounds")
    parser.add_argument("--decay_type", default="linear", help="'linear' or 'exp' epsilon decay")
    parser.add_argument("--k", default=4, type=int)                             # k number of augmentations for HER
    parser.add_argument("--plot", default=False, action='store_true')           # include this flag to auto plot after running
    args = parser.parse_args()
    return args