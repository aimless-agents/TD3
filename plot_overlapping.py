import sys
import os
import numpy as np
import matplotlib.pyplot as plt 
from datetime import datetime
import argparse
from utils import epsilon_calc
from glob import glob
import matplotlib as mpl
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--f", nargs=2, type=str)       # vanilla, then her filenames
parser.add_argument("--env", default="")
parser.add_argument("--custom_env", default=False, action="store_true")
args = parser.parse_args()

# vanilla, HER, reward threshold colors
colors = [
    "#41bbd9", "#99c24d", "#442B48"
]

if not os.path.exists("./plots"):
    os.makedirs("./plots")
outfile_stem = f"./plots/{args.f[0]}_v_{args.f[1]}"

files_to_load = []
for file_name in args.f:
    files = []
    for f in glob(f"{sys.path[0]}/final_results/{file_name}*"):
        if f.endswith("npy"):
            files.append(f)
    files.sort(reverse=True)        # most recent -> least recent
    files_to_load.append(files[0])

eval_freq = 5000

runs = {}
runs["Vanilla TD3"] = np.load(files_to_load[0])
runs["HER TD3"] = np.load(files_to_load[1])

x = np.arange(0, len(runs["Vanilla TD3"]) * eval_freq, eval_freq)
if args.custom_env:         # first plot original reward curve
    plt.plot(x, [18] * len(x), color=colors[2])
    i = 0
    for name, results in runs.items():
        plt.fill_between(x, results[:, 2] - results[:, 3], results[:, 2] + results[:, 3], color=colors[i]+"88")
        plt.plot(x, results[:, 2], color=colors[i], label=name)
        i += 1
    plt.xlabel("Timesteps")
    plt.ylabel("Original Returns")
    plt.legend()
    plt.savefig(f"{outfile_stem}_original_rewards.png", bbox_inches='tight')
    print("Output to:", f"{outfile_stem}_original_rewards.png")
    plt.clf()

if not args.custom_env:
    reward_thresh = 0
    if "FetchReach" not in args.env:
        # plot the registered reward threshold
        import gym, pybulletgym; env = gym.make(args.env)
        plt.plot(x, [env.unwrapped.spec.reward_threshold] * len(x))
    plt.plot(x, [reward_thresh] * len(x), color=colors[2])

i = 0
for name, results in runs.items():
    plt.fill_between(x, results[:, 0] - results[:, 1], results[:, 0] + results[:, 1], color=colors[i]+"88")
    plt.plot(x, results[:, 0], color=colors[i], label=name)
    i += 1

plt.xlabel("Timesteps")
plt.ylabel("Returns")
plt.legend()
plt.savefig(f"{outfile_stem}.png", bbox_inches='tight')
print("Output to:", f"{outfile_stem}.png")
