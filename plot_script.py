import sys
import os
import numpy as np
import matplotlib.pyplot as plt 
from datetime import datetime
from parser import *
from utils import epsilon_calc
from glob import glob
import matplotlib as mpl
from pathlib import Path

args = parse_our_args()

colors = [
    "#41bbd9", "#99c24d"
]

if args.policy:
    file_name = args.policy

else:
    eps_bounds = args.reacher_epsilon_bounds
    exp_descriptors = [
        args.policy, 'CustomReacher' if args.custom_env else args.env,
        f"{'rank' if args.use_rank else 'proportional'}PER" if args.prioritized_replay else '', 
        'HER' if args.use_hindsight else '',
        f"{args.decay_type}decay-eps{f'{eps_bounds[0]}-{eps_bounds[1]}' if eps_bounds[0] != eps_bounds[1] else f'{eps_bounds[0]}'}" if args.custom_env else "",
        f"k{args.k}",
    ]
    exp_descriptors = [x for x in exp_descriptors if len(x) > 0]
    file_name = "_".join(exp_descriptors)       # file name root (minus timestamp)


if not os.path.exists("./plots"):
    os.makedirs("./plots")

files = []
for f in glob(f"{sys.path[0]}/results/{file_name}*"):
    if f.endswith("npy"):
        files.append(f)
files.sort(reverse=True)        # most recent -> least recent

file_to_load = files[0]         # just use the most recent one; we can add options later
outfile_stem = f"./plots/{Path(file_to_load).stem}"      # output file stem, with timestamp

eval_freq = 5000

results = np.load(file_to_load)

x = np.arange(0, len(results) * eval_freq, eval_freq)
if args.custom_env:
    # plot original reward curve
    plt.fill_between(x, results[:, 2] - results[:, 3], results[:, 2] + results[:, 3], color=colors[0]+"88")
    plt.plot(x, results[:, 2], color=colors[0])
    plt.plot(x, [18] * len(x), color=colors[1])
    plt.xlabel("Timesteps")
    plt.ylabel("Original Returns")
    if not args.no_title:
        plt.title(f"Original Rewards")
    plt.savefig(f"{outfile_stem}_original_rewards.png", bbox_inches='tight')
    print("Output to:", f"{outfile_stem}_original_rewards.png")
    # plt.show()
    plt.clf()

plt.fill_between(x, results[:, 0] - results[:, 1], results[:, 0] + results[:, 1], color=colors[0]+"88")
plt.plot(x, results[:, 0], color=colors[0])

if "FetchReach" in args.env:
    plt.plot(x, [0] * len(x), color=colors[1])
    
plt.xlabel("Timesteps")
plt.ylabel("Returns")
if not args.no_title:
    plt.title(f"Rewards")
plt.savefig(f"{outfile_stem}.png", bbox_inches='tight')
print("Output to:", f"{outfile_stem}.png")
