import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from utils import epsilon_calc
from glob import glob
from pathlib import Path

def plot(file_name, custom_env):
    graph_title = file_name.replace("_", " ")

    if not os.path.exists("./plots"):
        os.makedirs("./plots")
    outfile_stem = f"./plots/{file_name}"

    eval_freq = 5000

    results = np.load(f"./results/{file_name}")

    x = np.arange(0, len(results) * eval_freq, eval_freq)
    if custom_env:
        # plot original reward curve
        plt.fill_between(x, results[:, 2] - results[:, 3], results[:, 2] + results[:, 3], alpha=0.5)
        plt.plot(x, results[:, 2])
        plt.plot(x, [18] * len(x))
        plt.xlabel("Timesteps")
        plt.ylabel("Original Returns")
        plt.title(f"{graph_title} Original Rewards")
        plt.savefig(f"{outfile_stem}_original_rewards.png", bbox_inches='tight')
        print("Output to:", f"{outfile_stem}_original_rewards.png")
        plt.clf()

        # plot epsilon
        plt.plot(x, results[:, 4])
        plt.xlabel("Timesteps")
        plt.ylabel("Epsilon")
        plt.title(f"{graph_title} Epsilon Values")
        plt.savefig(f"{outfile_stem}_epsilon.png", bbox_inches='tight')
        print("Output to:", f"{outfile_stem}_epsilon.png")
        plt.show()
        plt.clf()

    plt.fill_between(x, results[:, 0] - results[:, 1], results[:, 0] + results[:, 1], alpha=0.5)
    plt.plot(x, results[:, 0])
    plt.xlabel("Timesteps")
    plt.ylabel("Returns")
    plt.title(f"{graph_title} Rewards")
    plt.savefig(f"{outfile_stem}.png", bbox_inches='tight')
    print("Output to:", f"{outfile_stem}.png")
    if not custom_env:
        plt.show()