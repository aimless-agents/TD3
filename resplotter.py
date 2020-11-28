import numpy as np
import matplotlib.pyplot as plt
import sys

test = sys.argv[1]

a = np.load(f"results/{test}.npy")
x = np.arange(len(a)) * 5e3

plt.plot(x, a)
plt.xlabel("Timesteps")
plt.ylabel("Reward")
plt.show()