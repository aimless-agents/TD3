import numpy as np
import matplotlib.pyplot as plt
import sys

test = sys.argv[1]

a = np.load(f"results/{test}.npy")
means = a[:,0]
stds = a[:,1]
x = np.arange(len(a)) * 5e3

plt.title(f"{test} Results")
plt.plot(x, means, color='black')
plt.fill_between(x, means - stds, means + stds, color='gray')
plt.xlabel("Timesteps")
plt.ylabel("Reward")
plt.show()