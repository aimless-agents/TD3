import sys
import os
import numpy as np
import matplotlib.pyplot as plt 

filename = "TD3_CustomReacher_0.05_beta.npy"
file_to_load = os.path.join(sys.path[0], "results", filename)
eval_freq = 5000

nparray = np.load(file_to_load)
nparray = nparray[:, 2]

plt.plot(np.arange(0, len(nparray) * eval_freq, eval_freq), nparray)
plt.xlabel("timesteps")
plt.ylabel("rewards")
plt.title("TD3 Hopper Learning Curve")
plt.show()