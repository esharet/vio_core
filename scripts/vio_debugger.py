import matplotlib.pyplot as plt
import numpy as np

fig, axs = plt.subplots(2, 2, figsize=(4, 3), layout="constrained")

x, y = [1, 2], [4, 5]

ax = axs[0, 0]
linesx = ax.scatter(x, y)

ax = axs[0, 1]
linesx = ax.plot(x, y)

ax = axs[1, 0]
linesx = ax.bar(x, y)

ax = axs[1, 1]
img_2d_array = np.random.rand(10, 10)
linesx = ax.imshow(img_2d_array)

plt.show()