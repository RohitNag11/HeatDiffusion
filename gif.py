import os
from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
import imageio

path = 'plots'

# y = np.random.randint(30, 40, size=(40))
# plt.plot(y)
# plt.ylim(20,50)

# Build GIF
with imageio.get_writer('mygif.gif', mode='I') as writer:
    for i in range(0, 6000, 10):
        filename = f'plots/{i}.png'
        image = imageio.imread(filename)
        writer.append_data(image)
