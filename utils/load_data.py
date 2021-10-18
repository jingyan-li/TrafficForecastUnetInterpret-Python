#  Author: 2021. Jingyan Li

import h5py
import numpy as np
import os.path
import matplotlib.pyplot as plt

#%%
DATA_DIR = r"E:\ETH_Workplace\IPA\data\2021_IPA\ori\Berlin\training"
FILE = "2019-04-05_berlin_9ch.h5"

f = h5py.File(os.path.join(DATA_DIR, FILE), "r")
dset = f["array"]

#%%
np.random.seed(0)
rand_idx = np.random.randint(low=0, high=288, size=15)
for i in rand_idx:
    plt.imshow(dset[i,:,:,0])
    plt.show()

#%%