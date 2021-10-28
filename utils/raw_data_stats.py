#  Author: 2021. Jingyan Li

import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
from utils import dataloader

#%%
DATA_DIR = r"C:\Users\jingyli\OwnDrive\IPA\data\2021_IPA\ori\Berlin\validation"
FILE = "2019-07-11_berlin_9ch.h5"

d = dataloader.load_h5_file(os.path.join(DATA_DIR, FILE))
#%%
incident = d[:,:,:,-1]
incident_level = np.sum(incident, axis=(1,2))/(495*436)

plt.plot(incident_level)
plt.show()

#%%
