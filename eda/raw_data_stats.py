#  Author: 2021. Jingyan Li
#  Calculate abnormality level for validation days

import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
from utils import dataload_utils
import glob
import pickle
#%%
DATA_DIR = r"C:\Users\jingyli\OwnDrive\IPA\data\2021_IPA\ori\Berlin\validation"
FIG_DIR = r"C:\Users\jingyli\OwnDrive\IPA\result\raw_data_stats"
#%%
# # Iterate through all days and see days with highest incident level
# for file in glob.glob(os.path.join(DATA_DIR,"*.h5")):
#     d = dataload_utils.load_h5_file(file)
#     incident = d[:, :, :, -1]
#     incident_level = np.sum(incident, axis=(1, 2)) / (495 * 436)  # Average incident level per time
#     # incident_level = np.max(incident, axis=(1, 2))  # Max incident level per time stamp: almost 255
#     print(file.split("\\")[-1])
#     print(incident_level.max())

#%%
# Explore abnormal days
FILES = ["2019-07-11_berlin_9ch.h5",
         "2019-09-19_berlin_9ch.h5",
         "2019-09-29_berlin_9ch.h5",
         "2019-11-28_berlin_9ch.h5"
         ]
incident_dict = {}
#%%
# Read and Plot incident level per day
plot = False
for FILE in FILES:
    DATE = FILE.split("_")[0]
    d = dataload_utils.load_h5_file(os.path.join(DATA_DIR, FILE))

    incident = d[:, :, :, -1]
    incident_level = np.sum(incident, axis=(1, 2))/(495*436)
    incident_dict[DATE] = incident
    if plot:
        plt.plot(incident_level)
        plt.savefig(os.path.join(FIG_DIR, DATE+".png"), bbox_inches="tight")
        plt.show()

#%%
selected_day = "2019-09-19"

abnormal_range = np.sum(incident_dict[selected_day], axis=(1,2))/(495*436)

startt = 48
#%%
# Incident level -  time correlation



#%%
# Heat map of incident map per time epoch
def plot_heatmap(ax, data, title):
    ax.imshow(data, vmin=0, vmax=250, cmap='RdBu_r')
    ax.set_title(title)


fig, axes = plt.subplots(1,10, figsize=(50,5))
for i in range(len(axes)):
    plot_heatmap(axes[i], incident_dict[selected_day][startt+i,:,:], f"selected_day-{startt+i}")
plt.savefig(os.path.join(FIG_DIR, f"{selected_day}-{startt}.png"), bbox_inches="tight")
plt.show()

#%%
import matplotlib.patches as patches
# Zoom into small area
window = [16, 13]
window_size = 21
startt_ = 146
fig, axes = plt.subplots(1,1, figsize=(10,10))
plot_heatmap(axes, incident_dict[selected_day][startt_,:,:], f"selected_day-{startt_}")
rect = patches.Rectangle((window[1] * window_size, window[0] * window_size),
                         window_size, window_size,
                         linewidth=2.5, edgecolor="red", facecolor="none")
axes.add_patch(rect)
# axes.set_xlim(window[1]*window_size-100,window[1]*window_size+100)
# axes.set_ylim(window[0]*window_size-100,window[0]*window_size+100)
plt.savefig(os.path.join(FIG_DIR, f"{selected_day}-{startt_}-zoomin.png"), bbox_inches="tight")
plt.show()

#%%
# Road Mask
MASK_PATH = r"C:\Users\jingyli\OwnDrive\IPA\python-eda-code\utils\Berlin.mask"

road = pickle.load(open(MASK_PATH, "rb"))

# #%%
# # Visualize road network
# fig, axes = plt.subplots(1, 1,)
# axes.imshow(road, vmin=0, vmax=1, cmap="binary")
# plt.axis("off")
# plt.savefig(os.path.join(FIG_DIR, "road_network.png"), bbox_inches="tight", pad_inches=0, dpi=150)

#%%
LOG_DIR = r"C:\Users\jingyli\OwnDrive\IPA\python-eda-code\eda\log\abnormal_pickle"
# If there is any overlap between road network and abnormal region
incident_sample = incident_dict[selected_day][startt]
# Select high incident and within road network
abnormal_region = np.where((incident_sample > 250) & road)
pickle.dump(abnormal_region, open(os.path.join(LOG_DIR, f"{selected_day}-{startt}"), "wb" ))
#%%
# Filter some abnormal regions
import pandas as pd
_ = np.array(abnormal_region).transpose()
abnormal_df = pd.DataFrame(_, columns=["y", "x"])

# #%%
# # Print abnormal region
# idx = 52
# print(f"x: {abnormal_region[0][idx]}, y: {abnormal_region[1][idx]}")


