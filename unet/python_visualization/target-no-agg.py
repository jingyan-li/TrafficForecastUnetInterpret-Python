#  Author: 2021. Jingyan Li
#  Visualize attribution by matplotlib when targeting to a single pixel

import numpy as np
import os
import matplotlib.pyplot as plt

from utils import visualize_utils


log_root = r"C:\Users\jingyli\OwnDrive\IPA\python-eda-code\unet\log\attribution_pickle\resUnet"
figure_log_root = r"C:\Users\jingyli\OwnDrive\IPA\python-eda-code\unet\log\figures\resUnet"
window = [16, 13]
file_path = f"2019-09-29_berlin_9ch142-saliency-target-channel0-W{window[0]}-{window[1]}.npy"
window_size = 21
attr = np.load(os.path.join(log_root, file_path))

file_format = "png"

#%%
# # Aggregate by channel: Which channel contributes a lot in predicting the selected pixel of channel 0 (first time slot, volume of NE)
#
agg_channel = np.sum(attr[0].reshape(attr[0].shape[0], -1), axis=1)
#
# plt.bar(x=np.arange(agg_channel.shape[0]), height=agg_channel)
# plt.ylabel("Gradient value")
# plt.xlabel("Channels")
# plt.title("Attribution aggregated per channel")
# plt.savefig(os.path.join(figure_log_root, os.path.split(file_path)[-1][:-3] + f"-attr-channel.{file_format}"),
#             bbox_inches="tight")
# plt.show()
#
# #%%
# channel_idx = np.argwhere(agg_channel > np.max(agg_channel)*0.5)
# print("Those channels show higher gradients: ")
# print(visualizer.input_indices_to_semantics(channel_idx.reshape(-1)))

#%%
# # Aggregate by time epoch
#
# agg_tepoch = np.sum(agg_channel[:108].reshape(12, -1), axis=1)
#
#
# plt.bar(x=np.arange(12), height=agg_tepoch)
# plt.ylabel("Gradient value")
# plt.xlabel("Time epochs")
# plt.title("Attribution aggregated per time epoch")
# plt.savefig(os.path.join(figure_log_root, os.path.split(file_path)[-1][:-3] + f"-attr-tepoch.{file_format}"),
#             bbox_inches="tight")
# plt.show()

# By static features

agg_static = agg_channel[108:]
plt.bar(x=[v for k, v in sorted(visualize_utils.input_static_semantic_dict.items(), key=lambda _:_[0])], height=agg_static)
plt.ylabel("Gradient value")
plt.xlabel("Static features")
plt.savefig(os.path.join(figure_log_root, os.path.split(file_path)[-1][:-3] + f"-attr-static.{file_format}"),
            bbox_inches="tight")
plt.title("Attribution per static feature")
plt.show()

#%%
# Aggregate by speed / volume / incident level

agg_incident = agg_channel[:108].reshape(12, -1)[:, -1]

agg_volume_speed = np.sum(agg_channel[:108].reshape(12, -1)[:, :-1].reshape(12, -1, 2), axis=1)/4

BAR_WIDTH = 0.2
x = np.arange(agg_incident.shape[0])
plt.bar(x-BAR_WIDTH, height=agg_volume_speed[:,0], label="Volume", width=BAR_WIDTH)
plt.bar(x, height=agg_volume_speed[:,1], label="Speed", width=BAR_WIDTH)
plt.bar(x+BAR_WIDTH, height=agg_incident, label="Incident Level", width=BAR_WIDTH)
plt.ylabel("Gradient value")
plt.xlabel("Time epoch")
plt.xticks(x)
plt.title("Attribution aggregated per features")
plt.legend()
plt.savefig(os.path.join(figure_log_root, os.path.split(file_path)[-1][:-3] + f"-attr-features.{file_format}"),
            bbox_inches="tight")
plt.show()


#%%
import matplotlib.patches as patches
import pickle
# Visualize attribution by volume, speed, incident level per time epoch
# only the target pixel or its surroundings (one to two pixels) impact its prediction
# Add road mask
ROAD_MASK = True
if ROAD_MASK:
    # Road Mask
    MASK_PATH = r"C:\Users\jingyli\OwnDrive\IPA\python-eda-code\utils\Berlin.mask"
    road = pickle.load(open(MASK_PATH, "rb"))
    road_ = np.pad(road, pad_width=((8, 9), (6, 6)))
    # Visualize road network
    fig, axes = plt.subplots(1, 1,)
    axes.imshow(road, vmin=0, vmax=1, cmap="binary")
    rect = patches.Rectangle((window[1] * window_size, window[0] * window_size),
                             window_size, window_size,
                             linewidth=2.5, edgecolor="red", facecolor="none")
    axes.add_patch(rect)
    axes.set_xlim(window[1] * window_size - 100, window[1] * window_size + 100)
    axes.set_ylim(window[0] * window_size - 100, window[0] * window_size + 100)
    plt.savefig(os.path.join(figure_log_root,
                             f"road_network_center{window[0]}-{window[1]}.png"), bbox_inches="tight", pad_inches=0, dpi=150)

#%%
for i in range(12):
    start_epoch = i
    x = attr[0, start_epoch*9:(start_epoch+1)*9, :, :]
    if ROAD_MASK:
        x = np.where(road_[None, :, :], x, 10**(-10))
    fig, axes = plt.subplots(1, 3, sharey=True)
    visualize_utils.attr_one_time_epoch(fig, axes, x, max=0.5, min=10**(-9))
    for i in range(3):
        rect = patches.Rectangle((window[1]*window_size, window[0]*window_size),
                                 window_size, window_size,
                                 linewidth=0.5, edgecolor="black", facecolor="none")
        axes[i].add_patch(rect)
        axes[i].set_xlim(window[1]*window_size-100,window[1]*window_size+100)
        axes[i].set_ylim(window[0]*window_size-100,window[0]*window_size+100)
        filename = f"-attr-space-startt{start_epoch}.png" if not ROAD_MASK else f"-attr-space-road-startt{start_epoch}.png"
    plt.savefig(os.path.join(figure_log_root,
                             os.path.split(file_path)[-1][:-3]
                             +filename),
                bbox_inches="tight")

    # # Road Junction_1
    # fig, axes = plt.subplots(1, 1)
    # feature = attr[0, -6, :, :]
    # cbar = axes.imshow(feature,  cmap='RdBu_r', norm=LogNorm(vmin=feature.min(), vmax=feature.max()))
    # rect = patches.Rectangle((window[1] * window_size, window[0] * window_size),
    #                          window_size, window_size,
    #                          linewidth=0.5, edgecolor="black", facecolor="none")
    # axes.add_patch(rect)
    # axes.set_xlim(window[1] * window_size - 100, window[1] * window_size + 100)
    # axes.set_ylim(window[0] * window_size - 100, window[0] * window_size + 100)
    # fig.colorbar(cbar, ax=axes, location="bottom", orientation="horizontal", pad=0.1, aspect=60)
    # plt.savefig(os.path.join(figure_log_root,
    #                          os.path.split(file_path)[-1][:-3]
    #                          +f"-attr-space-junction-startt{start_epoch}.png"),
    #             bbox_inches="tight")