#  Author: 2021. Jingyan Li
# Reference:

import numpy as np
import torch
import torch.nn as nn
import h5py
from pathlib import Path
import pickle
import sys, os, glob
import datetime as dt
from tqdm import tqdm

sys.path.append(os.getcwd())

from unet.model.config import config
from unet.model.Unet import UNet
from utils import visualizer, dataloader

import matplotlib.pyplot as plt

# simplified depth 5 model

# please enter the source data root and submission root
source_root = r"C:\Users\jingyli\OwnDrive\IPA\data\2021_IPA\ori"

model_root = r"C:\Users\jingyli\OwnDrive\IPA\data\2021_IPA\UnetDeep_1632078207\checkpoint_5.pt"

figure_log_root = "unet/log/figures/"
arr_log_root = "unet/log/attribution_pickle/"


print(f"CUDA is available: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet(img_ch=config["in_channels"], output_ch=config["n_classes"]).to(device)
city = "Berlin"


padd = torch.nn.ZeroPad2d((6, 6, 8, 9))

state_dict = torch.load(model_root, map_location=device)
model.load_state_dict(state_dict)
model.eval()

# load static data
filepath = glob.glob(os.path.join(source_root, city, f"{city}_static_2019.h5"))[0]
static = dataloader.load_h5_file(filepath)
static = torch.from_numpy(static).permute(2, 0, 1).unsqueeze(0).to(device).float()

#%%
# Load train data
file_path = glob.glob(os.path.join(source_root, city, "training", f"2019-01-12_{city.lower()}_9ch.h5"))[0]
all_data = dataloader.load_h5_file(file_path)
all_data = np.moveaxis(all_data, -1, 1)
#%%
# Choose a time epoch in train data as a train sample
startt = 0
# Train sample
tepoch = all_data[startt:startt+12,:,:,:]
# Ground truth (the following 6 time epochs)
gt_epoch = np.expand_dims(all_data[startt+12:startt+18,:8,:,:].reshape(-1, 495, 436), axis=0)

tepoch = torch.from_numpy(tepoch).to(device)

# reduce / stack 12 timeslots into 1
tepoch = tepoch.reshape(-1, tepoch.shape[-2], tepoch.shape[-1]).unsqueeze(0)

# concat the static data
tepoch = torch.cat([tepoch, static.repeat(tepoch.shape[0], 1, 1, 1)], axis=1)
tepoch = tepoch / 255


#%%
# # Visualize the input data
# for i in range(12):
#     start_epoch = i
#     x = tepoch[0, start_epoch*9:(start_epoch+1)*9, :, :].cpu().float().numpy()
#
#     fig, axes = plt.subplots(1, 3, sharey=True)
#     visualizer.one_time_epoch(fig, axes, x)
#     plt.savefig(os.path.join(figure_log_root, os.path.split(file_path)[-1][:-3]+f"{startt}-input-startt{start_epoch}.png"),
#                 bbox_inches="tight")
#     plt.show()
#%%
# Visualize ground truth
for i in range(6):
    start_epoch = i
    x = gt_epoch[0, start_epoch*8:(start_epoch+1)*8, :, :]/255

    fig, axes = plt.subplots(1, 2, sharey=True)
    visualizer.one_time_epoch(fig, axes, x, incidence=False)
    plt.savefig(os.path.join(figure_log_root, os.path.split(file_path)[-1][:-3]+f"{startt}-gt-startt{start_epoch}.png"),
                bbox_inches="tight")
    plt.show()
#%%

# Preprocess of input
inputs = padd(tepoch[:1,:,:,:])

# Forward
with torch.no_grad():
    pred = model(inputs)

#%%
# Visualize the prediction result
for i in range(6):
    start_epoch = i
    x = pred[0, start_epoch*8:(start_epoch+1)*8, :, :].cpu().float().numpy()/255

    fig, axes = plt.subplots(1, 2, sharey=True)
    visualizer.one_time_epoch(fig, axes, x, incidence=False)
    plt.savefig(os.path.join(figure_log_root, os.path.split(file_path)[-1][:-3]+f"{startt}-pred-startt{start_epoch}.png"),
                bbox_inches="tight")
    plt.show()


#%%

# # Test captum
#
# # Input single sample
# inputs.requires_grad = True
#
# # Attribution by Saliency
# from captum.attr import Saliency
#
# TARGET_CHANNEL = 0
# X = 256
# Y = 256
# sa = Saliency(model)
#
# attr = sa.attribute(inputs, abs=True, target=(TARGET_CHANNEL, X, Y))
# attr = attr.detach().numpy()
#
# np.save(os.path.join(arr_log_root, os.path.split(file_path)[-1][:-3] + f"{startt}-saliency-target{TARGET_CHANNEL}-{X}-{Y}"), attr)
#

#%%

# # Aggregate the target by channel
#
# def model_wrapper_channel(inp):
#     '''
#     Wrap the model, the output becomes one value per each channel
#     '''
#     model_out = model(inp)
#     return model_out.sum(axis=(2, 3))
#
# # Forward by wrapper
# with torch.no_grad():
#     pred_wrapper = model_wrapper_channel(inputs)
#
# from captum.attr import Saliency
#
# TARGET_CHANNEL = 0
#
# # Preserve gradients
# inputs.requires_grad = True
# sa = Saliency(model_wrapper_channel)
#
# attr = sa.attribute(inputs, abs=True, target=TARGET_CHANNEL)
# attr = attr.detach().numpy()
#
# np.save(os.path.join(arr_log_root, os.path.split(file_path)[-1][:-3] + f"{startt}-saliency-target-channel{TARGET_CHANNEL}"), attr)


#%%

# Aggregate target by channel and local windows (9*9)
WINDOW_SIZE = 9