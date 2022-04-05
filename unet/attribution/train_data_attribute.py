#  Author: 2021. Jingyan Li

# Conduct attribution

import numpy as np
import torch
import torch.nn as nn
import sys, os, glob

sys.path.append(os.getcwd())

from unet.model.config.config import config
from unet.model.Unet import UNet
from utils import visualize_utils, dataload_utils

import matplotlib.pyplot as plt


# please enter the source data root and submission root
source_root = r"C:\Users\jingyli\OwnDrive\IPA\data\2021_IPA\ori"

model_root = r"C:\Users\jingyli\OwnDrive\IPA\data\2021_IPA\UnetDeep_Good\checkpoint.pt"

figure_log_root = r"C:\Users\jingyli\OwnDrive\RA\unet_good\figures"
arr_log_root = r"C:\Users\jingyli\OwnDrive\RA\unet_good\attribution_pickle"


if not os.path.exists(figure_log_root):
    os.makedirs(figure_log_root)

if not os.path.exists(arr_log_root):
    os.makedirs(arr_log_root)


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
static = dataload_utils.load_h5_file(filepath)
static = torch.from_numpy(static).permute(2, 0, 1).unsqueeze(0).to(device).float()

#%%
# Load train data
DATE = "2019-09-19"
TIME = 142
# Output path
arr_log_root = os.path.join(arr_log_root, f"{DATE}_{TIME}")
if not os.path.exists(arr_log_root):
    os.makedirs(arr_log_root)
figure_log_root = os.path.join(figure_log_root, f"{DATE}_{TIME}")
if not os.path.exists(figure_log_root):
    os.makedirs(figure_log_root)

file_path = glob.glob(os.path.join(source_root, city, "validation", f"{DATE}_{city.lower()}_9ch.h5"))[0]
all_data = dataload_utils.load_h5_file(file_path)
all_data = np.moveaxis(all_data, -1, 1)

# Choose a time epoch in train data as a train sample
startt = TIME
# Train sample
tepoch = all_data[startt:startt+12,:,:,:]
# Ground truth (the following 6 time epochs)
gt_epoch = np.expand_dims(all_data[startt+12:startt+18,:8,:,:].reshape(-1, 495, 436), axis=0)

tepoch = torch.from_numpy(tepoch).to(device)
gt_epoch = torch.from_numpy(gt_epoch).to(device)

# reduce / stack 12 timeslots into 1
tepoch = tepoch.reshape(-1, tepoch.shape[-2], tepoch.shape[-1]).unsqueeze(0)

# concat the static data
tepoch = torch.cat([tepoch, static.repeat(tepoch.shape[0], 1, 1, 1)], axis=1)
tepoch = tepoch / 255


# #%%
# # Visualize the input data
# input_figure_path = os.path.join(figure_log_root, "input")
# if not os.path.exists(input_figure_path):
#     os.makedirs(input_figure_path)
# for i in range(12):
#     start_epoch = i
#     # For 1 timestamp
#     x = tepoch[0, start_epoch*9:(start_epoch+1)*9, :, :].cpu().float().numpy()
#
#     fig, axes = plt.subplots(1, 3, sharey=True)
#     visualize_utils.one_time_epoch(fig, axes, x)
#     plt.savefig(os.path.join(input_figure_path, os.path.split(file_path)[-1][:-3]+f"{startt}-input-startt{start_epoch}.png"),
#                 bbox_inches="tight")
#
# #%%
# # Visualize ground truth
# gt_figure_path = os.path.join(figure_log_root, "gt")
# if not os.path.exists(gt_figure_path):
#     os.makedirs(gt_figure_path)
# for i in range(6):
#     start_epoch = i
#     x = gt_epoch[0, start_epoch*8:(start_epoch+1)*8, :, :]/255
#
#     fig, axes = plt.subplots(1, 2, sharey=True)
#     visualize_utils.one_time_epoch(fig, axes, x, incidence=False)
#     plt.savefig(os.path.join(gt_figure_path, os.path.split(file_path)[-1][:-3]+f"{startt}-gt-startt{start_epoch}.png"),
#                 bbox_inches="tight")
#
#%%

# Preprocess of input
inputs = padd(tepoch[:1,:,:,:])

# Forward
with torch.no_grad():
    pred = model(inputs)

#%%
# Save prediction & error map for the prediction
gt_epoch_pad = padd(gt_epoch).numpy()
pred = pred.detach().numpy()
# Agg volume/speed
volume_idx = np.arange(0, 8, 2)
speed_idx = np.arange(1, 8, 2)
v_gt_epoch = np.mean(gt_epoch_pad[:,volume_idx, :, :], axis=1)
s_gt_epoch = np.mean(gt_epoch_pad[:,speed_idx, :, :], axis=1)
v_pred = np.mean(pred[:,volume_idx, :, :], axis=1)
s_pred = np.mean(pred[:,speed_idx, :, :], axis=1)

out = np.concatenate([v_pred,
                      s_pred,
                      v_gt_epoch-v_pred,
                      s_gt_epoch-s_pred
                      ])
#%%
np.save(os.path.join(arr_log_root,
                     os.path.split(file_path)[-1][:-3]
                     + f"{startt}-err-pred"),
        out)

#%%
# Visualize the prediction result
pred_figure_path = os.path.join(figure_log_root, "pred")
if not os.path.exists(pred_figure_path):
    os.makedirs(pred_figure_path)
for i in range(6):
    start_epoch = i
    x = pred[0, start_epoch*8:(start_epoch+1)*8, :, :]/255

    fig, axes = plt.subplots(1, 2, sharey=True)
    visualize_utils.one_time_epoch(fig, axes, x, incidence=False)
    plt.savefig(os.path.join(pred_figure_path, os.path.split(file_path)[-1][:-3]+f"{startt}-pred-startt{start_epoch}.png"),
                bbox_inches="tight")



#%%

# Aggregate target by channel and local windows (WINDOWSIZE*WINDOWSIZE)
WINDOW_SIZE = 21

def model_wrapper_window(inp):
    '''
    Wrap the model by down sampling the spatial resolution and agg speed/volume of 4 directions
    '''
    pooling_layer = nn.AvgPool2d(kernel_size=WINDOW_SIZE, stride=WINDOW_SIZE)
    model_out = pooling_layer(model(inp))
    model_agg = model_out.reshape((model_out.shape[0],-1,4,model_out.shape[2],model_out.shape[3])).sum(axis=2)
    return model_agg

# Forward by wrapper
with torch.no_grad():
    pred_wrapper = model_wrapper_window(inputs)

from captum.attr import Saliency

TARGET_CHANNEL = 1
TARGET_X = 286//WINDOW_SIZE
TARGET_Y = 121//WINDOW_SIZE

# Attr_log_root
attr_log_path = os.path.join(arr_log_root, f"W{TARGET_X}-{TARGET_Y}")
if not os.path.exists(attr_log_path):
    os.makedirs(attr_log_path)

# Preserve gradients
inputs.requires_grad = True
sa = Saliency(model_wrapper_window)

attr = sa.attribute(inputs, abs=True, target=(TARGET_CHANNEL,TARGET_X,TARGET_Y))
attr = attr.detach().numpy()

np.save(os.path.join(attr_log_path,
                     os.path.split(file_path)[-1][:-3]
                     + f"{startt}-saliency-target-channel{TARGET_CHANNEL}-W{TARGET_X}-{TARGET_Y}"),
        attr)

#%%
import matplotlib.patches as patches
# Visualize error map of speed and volume
# Attr_figure_root
figure_attr_path = os.path.join(figure_log_root, "attr", f"W{TARGET_X}-{TARGET_Y}")
if not os.path.exists(figure_attr_path):
    os.makedirs(figure_attr_path)

window = [TARGET_X, TARGET_Y]
fig, axes = plt.subplots(7, 2, sharey=True, figsize=(10,34))
cbar = None
for i in range(6):
    start_epoch = i
    # Prediction
    x = pred[0, start_epoch*8:(start_epoch+1)*8, :, :] / 255
    # Ground Truth
    gt_epoch_pad = padd(gt_epoch[:1, :, :, :])
    x_ = gt_epoch_pad[0, start_epoch * 8:(start_epoch + 1) * 8, :, :].numpy() / 255
    # Error
    err = x - x_
    cbar = visualize_utils.one_time_epoch(fig, axes[i], err, incidence=False, vmin=-1, vmax=1, colorbar=False)
    for ax in axes[i]:
        rect = patches.Rectangle((window[1] * WINDOW_SIZE, window[0] * WINDOW_SIZE),
                                 WINDOW_SIZE, WINDOW_SIZE,
                                 linewidth=2.5, edgecolor="red", facecolor="none")
        ax.add_patch(rect)
        ax.set_xlim(window[1] * WINDOW_SIZE - 100, window[1] * WINDOW_SIZE + 100)
        ax.set_ylim(window[0] * WINDOW_SIZE - 100, window[0] * WINDOW_SIZE + 100)
fig.colorbar(cbar, ax=axes[-1], location="bottom", orientation="horizontal", pad=0.1, aspect=60)

plt.savefig(os.path.join(figure_attr_path, os.path.split(file_path)[-1][:-3]
                         + f"{startt}-w{TARGET_X}_{TARGET_Y}-err.png"),
            bbox_inches="tight")
