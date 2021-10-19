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

# simplified depth 5 model

# please enter the source data root and submission root
source_root = r"E:/ETH_Workplace/IPA/data/2021_IPA/ori"
submission_root = r"E:/ETH_Workplace/IPA/data/2021_IPA/submission"

model_root = r"E:\ETH_Workplace\IPA\data\2021_IPA\UnetDeep_1632078207\checkpoint.pt"
mask_root = r"utils/masks.dict"


def load_h5_file(file_path):
    """
    Given a file path to an h5 file assumed to house a tensor,
    load that tensor into memory and return a pointer.
    """
    # load
    fr = h5py.File(file_path, "r")
    a_group_key = list(fr.keys())[0]
    data = list(fr[a_group_key])
    # transform to appropriate numpy array
    data = data[0:]
    data = np.stack(data, axis=0)
    return data


class WrappedModel(torch.nn.Module):
    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module  # that I actually define.

    def forward(self, x):
        return self.module(x)

#%%
print(f"CUDA is available: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet(img_ch=config["in_channels"], output_ch=config["n_classes"]).to(device)
# model = WrappedModel(model).to(device)
city = "Berlin"

# TODO: ASK: What is mask_dict here?
# mask_dict = pickle.load(open(mask_root, "rb"))

padd = torch.nn.ZeroPad2d((6, 6, 8, 9))

state_dict = torch.load(model_root, map_location=device)
model.load_state_dict(state_dict)
model.eval()
#%%
# load static data
filepath = glob.glob(os.path.join(source_root, city, f"{city}_static_2019.h5"))[0]
static = load_h5_file(filepath)
static = torch.from_numpy(static).permute(2, 0, 1).unsqueeze(0).to(device).float()

# load mask
# mask_ = torch.from_numpy(mask_dict[city]["sum"] > 0).bool()

#%%
# use an example from training data
# file_path = glob.glob(os.path.join(source_root, city, "training", f"2019-01-12_{city.lower()}_9ch.h5"))[0]
file_path = glob.glob(os.path.join(source_root, city, "testing", f"2019-07-02_test.h5"))[0]
# the date of the file
all_data = load_h5_file(file_path)  # of size (3,12,495,436,9)
x = np.moveaxis(all_data, -1, 2)  # of size (3,12,9,495,436)
#%%
x = torch.from_numpy(x).to(device)
# reduce / stack 12 timeslots into 1
x = x.reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])  # of size (3,12*9,495,436)

# concat the static data
x = torch.cat([x, static.repeat(x.shape[0], 1, 1, 1)], axis=1)
x = x / 255
#%%
# Visualize the input data
import matplotlib.pyplot as plt

tepoch = x[0, :9, :, :].cpu().float()
tepoch = tepoch.numpy()*255

print(f"Sum of the time epoch: {np.sum(tepoch)}")
plt.imshow(tepoch[0,:,:])
plt.colorbar()
plt.show()
#%%
# Forward
# with torch.no_grad():
inputs = padd(x)
pred = model(inputs)
# expand
pred = pred.view(pred.shape[0], 6, 8, pred.shape[-2], pred.shape[-1])
res = pred[:, :, :, 1:, 6:-6].cpu().float()

res = torch.clamp(res, 0, 255).permute(0, 1, 3, 4, 2).detach().numpy().astype(np.uint8) # of size

#%%
# Test Captum
from captum.attr import IntegratedGradients
ig = IntegratedGradients(model)
x.requires_grad_()
attr, delta = ig.attribute(x, return_convergence_delta=True)
attr = attr.detach().numpy()

#%%
# get test data
file_paths = glob.glob(os.path.join(source_root, city, "testing", "*.h5"))
for path in tqdm(file_paths):
    # the date of the file
    all_data = load_h5_file(path)
    x = np.moveaxis(all_data, -1, 2)

    x = torch.from_numpy(x).to(device)
    # reduce
    x = x.reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])

    # concat the static data
    x = torch.cat([x, static.repeat(x.shape[0], 1, 1, 1)], axis=1)
    x = x / 255

    with torch.no_grad():
        inputs = padd(x)
        pred = model(inputs)

        # expand
        pred = pred.view(pred.shape[0], 6, 8, pred.shape[-2], pred.shape[-1])
        res = pred[:, :, :, 1:, 6:-6].cpu().float()

    # apply mask
    # masks = mask_.expand(res.shape)
    # res[~masks] = 0

    res = torch.clamp(res, 0, 255).permute(0, 1, 3, 4, 2).numpy().astype(np.uint8)

    # create saving root
    root = os.path.join(submission_root, city.upper())
    if not os.path.exists(root):
        os.makedirs(root)

    # save predictions
    target_file = os.path.join(root, path.split("\\")[-1])
    with h5py.File(target_file, "w", libver="latest",) as f:
        f.create_dataset("array", shape=(res.shape), data=res, compression="gzip", compression_opts=4)