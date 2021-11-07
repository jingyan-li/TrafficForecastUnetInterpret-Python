import time
import os
import random
import torch
from collections import OrderedDict
from covlstm.model.covlstm import CGRU_cell


config = dict()

##################################################################

# Please enter where raw data are stored.
config["source_dir"] = r"D:\Traffic4\Data\2020\ori"

config["debug"] = False
config["city"] = "Berlin"

config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
##################################################################

# data loader configuration
config["num_workers"] = 10


# Hyper-parameters and training configuration.
config["batch_size"] = 8
config["learning_rate"] = 1e-2

# early stopping and lr schedule
config["patience"] = 2
config["lr_step_size"] = 1
config["lr_gamma"] = 0.1

config["num_epochs"] = 50

config["iters_to_accumulate"] = 1

if config["debug"] == False:
    config["print_every_step"] = 50
else:
    config["print_every_step"] = 5

##############################################################
# Model params
config["encoder_params"] = [
    [
        OrderedDict({"conv1_relu_1": [16, 16, 3, 1, 1], "pool_2": [2, 2, 0]}),
        OrderedDict({"conv2_relu_1": [16, 32, 3, 1, 1], "pool_2": [2, 2, 0]}),
        OrderedDict({"conv3_relu_1": [32, 96, 3, 1, 1], "pool_2": [2, 2, 0]}),
        OrderedDict({"conv4_relu_1": [96, 192, 3, 1, 1], "pool_2": [2, 2, 0]}),
    ],
    [
        CGRU_cell(input_channels=16, num_features=16, shape=(248, 224), filter_size=3),
        CGRU_cell(input_channels=32, num_features=32, shape=(124, 112), filter_size=3),
        CGRU_cell(input_channels=96, num_features=96, shape=(62, 56), filter_size=3),
        CGRU_cell(input_channels=192, num_features=192, shape=(31, 28), filter_size=3),
    ],
]

config["forecast_params"] = [
    [
        OrderedDict({"deconv1_relu_1": [192, 96, 3, 1, 1]}),
        OrderedDict({"deconv2_relu_1": [96, 32, 3, 1, 1]}),
        OrderedDict({"deconv3_relu_1": [32, 16, 3, 1, 1]}),
        OrderedDict(
            {"deconv4_relu_1": [16, 16, 3, 1, 1], "conv3_relu_2": [16, 16, 3, 1, 1], "conv3_3": [16, 8, 1, 1, 0]}
        ),
    ],
    [
        CGRU_cell(input_channels=192, num_features=192, shape=(31, 28), filter_size=3),
        CGRU_cell(input_channels=96, num_features=96, shape=(62, 56), filter_size=3),
        CGRU_cell(input_channels=32, num_features=32, shape=(124, 112), filter_size=3),
        CGRU_cell(input_channels=16, num_features=16, shape=(248, 224), filter_size=3),
    ],
]