# Do validation over the final model

import numpy as np
import torch
import torch.nn as nn
import h5py
from pathlib import Path
import pickle
import sys, os, glob
from datetime import datetime
from tqdm import tqdm
import random
from videoloader import trafic4cast_dataset
# from visualizer import Visualizer

sys.path.append(os.getcwd())

from unet.model.config_train import config
from unet.model.config_validate import config_val
from unet.model.Unet import UNet


city = config["city"]
if config_val["debug"] == True:
    networkName = "Test"
else:
    networkName = "UnetDeep_"


def validate(model, val_loader, device, writer, mse_arr, mask=None):
    # random_visualize = random.randint(0, len(val_loader))
    # if config["debug"] == True:
    #     random_visualize = 0

    padd = torch.nn.ZeroPad2d((6, 6, 8, 9))

    total_val_loss = 0
    # change to validation mode
    model.eval()
    with torch.no_grad():
        for i, (val_inputs, val_y, startt_idx) in tqdm(enumerate(val_loader, 0)):
            val_inputs = val_inputs / 255

            val_inputs = padd(val_inputs).to(device)
            val_y = val_y.to(device)

            val_output = model(val_inputs)
            if mask is not None:
                masks = padd(mask).expand(val_output.shape)
                val_output[~masks] = 0

            val_loss_size = torch.nn.functional.mse_loss(val_output[:, :, 8:-9, 6:-6], val_y)
            mse_arr[startt_idx[0]] = val_loss_size
            # write the validation loss to tensorboard
            if writer is not None:
                writer.write_loss_validation(val_loss_size, startt_idx[0])

            if i % 270 == 0:
                print("Validation mse = {:.2f}".format(val_loss_size))

            total_val_loss += val_loss_size.item()

            # # each epoch select one prediction set (one batch) to visualize
            # if i == random_visualize:
            #     writer.write_video(val_output.cpu(), epoch, if_predict=True)
            #     writer.write_video(val_y.cpu(), epoch, if_predict=False)
            if config_val["debug"] == True:
                break

    val_loss = total_val_loss / len(val_loader)
    print("Validation loss = {:.2f}".format(val_loss))
    # # write the validation loss to tensorboard
    # writer.write_loss_validation(val_loss, epoch)
    return mse_arr


if __name__ == "__main__":
    dataset_val = trafic4cast_dataset(
        source_root=config_val["source_root"], split_type="validation", cities=[city], reduce=True, include_static=True
    )

    val_loader = torch.utils.data.DataLoader(
        dataset_val, batch_size=config_val["batch_size"], shuffle=False, num_workers=config_val["num_workers"]
    )

    device = config["device"]

    # define the network structure -- UNet
    # the output size is not always equal to your input size !!!
    model = UNet(img_ch=config["in_channels"], output_ch=config["n_classes"])
    # model = nn.DataParallel(model)
    model.to(device)

    # please enter the mask dir
    # mask_dir = ""
    # mask_dict = pickle.load(open(mask_dir, "rb"))
    # mask_ = torch.from_numpy(mask_dict[city]["sum"] > 0).bool()

    if config_val["tensor_board"]:
        from visualizer import Visualizer

        log_dir = "../runs/" + networkName + str(int(datetime.now().timestamp()))
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        writer = Visualizer(log_dir)
    else:
        writer = None

    # # get the trainable paramters
    # model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    # params = sum([np.prod(p.size()) for p in model_parameters])
    # print("# of parameters: ", params)

    # Store log
    mse_arr = np.zeros(dataset_val.__len__())

    mask = None
    epoch = 0
    val_loss_arr = validate(model, val_loader, device, writer, mse_arr, mask)

    np.save(os.path.join(config_val["log_root"], networkName + "validation_mse.npy"), val_loss_arr)