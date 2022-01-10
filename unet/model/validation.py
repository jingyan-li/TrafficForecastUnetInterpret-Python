# Do validation over the final model for all validation days

import numpy as np
import pickle
import torch
import sys, os
sys.path.append(os.getcwd())
from datetime import datetime
from tqdm import tqdm
from videoloader import trafic4cast_dataset
# from visualizer import Visualizer

sys.path.append(os.getcwd())

from unet.model.config.config_train import config
# from unet.model.config.config_validate import config_val
from unet.model.config.config_local import config_val
from unet.model.Unet import UNet


city = config["city"]
if config_val["debug"] == True:
    networkName = "Test"
else:
    networkName = "UnetDeep_"

volume_index= np.arange(0,8,2)
speed_index= np.arange(1,8,2)


def validate(model, val_loader, device, writer, mse_arr, mask=None):
    # random_visualize = random.randint(0, len(val_loader))
    # if config["debug"] == True:
    #     random_visualize = 0

    # MSE of volume and speed at different timestamps (6 in total) in prediction
    mse_arr_vol = np.array([mse_arr]*6)
    mse_arr_speed = np.array([mse_arr] * 6)

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
            for t in range(6):
                # In each predicted time epoch, calculate mse
                val_loss_curVol = torch.nn.functional.mse_loss(val_output[:, t*8+volume_index, 8:-9, 6:-6],
                                                               val_y[:,t*8+volume_index,:,:])
                val_loss_curSpeed = torch.nn.functional.mse_loss(val_output[:, t*8+speed_index, 8:-9, 6:-6],
                                                               val_y[:,t*8+speed_index,:,:])
                mse_arr_vol[t, startt_idx[0]] = val_loss_curVol
                mse_arr_speed[t, startt_idx[0]] = val_loss_curSpeed

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
    return mse_arr, mse_arr_speed, mse_arr_vol


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
    mask_dir = config_val["mask_root"]
    mask = pickle.load(open(mask_dir, "rb"))
    mask_ = torch.from_numpy(mask > 0).bool()

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
    val_loss_arr, val_loss_arr_speed, val_loss_arr_vol = validate(model, val_loader, device, writer, mse_arr, mask)

    np.save(os.path.join(config_val["log_root"], networkName + "validation_mse.npy"), val_loss_arr)
    np.save(os.path.join(config_val["log_root"], networkName + "validation_mse_speed.npy"), val_loss_arr_speed)
    np.save(os.path.join(config_val["log_root"], networkName + "validation_mse_volume.npy"), val_loss_arr_vol)