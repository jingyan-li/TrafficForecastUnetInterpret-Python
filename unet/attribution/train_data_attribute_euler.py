#  Author: 2021. Jingyan Li
#  Code wrapped for attribution
#  Can be run either locally (CPU/GPU) or on server (GPU)

import numpy as np
import torch
import torch.nn as nn
import sys, os, glob
from captum.attr import Saliency
sys.path.append(os.getcwd())

from unet.model.config.config import config
from unet.model.Unet import UNet
from utils import dataload_utils

import time
import argparse

# simplified depth 5 model
####### training settings #########
parser = argparse.ArgumentParser()
parser.add_argument("--date", default="2019-09-19", type=str, help="dates in validation set you want to conduct attribution; can be a list of dates, splited by comma")
parser.add_argument("--timestamp", default="142", type=str, help='[0,288] first timestamp in input data; "142,123" can be list of timestamps, splited by comma')
parser.add_argument("--run_local", default=False, type=bool, help="True if run locally")
parser.add_argument("--window", default="12-12", type=str, help="Targets Window;")
parser.add_argument("--source_root", default="/cluster/scratch/jingyli/ipa/data/2021_IPA/ori", type=str, help="path to data")
parser.add_argument("--model_root", default="/cluster/scratch/jingyli/ipa/data/2021_IPA/resUnet_1634736536/checkpoint.pt", type=str, help="path to model")
parser.add_argument("--arr_log_root", default="/cluster/scratch/jingyli/ipa/attribution_result/unet/attribution_pickle/resUnet", type=str, help="model output root")

args = parser.parse_args()
###################################
# please enter the source data root and submission root

source_root = args.source_root
model_root = args.model_root
arr_log_root = args.arr_log_root

if args.run_local:
    # Use local path
    source_root = r"C:\Users\jingyli\OwnDrive\IPA\data\2021_IPA\ori"
    model_root = r"C:\Users\jingyli\OwnDrive\IPA\data\2021_IPA\resUnet_1634736536\checkpoint.pt"
    arr_log_root = r"C:\Users\jingyli\OwnDrive\IPA\attribution_Result\unet\attribution_pickle\resUnet"

print(f"CUDA is available: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet(img_ch=config["in_channels"], output_ch=config["n_classes"]).to(device)
city = "Berlin"

padd = torch.nn.ZeroPad2d((6, 6, 8, 9))

state_dict = torch.load(model_root, map_location=device)
model.load_state_dict(state_dict)
model.eval()


def load_data(TIME, file_path):
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


    # Preprocess of input
    inputs = padd(tepoch[:1,:,:,:])

    # Forward
    with torch.no_grad():
        pred = model(inputs)

    return inputs, gt_epoch, pred


def save_prediction_error(gt_epoch, pred, path):
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

    np.save(path, out)


def save_incident_gt(inputs, path):
    # Save incident in ground truth for 12 time epochs
    inputs = inputs.numpy()
    # Select incidents
    inc_idx = np.arange(8,9*12,9)
    inc_gt_epoch = inputs[:, inc_idx, :, :][0]
    np.save(path, inc_gt_epoch)


def model_wrapper_window(inp):
    '''
    Wrap the model by down sampling the spatial resolution and agg speed/volume of 4 directions in the output
    '''
    pooling_layer = nn.AvgPool2d(kernel_size=WINDOW_SIZE, stride=WINDOW_SIZE)
    model_out = pooling_layer(model(inp))
    model_agg = model_out.reshape((model_out.shape[0],-1,4,model_out.shape[2],model_out.shape[3])).sum(axis=2)
    return model_agg


def do_attribution(inputs,TARGET_CHANNEL,TARGET_X,TARGET_Y):
    '''
    TARGET_X: Window index in Height (Top:0; Bottom:495//21)
    TARGET_Y: Window index in Width
    '''
    # Preserve gradients
    inputs.requires_grad = True
    sa = Saliency(model_wrapper_window)
    # Do attribution
    attr = sa.attribute(inputs, abs=True, target=(TARGET_CHANNEL,TARGET_X,TARGET_Y))
    attr = attr.detach().numpy()
    return attr


if __name__=="__main__":


    # load static data
    filepath = glob.glob(os.path.join(source_root, city, f"{city}_static_2019.h5"))[0]
    static = dataload_utils.load_h5_file(filepath)
    static = torch.from_numpy(static).permute(2, 0, 1).unsqueeze(0).to(device).float()

    # Load train data
    DATELIST = [d for d in args.date.split(",")]
    WINDOW = [int(w) for w in args.window.split("-")]
    # TIMELIST = [int(t) for t in args.timestamp.split(",")]
    # TIMELIST = [_ for _ in range(0,288,12)]
    # DATELIST = ['2019-07-01',
    #              '2019-07-11',
    #              '2019-07-21',
    #              '2019-07-31',
    #              '2019-08-10',
    #              '2019-08-20',
    #              '2019-08-30',
    #              '2019-09-09',
    #              '2019-09-19',
    #              '2019-09-29',
    #              '2019-10-09',
    #              '2019-10-19',
    #              '2019-10-29',
    #              '2019-11-08',
    #              '2019-11-18',
    #              '2019-11-28',
    #              '2019-12-08',
    #              '2019-12-18']
    TIMELIST = [84, 132, 204, 264]
    for DATE in DATELIST:
        for TIME in TIMELIST:
            arr_log_root_cur = os.path.join(arr_log_root, f"{DATE}_{TIME}")
            if not os.path.exists(arr_log_root_cur):
                os.mkdir(arr_log_root_cur)
            # else:
            #     continue
            print(f"Attribution for {DATE} at timeepoch {TIME}")
            file_path = glob.glob(os.path.join(source_root, city, "validation", f"{DATE}_{city.lower()}_9ch.h5"))[0]
            pred_err_path = os.path.join(arr_log_root_cur,
                                         os.path.split(file_path)[-1][:-3]
                                         + f"{TIME}-err-pred.npy")
            inc_path = os.path.join(arr_log_root_cur,
                                         os.path.split(file_path)[-1][:-3]
                                         + f"{TIME}-gt-inc.npy")
            inputs, gt_epoch, pred = load_data(TIME, file_path)
            # Save prediction result and error map
            if not os.path.isfile(pred_err_path):
                save_prediction_error(gt_epoch, pred, pred_err_path)
                print("Prediction and err map saved!")
            # Save incident level in ground truth
            if not os.path.isfile(inc_path):
                save_incident_gt(inputs, inc_path)
                print("GT saved!")
            # Aggregate target by channel and local windows (WINDOWSIZE*WINDOWSIZE)
            WINDOW_SIZE = 21
            target_channels = [0, 1]  # Volume/Speed
            windows = 0
            target_xs = np.arange(WINDOW[0]-windows, WINDOW[0]+windows+1)  # np.arange(10, 13)  # H
            target_ys = np.arange(WINDOW[1]-windows, WINDOW[1]+windows+1)  # np.arange(10, 13)   # W
            print("Start Attribution")
            start_ = time.time()
            for TARGET_CHANNEL in target_channels:
                for TARGET_X in target_xs:
                    for TARGET_Y in target_ys:
                        # Forward by wrapper
                        attr_file_path = os.path.join(arr_log_root_cur,
                                             os.path.split(file_path)[-1][:-3]
                                             + f"{TIME}-saliency-target-channel{TARGET_CHANNEL}-W{TARGET_X}-{TARGET_Y}.npy")
                        if not os.path.isfile(attr_file_path):
                            attr = do_attribution(inputs,TARGET_CHANNEL,TARGET_X,TARGET_Y)
                            np.save(attr_file_path,
                                    attr)
                            print(f"Cumulated time used - Channel{TARGET_CHANNEL}-X{TARGET_X}-Y{TARGET_Y}: {time.time()-start_}")