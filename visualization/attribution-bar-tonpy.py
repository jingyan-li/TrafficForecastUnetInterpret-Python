# Author: 2021. Jingyan Li
# Analyze feature importance (python_visualization block 1)
# Loop through all validation days;
# Target to either urban / suburban window
# Read and aggregate attribution on weekdays / weekends during morning peak, noon peak, afternoon peak and late at night
# Store the mean attribution and standard deviations for website python_visualization

import numpy as np
import matplotlib.pyplot as plt
import os

import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

window = [2, 10]  # [16,10]
loc_type = "suburban"  # "urban"
day_dict = {
    "weekday": ['2019-07-01', '2019-07-11', '2019-07-31', '2019-08-20', '2019-08-30', '2019-09-09', '2019-09-19',
                '2019-10-09', '2019-10-29', '2019-11-08', '2019-11-18', '2019-11-28', '2019-12-18'],
    "weekend": ['2019-07-21', '2019-08-10', '2019-09-29', '2019-10-19', '2019-12-08']}

time = {
    84: "7AM-8AM",
    132: "11AM-12PM",
    204: "5PM-6PM",
    264: "10PM-11PM"
}
channels = [0, 1]
channel_dict = {
    0: "volume",
    1: "speed"
}

for c in channels:
    for t, tname in time.items():
        for dtype, days in day_dict.items():
            print("Processing: "+f"{dtype}_{tname}_{loc_type}_{channel_dict[c]}")
            dynamic_arr = []
            static_arr = []
            for date in days:
                log_root = f"../../attribution_Result/unet/attribution_pickle/resUnet/{date}_{t}"
                file_path = f"{date}_berlin_9ch{t}-saliency-target-channel{c}-W{window[0]}-{window[1]}.npy"

                attr = np.load(os.path.join(log_root, file_path))

                agg_channel = np.sum(attr[0].reshape(attr[0].shape[0], -1), axis=1)
                # Incident level per time epoch
                agg_incident = agg_channel[:108].reshape(12, -1)[:, -1]
                # Volume / speed per time epoch
                agg_volume_speed = np.sum(agg_channel[:108].reshape(12, -1)[:, :-1].reshape(12, -1, 2), axis=1) / 4
                # Static features
                agg_static = agg_channel[108:]
                # Save as npy array
                dynamic_arr.append(np.concatenate((agg_volume_speed, agg_incident[:, None]), axis=1))
                static_arr.append(agg_static)

            dynamic_arr = np.asarray(dynamic_arr)
            static_arr = np.asarray(static_arr)
            # Errors
            err_dynamic = np.std(dynamic_arr, axis=0)
            err_static = np.std(static_arr, axis=0)
            # Merge days
            dynamic_arr = np.mean(dynamic_arr, axis=0)
            static_arr = np.mean(static_arr, axis=0)

            output_path = os.path.join("../../attribution_Result/unet/attribution_pickle/resUnet/",
                                       f"{dtype}_{tname}_{loc_type}_{channel_dict[c]}")
            # Save dynamic features
            np.save(output_path + "_dynamic.npy", dynamic_arr)
            np.save(output_path + "_dynamic_err.npy", err_dynamic)
            # Save static features
            np.save(output_path + "_static.npy", static_arr)
            np.save(output_path + "_static_err.npy", err_static)