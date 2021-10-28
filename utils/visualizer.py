import matplotlib.pyplot as plt
import numpy as np
import torch


def one_prediction_sample(sample, path):
    '''
    Visualize all time epochs in one prediction
    '''
    fig, axes = plt.subplots(12, 3, sharey=True)

    for i in range(12):
        start_epoch = i
        x = sample[0, start_epoch * 9:(start_epoch + 1) * 9, :, :].cpu().float().numpy()
        one_time_epoch(fig, axes[start_epoch], x)

    plt.savefig(path,
                bbox_inches="tight")
    plt.show()


def one_time_epoch(fig, axes, data, incidence=True):
    '''
    Three plots per one time epoch (5 mins): average volume, average speed, incident level
    '''
    data = data * 250
    volume_idx = np.arange(0, 8, 2)
    speed_idx = np.arange(1, 8, 2)
    if incidence:
        incident_idx = 8

    cbar = axes[0].imshow(np.mean(data[volume_idx, :, :], axis=0), vmin=0, vmax=250, cmap='RdBu_r')
    axes[1].imshow(np.mean(data[speed_idx, :, :], axis=0), vmin=0, vmax=250, cmap='RdBu_r')

    axes[0].set_title("Average volume")
    axes[1].set_title("Average speed")

    axes[0].tick_params("y", left=True, right=True, labelleft=True, labelright=False)
    axes[1].tick_params("y", left=True, right=True, labelleft=False, labelright=False)

    if incidence:
        axes[2].imshow(data[incident_idx, :, :], vmin=0, vmax=250, cmap='RdBu_r')
        axes[2].set_title("Incident level")
        axes[2].tick_params("y", left=True, right=True, labelleft=False, labelright=True)

    fig.colorbar(cbar, ax=axes, location="bottom", orientation="horizontal", pad=0.1, aspect=60)


def attr_one_time_epoch(fig, axes, data, max, min):
    '''
        Three plots per one time epoch (5 mins): average volume, average speed, incident level
        '''

    volume_idx = np.arange(0, 8, 2)
    speed_idx = np.arange(1, 8, 2)
    incident_idx = 8

    cbar = axes[0].imshow(np.mean(data[volume_idx, :, :], axis=0), vmin=min, vmax=max, cmap='RdBu_r')
    axes[1].imshow(np.mean(data[speed_idx, :, :], axis=0), vmin=min, vmax=max, cmap='RdBu_r')
    axes[2].imshow(data[incident_idx, :, :], vmin=min, vmax=max, cmap='RdBu_r')

    axes[0].set_title("Average volume")
    axes[1].set_title("Average speed")
    axes[2].set_title("Incident level")

    axes[0].tick_params("y", left=True, right=True, labelleft=True, labelright=False)
    axes[1].tick_params("y", left=True, right=True, labelleft=False, labelright=False)
    axes[2].tick_params("y", left=True, right=True, labelleft=False, labelright=True)

    fig.colorbar(cbar, ax=axes, location="bottom", orientation="horizontal", pad=0.1, aspect=60)


input_feature_semantic_dict = {
    0: "NE",
    1: "SE",
    2: "SW",
    3: "NW",
    4: "IncidentLevel"
}

input_static_semantic_dict = {
    0: "JunctionCount_0",
    1: "JunctionCount_1",
    2: "Casual",
    3: "Hospital",
    4: "Parking",
    5: "Shops",
    6: "PT"
}


def input_indices_to_semantics(idx_arr):
    '''
    A list of indices between [0, 115], return the semantics of idx
    '''
    semantics = []
    for i in idx_arr:
        if i < 108:
            time_epoch = i//9
            feature_at_epoch = i % 9

            type = "volume" if feature_at_epoch % 2 == 0 else "speed"

            direction = input_feature_semantic_dict[feature_at_epoch // 2]

            if direction == input_feature_semantic_dict[4]:
                semantics.append(f"{time_epoch}-{direction}")
            else:
                semantics.append(f"{time_epoch}-{direction}-{type}")

        else:
            semantics.append(input_static_semantic_dict[i-108])

    return semantics