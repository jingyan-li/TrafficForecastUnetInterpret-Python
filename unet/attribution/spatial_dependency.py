import numpy as np
import os
import matplotlib.pyplot as plt

DATE = "2019-09-19"
TIME = "142"
CHANNEL = "1"
W = f"{13}-{5}"

def read_attr(DATE, TIME, CHANNEL, W, MODEL):
    file_path = f"{DATE}_berlin_9ch{TIME}-saliency-target-channel{CHANNEL}-W{W}.npy"
    figure_log_root = os.path.join(r"C:\Users\jingyli\OwnDrive\RA",f"{MODEL}/figures",
                                   f"{DATE}_{TIME}", "attr", f"W{W}")
    if not os.path.exists(figure_log_root):
        os.makedirs(figure_log_root)
    log_root = os.path.join(r"C:\Users\jingyli\OwnDrive\RA", f"{MODEL}/attribution_pickle",
                            f"{DATE}_{TIME}", f"W{W}")
    attr = np.load(os.path.join(log_root, file_path))[0]
    return attr, figure_log_root

def get_space_mat(attr):
    # Get contribution per timeepoch
    C, H, W = attr.shape
    time_attr = np.sum(attr[:108].reshape(12, -1, H, W)[:, :-1, ...].reshape(12, -1, H, W), axis=1)
    static_attr = attr[108:]
    stacked_attr = np.concatenate((time_attr, static_attr), axis=0)

    C, H, W = stacked_attr.shape

    sample_mat = stacked_attr.reshape(C, -1)
    return sample_mat

file_format = "png"
#%%
attr, _ = read_attr(DATE, TIME, CHANNEL, W, "unet_good")
sample_mat = get_space_mat(attr)

attr_, fig_path = read_attr(DATE, TIME, CHANNEL, W, "unet_random")
sample_mat_ = get_space_mat(attr_)
#%%
# Pearson correlation
coef = np.corrcoef(sample_mat, sample_mat_)

#%%
# Visualize in heatmap
import seaborn as sns
from matplotlib.colors import LogNorm
import pandas as pd
from utils.visualize_utils import input_static_semantic_dict
COLUMN = [f"input t{i}" for i in range(12)] + [input_static_semantic_dict[i] for i in range(7)]
COLUMN_GOOD = [f"model_{_}"for _ in COLUMN]
COLUMN_RANDOM = [f"random_{_}"for _ in COLUMN]
df = pd.DataFrame(data=coef, columns=COLUMN_GOOD+COLUMN_RANDOM, index=COLUMN_GOOD+COLUMN_RANDOM)
sns.set(rc={'figure.figsize':(30,25)})
ax = sns.heatmap(df, annot=True, fmt=".3f", linewidths=.5, square=True, norm=LogNorm(), cmap="YlGn")
plt.savefig(os.path.join(fig_path, f"random-good_confmat.{file_format}"), bbox_inches="tight")
plt.show()