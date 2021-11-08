import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd

FIG_DIR = r"C:\Users\jingyli\OwnDrive\IPA\result\raw_data_stats"
DATA_DIR = r"C:\Users\jingyli\OwnDrive\IPA\data\2021_IPA\ori\Berlin\validation"

mse = np.load(r"C:\Users\jingyli\OwnDrive\IPA\python-eda-code\unet\log\UnetDeep_validation_mse.npy")

# Get files
file_paths = glob.glob(os.path.join(DATA_DIR,'*.h5'))
file_names = [f.split("\\")[-1].split("_")[0] for f in file_paths]
#%%
x_ticks = [i for i in range(mse.shape[0]) if i % 270 == 0]
x_ticklabel = file_names
plt.figure(1, figsize=(15,6))
plt.plot(mse)
plt.xticks(x_ticks, x_ticklabel, rotation=90)
plt.grid("on")
plt.savefig(os.path.join(FIG_DIR, "mse.pdf"), bbox_inches="tight")
plt.show()
#%%
idx = np.where(mse >= 275)[0]
date = idx // 270
stamp = idx % 270
hour = stamp // 12
mint = stamp % 12

stats = pd.DataFrame()
stats["date"] = [file_names[_] for _ in date]
stats["hour"] = hour
stats["mint"] = mint
stats["mse"] = mse[idx]


# days
uniqueDay, dayCount = np.unique(date, return_counts=True)
uniqueDay = [file_names[_] for _ in uniqueDay]
uniqueDay = sorted(zip(uniqueDay, dayCount), key=lambda _:_[1], reverse=True)

uniqueHour, hourCount = np.unique(hour, return_counts=True)
uniqueHour = sorted(zip(uniqueHour, hourCount), key=lambda _:_[1], reverse=True)
#%%
# 07-11
abnormal_df = stats[stats["date"]=="2019-09-29"]
