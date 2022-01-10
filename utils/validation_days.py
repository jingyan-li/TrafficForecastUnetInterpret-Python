# Get weekday number of days in validation set

import os
import pandas as pd
from datetime import datetime

#%%
directory = r"C:\Users\jingyli\OwnDrive\IPA\data\2021_IPA\ori\Berlin\validation"

weekdays = []
weekends = []
days = []
for file in os.listdir(directory):
    daystr = file.split("_")[0]
    days.append(daystr)
    date = datetime.strptime(daystr, "%Y-%m-%d")
    weekday = date.weekday()
    if weekday >= 5:
        weekends.append(daystr)
    else:
        weekdays.append(daystr)


#%%