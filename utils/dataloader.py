#  Author: 2021. Jingyan Li
import h5py
import numpy as np


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