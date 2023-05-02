# this file takes a subset of the data of the following size and creates a new file
SUBSET_SIZE = 1000

import h5py
import numpy as np

DATA_FILEPATH = "./data/data.h5"
SUB_FILEPATH = './data/subset.h5'
with h5py.File(DATA_FILEPATH) as h5file:
    with h5py.File(SUB_FILEPATH, 'w') as subset:
        for key in h5file.keys():
            data = h5file[key]
            data_sub = data[:SUBSET_SIZE,]
            subset.create_dataset(key, data=data_sub)

with h5py.File(SUB_FILEPATH) as h5file:
    for key in h5file.keys():
        print(key)
        data = h5file[key]
        print(data.shape)