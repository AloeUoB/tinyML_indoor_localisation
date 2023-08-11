import os
import numpy as np
from data_util import MinMaxScaler
import sys
# sys.path.insert(0, '/data_BLE/')

data_directory = 'data_BLE/data/'
house_name = 'C'
datatype = 'fp'
house_file = 'csv_house_' + house_name + '_' + datatype + '.csv'
house_rssi = np.loadtxt(data_directory + house_file, delimiter=",", skiprows=1, usecols=list(range(1, 12)))
house_label = np.loadtxt(data_directory + house_file, delimiter=",", skiprows=1, usecols=[12])
house_rssi_norm, min_val, max_val = MinMaxScaler(house_rssi)

random_indices = np.random.choice(len(house_rssi_norm), size=100, replace=False)
temp_sample = house_rssi_norm[random_indices]
temp_label = house_label [random_indices]

