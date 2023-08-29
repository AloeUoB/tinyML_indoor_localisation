import numpy as np
from scipy import stats

def MinMaxScaler(data):
    """Min-Max Normalizer.
    Args:
      - data: raw data
    Returns:
      - norm_data: normalized data
      - min_val: minimum values (for renormalization)
      - max_val: maximum values (for renormalization)
    """
    min_val = np.min(np.min(data, axis=0), axis=0)
    data = data - min_val

    max_val = np.max(np.max(data, axis=0), axis=0)
    norm_data = data / (max_val + 1e-7)

    return norm_data, min_val, max_val

def windowing(ori_data, y, seq_len = 20, hop_size = 10, shuffle=True):
    windowed_data = []
    windowed_label = []
    # Cut data by sequence length
    for i in range(0, len(ori_data) - seq_len, hop_size):
        _x = ori_data[i:i + seq_len]
        _y = stats.mode(y[i:i + seq_len])[0][0]
        windowed_data.append(_x)
        windowed_label.append(_y)
    if shuffle:
        idx = np.random.permutation(len(windowed_data))
        data = []
        label = []
        for i in range(len(windowed_data)):
            data.append(windowed_data[idx[i]])
            label.append(windowed_label[idx[i]])
    else:
        data = windowed_data
        label = windowed_label
    data = np.asarray(data)
    label = np.asarray(label)

    return data, label

def load_rssi(data_directory, house_name, datatype, shuffle_data, window_size=False, hop_size=False, flatten=False):
    if house_name == 'A':
        APs = 8
    else:
        APs = 11
    house_file = 'csv_house_' + house_name + '_' + datatype + '.csv'
    X_train = np.loadtxt(data_directory + house_file, delimiter=",", skiprows=1, usecols=list(range(1, APs+1)))
    y_train = np.loadtxt(data_directory + house_file, delimiter=",", skiprows=1, usecols=[APs+1])
    y_train = y_train - 1  # change range of labels
    # X_train, min_val, max_val = MinMaxScaler(X_train)  # min-max normalisation
    if window_size:
        X_train, y_train = windowing(X_train, y_train, window_size, hop_size, shuffle_data)
        # X_train = np.transpose(X_train, (0, 2, 1))  # set shape to (N, APs, window)
        if flatten:
            X_train = np.reshape(X_train, (-1, APs * window_size))  # flatten for MLP model?
    return X_train, y_train

# house_rssi_norm, min_val, max_val = MinMaxScaler(house_rssi)
# random_indices = np.random.choice(len(house_rssi_norm), size=100, replace=False)
# X_train = house_rssi_norm[random_indices]
# y_train = house_label [random_indices]
#
# random_indices_test = np.random.choice(len(house_rssi_norm), size=100, replace=False)
# X_test = house_rssi_norm[random_indices_test]
# y_test =house_label [random_indices_test]


# def load_house_data(data_directory, house_name, datatype='fp', reduce_ap=False, shuffle_mode=True):
#     """
#     Load data and apply window of 20 timestamp with 10 hop
#
#     :return: numpy array - RSSI(B, W, C), room_label(float)| int - AP, number of rooms
#     """
#     col_idx_use, col_idx_use_label = get_col_use(house_name, reduce_ap)
#     # load raw data
#     house_file = 'csv_house_' + house_name + '_'+datatype+'.csv'
#     ori_data = np.loadtxt(data_directory + house_file, delimiter=",", skiprows=1, usecols=col_idx_use)
#     label = np.loadtxt(data_directory + house_file, delimiter=",", skiprows=1, usecols=[col_idx_use_label])
#     # data normalisation
#     norm_data, min_val, max_val = MinMaxScaler(ori_data)
#     # get window data
#     windowed_data, windowed_label = windowing(norm_data, label, seq_len=20, hop_size=10, shuffle=shuffle_mode)
#     # get data information
#     NUM_CLASSES = num_room_house[house_name]
#     APs = len(col_idx_use)
#
#     return windowed_data, windowed_label, APs, NUM_CLASSES