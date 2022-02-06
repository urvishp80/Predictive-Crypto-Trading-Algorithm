import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, OneHotEncoder

import config
from src.logger import LOGGER


def fill_nan(train_df, exc_feat, is_train=True, means_dir=None):
    df = train_df.copy()

    if is_train:
        means_dir = {}
        for col in df.columns:
            if col not in exc_feat:
                mean_score = df[col].mean() if df[col].dtype == 'float64' else "Sunday"
                df[col] = df[col].fillna(mean_score)
                means_dir[col] = mean_score
    else:
        for col in df.columns:
            if col not in exc_feat:
                df[col] = df[col].fillna(means_dir[col])
    return df, means_dir


def fe_normalize(train_df, exc_feat, is_train=True, scalers=None):
    df = train_df.copy()

    if is_train:
        scalers = {}
        for col in df.columns:
            if col not in exc_feat:
                scaler = MinMaxScaler() if df[col].dtype == 'float64' else LabelEncoder()
                data = df[col].values.reshape(-1, 1) if df[col].dtype == 'float64' else df[col].values.tolist()
                df[col] = scaler.fit_transform(data)
                scalers[col] = scaler
    else:
        for col in df.columns:
            if col not in exc_feat:
                data = df[col].values.reshape(-1, 1) if df[col].dtype == 'float64' else df[col].values.tolist()
                df[col] = scalers[col].transform(data)

    return df, scalers

# for undersampling we need a portion of majority class and will take whole data of minority class
# count fraud transaction is the total number of fraud transaction
# now lets us see the index of fraud cases
# now let us a define a function for make undersample data with different proportion
# different proportion means with different proportion of normal classes of data


def undersample(data, times, count_positive_labels):  # times denote the normal data = times*fraud data
    positive_indices = np.array(data[data[config.TARGET] == True].index)
    negative_indices = np.array(data[data[config.TARGET] == False].index)
    negative_indices_undersample = np.array(np.random.choice(negative_indices, (times * count_positive_labels), replace=False))
    undersample_data = np.concatenate([positive_indices, negative_indices_undersample])
    undersample_data = data.iloc[undersample_data, :]

    LOGGER.info(f"the normal transacation proportion is : {len(undersample_data[undersample_data[config.TARGET] == False])/len(undersample_data[undersample_data.Class])}")
    LOGGER.info(f"the fraud transacation proportion is : {len(undersample_data[undersample_data[config.TARGET] == True])/len(undersample_data[undersample_data.Class])}")
    LOGGER.info(f"total number of record in resampled data is: {len(undersample_data[undersample_data.Class])}")
    return undersample_data


def create_balanced_data(features, targets):
    features[np.isnan(features)] = 0

    positive_indices = np.argwhere(targets)[:, 0]
    negative_indices = np.argwhere(targets == 0)[:, 0]

    total_pos_samples = len(positive_indices)
    if len(positive_indices) < len(negative_indices):
        negative_indices = negative_indices[-total_pos_samples:]

    sorted_indices = []
    for i, v in enumerate(positive_indices):
        sorted_indices.append(v)
        sorted_indices.append(negative_indices[i])
    all_indices = np.concatenate((positive_indices, negative_indices), axis=0)
    np.random.shuffle(all_indices)
    return features[all_indices], targets[all_indices]


def get_processed_data(strt_idx, total_len, gaps):
    x = []
    y = []
    for i in range(strt_idx, total_len, gaps):
        features = np.load(f'./data/train_{i}.npy')
        x.append(features)
        targets = np.load(f'./data/targets_{i}.npy')
        y.append(targets)

    features = np.concatenate(x)
    targets = np.concatenate(y)
    features, targets = create_balanced_data(features, targets)
    return features, targets
