import pandas as pd
import numpy as np
from tqdm import tqdm

import config


def get_data(path, drop_col=None):
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['unix'], unit='ms')
    df = df.sort_values(by='Date', ascending=True).reset_index(drop=True)
    if drop_col is not None:
        df = df.drop(drop_col, axis=1)
    return df


def get_features_targets(data, target_col, features_names, date_col=None):
    if date_col is not None:
        cols = [target_col] + [date_col]
        features = data.drop(cols, axis=1)
    else:
        features = data.drop(target_col, axis=1)
    targets = np.where(data[target_col] == False, 0, 1)
    # cols = [col for col in features.columns if col not in features_names]
    # print(cols)
    d = features[features_names]
    return d, targets.reshape(-1, 1)


def split_data(df, date):
    # considering that the data is in ascending order
    train = df[df['Date'] <= date]
    test = df[df['Date'] > date]
    return train, test


def merge_data(data, indicators, price_patterns, additional_indicators=None):
    if additional_indicators is not None:
        data = pd.concat((data, indicators.shift(), price_patterns.shift(), additional_indicators.shift()), axis=1) #.fillna(method='bfill').fillna(method='ffill')
        data = data.dropna(subset=['Date', config.TARGET])
        return data.fillna(0)
    else:
        data = pd.concat((data, indicators.shift(), price_patterns.shift()), axis=1) #.fillna(0)
        data = data.dropna(subset=['Date', config.TARGET])
        return data.fillna(0)


def extract_most_important_features(df):
    df_local = df  # make a local copy
    context_len = None

    # Cutting data outside trader actions band by time
    start_index, end_index = 0, df[df['Date'] <= config.SPLIT_DATE.iloc[0]].reset_index(drop=True).shape[0]
    df_local = df_local.iloc[start_index: end_index + 1]
    # Determining correlations
    corr_coef = df_local.corr()
    corr_vec_sorted = corr_coef[config.TARGET].abs().fillna(0).sort_values()[:-1]
    corr_vec_sorted = corr_vec_sorted.loc[corr_vec_sorted > 0]
    the_low_limit_of_feats = 20
    assert len(corr_vec_sorted) >= the_low_limit_of_feats
    sum_corrs = corr_vec_sorted.sum()
    index_of_feature = None
    for i in range(len(corr_vec_sorted)):
        cum_sum = corr_vec_sorted.iloc[:i].sum() / sum_corrs
        if cum_sum > 0.15:  # Here 10% bumping threshold is used
            index_of_feature = i
            break
    if index_of_feature is None:
        result = list(corr_vec_sorted.index)[-the_low_limit_of_feats:], context_len, corr_coef
    else:
        result = list(corr_vec_sorted.index)[index_of_feature:], context_len, corr_coef
    return result


def create_lstm_features(data, targets, n_context, features_names):
    # features = data[features_names]
    # features = features.values
    data[np.isnan(data)] = 0
    features = data
    all_features = []
    all_targets = []

    for i in tqdm(range(n_context, len(data))):
        data_slice = features[i - n_context: i, :]
        all_features.append(data_slice)
        tar = targets[i]
        all_targets.append(tar)
    all_features = np.array(all_features)
    all_targets = np.array(all_targets)
    return all_features, all_targets


def create_flatten_features(data, targets, n_context, features_names):
    features = data
    features[np.isnan(features)] = 0
    print(features.shape, len(features))
    all_features = []
    all_targets = []

    for i in tqdm(range(n_context, len(features))):
        data_slice = features[i - n_context: i, :].reshape(1, -1)
        all_features.append(data_slice)
        tar = targets[i]
        all_targets.append(tar)
    all_features = np.squeeze(np.asarray(all_features))
    all_targets = np.asarray(all_targets)
    print(all_targets.shape, all_features.shape)
    return all_features, all_targets


def get_lgbm_features(data, targets, n_context, features_names):
    features = data[features_names]
    features = features.values
    features[np.isnan(features)] = 0

    # For xgboost we flatten all data and predict if there is a transaction
    all_features = np.zeros(shape=(0, (features.shape[1]) * n_context))
    all_targets = np.zeros(shape=(0, targets.shape[1]))
    for p_el in tqdm(np.argwhere(targets)[:, 0]):
        if p_el < n_context:
            continue
        all_features = np.concatenate((all_features, features[p_el - n_context:p_el, :].reshape((1, -1))), axis=0)
        all_targets = np.concatenate((all_targets, targets[p_el:p_el + 1, :]), axis=0)
        for _ in range(config.neg_samples_factor):
            p_el = np.random.randint(features.shape[0] - n_context) + n_context
            all_features = np.concatenate((all_features, features[p_el - n_context:p_el, :].reshape((1, -1))),
                                            axis=0)
            all_targets = np.concatenate((all_targets, targets[p_el:p_el + 1, :]), axis=0)
    # Defining the split of data
    # split = int(len(all_targets) * 0.7)
    print(targets.sum())
    return all_features, all_targets
