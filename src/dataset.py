import pandas as pd
import numpy as np


def get_data(path, drop_col=None):
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['unix'], unit='ms')
    df = df.sort_values(by='Date', ascending=True)
    if drop_col is not None:
        df = df.drop(drop_col, axis=1)
    return df


def get_features_targets(data, target_col, date_col=None):
    if date_col is not None:
        cols = [target_col] + [date_col]
        features = data.drop(cols, axis=1).values
    else:
        features = data.drop(target_col, axis=1).values
    targets = np.where(data[target_col] == False, 0, 1)
    return features, targets.reshape(-1, 1)


def split_data(df, date):
    # considering that the data is in ascending order
    train = df[df['Date'] <= date]
    test = df[df['Date'] > date]
    return train, test


def merge_data(data, indicators, price_patterns):
    return pd.concat((data, indicators.shift(), price_patterns.shift()), axis=1)
