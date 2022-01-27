import pandas as pd
# import numpy as np


def get_data(path, drop_col=None):
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['unix'], unit='ms')
    if drop_col is not None:
        df = df.drop(drop_col, axis=1)
    return df


def get_features_targets(data, target_col, date_col=None):
    if date_col is not None:
        features = data.drop([target_col] + [date_col], axis=1).values
    else:
        features = data.drop(target_col, axis=1).values
    targets = data[target_col].values
    return features, targets
