import pandas as pd
import numpy as np


def upper_shadow(df):
    return df['High'] - np.maximum(df['Close'], df['Open'])


def lower_shadow(df):
    return np.minimum(df['Close'], df['Open']) - df['Low']


# A utility function to build features from the original df
# It works for rows to, so we can reutilize it.
def get_features(df):
    df_feat = df[['open', 'high', 'low', 'close']].copy()
    df_feat['Upper_Shadow'] = upper_shadow(df_feat)
    df_feat['Lower_Shadow'] = lower_shadow(df_feat)
    return df_feat
