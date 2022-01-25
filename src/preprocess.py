import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, OneHotEncoder

def fill_nan(train_df, exc_feat,is_train=True, means_dir=None):
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