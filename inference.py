import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import pickle

import config

from src.dataset import get_data, get_features_targets, merge_data, split_data, extract_most_important_features, get_lgbm_features, \
    create_flatten_features, create_lstm_features
from src.indicators import get_indicators, get_price_patterns, get_additional_indicators
from src.logger import setup_logger
from src.models import BiLSTMModel, LightGBMModel, evaluate

log = setup_logger(stderr_level=logging.INFO)
scaler = StandardScaler()

if __name__ == '__main__':

    log.info("Starting test data reading.")
    df = get_data(config.TEST_DATA_PATH, drop_col=config.TEST_DROP_COLS)
    log.info("Getting indicator for data.")
    df_indicators = get_indicators(df, intervals=config.INTERVALS)
    log.info("Getting price pattern for data.")
    df_price_pattern = get_price_patterns(df)
    log.info("Getting additional indicators.")
    df_add_indicators = get_additional_indicators(df)
    log.info("Merging all data into one.")
    data = merge_data(df, df_indicators, df_price_pattern, df_add_indicators, test=True)
    print(data.head())

    log.info("Loading features names from pickle file.")
    # with open(config.feature_save_path, 'rb') as f:
    #     features_names = pickle.load(f)

    features_names = config.test_fe_names
    log.info("Getting features and targets for training data.")
    features, _ = get_features_targets(data, None, features_names, date_col='Date')
    log.info(f"Shape of test features: {features.shape}")

    features = features.values
    features, _ = create_flatten_features(features, None, config.n_context, features_names)
    log.info(f"Shape of test features: {features.shape}.")

    log.info("Initializing LightGBM model.")
    model = LightGBMModel(config.model_parameters, config.fit_parameters, config, inference=True)

    predictions = model.predict(features)
    log.info(f"Shape of the predictions {predictions.shape}")

    df['Predictions'] = list(range(config.n_context)) + predictions.tolist()
    df['Class'] = list(range(config.n_context)) + np.round(predictions).tolist()
    # df['Class'] = np.where(df['Class'] == 0, False, True)
    print(df.head())
    df.to_csv(f"./data/btc_predictions_{config.TARGET}_{config.saved_path.split('/')[-1][:-4]}.csv", index=False)
