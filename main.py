import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

import config

from src.dataset import get_data, get_features_targets, merge_data, split_data, extract_most_important_features, get_lgbm_features, \
    create_flatten_features, create_lstm_features
from src.indicators import get_indicators, get_price_patterns, get_additional_indicators
from src.logger import setup_logger
from src.models import BiLSTMModel, LightGBMModel, evaluate


log = setup_logger(stderr_level=logging.INFO)
scaler = StandardScaler()

if __name__ == '__main__':

    log.info("Starting data reading.")
    df = get_data(config.DATA_PATH, drop_col=config.DROP_COLS)
    log.info("Getting indicator for data.")
    df_indicators = get_indicators(df, intervals=config.INTERVALS)
    log.info("Getting price pattern for data.")
    df_price_pattern = get_price_patterns(df)
    log.info("Getting additional indicators.")
    df_add_indicators = get_additional_indicators(df)
    log.info("Merging all data into one.")
    data = merge_data(df, df_indicators, df_price_pattern, df_add_indicators)

    log.info("Performing features importance on data.")
    features_names, _, corr_mtrx = extract_most_important_features(data)
    log.info(f"Important features are {features_names}. Total features: {len(features_names)}")

    log.info(f"Spliting data for training and testing based on the date {config.SPLIT_DATE.iloc[0]}")
    train_df, test_df = split_data(data, config.SPLIT_DATE.iloc[0])
    log.info(f"Count of target in training {train_df[config.TARGET].value_counts()}")
    log.info(f"Count of target in testing {test_df[config.TARGET].value_counts()}")

    log.info("Getting features and targets for training data.")
    features, targets = get_features_targets(train_df, config.TARGET, features_names, date_col='Date')
    log.info("Getting features and targets for testing data.")
    valid_feat, valid_targets = get_features_targets(test_df, config.TARGET, features_names, date_col='Date')

    log.info(f"Shape of train features: {features.shape}, Shape of train targets: {targets.shape}")
    log.info(f"Shape of test features: {valid_feat.shape}, Shape of the test targets: {valid_targets.shape}")

    if config.use_lstm:
        log.info("Normalizing features for LSTM model.")
        features = scaler.fit_transform(features[features_names])
        valid_feat = scaler.transform(valid_feat[features_names])

    if not config.use_lstm:
        features = features.values
        valid_feat = valid_feat.values
        targets = targets
        for i in range(0, len(train_df), 150000):
            log.info(f"Preparing data for LGBM with previous context {config.n_context}.")
            # x = features[i:i+150000], y = targets[i:i+150000]
            x, y = create_flatten_features(features[i:i+150000], targets[i:i+150000], config.n_context, features_names)
            np.save(f'./data/train_{i}.npy', x)
            np.save(f'./data/targets_{i}.npy', y)
        valid_feat, valid_targets = create_flatten_features(valid_feat, valid_targets, config.n_context, features_names)
        np.save('./data/valid_feat.npy', valid_feat)
        np.save('./data/valid_targets.npy', valid_targets)
    else:
        log.info(f"Preparing data for LSTM with previous context {config.n_context}.")
        features, targets = create_lstm_features(features[0:200000], targets[0:200000], config.n_context, features_names)
        valid_feat, valid_targets = create_lstm_features(valid_feat, valid_targets, config.n_context, features_names)

    log.info(f"Shape of train features: {features.shape}, Shape of train targets: {targets.shape}")
    log.info(f"Shape of test features: {valid_feat.shape}, Shape of the test targets: {valid_targets.shape}")

    if config.use_lstm:
        log.info("Initializing LSTM model.")
        model = BiLSTMModel(input_shape=(config.n_context, features.shape[2]), output_shape=(1))
        model = model.create_model()

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=tf.keras.metrics.BinaryAccuracy())

        model.fit(features, targets, validation_data=(valid_feat, valid_targets), batch_size=config.lstm_config['batch_size'],
                  epochs=config.lstm_config['epochs'])
        model.evaluate(valid_feat, valid_targets)
        evaluate(model, valid_feat, valid_targets)
    else:
        log.info("Initializing LightGBM model.")
        model = LightGBMModel(config.model_parameters, config.fit_parameters, config)

        # model.train(features, targets, (valid_feat, valid_targets), save=False)
        # acc, f1, precision, recall = model.eval(valid_feat, valid_targets)
