import pandas as pd
import numpy as np
import logging
import pickle
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.model_selection import train_test_split
import time

import config

from src.dataset import get_data, get_features_targets, split_data, extract_most_important_features, \
    create_flatten_features, create_lstm_features, prepare_data
from src.logger import setup_logger
from src.models import BiLSTMModel, LightGBMModel, evaluate
from src.preprocess import get_processed_data, create_balanced_data


log = setup_logger(out_file=f'./logs/{config.TARGET}_training_{str(time.time())}.txt', stderr_level=logging.INFO)
log.info(f"{config.model_parameters}.")
scaler = StandardScaler()

if __name__ == '__main__':

    log.info("Starting data reading.")
    df = get_data(config.DATA_PATH, drop_col=config.DROP_COLS)
    log.info(f"Spliting data for training and testing based on the date {config.SPLIT_DATE.iloc[0]}")
    train_df, test_df = split_data(df, config.SPLIT_DATE.iloc[0])
    train_df = prepare_data(train_df)
    test_df = prepare_data(test_df)
    count_positive_labels = len(train_df[train_df[config.TARGET] == True])
    log.info(f"Total number of positive labels in data {count_positive_labels}.")

    log.info("Performing features importance on full data.")
    features_names, _, corr_mtrx = extract_most_important_features(train_df)
    log.info(f"Important features are {features_names}. Total features: {len(features_names)}")

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
            x = features[i:i+150000]
            y = targets[i:i+150000]
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

        valid_feat = np.load('./data/valid_feat.npy')
        valid_targets = np.load('./data/valid_targets.npy')

        x = []
        y = []
        for i in range(300000, len(train_df), 150000):
            features = np.load(f'./data/train_{i}.npy')
            x.append(features)
            targets = np.load(f'./data/targets_{i}.npy')
            y.append(targets)

        features = np.concatenate(x)
        targets = np.concatenate(y)
        # features, targets = create_balanced_data(features, targets)

        # X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.30, random_state=42, shuffle=True)
        # features = np.concatenate([X_train, X_test])
        # targets = np.concatenate([y_train, y_test])
        print(features.shape, targets.shape)

        model.train(features, targets, (valid_feat, valid_targets), save=False)

        for i in range(0, len(features), 50000):
            log.info(f'**************** {i} *****************')
            acc, f1, precision, recall = model.eval(features[i:i+50000], targets[i:i+50000])

        for i in range(0, len(valid_feat), 30000):
            log.info(f'**************** {i} *****************')
            acc, f1, precision, recall = model.eval(valid_feat[i:i+30000], valid_targets[i:i+30000])
        model.save()
        log.info(f"Saving features names to {config.feature_save_path} for future use.")
        with open(config.feature_save_path, 'wb') as f:
            pickle.dump(features_names, f)
