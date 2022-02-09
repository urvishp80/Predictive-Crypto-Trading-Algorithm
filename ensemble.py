import pandas as pd
import numpy as np
import os
import logging
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
# import pickle
import time
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score

import config

from src.dataset import get_data, get_features_targets, merge_data, split_data, extract_most_important_features, get_lgbm_features, \
    create_flatten_features, create_lstm_features
from src.indicators import get_indicators, get_price_patterns, get_additional_indicators
from src.logger import setup_logger
from src.models import BiLSTMModel, LightGBMModel, evaluate

log = setup_logger(out_file=f'./logs/{config.TARGET}_ensemble_logs.txt', stderr_level=logging.INFO)
scaler = StandardScaler()


def evaluate_validation_data(model, j, valid_feat, valid_targets):
    log.info(f"Running model {j}")
    all_preds = []
    for i in range(0, len(valid_feat), 30000):
        log.info(f'**************** {i} *****************')
        preds, acc, f1, precision, recall = evaluate(model, valid_feat[i:i+30000], valid_targets[i:i+30000])
        all_preds.append(preds)
    return np.concatenate(all_preds)


if __name__ == '__main__':
    log.info("Doing model predictions and testing on validation data overall.")
    log.info("Loading validation data.")
    valid_feat = np.load('./data/valid_feat.npy')
    valid_targets = np.load('./data/valid_targets.npy')

    valid_preds = []
    models = []
    log.info(f"Loading models {config.models_path_list}.")
    for i, path in enumerate(config.models_path_list):
        model = LightGBMModel(config.model_parameters, config.fit_parameters, config, inference=False)
        model = model.load(path)
        models.append(model)

        preds = evaluate_validation_data(model, i, valid_feat, valid_targets)
        valid_preds.append(preds.reshape(-1, 1))

    preds = np.concatenate(valid_preds, axis=1)
    if config.mode == 'mean':
        preds = preds.mean(axis=1)
    else:
        preds = np.median(preds, axis=1)

    auc = roc_auc_score(valid_targets, preds)
    acc = accuracy_score(valid_targets, np.round(preds))
    f1 = f1_score(valid_targets, np.round(preds), average='binary')
    precision = precision_score(valid_targets, np.round(preds), average='binary')
    recall = recall_score(valid_targets, np.round(preds), average='binary')

    log.info("Overall performance on validation data for blending.")
    log.info(f"AUC score of model is {round(auc, 4)}.")
    log.info(f"Accuracy score of model is {round(acc, 4)}.")
    log.info(f"F1 score of model is {round(f1, 4)}.")
    log.info(f"Precision score is {round(precision, 4)}.")
    log.info(f"Recall score is {round(recall, 4)}")

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

    test_preds = []
    for i, model in enumerate(models):
        predictions = model.predict(features)
        df[f'Predictions_{i}'] = list(range(config.n_context)) + predictions.tolist()
        df[f'Class_{i}'] = list(range(config.n_context)) + np.round(predictions).tolist()
        log.info(f'Model name {config.models_path_list[i]}')
        log.info(f"Column name Predictions_{i} and Class_{i}")
        log.info(f"{df[f'Class_{i}'].iloc[20:].value_counts()}")

        test_preds.append(predictions.reshape(-1, 1))

    test_preds = np.concatenate(test_preds, axis=1)
    mean_preds = test_preds.mean(axis=1)
    median_preds = np.median(test_preds, axis=1)
    df['Ensemble_mean'] = list(range(config.n_context)) + mean_preds.tolist()
    df['Ensemble_mean_class'] = list(range(config.n_context)) + np.round(mean_preds).tolist()
    df['Ensemble_median'] = list(range(config.n_context)) + median_preds.tolist()
    df['Ensemble_median_class'] = list(range(config.n_context)) + np.round(median_preds).tolist()

    print(df.head())
    df = df.sort_values(by='Date', ascending=False).reset_index(drop=True)
    os.makedirs(f'./data/{config.TARGET}', exist_ok=True)
    df.to_csv(f"./data/{config.TARGET}/btc_predictions_{config.TARGET}_ensemble_{str(time.time())}.csv", index=False)
