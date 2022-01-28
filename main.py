import pandas as pd
import numpy as np
import logging

import config

from src.dataset import get_data, get_features_targets, merge_data, split_data
from src.indicators import get_indicators, get_price_patterns
from src.logger import setup_logger
from src.models import LightGBMModel

log = setup_logger(stderr_level=logging.INFO)

if __name__ == '__main__':

    log.info("Starting data reading.")
    df = get_data(config.DATA_PATH, drop_col=config.DROP_COLS)
    log.info("Getting indicator for data.")
    df_indicators = get_indicators(df, intervals=config.INTERVALS)
    log.info("Getting price pattern for data.")
    df_price_pattern = get_price_patterns(df)
    log.info("Merging all data into one.")
    data = merge_data(df, df_indicators, df_price_pattern)

    log.info(f"Spliting data for training and testing based on the date {config.SPLIT_DATE.iloc[0]}")
    train_df, test_df = split_data(data, config.SPLIT_DATE.iloc[0])
    log.info(f"Count of target in training {train_df[config.TARGET].value_counts()}")
    log.info(f"Count of target in testing {test_df[config.TARGET].value_counts()}")

    log.info("Getting features and targets for training data.")
    features, targets = get_features_targets(train_df, config.TARGET, date_col='Date')
    log.info("Getting features and targets for testing data.")
    valid_feat, valid_targets = get_features_targets(test_df, config.TARGET, date_col='Date')

    log.info(f"Shape of train features: {features.shape}, Shape of train targets: {targets.shape}")
    log.info(f"Shape of test features: {valid_feat.shape}, Shape of the test targets: {valid_targets.shape}")

    log.info("Initializing LightGBM model.")
    model = LightGBMModel(config.model_parameters, config.fit_parameters, config)

    model.train(features, targets, (valid_feat, valid_targets), save=False)
    acc, f1, precision, recall = model.eval(valid_feat, valid_targets)
