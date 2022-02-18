import numpy as np
import os
import time
import logging

import config
from src.dataset import get_data, get_features_targets, merge_data, create_flatten_features
from src.indicators import get_indicators, get_price_patterns, get_additional_indicators
from src.logger import setup_logger

log = setup_logger(stderr_level=logging.INFO)


def predict_single(df, models_dir):
    """A simple function to get the prediction for current one minute.

    Args:
        df (dataframe): A simple dataframe, which should have columns Date, OHLCV, Volume BTC, Volume USD, Tradecount
        models_dir (dict): A dictionary with models and for each objective so the structure would be {'obj5': [model1, model2]}
    """
    preds_dict = {}

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

    for key, models in models_dir:
        features_names = config.prod_features[f'{key}']
        features, _ = get_features_targets(data, None, features_names, date_col='Date')
        log.info(f"Shape of test features: {features.shape}")

        features = features.values
        features, _ = create_flatten_features(features, None, config.n_context, features_names, return_fe_list=False)
        log.info(f"Shape of test features: {features.shape}.")
        single_preds = {}
        for i, model in models:
            pred = model.predict(features)
            single_preds['prediction'] = {'preds': pred, 'class': np.round(pred)}
        preds_dict[key] = single_preds

    return preds_dict
