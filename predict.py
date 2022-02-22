import numpy as np
import pandas as pd
import logging
from tqdm import tqdm

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
    df['Date'] = pd.to_datetime(df['unix'], unit='ms')
    df = df.sort_values(by='Date', ascending=True).reset_index(drop=True)
    df = df.resample('1Min', on='Date').mean().dropna()
    df['Date'] = df.index
    df = df.drop(config.TEST_DROP_COLS, axis=1)
    log.info("Getting indicator for data.")
    df_indicators = get_indicators(df, intervals=config.INTERVALS)
    log.info("Getting price pattern for data.")
    df_price_pattern = get_price_patterns(df)
    log.info("Getting additional indicators.")
    df_add_indicators = get_additional_indicators(df)
    log.info("Merging all data into one.")
    data = merge_data(df, df_indicators, df_price_pattern, df_add_indicators, test=True)

    for key, models_dict in models_dir.items():
        obj_predictions = {}
        for obj_name, obj_models_list in models_dict.items():
            features_names = config.prod_features[obj_name]
            features, _ = get_features_targets(data, None, features_names, date_col='Date')

            features = features.values
            features, _ = create_flatten_features(features, None, config.n_context, features_names, return_fe_list=False)
            models_preds = []
            for i, model in tqdm(enumerate(obj_models_list)):
                pred = model.predict(features)
                models_preds.append((pred.tolist(), np.round(pred).tolist()))
            obj_predictions[obj_name] = models_preds
        preds_dict[key] = obj_predictions
    return preds_dict
