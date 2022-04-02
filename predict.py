import numpy as np
import pandas as pd
import logging
import time

import config
from src.dataset import get_features_targets, merge_data, create_flatten_features
from src.indicators import get_indicators, get_price_patterns, get_additional_indicators
from src.logger import setup_logger

log = setup_logger(stderr_level=logging.INFO)


def predict_single_objective(df, models_dir):
    """A simple function to get the prediction for current one minute.

    Args:
        df (dataframe): A simple dataframe, which should have columns Date, OHLCV, Volume BTC, Volume USD, Tradecount
        models_dir (dict): A dictionary with models and for each objective so the structure would be {'obj5': [model1, model2]}
    """
    preds_dict = {}

    start_time = time.time()
    log.info("Starting test data reading.")
    df['Date'] = pd.to_datetime(df['unix'], unit='ms')
    df = df.sort_values(by='Date', ascending=True).reset_index(drop=True)
    df = df.resample('1Min', on='Date').mean().dropna()
    df['Date'] = df.index
    df = df.drop(config.TEST_DROP_COLS, axis=1)
    log.info("Getting indicator for data.")
    df_indicators = get_indicators(df, intervals=config.INTERVALS, PROD_MODE=True)
    log.info("Getting additional indicators.")
    df_add_indicators = get_additional_indicators(df)
    log.info("Merging all data into one.")
    data = merge_data(df, df_indicators, df_add_indicators, test=True)
    end_time = time.time()
    log.info(f"time to make predictions. {(end_time - start_time)}")

    start_time = time.time()
    for key, model_list in models_dir.items():
        features_names = config.prod_features[key]
        features, _ = get_features_targets(data, None, features_names, date_col='Date')

        features = features.values
        features, _ = create_flatten_features(features, None, config.n_context, features_names, return_fe_list=False)

        obj_predictions = {}
        if len(model_list) >= 1:
            for model_name, model in model_list:
                pred = model.predict(features[-1:])
                obj_predictions[model_name] = (pred.tolist(), np.round(pred).tolist())
        preds_dict[key] = obj_predictions
    end_time = time.time()
    log.info(f"time to make predictions. {(end_time - start_time)}")
    return preds_dict


def map_preds_to_model_names(preds_dict, model_mapping, if_binary):
    if if_binary:
        mapped_preds = {}
        for score_type in ["binary_score", "min_max_consecutive_losses", "martingale_return"]:
            score_type_preds = {}
            for key, models_dict in model_mapping.items():
                try:
                    models_name_list = models_dict[score_type]

                    for obj_models_list in models_name_list:
                        for obj_name, model_names in obj_models_list.items():
                            for temp_name, model in model_names:
                                model_num = temp_name.split('_')[1]
                                preds_from_dict = preds_dict[obj_name][model]
                                score_type_preds[f"{key.upper()}{obj_name.upper()}C{model_num}C"] = preds_from_dict[0][0]
                                score_type_preds[f"{key.upper()}{obj_name.upper()}C{model_num}P"] = preds_from_dict[1][0]
                except KeyError:
                    pass
            mapped_preds[score_type] = score_type_preds
        return mapped_preds

    if not if_binary:
        mapped_preds = {}
        score_type_preds = {}
        for key, models_dict in model_mapping.items():
            try:
                models_name_list = models_dict["trading_score"]
                for obj_models_list in models_name_list:
                    for obj_name, model_names in obj_models_list.items():
                        for temp_name, model in model_names:
                            model_num = temp_name.split('_')[1]
                            preds_from_dict = preds_dict[obj_name][model]
                            score_type_preds[f"{key.upper()}{obj_name.upper()}C{model_num}C"] = preds_from_dict[0][0]
                            score_type_preds[f"{key.upper()}{obj_name.upper()}C{model_num}P"] = preds_from_dict[1][0]
            except KeyError:
                log.info("Error while finding keys for model map.")
                pass
        mapped_preds["trading_score"] = score_type_preds
        return mapped_preds
