import warnings
import pandas as pd
import os
from flask import Flask, request, Response
from glob import glob
import logging

import config
from src.models import LightGBMModel
from predict import predict_single_objective, map_preds_to_model_names
from prod_models_list import models_full_list
from src.logger import setup_logger

# Ignore the warning
warnings.simplefilter("ignore")
log = setup_logger(stderr_level=logging.INFO)

app = Flask(__name__)

# gloabal variable to store models and their mapping with names
binary_score_models_dir = None
trading_score_models_dir = None

# global variable to store previous 300 min data
cache = {}


def create_model_map(is_binary):
    """Maps the models for production to simple dictionary for each objecive so we can later use it to load and do prediction.

    Returns:
        dict: dictionary with mapping of names for each objecive type and its models list.
    """
    models_map_dir = {}

    for key, val in config.objectives_to_run.items():
        if val:
            models_map_dir[key] = []

    for key, value in config.objectives_to_run.items():
        if value:
            if is_binary:
                models_name_list = models_full_list[key]["binary_score"]
                for model_names in models_name_list:
                    for obj_name, obj_model_list in model_names.items():
                        for i in obj_model_list:
                            if i not in models_map_dir[obj_name]:
                                models_map_dir[obj_name].append(i)
            else:
                models_name_list = models_full_list[key]["trading_score"]
                for model_names in models_name_list:
                    for obj_name, obj_model_list in model_names.items():
                        for i in obj_model_list:
                            if i not in models_map_dir[obj_name]:
                                models_map_dir[obj_name].append(i)
    return models_map_dir


def load_binary_score_models():
    obj_wise_models_loaded = {}
    obj_wise_models = create_model_map(True)
    model = LightGBMModel(config.model_parameters, config.fit_parameters, config, inference=False)
    for key, model_names in obj_wise_models.items():
        loaded_model = []
        for i in model_names:
            m = model.load(os.path.join(os.path.join(config.PROD_MODELS_DIR, key), f"{i}.txt"))
            loaded_model.append((i, m))
        obj_wise_models_loaded[key] = loaded_model
    return obj_wise_models_loaded


def load_trading_score_models():
    obj_wise_models_loaded = {}
    obj_wise_models = create_model_map(False)
    model = LightGBMModel(config.model_parameters, config.fit_parameters, config, inference=False)
    for key, model_names in obj_wise_models.items():
        loaded_model = []
        for i in model_names:
            m = model.load(os.path.join(os.path.join(config.PROD_MODELS_DIR, key), f"{i}.txt"))
            loaded_model.append((i, m))
        obj_wise_models_loaded[key] = loaded_model
    return obj_wise_models_loaded


# calling the load function as we need models loaded into memory as soon as the app starts.
binary_score_models_dir = load_binary_score_models()
trading_score_models_dir = load_trading_score_models()


# get category of the app from play store
@app.route('/api/predict', methods=['POST'])
def getPredictions():
    try:
        data = request.json
        if_binary = data["is_binary"]
        current_data = data["data"]

        try:
            prev_data_dict = cache["prev_data"]
        except KeyError:
            prev_data_dict = None

        if prev_data_dict is None:
            cache["prev_data"] = current_data
            df = pd.DataFrame.from_dict(current_data)
        else:
            prev_data_dict.extend(current_data)
            prev_data_dict.pop(0)
            df = pd.DataFrame.from_dict(prev_data_dict)

        if if_binary.lower() == "true":
            log.info("Doing prediction for binary trading score.")
            preds = predict_single_objective(df, binary_score_models_dir)
            mapped_response = map_preds_to_model_names(preds, models_full_list, True)
        else:
            log.info("Doing prediction for trading score.")
            preds = predict_single_objective(df, trading_score_models_dir)
            mapped_response = map_preds_to_model_names(preds, models_full_list, False)
        return mapped_response
    except Exception as ex:
        return Response(f"An error occurred. {ex}.", status=400)


if __name__ == "__main__":
    app.run(debug=False)
