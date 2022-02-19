import warnings
import pandas as pd
import os
from flask import Flask, request, jsonify, Response
from glob import glob
import logging

import config
from src.models import LightGBMModel
from predict import predict_single
from prod_models_list import models_full_list
from src.logger import setup_logger

# Ignore the warning
warnings.simplefilter("ignore")
log = setup_logger(stderr_level=logging.INFO)

app = Flask(__name__)

binary_score_models_dir = None
trading_score_models_dir = None


def load_binary_score_models():
    """Loads the models for each of the objective and then create a dictionary

    Returns:
        dict: dictionary with models list as key.
    """
    models_dir = {}
    model = LightGBMModel(config.model_parameters, config.fit_parameters, config, inference=False)

    for key, value in config.objectives_to_run.items():
        if value:
            obj_models = {}
            models_name_list = models_full_list[key]["binary_score"]
            for model_names in models_name_list:
                cocktail_models = []
                for obj_name, obj_model_list in model_names.items():
                    for i in obj_model_list:
                        m = model.load(os.path.join(os.path.join(config.PROD_MODELS_DIR, key), f"{i}.txt"))
                        cocktail_models.append(m)
                obj_models[obj_name] = cocktail_models
            models_dir[key] = obj_models
    return models_dir


def load_trading_score_models():
    """Loads the models for each of the objective and then create a dictionary

    Returns:
        dict: dictionary with models list as key.
    """
    models_dir = {}
    model = LightGBMModel(config.model_parameters, config.fit_parameters, config, inference=False)

    for key, value in config.objectives_to_run.items():
        if value:
            obj_models = {}
            models_name_list = models_full_list[key]["trading_score"]
            for model_names in models_name_list:
                cocktail_models = []
                for obj_name, obj_model_list in model_names.items():
                    for i in obj_model_list:
                        m = model.load(os.path.join(os.path.join(config.PROD_MODELS_DIR, key), f"{i}.txt"))
                        cocktail_models.append(m)
                obj_models[obj_name] = cocktail_models
            models_dir[key] = obj_models
    return models_dir


# calling the load function as we need models loaded into memory as soon as the app starts.
binary_score_models_dir = load_binary_score_models()
trading_score_models_dir = load_trading_score_models()


# get category of the app from play store
@app.route('/api/predict', methods=['POST'])
def getPredictions():
    try:
        data = request.json
        if_binary = data["is_binary"]
        df = pd.DataFrame.from_dict(data["data"])
        if if_binary.lower() == "true":
            log.info("Doing prediction for binary trading score.")
            preds = predict_single(df, binary_score_models_dir)
        else:
            log.info("Doing prediction for trading score.")
            preds = predict_single(df, trading_score_models_dir)
        return preds
    except Exception as ex:
        return Response(f"An error occurred. {ex}.", status=400)


if __name__ == "__main__":
    app.run(debug=False)
