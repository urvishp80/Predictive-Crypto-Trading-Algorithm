import json
import requests
import warnings
import os
import datetime
from flask import Flask, flash, request, jsonify, render_template
from glob import glob

import config
from src.models import LightGBMModel
from predict import predict_single

# Ignore the warning
warnings.simplefilter("ignore")


def load_models():
    models_dir = {}
    model = LightGBMModel(config.model_parameters, config.fit_parameters, config, inference=False)

    for key, value in config.objectives_to_run.items():
        if value:
            obj_models = []
            models_name_list = glob(os.path.join(config.PROD_MODELS_DIR, key))
            for i in models_name_list:
                m = model.load(os.path.join(os.path.join(config.PROD_MODELS_DIR, key), i))
                obj_models.append(m)
        models_dir[key] = obj_models
    return models_dir
