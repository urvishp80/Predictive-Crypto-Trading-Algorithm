from src.logger import LOGGER
import abc
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, GRU, RNN, Embedding, Dropout, Flatten, Bidirectional, Conv1D, Add, Multiply, Input
from tensorflow.keras.models import Model
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
import lightgbm as lgb
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class LightGBMModel:

    def __init__(self, model_parameters, fit_parameters, config, inference=False):

        self.model_parameters = model_parameters
        self.fit_parameters = fit_parameters
        self.config = config
        self.logger = LOGGER
        self.inference = inference

        if self.inference:
            self._model = self.load(self.config.saved_path)
        else:
            self._model = None   # if self.config.saved_path is None else self.load(self.load(self.config.saved_path))

    def save(self):
        self._model.save_model(self.config.model_save_path)
        self.logger.info('Saved model to {}'.format(
            self.config.model_save_path))

    def load(self, path):
        model = lgb.Booster(model_file=path)  # init model
        self.logger.info('Loaded model from {}'.format(path))
        return model

    def train(self, features, targets, evaluation_set, seed=2022, save=True):
        trn = lgb.Dataset(features, label=targets)
        val = lgb.Dataset(evaluation_set[0], label=evaluation_set[1])

        self.model_parameters['seed'] = seed
        self.model_parameters['feature_fraction_seed'] = seed
        self.model_parameters['bagging_seed'] = seed
        self.model_parameters['drop_seed'] = seed
        self.model_parameters['data_random_seed'] = seed

        self._model = lgb.train(
            params=self.model_parameters,
            train_set=trn,
            valid_sets=[trn, val],
            num_boost_round=self.fit_parameters['boosting_rounds'],
            early_stopping_rounds=self.fit_parameters['early_stopping_rounds'],
            verbose_eval=self.fit_parameters['verbose_eval']
        )
        if save:
            self.save()

    def predict(self, features):
        preds = self._model.predict(features)
        return preds

    def eval(self, features, labels):
        preds = self._model.predict(features)
        # preds = np.where(preds > 0.35, 1, 0)
        auc = roc_auc_score(labels, preds)
        acc = accuracy_score(labels, np.round(preds))
        f1 = f1_score(labels, np.round(preds), average='binary')
        precision = precision_score(labels, np.round(preds), average='binary')
        recall = recall_score(labels, np.round(preds), average='binary')

        self.logger.info(f"AUC score of model is {round(auc, 4)}.")
        self.logger.info(f"Accuracy score of model is {round(acc, 4)}.")
        self.logger.info(f"F1 score of model is {round(f1, 4)}.")
        self.logger.info(f"Precision score is {round(precision, 4)}.")
        self.logger.info(f"Recall score is {round(recall, 4)}")
        return auc, f1, precision, recall


def evaluate(model, features, labels):
    preds = model.predict(features)
    acc = roc_auc_score(labels, preds)
    f1 = f1_score(labels, np.round(preds), average='micro')
    precision = precision_score(labels, np.round(preds), average='micro')
    recall = recall_score(labels, np.round(preds), average='micro')

    print(f"AUC score of model is {round(acc, 4)}.")
    print(f"F1 score of model is {round(f1, 4)}.")
    print(f"Precision score is {round(precision, 4)}.")
    print(f"Recall score is {round(recall, 4)}")
    return acc, f1, precision, recall


class BaseModel(abc.ABC):
    """Base class for models."""

    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    @abc.abstractmethod
    def create_model(self):
        pass


class BiLSTMModel(BaseModel):
    """
     Bidirectional LSTM model class to create model.

     Arguments:
      * input_shape: Shape of the input tensor.
      * output_shape: Shape of the output tensor.
    """

    def __init__(self, input_shape, output_shape, *args, **kwargs):
        self.input_shape = input_shape
        self.output_shape = output_shape

        # model initialization
        self.model = None

    def create_model(self):

        inputs = tf.keras.Input(shape=self.input_shape)

        x = Bidirectional(
            LSTM(64, activation='elu', return_sequences=True))(inputs)
        x = Bidirectional(LSTM(64, activation='elu'))(x)

        x = Flatten()(x)
        x = Dense(64, activation='elu')(x)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='elu')(x)
        x = Dropout(0.3)(x)
        x = Dense(16, activation='elu')(x)
        output = Dense(self.output_shape, activation='sigmoid')(x)

        self.model = Model(inputs=inputs, outputs=output)
        return self.model
