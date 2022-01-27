from src.logger import Logger
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')


class LightGBMModel:

    def __init__(self, model_parameters, fit_parameters, config):

        self.model_parameters = model_parameters
        self.fit_parameters = fit_parameters
        self.config = config
        self.logger = Logger

        self._model = None if self.config.saved_path is None else self.load(
            self.load(self.config.saved_path))

    def save(self):
        self._model.save_model(self.config.model_save_path)
        self.logger.info('Saved model to {}'.format(
            self.config.model_save_path))

    def load(self, path):
        self._model = lgb.Booster(model_file=path)  # init model
        self.logger.info('Loaded model from {}'.format(path))

    def train(self, features, targets, evaluation_set, seed=42, save=True):
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
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds)
        precision = precision_score(labels, preds, average='micro')
        recall = recall_score(labels, preds, average='micro')

        self.logger.info(f"Accuracy score of model is {round(acc, 4)}.")
        self.logger.infof(f"F1 score of model is {round(f1, 4)}.")
        self.logger.info(f"Precision score is {round(precision, 4)}.")
        self.logger.info(f"Recall score is {round(recall, 4)}")
        return acc, f1, precision, recall
