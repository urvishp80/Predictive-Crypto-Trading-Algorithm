import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

import lightgbm as lgb

class LightGBMModel:

    def __init__(self, features, target, model_parameters, fit_parameters, categorical_features, seeds):

        self.features = features
        self.target = target
        self.model_parameters = model_parameters
        self.fit_parameters = fit_parameters
        self.categorical_features = categorical_features
        self.seeds = seeds

    def train_and_predict(self, X_train, y_train, X_test, df_train, df_test):

        seed_avg_oof_predictions = np.zeros(X_train.shape[0])
        seed_avg_test_predictions = np.zeros(X_test.shape[0])
        seed_avg_importance = pd.DataFrame(data=np.zeros(len(self.features)), index=self.features, columns=['Importance'])

        for seed in self.seeds:
            print(f'{"-" * 30}\nRunning LightGBM model with seed: {seed}\n{"-" * 30}\n')
            self.model_parameters['seed'] = seed
            self.model_parameters['feature_fraction_seed'] = seed
            self.model_parameters['bagging_seed'] = seed
            self.model_parameters['drop_seed'] = seed
            self.model_parameters['data_random_seed'] = seed

            for fold in sorted(X_train['fold'].unique()):

                trn_idx, val_idx = X_train.loc[X_train['fold'] != fold].index, X_train.loc[X_train['fold'] == fold].index
                trn = lgb.Dataset(X_train.loc[trn_idx, self.features], label=y_train.loc[trn_idx], categorical_feature=self.categorical_features)
                val = lgb.Dataset(X_train.loc[val_idx, self.features], label=y_train.loc[val_idx], categorical_feature=self.categorical_features)

                model = lgb.train(
                    params=self.model_parameters,
                    train_set=trn,
                    valid_sets=[trn, val],
                    num_boost_round=self.fit_parameters['boosting_rounds'],
                    early_stopping_rounds=self.fit_parameters['early_stopping_rounds'],
                    verbose_eval=self.fit_parameters['verbose_eval']
                )

                val_predictions = model.predict(X_train.loc[val_idx, self.features])
                seed_avg_oof_predictions[val_idx] += (val_predictions / len(self.seeds))
                test_predictions = model.predict(X_test[self.features])
                seed_avg_test_predictions += (test_predictions / X_train['fold'].nunique() / len(self.seeds))
                seed_avg_importance['Importance'] += (model.feature_importance(importance_type='gain') / X_train['fold'].nunique() / len(self.seeds))

                fold_score = mean_squared_error(y_train.loc[val_idx], val_predictions, squared=False)
                print(f'\nLGB Fold {int(fold)} - X_trn: {X_train.loc[trn_idx, self.features].shape} X_val: {X_train.loc[val_idx, self.features].shape} - Score: {fold_score:.6f} - Seed: {seed}\n')

        df_train['lgb_predictions'] = seed_avg_oof_predictions
        df_test['lgb_predictions'] = seed_avg_test_predictions
        oof_score = mean_squared_error(y_train, df_train['lgb_predictions'], squared=False)
        print(f'{"-" * 30}\nLGB OOF RMSE: {oof_score:.6f} ({len(self.seeds)} Seed Average)\n{"-" * 30}')