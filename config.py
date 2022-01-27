import os

# data paths
DATA_FOLDER = 'data'
FILE_NAME = 'btc_dataset.csv'
DATA_PATH = os.path.join(DATA_FOLDER, FILE_NAME)

# target definition and columns to drop
TARGET = 'Objective 1'
OTHER_TARGETS = ['Objective 2', 'Objective 3', 'Objective 4', 'Objective 5', 'Objective 6',
                 'Objective 7', 'Objective 8', 'Obj 1 Linked', 'Obj 2 Linked', 'Obj 3 Linked',
                 'Obj 4 Linked', 'Obj 5 Linked', 'Obj 6 Linked', 'Obj 7 Linked', 'Obj 8 Linked']
DROP_COLS = ['unix'] + OTHER_TARGETS

# features and indicators
INTERVALS = (5, 10, 15, 20, 30)

# model parameters
model_parameters = {'num_leaves': 2 ** 6,
                    'learning_rate': 0.05,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 1,
                    'feature_fraction': 0.7,
                    'feature_fraction_bynode': 1,
                    'min_data_in_leaf': 10,
                    'min_gain_to_split': 0,
                    'lambda_l1': 0.01,
                    'lambda_l2': 0,
                    'max_bin': 512,
                    'max_depth': -1,
                    'objective': 'binary',
                    'seed': None,
                    'feature_fraction_seed': None,
                    'bagging_seed': None,
                    'drop_seed': None,
                    'data_random_seed': None,
                    'boosting_type': 'gbdt',
                    'verbose': 1,
                    'metric': ['auc', 'binary_logloss'],
                    'n_jobs': -1,
                    }

# training parameters
fit_parameters = {
    'boosting_rounds': 20000,
    'early_stopping_rounds': 200,
    'verbose_eval': 500
}
