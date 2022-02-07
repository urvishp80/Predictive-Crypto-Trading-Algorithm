import os
import pandas as pd
import tensorflow as tf
from src.utils import get_models_path

# data paths
DATA_FOLDER = 'data'
MODELS_FOLDER = 'weights'
FILE_NAME = 'btc_dataset.csv'
DATA_PATH = os.path.join(DATA_FOLDER, FILE_NAME)

# test data
TEST_DATA = 'btc_test_data3.csv'
TEST_DATA_PATH = os.path.join(DATA_FOLDER, TEST_DATA)
TEST_DROP_COLS = ['unix']

# target definition and columns to drop
TARGET = 'Obj 8 Linked'
OTHER_TARGETS = ['Objective 2', 'Objective 3', 'Objective 4', 'Objective 5', 'Objective 6',
                 'Objective 7', 'Objective 8', 'Objective 1', 'Obj 2 Linked', 'Obj 3 Linked',
                 'Obj 4 Linked', 'Obj 5 Linked', 'Obj 6 Linked', 'Obj 1 Linked', 'Obj 7 Linked']
DROP_COLS = ['unix'] + OTHER_TARGETS

# features and indicators
INTERVALS = (5, 15, 30, 60, 100)

# path to save feature names for future use
feature_save_path = './data/feature_names.pkl'

# feature selection threshold
fe_threshold = 0.3

# data splitting
SPLIT_DATE = pd.to_datetime(pd.Series(['2021/09/30']), format='%Y/%m/%d')

# model parameters
model_parameters = {'num_leaves': 2 ** 7,
                    'learning_rate': 0.05,
                    'bagging_fraction': 0.7,
                    'bagging_freq': 10,
                    'feature_fraction': 0.4,
                    'feature_fraction_bynode': 0.8,
                    'min_data_in_leaf': 500,
                    'min_gain_to_split': 0.1,
                    'lambda_l1': 0.01,
                    'lambda_l2': 0.001,
                    'max_bin': 1024,
                    'max_depth': -1,
                    'objective': 'binary',
                    'seed': None,
                    'feature_fraction_seed': None,
                    'bagging_seed': None,
                    'drop_seed': None,
                    'data_random_seed': None,
                    'boosting_type': 'gbdt',
                    'verbose': 1,
                    'metric': ['auc'],
                    'n_jobs': -1,
                    'force_col_wise': True,
                    # 'is_unbalance': True
                    # 'tree_learner': 'voting',
                    'scale_pos_weight': 5.00
                    }

# training parameters
fit_parameters = {
    'boosting_rounds': 200000,
    'early_stopping_rounds': 200,
    'verbose_eval': 500
}

# neg_samples_factor-
neg_samples_factor = 1

# previous context
n_context = 20

# model save and load paths
model_save_path = f'./{MODELS_FOLDER}/{TARGET}/lgb_{str(TARGET)}.txt'
saved_path = f'./{MODELS_FOLDER}/{TARGET}/lgb_{str(TARGET)}.txt'

# Use LSTM
use_lstm = False

# LSTM model config
lstm_config = {'optimizer': tf.keras.optimizers.SGD(learning_rate=0.01),
               'epochs': 10,
               'batch_size': 128,
               }

# use these features for testing
test_fe_names = ['MIDPRICE_30', 'ULTOSC_100', 'KAMA_30', 'CKSPl_10_3_20', 'CKSPs_10_3_20', 'ULTOSC_60', 'EMA_30', 'MIDPRICE_60', 'BBANDS_upper_15', 'BBANDS_upper_5', 'BBANDS_upper_100', 'BBANDS_upper_30', 'BBANDS_upper_60', 'KAMA_60', 'KAMA_100', 'MIDPRICE_100', 'EMA_60', 'MOM_15', 'ISA_9', 'DCU_10_15', 'T3_100', 'ISB_26', 'EMA_100', 'tradecount', 'AROON_up_60', 'RSI_15', 'MFI_14', 'CCI_60', 'open_VAR_15', 'close_VAR_15', 'low_VAR_15', 'HT_DCPERIOD', 'WILLR_60', 'AROON_up_100', 'LINEARREG_ANGLE_60', 'high_VAR_30', 'TSI_13_25_13', 'ROCP_60', 'high_VAR_60', 'ADX_100', 'TSIs_13_25_13', 'open_VAR_30', 'MOM_30', 'close_VAR_30', 'open_VAR_60', 'close_VAR_60', 'low_VAR_30', 'low_VAR_60', 'CCI_100', 'low_macd', 'WILLR_100', 'low_macdsignal', 'RSI_30', 'open_macd',
                 'close_macd', 'MASSI_9_25', 'LINEARREG_ANGLE_100', 'open_macdsignal', 'close_macdsignal', 'high_macd', 'high_macdsignal', 'ROCP_100', 'high_VAR_100', 'MOM_60', 'open_VAR_100', 'close_VAR_100', 'low_VAR_100', 'BEARP_13', 'MOM_100', 'RSI_60', 'high_STDDEV_5', 'low_STDDEV_5', 'TRUERANGE_1', 'high_STDDEV_15', 'close_STDDEV_5', 'open_STDDEV_5', 'DMP_14', 'CUMLOGRET_1', 'close_STDDEV_15', 'open_STDDEV_15', 'low_STDDEV_15', 'RSI_100', 'high_STDDEV_30', 'close_STDDEV_30', 'open_STDDEV_30', 'low_STDDEV_30', 'ATR_5', 'high_STDDEV_60', 'CHOP_14_1_100', 'close_STDDEV_60', 'open_STDDEV_60', 'low_STDDEV_60', 'ATRr_14', 'ATR_15', 'high_STDDEV_100', 'DMN_14', 'ATR_30', 'close_STDDEV_100', 'open_STDDEV_100', 'low_STDDEV_100', 'Volume BTC', 'ATR_60', 'ATR_100', 'OBV', 'AD']
# for blending models on test and validation data
ensemble = False
if ensemble:
    models_path_list = get_models_path(
        f'./{MODELS_FOLDER}/{TARGET}', [f'{TARGET}'])
mode = 'mean'
