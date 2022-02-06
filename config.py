import os
import pandas as pd
import tensorflow as tf

# data paths
DATA_FOLDER = 'data'
FILE_NAME = 'btc_dataset.csv'
DATA_PATH = os.path.join(DATA_FOLDER, FILE_NAME)

# test data
TEST_DATA = 'btc_test_data3.csv'
TEST_DATA_PATH = os.path.join(DATA_FOLDER, TEST_DATA)
TEST_DROP_COLS = ['unix']

# target definition and columns to drop
TARGET = 'Obj 7 Linked'
OTHER_TARGETS = ['Objective 2', 'Objective 3', 'Objective 4', 'Objective 5', 'Objective 6',
                 'Objective 7', 'Objective 8', 'Objective 1', 'Obj 2 Linked', 'Obj 3 Linked',
                 'Obj 4 Linked', 'Obj 5 Linked', 'Obj 6 Linked', 'Obj 1 Linked', 'Obj 8 Linked']
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
                    'is_unbalance': True
                    # 'tree_learner': 'voting',
                    # 'scale_pos_weight': 0.3
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
model_save_path = f'./weights/lgb3_{str(TARGET)}.txt'
saved_path = f'./weights/lgb3_{str(TARGET)}.txt'

# Use LSTM
use_lstm = False

# LSTM model config
lstm_config = {'optimizer': tf.keras.optimizers.SGD(learning_rate=0.01),
               'epochs': 10,
               'batch_size': 128,
               }

# use these features for testing
test_fe_names = ['MIDPRICE_60', 'KAMA_100', 'EMA_60', 'MIDPRICE_100', 'ISA_9', 'T3_100', 'EMA_100', 'ISB_26', 'MOM_5', 'HT_DCPHASE', 'AROON_down_15', 'LINEARREG_ANGLE_15', 'high_VAR_30', 'high_VAR_60', 'open_VAR_15', 'AROONOSC_14', 'NATR_30', 'close_VAR_15', 'high_VAR_100', 'ADX_15', 'low_VAR_15', 'ULTOSC_30', 'WILLR_30', 'open_VAR_60', 'AROON_down_30', 'close_VAR_60', 'open_VAR_30', 'open_VAR_100', 'CCI_30', 'close_VAR_100', 'close_VAR_30', 'low_VAR_60', 'NATR_15', 'low_VAR_100', 'ROCP_15', 'low_VAR_30', 'ULTOSC_60', 'DMP_14', 'COPC_11_14_10', 'ADX_30', 'MFI_14', 'RSI_15', 'high_STDDEV_15', 'CCI_60', 'Volume USDT', 'AROON_up_30', 'NATR_5', 'WILLR_60', 'high_STDDEV_5', 'ULTOSC_100', 'high_STDDEV_30', 'close_STDDEV_15', 'open_STDDEV_15', 'open_STDDEV_5', 'close_STDDEV_5',
                 'low_STDDEV_5', 'AROON_up_60', 'LINEARREG_ANGLE_30', 'TSI_13_25_13', 'low_STDDEV_15', 'ROCP_30', 'CCI_100', 'LINEARREG_ANGLE_60', 'TRUERANGE_1', 'close_STDDEV_30', 'open_STDDEV_30', 'high_STDDEV_60', 'WILLR_100', 'TSIs_13_25_13', 'AROON_up_100', 'low_STDDEV_30', 'MOM_15', 'RSI_30', 'close_STDDEV_60', 'open_STDDEV_60', 'high_STDDEV_100', 'ROCP_60', 'low_STDDEV_60', 'ATR_5', 'LINEARREG_ANGLE_100', 'close_STDDEV_100', 'open_STDDEV_100', 'BEARP_13', 'CUMLOGRET_1', 'low_STDDEV_100', 'ATRr_14', 'ATR_15', 'CHOP_14_1_100', 'MOM_30', 'ROCP_100', 'ATR_30', 'MASSI_9_25', 'low_macd', 'RSI_60', 'low_macdsignal', 'open_macd', 'close_macd', 'ATR_60', 'DMN_14', 'open_macdsignal', 'close_macdsignal', 'high_macd', 'ATR_100', 'high_macdsignal', 'MOM_60', 'RSI_100', 'OBV', 'MOM_100', 'Volume BTC', 'AD']
