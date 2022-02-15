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
TARGET = 'Obj 3 Linked'
OTHER_TARGETS = ['Objective 2', 'Objective 3', 'Objective 4', 'Objective 5', 'Objective 6',
                 'Objective 7', 'Objective 8', 'Objective 1', 'Obj 2 Linked', 'Obj 4 Linked',
                 'Obj 5 Linked', 'Obj 6 Linked', 'Obj 8 Linked', 'Obj 1 Linked', 'Obj 7 Linked']
DROP_COLS = ['unix'] + OTHER_TARGETS

# features and indicators
INTERVALS = (5, 15, 30, 60, 100)

# path to save feature names for future use
feature_save_path = f'./data/feature_names_{TARGET}.pkl'

# feature selection threshold
fe_threshold = 0.3

# data splitting
SPLIT_DATE = pd.to_datetime(pd.Series(['2021/09/30']), format='%Y/%m/%d')

# model parameters
model_parameters = {'num_leaves': 2**7,
                    'learning_rate': 0.024,
                    'bagging_fraction': 0.7,
                    'bagging_freq': 10,
                    'feature_fraction': 0.4,
                    'feature_fraction_bynode': 0.8,
                    'min_data_in_leaf': 500,
                    'min_gain_to_split': 0.1,
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
                    'metric': ['auc'],
                    'n_jobs': -1,
                    'force_col_wise': True,
                    # 'is_unbalance': True,
                    # 'tree_learner': 'voting',
                    'scale_pos_weight': 1.75
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
test_fe_names = ['close_VAR_100', 'close_VAR_60', 'low_VAR_100', 'LINEARREG_ANGLE_60', 'low_VAR_60', 'low_macd', 'NATR_15', 'ATRr_14', 'ADX_5', 'ATR_15', 'high_VAR_15', 'MOM_60', 'high_VAR_30', 'RSI_100', 'close_macd', 'DEMA_60', 'T3_15', 'TSF_100', 'TSF_60', 'DEMA_30', 'TSF_30', 'DEMA_100', 'DEMA_15', 'SAR', 'T3_5', 'TSF_15', 'T3_30', 'high', 'MIDPRICE_5', 'TYPPRICE', 'HMA_10', 'CKSPl_10_3_20', 'open', 'BBANDS_lower_100', 'BBANDS_lower_5', 'BBANDS_lower_30', 'BBANDS_lower_15', 'BBANDS_lower_60', 'BBANDS_upper_5', 'BBANDS_upper_15', 'BBANDS_upper_100', 'BBANDS_upper_30', 'BBANDS_upper_60', 'BBANDS_middle_15', 'BBANDS_middle_60', 'BBANDS_middle_100', 'BBANDS_middle_30', 'BBANDS_middle_5', 'DEMA_5', 'TSF_5', 'CUMPCTRET_1',
                 'low', 'ITS_9', 'HT_TRENDLINE', 'close', 'EMA_5', 'DCL_10_15', 'KAMA_10_2_30', 'KAMA_5', 'DCM_10_15', 'DCU_10_15', 'MIDPRICE_15', 'ALMA_10_6.0_0.85', 'KAMA_15', 'IKS_26', 'MIDPRICE_30', 'EMA_15', 'T3_60', 'KAMA_30', 'EMA_30', 'MIDPRICE_60', 'KAMA_100', 'MIDPRICE_100', 'KAMA_60', 'T3_100', 'EMA_60', 'CKSPs_10_3_20', 'EMA_100', 'ICS_26', 'ISA_9', 'ISB_26', 'open_macd', 'open_VAR_30', 'close_VAR_30', 'close_VAR_15', 'DMN_14', 'open_VAR_15', 'low_macdsignal', 'ATR_30', 'high_macd', 'MOM_100', 'low_VAR_15', 'low_VAR_30', 'close_macdsignal', 'open_macdsignal', 'NATR_5', 'high_macdsignal', 'ATR_60', 'ATR_100', 'AROON_up_60', 'Volume USDT', 'CUMLOGRET_1', 'ADX_30', 'AROON_up_100', 'ADX_15', 'OBV', 'MASSI_9_25', 'Volume BTC', 'AD']

# for blending models on test and validation data
ensemble = False

if ensemble:
    models_path_list = get_models_path(
        f'./{MODELS_FOLDER}/{TARGET}', [f'{TARGET}'])
mode = 'mean'
