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
                 'Objective 7', 'Objective 8', 'Objective 1', 'Obj 2 Linked', 'Obj 5 Linked',
                 'Obj 7 Linked', 'Obj 8 Linked', 'Obj 4 Linked', 'Obj 1 Linked', 'Obj 6 Linked']
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
ensemble = True

if ensemble:
    models_path_list = get_models_path(
        f'./{MODELS_FOLDER}/{TARGET}', [f'{TARGET}'])
mode = 'mean'

##################################
# production related configuration
##################################

# Feature names for each objective

prod_features = {'obj5': ['BEARP_13', 'AROON_up_30', 'ROCP_100', 'open_VAR_30', 'low_VAR_60', 'LINEARREG_ANGLE_100', 'close_VAR_30', 'high_STDDEV_100', 'LINEARREG_ANGLE_60', 'open_VAR_15', 'low_VAR_30', 'close_VAR_15', 'DEMA_30', 'TSF_30', 'TSF_60', 'DEMA_60', 'T3_15', 'TSF_100', 'close_STDDEV_100', 'DEMA_15', 'low', 'TSF_15', 'TSF_5', 'close', 'open', 'CUMPCTRET_1', 'BBANDS_lower_15', 'BBANDS_lower_60', 'BBANDS_lower_5', 'BBANDS_lower_30', 'BBANDS_lower_100', 'TYPPRICE', 'T3_5', 'DEMA_5', 'HMA_10', 'high', 'DEMA_100', 'DCL_10_15', 'BBANDS_middle_5', 'BBANDS_middle_15', 'BBANDS_middle_60', 'BBANDS_middle_30',
                          'BBANDS_middle_100', 'MIDPRICE_5', 'EMA_5', 'KAMA_5', 'ITS_9', 'BBANDS_upper_100', 'BBANDS_upper_5', 'BBANDS_upper_30', 'BBANDS_upper_15', 'BBANDS_upper_60', 'SAR', 'KAMA_10_2_30', 'DCM_10_15', 'MIDPRICE_15', 'KAMA_15', 'HT_TRENDLINE', 'CKSPl_10_3_20', 'T3_30', 'EMA_15', 'ALMA_10_6.0_0.85', 'DCU_10_15', 'IKS_26', 'MIDPRICE_30', 'KAMA_30', 'EMA_30', 'T3_60', 'MIDPRICE_60', 'KAMA_60', 'ICS_26', 'KAMA_100', 'CKSPs_10_3_20', 'MIDPRICE_100', 'EMA_60', 'EMA_100', 'T3_100', 'open_STDDEV_100', 'ISA_9', 'ISB_26', 'RSI_60', 'ATR_5', 'low_STDDEV_100', 'low_VAR_15', 'Volume USDT', 'MOM_30', 'ATRr_14', 'low_macd', 'ATR_15', 'MOM_60', 'RSI_100', 'close_macd', 'open_macd', 'MOM_100', 'low_macdsignal', 'ATR_30', 'high_macd', 'close_macdsignal', 'open_macdsignal', 'AROON_up_60', 'high_macdsignal', 'DMN_14', 'ATR_60', 'ADX_30', 'ATR_100', 'CUMLOGRET_1', 'AROON_up_100', 'ADX_15', 'MASSI_9_25', 'OBV', 'Volume BTC', 'AD'],
                 'obj6': ['open_VAR_30', 'close_VAR_100', 'close_VAR_30', 'open_STDDEV_5', 'open_STDDEV_30', 'ICS_26', 'TSF_30', 'DEMA_30', 'DEMA_15', 'DEMA_60', 'TSF_15', 'DEMA_100', 'TSF_60', 'TSF_100', 'low', 'T3_15', 'TSF_5', 'BBANDS_lower_5', 'BBANDS_lower_15', 'BBANDS_lower_100', 'BBANDS_lower_30', 'BBANDS_lower_60', 'HMA_10', 'T3_5', 'DEMA_5', 'DCL_10_15', 'close', 'CUMPCTRET_1', 'open', 'TYPPRICE', 'high', 'BBANDS_middle_15', 'BBANDS_middle_30',
                          'BBANDS_middle_5', 'BBANDS_middle_60', 'BBANDS_middle_100', 'EMA_5', 'MIDPRICE_5', 'KAMA_5', 'ITS_9', 'KAMA_10_2_30', 'DCM_10_15', 'BBANDS_upper_100', 'BBANDS_upper_5', 'BBANDS_upper_15', 'BBANDS_upper_60', 'BBANDS_upper_30', 'MIDPRICE_15', 'EMA_15', 'T3_30', 'HT_TRENDLINE', 'SAR', 'ALMA_10_6.0_0.85', 'KAMA_15', 'CKSPl_10_3_20', 'T3_60', 'IKS_26', 'MIDPRICE_30', 'DCU_10_15', 'KAMA_30', 'EMA_30', 'MIDPRICE_60', 'KAMA_60', 'CKSPs_10_3_20', 'T3_100', 'EMA_60', 'KAMA_100', 'MIDPRICE_100', 'EMA_100', 'ISA_9', 'ISB_26', 'low_STDDEV_30', 'low_VAR_30', 'open_VAR_60', 'TRUERANGE_1', 'close_VAR_60', 'low_VAR_60', 'high_STDDEV_60', 'ADX_30', 'AROON_up_60', 'close_STDDEV_60', 'BEARP_13', 'open_STDDEV_60', 'low_STDDEV_60', 'RSI_60', 'MOM_30', 'MOM_60', 'low_macd', 'low_macdsignal', 'ADX_15', 'ATR_5', 'open_macdsignal', 'open_macd', 'close_macd', 'close_macdsignal', 'high_STDDEV_100', 'high_macdsignal', 'high_macd', 'close_STDDEV_100', 'open_STDDEV_100', 'low_STDDEV_100', 'MOM_100', 'ATRr_14', 'RSI_100', 'ATR_15', 'AROON_up_100', 'ATR_30', 'DMN_14', 'ATR_60', 'CUMLOGRET_1', 'ATR_100', 'MASSI_9_25', 'OBV', 'Volume BTC', 'AD'],
                 'obj7': ['MIDPRICE_60', 'KAMA_100', 'EMA_60', 'MIDPRICE_100', 'ISA_9', 'T3_100', 'EMA_100', 'ISB_26', 'MOM_5', 'HT_DCPHASE', 'AROON_down_15', 'LINEARREG_ANGLE_15', 'high_VAR_30', 'high_VAR_60', 'open_VAR_15', 'AROONOSC_14', 'NATR_30', 'close_VAR_15', 'high_VAR_100', 'ADX_15', 'low_VAR_15', 'ULTOSC_30', 'WILLR_30', 'open_VAR_60', 'AROON_down_30', 'close_VAR_60', 'open_VAR_30', 'open_VAR_100', 'CCI_30', 'close_VAR_100', 'close_VAR_30', 'low_VAR_60', 'NATR_15', 'low_VAR_100', 'ROCP_15', 'low_VAR_30', 'ULTOSC_60', 'DMP_14', 'COPC_11_14_10', 'ADX_30', 'MFI_14', 'RSI_15', 'high_STDDEV_15', 'CCI_60', 'Volume USDT', 'AROON_up_30', 'NATR_5', 'WILLR_60', 'high_STDDEV_5', 'ULTOSC_100', 'high_STDDEV_30', 'close_STDDEV_15', 'open_STDDEV_15', 'open_STDDEV_5', 'close_STDDEV_5',
                          'low_STDDEV_5', 'AROON_up_60', 'LINEARREG_ANGLE_30', 'TSI_13_25_13', 'low_STDDEV_15', 'ROCP_30', 'CCI_100', 'LINEARREG_ANGLE_60', 'TRUERANGE_1', 'close_STDDEV_30', 'open_STDDEV_30', 'high_STDDEV_60', 'WILLR_100', 'TSIs_13_25_13', 'AROON_up_100', 'low_STDDEV_30', 'MOM_15', 'RSI_30', 'close_STDDEV_60', 'open_STDDEV_60', 'high_STDDEV_100', 'ROCP_60', 'low_STDDEV_60', 'ATR_5', 'LINEARREG_ANGLE_100', 'close_STDDEV_100', 'open_STDDEV_100', 'BEARP_13', 'CUMLOGRET_1', 'low_STDDEV_100', 'ATRr_14', 'ATR_15', 'CHOP_14_1_100', 'MOM_30', 'ROCP_100', 'ATR_30', 'MASSI_9_25', 'low_macd', 'RSI_60', 'low_macdsignal', 'open_macd', 'close_macd', 'ATR_60', 'DMN_14', 'open_macdsignal', 'close_macdsignal', 'high_macd', 'ATR_100', 'high_macdsignal', 'MOM_60', 'RSI_100', 'OBV', 'MOM_100', 'Volume BTC', 'AD'],
                 'obj8': ['MIDPRICE_30', 'ULTOSC_100', 'KAMA_30', 'CKSPl_10_3_20', 'CKSPs_10_3_20', 'ULTOSC_60', 'EMA_30', 'MIDPRICE_60', 'BBANDS_upper_15', 'BBANDS_upper_5', 'BBANDS_upper_100', 'BBANDS_upper_30', 'BBANDS_upper_60', 'KAMA_60', 'KAMA_100', 'MIDPRICE_100', 'EMA_60', 'MOM_15', 'ISA_9', 'DCU_10_15', 'T3_100', 'ISB_26', 'EMA_100', 'tradecount', 'AROON_up_60', 'RSI_15', 'MFI_14', 'CCI_60', 'open_VAR_15', 'close_VAR_15', 'low_VAR_15', 'HT_DCPERIOD', 'WILLR_60', 'AROON_up_100', 'LINEARREG_ANGLE_60', 'high_VAR_30', 'TSI_13_25_13', 'ROCP_60', 'high_VAR_60', 'ADX_100', 'TSIs_13_25_13', 'open_VAR_30', 'MOM_30', 'close_VAR_30', 'open_VAR_60', 'close_VAR_60', 'low_VAR_30', 'low_VAR_60', 'CCI_100',
                          'low_macd', 'WILLR_100', 'low_macdsignal', 'RSI_30', 'open_macd', 'close_macd', 'MASSI_9_25', 'LINEARREG_ANGLE_100', 'open_macdsignal', 'close_macdsignal', 'high_macd', 'high_macdsignal', 'ROCP_100', 'high_VAR_100', 'MOM_60', 'open_VAR_100', 'close_VAR_100', 'low_VAR_100', 'BEARP_13', 'MOM_100', 'RSI_60', 'high_STDDEV_5', 'low_STDDEV_5', 'TRUERANGE_1', 'high_STDDEV_15', 'close_STDDEV_5', 'open_STDDEV_5', 'DMP_14', 'CUMLOGRET_1', 'close_STDDEV_15', 'open_STDDEV_15', 'low_STDDEV_15', 'RSI_100', 'high_STDDEV_30', 'close_STDDEV_30', 'open_STDDEV_30', 'low_STDDEV_30', 'ATR_5', 'high_STDDEV_60', 'CHOP_14_1_100', 'close_STDDEV_60', 'open_STDDEV_60', 'low_STDDEV_60', 'ATRr_14', 'ATR_15', 'high_STDDEV_100', 'DMN_14', 'ATR_30', 'close_STDDEV_100', 'open_STDDEV_100', 'low_STDDEV_100', 'Volume BTC', 'ATR_60', 'ATR_100', 'OBV', 'AD']
                 }


# objectives to run on
objectives_to_run = {'obj5': True, 'obj6': True, 'obj7': True, 'obj8': True}

# models dir
PROD_MODELS_DIR = './models'
