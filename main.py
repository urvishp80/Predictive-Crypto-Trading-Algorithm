import pandas as pd

import config

from src.dataset import get_data, get_features_targets
from src.indicators import get_indicators, get_price_patterns

if __name__ == '__main__':

    df = get_data(config.DATA_PATH, drop_col=config.DROP_COLS)
    df_indicators = get_indicators(df, intervals=config.INTERVALS)
    df_price_pattern = get_price_patterns(df)

    features, targets = get_features_targets(df, [config.TARGET])
    print(features.shape, targets.shape, df_price_pattern.shape, df_indicators.shape)
