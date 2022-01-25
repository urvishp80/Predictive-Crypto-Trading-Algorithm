import talib
import pandas as pd


def enrich_data(data):
    """
    Enhances data frame with information on indicators and price patterns. Indicators and patterns are align so
    that they represent details for previous minute.

    :param data: DataFrame
    :return: DataFrame
    """
    # We specifically do shifting here so that all additional data represents information about past history.
    return pd.concat((data, get_indicators(data).shift(), get_price_patterns(data).shift()), axis=1)


def get_indicators(data, intervals=(5, 10, 20, 50, 100)):
    """
    Computes technical indicators given ticks data.
    These indicators are computed with fixed parameters, i.e. intervals argument shouldn't affect them:
    * Parabolic SAR
    * Chaikin A/D Line
    * On Balance Volume
    * Hilbert Transform - Instantaneous Trendline
    * Hilbert Transform - Trend vs Cycle Mode
    * Hilbert Transform - Dominant Cycle Period
    * Hilbert Transform - Dominant Cycle Phase
    * Typical Price
    These indicators are computed for each of periods given in intervals argument:
    * Exponential Moving Average
    * Double Exponential Moving Average
    * Kaufman Adaptive Moving Average
    * Midpoint Price over period
    * Triple Exponential Moving Average
    * Average Directional Movement Index
    * Aroon
    * Commodity Channel Index
    * Momentum
    * Rate of change Percentage: (price-prevPrice)/prevPrice
    * Relative Strength Index
    * Ultimate Oscillator (based on T, 2T, 3T periods)
    * Williams' %R
    * Normalized Average True Range
    * Time Series Forecast (linear regression)
    * Bollinger Bands
    For more details see TA-lib documentation.
    When there are options in indicator API, Close prices are used for computation. For volume TickVol is used.
    Note that some of the indicators are not stable and could output unexpected results if fed with NaNs or long series.

    :param data DataFrame with ticks data. Could be with or without embed data transactions.
    :param intervals Iterable with time periods to use for computation.
                     Periods should be in the same sample units as ticks data, i.e. in minutes.
                     Default values: 5, 10, 20, 50 and 100 minutes.
    :return DataFrame with indicators. For interval-based indicators, interval is mentioned in column name, e.g. CCI_5.
    """
    indicators = {}
    # Time period based indicators.
    for i in intervals:
        indicators['DEMA_{}'.format(i)] = talib.DEMA(data['Close'], timeperiod=i)
        indicators['EMA_{}'.format(i)] = talib.EMA(data['Close'], timeperiod=i)
        indicators['KAMA_{}'.format(i)] = talib.KAMA(data['Close'], timeperiod=i)
        indicators['MIDPRICE_{}'.format(i)] = talib.MIDPRICE(data['High'], data['Low'], timeperiod=i)
        indicators['T3_{}'.format(i)] = talib.T3(data['Close'], timeperiod=i)
        indicators['ADX_{}'.format(i)] = talib.ADX(data['High'], data['Low'], data['Close'], timeperiod=i)
        indicators['AROON_down_{}'.format(i)], indicators['AROON_up_{}'.format(i)] = talib.AROON(
            data['High'], data['Low'], timeperiod=i)
        indicators['CCI_{}'.format(i)] = talib.CCI(data['High'], data['Low'], data['Close'], timeperiod=i)
        indicators['MOM_{}'.format(i)] = talib.MOM(data['Close'], timeperiod=i)
        indicators['ROCP_{}'.format(i)] = talib.ROCP(data['Close'], timeperiod=i)
        indicators['RSI_{}'.format(i)] = talib.RSI(data['Close'], timeperiod=i)
        indicators['ULTOSC_{}'.format(i)] = talib.ULTOSC(data['High'], data['Low'], data['Close'],
                                                         timeperiod1=i, timeperiod2=2 * i, timeperiod3=4 * i)
        indicators['WILLR_{}'.format(i)] = talib.WILLR(data['High'], data['Low'], data['Close'], timeperiod=i)
        indicators['NATR_{}'.format(i)] = talib.NATR(data['High'], data['Low'], data['Close'], timeperiod=i)
        indicators['TSF_{}'.format(i)] = talib.TSF(data['Close'], timeperiod=i)
        indicators['BBANDS_upper_{}'.format(i)], indicators['BBANDS_middle_{}'.format(i)], indicators['BBANDS_lower_{}'.format(i)] = talib.BBANDS(
            data['Close'], timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
        indicators['ATR_{}'.format(i)] = talib.ATR(data['High'], data['Low'], data['Close'], timeperiod=i)
        indicators['NATR_{}'.format(i)] = talib.NATR(data['High'], data['Low'], data['Close'], timeperiod=i)
        indicators['BETA_{}'.format(i)] = talib.BETA(data['High'], data['Low'], timeperiod=i)
        indicators['CORREL_{}'.format(i)] = talib.CORREL(data['High'], data['Low'], timeperiod=1)
        indicators['LINEARREG_ANGLE_{}'.format(i)] = talib.LINEARREG_ANGLE(data['Close'], timeperiod=i)
        indicators['CLOSE_STDDEV_{}'.format(i)] = talib.STDDEV(data['Close'], timeperiod=i)
        indicators['HIGH_STDDEV_{}'.format(i)] = talib.STDDEV(data['High'], timeperiod=i)
        indicators['LOW_STDDEV_{}'.format(i)] = talib.STDDEV(data['Low'], timeperiod=i)
        indicators['OPEN_STDDEV_{}'.format(i)] = talib.STDDEV(data['Open'], timeperiod=i)
        indicators['CLOSE_VAR_{}'.format(i)] = talib.VAR(data['Close'], timeperiod=i, nbdev=1)
        indicators['OPEN_VAR_{}'.format(i)] = talib.VAR(data['Open'], timeperiod=i, nbdev=1)
        indicators['HIGH_VAR_{}'.format(i)] = talib.VAR(data['High'], timeperiod=i, nbdev=1)
        indicators['LOW_VAR_{}'.format(i)] = talib.VAR(data['Low'], timeperiod=i, nbdev=1)
    # Indicators that do not depend on time periods.
    indicators['Close_macd'], indicators['Close_macdsignal'], indicators['Close_macdhist'] = talib.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    indicators['Open_macd'], indicators['Open_macdsignal'], indicators['Open_macdhist'] = talib.MACD(data['Open'], fastperiod=12, slowperiod=26, signalperiod=9)
    indicators['High_macd'], indicators['High_macdsignal'], indicators['High_macdhist'] = talib.MACD(data['High'], fastperiod=12, slowperiod=26, signalperiod=9)
    indicators['Low_macd'], indicators['Low_macdsignal'], indicators['Low_macdhist'] = talib.MACD(data['Low'], fastperiod=12, slowperiod=26, signalperiod=9)
    indicators['SAR'] = talib.SAR(data['High'], data['Low'])
    indicators['AD'] = talib.AD(data['High'], data['Low'], data['Close'], data['TickVol'])
    indicators['OBV'] = talib.OBV(data['Close'], data['TickVol'])
    indicators['HT_TRENDLINE'] = talib.HT_TRENDLINE(data['Close'])
    indicators['HT_TRENDMODE'] = talib.HT_TRENDMODE(data['Close'])
    indicators['HT_DCPERIOD'] = talib.HT_DCPERIOD(data['Close'])
    indicators['HT_DCPHASE'] = talib.HT_DCPHASE(data['Close'])
    indicators['TYPPRICE'] = talib.TYPPRICE(data['High'], data['Low'], data['Close'])
    return pd.DataFrame(indicators)


def get_price_patterns(data):
    """
    Detects common price patterns using TA-lib, e.g. Two Crows, Belt-hold, Hanging Man etc.

    :param data: DataFrame with ticks data. Could be with or without embed transactions data.
    :return: DataFrame with pattern "likelihoods" on -200 - 200 scale.
    """
    patterns = {name: getattr(talib, name)(data['Open'], data['High'], data['Low'], data['Close'])
                for name in talib.get_function_groups()['Pattern Recognition']}
    return pd.DataFrame(patterns)