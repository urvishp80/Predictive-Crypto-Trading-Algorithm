import talib
import pandas as pd
import pandas_ta as ta

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
    When there are options in indicator API, close prices are used for computation. For volume Volume BTC is used.
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
        indicators['DEMA_{}'.format(i)] = talib.DEMA(data['close'], timeperiod=i)
        indicators['EMA_{}'.format(i)] = talib.EMA(data['close'], timeperiod=i)
        indicators['KAMA_{}'.format(i)] = talib.KAMA(data['close'], timeperiod=i)
        indicators['MIDPRICE_{}'.format(i)] = talib.MIDPRICE(data['high'], data['low'], timeperiod=i)
        indicators['T3_{}'.format(i)] = talib.T3(data['close'], timeperiod=i)
        indicators['ADX_{}'.format(i)] = talib.ADX(data['high'], data['low'], data['close'], timeperiod=i)
        indicators['AROON_down_{}'.format(i)], indicators['AROON_up_{}'.format(i)] = talib.AROON(
            data['high'], data['low'], timeperiod=i)
        indicators['CCI_{}'.format(i)] = talib.CCI(data['high'], data['low'], data['close'], timeperiod=i)
        indicators['MOM_{}'.format(i)] = talib.MOM(data['close'], timeperiod=i)
        indicators['ROCP_{}'.format(i)] = talib.ROCP(data['close'], timeperiod=i)
        indicators['RSI_{}'.format(i)] = talib.RSI(data['close'], timeperiod=i)
        indicators['ULTOSC_{}'.format(i)] = talib.ULTOSC(data['high'], data['low'], data['close'],
                                                         timeperiod1=i, timeperiod2=2 * i, timeperiod3=4 * i)
        indicators['WILLR_{}'.format(i)] = talib.WILLR(data['high'], data['low'], data['close'], timeperiod=i)
        indicators['NATR_{}'.format(i)] = talib.NATR(data['high'], data['low'], data['close'], timeperiod=i)
        indicators['TSF_{}'.format(i)] = talib.TSF(data['close'], timeperiod=i)
        indicators['BBANDS_upper_{}'.format(i)], indicators['BBANDS_middle_{}'.format(i)], indicators['BBANDS_lower_{}'.format(i)] = talib.BBANDS(
            data['close'], timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
        indicators['ATR_{}'.format(i)] = talib.ATR(data['high'], data['low'], data['close'], timeperiod=i)
        indicators['NATR_{}'.format(i)] = talib.NATR(data['high'], data['low'], data['close'], timeperiod=i)
        indicators['BETA_{}'.format(i)] = talib.BETA(data['high'], data['low'], timeperiod=i)
        indicators['CORREL_{}'.format(i)] = talib.CORREL(data['high'], data['low'], timeperiod=1)
        indicators['LINEARREG_ANGLE_{}'.format(i)] = talib.LINEARREG_ANGLE(data['close'], timeperiod=i)
        indicators['close_STDDEV_{}'.format(i)] = talib.STDDEV(data['close'], timeperiod=i)
        indicators['high_STDDEV_{}'.format(i)] = talib.STDDEV(data['high'], timeperiod=i)
        indicators['low_STDDEV_{}'.format(i)] = talib.STDDEV(data['low'], timeperiod=i)
        indicators['open_STDDEV_{}'.format(i)] = talib.STDDEV(data['open'], timeperiod=i)
        indicators['close_VAR_{}'.format(i)] = talib.VAR(data['close'], timeperiod=i, nbdev=1)
        indicators['open_VAR_{}'.format(i)] = talib.VAR(data['open'], timeperiod=i, nbdev=1)
        indicators['high_VAR_{}'.format(i)] = talib.VAR(data['high'], timeperiod=i, nbdev=1)
        indicators['low_VAR_{}'.format(i)] = talib.VAR(data['low'], timeperiod=i, nbdev=1)
    # Indicators that do not depend on time periods.
    indicators['close_macd'], indicators['close_macdsignal'], indicators['close_macdhist'] = talib.MACD(data['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    indicators['open_macd'], indicators['open_macdsignal'], indicators['open_macdhist'] = talib.MACD(data['open'], fastperiod=12, slowperiod=26, signalperiod=9)
    indicators['high_macd'], indicators['high_macdsignal'], indicators['high_macdhist'] = talib.MACD(data['high'], fastperiod=12, slowperiod=26, signalperiod=9)
    indicators['low_macd'], indicators['low_macdsignal'], indicators['low_macdhist'] = talib.MACD(data['low'], fastperiod=12, slowperiod=26, signalperiod=9)
    indicators['SAR'] = talib.SAR(data['high'], data['low'])
    indicators['AD'] = talib.AD(data['high'], data['low'], data['close'], data['Volume BTC'])
    indicators['OBV'] = talib.OBV(data['close'], data['Volume BTC'])
    indicators['HT_TRENDLINE'] = talib.HT_TRENDLINE(data['close'])
    indicators['HT_TRENDMODE'] = talib.HT_TRENDMODE(data['close'])
    indicators['HT_DCPERIOD'] = talib.HT_DCPERIOD(data['close'])
    indicators['HT_DCPHASE'] = talib.HT_DCPHASE(data['close'])
    indicators['TYPPRICE'] = talib.TYPPRICE(data['high'], data['low'], data['close'])
    return pd.DataFrame(indicators)


def get_additional_indicators(data):
    # additional_indicators = []
    # prev_col = data.columns
    log_return = data.ta.log_return(cumulative=True)
    percent_return = data.ta.percent_return(cumulative=True)
    cksp = data.ta.cksp()
    alma = data.ta.alma()
    adosc = data.ta.adosc()
    aroon = data.ta.aroon()
    atr = data.ta.atr()
    chop = data.ta.chop()
    tsi = data.ta.tsi()
    ichimoku = data.ta.ichimoku()
    donchian = data.ta.donchian(lower_length=10, upper_length=15)
    hma = data.ta.hma()
    kama = data.ta.kama()
    stochrsi = data.ta.stochrsi()
    coppock = data.ta.coppock()
    dm = data.ta.dm()
    massi = data.ta.massi()
    mfi = data.ta.mfi()
    eri = data.ta.eri()
    true_range = data.ta.true_range()
    all_indicators = [log_return, percent_return, cksp, alma, adosc, aroon, atr, chop, tsi,
                      ichimoku, donchian, hma, kama, stochrsi, coppock, dm, massi, mfi, eri,
                      true_range]
    df_and_series = []
    for df in all_indicators:
        if type(df) == tuple:
            for j in df:
                df_and_series.append(j)
        else:
            df_and_series.append(df)
    # [df for df in all_indicators if type(df) != tuple else j for j in df for df in all_indicators]
    indicators = pd.concat(df_and_series, axis=1)
    return indicators


def get_price_patterns(data):
    """
    Detects common price patterns using TA-lib, e.g. Two Crows, Belt-hold, Hanging Man etc.

    :param data: DataFrame with ticks data. Could be with or without embed transactions data.
    :return: DataFrame with pattern "likelihoods" on -200 - 200 scale.
    """
    patterns = {name: getattr(talib, name)(data['open'], data['high'], data['low'], data['close'])
                for name in talib.get_function_groups()['Pattern Recognition']}
    return pd.DataFrame(patterns)
