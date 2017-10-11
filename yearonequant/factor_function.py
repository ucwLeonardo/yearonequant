from yearonequant.factor import *

# # 大连商品交易所
# DCE = ['C99', 'CS99', 'A99', 'M99', 'Y99', 'P99', 'JD99', 'L99', 'V99', 'PP99', 'I99', 'J99', 'JM99']
# # 郑州商品交易所
# CZCE = ['CF99', 'SR99', 'MA99', 'ZC99']
# # 上海商品交易所
# SHFE = ['CU99', 'AL99', 'ZN99', 'RU99', 'AU99', 'AG99', 'RB99']
# # 中国金融期货交易所
# CFFEX = ['IH99', 'IC99', 'IF99', 'TF99', 'T99']
#
# order_book_ids = DCE + CZCE + SHFE + CFFEX

# start_date = '2010-01-01'
# end_date = '2017-09-30'

# panel_data = get_price(order_book_ids, start_date=start_date, end_date=end_date,
#                     fields=['open', 'high', 'low', 'volume'], adjust_type='post')
# close_df = get_price(order_book_ids, start_date=start_date, end_date=end_date,
#                     fields='close', adjust_type='post')
# returns_df = close_df / close_df.shift(1) - 1
# panel_data['close'] = close_df
# panel_data['returns'] = returns_df

def signF(df):
    return np.sign(df)

def absF(df):
    return abs(df)

def logF(df):
    return df.apply(np.log)

def rankF(df):
    return pd.DataFrame.rank(df, axis=1)

def delayF(df, past_days):
    return df.shift(past_days)

def correlationF(df1, df2, past_days):
    return df1.rolling(window=past_days).corr(other=df2)

def covarianceF(df1, df2, past_days):
    return df1.rolling(window=past_days).cov(other=df2)

# def scale(df, a):

def deltaF(df, past_days):
    return df - df.shift(past_days)

def signedpower(df, a):
    return df.pow(a)

def decay_linearF(df, past_days):
    wts = np.array(range(1, past_days+1))[::-1]
    wts = wts / wts.sum()

    # applied function, weight time series
    def f(w):
        def g(x):
            return (w * x).sum()
        return g

    factor_df = df.rolling(window=past_days).apply(f(wts))
    factor_df = factor_df.replace([np.inf, -np.inf], np.nan)
    return factor_df

def ts_minF(df, past_days):
    return df.rolling(window=past_days).apply(np.min)

def ts_maxF(df, past_days):
    return df.rolling(window=past_days).apply(np.max)

def ts_argminF(df, past_days):
    rolling_argmin = df.rolling(window=past_days).apply(np.argmin)
    factor_df = rolling_argmin
    return factor_df

def ts_argmaxF(df, past_days):
    rolling_argmax = df.rolling(window=past_days).apply(np.argmax)
    factor_df = rolling_argmax
    return factor_df

def ts_rankF(df, past_days):
    rank = lambda x: pd.Series(x).rank().iloc[-1]
    return df.rolling(window=past_days).apply(rank)

def minF(df, past_days):
    return df.rolling(window=past_days).apply(np.min)

def maxF(df, past_days):
    return df.rolling(window=past_days).apply(np.max)

def sumF(df, past_days):
    return df.rolling(window=past_days).sum()

def productF(df, past_days):
    return df.rolling(window=past_days).apply(np.prod)

def stddevF(df, past_days):
    return df.rolling(window=past_days).apply(np.std)

def factor002(panel):

    print('calculating factor...')
    df1 = rankF(deltaF(logF(panel.volume), 2))
    df2 = rankF((panel.close - panel.open) / panel.open)
    factor_df = -1 * correlationF(df1, df2, 6).replace([np.inf, -np.inf], np.nan)
    print('done')

    return factor_df

def factor003(panel):

    print('calculating factor...')
    df1 = rankF(panel.open)
    df2 = rankF(panel.volume)
    factor_df = -1 * correlationF(df1, df2, 10).replace([np.inf, -np.inf], np.nan)
    print('done')

    return factor_df

def factor004(panel):

    print('calculating factor...')
    factor_df = -1 * ts_rankF(rankF(panel.low), 9).replace([np.inf, -np.inf], np.nan)
    print('done')

    return factor_df

def factor006(panel):

    print('calculating factor...')
    factor_df = -1 * correlationF(panel.open, panel.volume, 10).replace([np.inf, -np.inf], np.nan)
    print('done')

    return factor_df

def factor008(panel):

    panel.returns = panel.close / panel.close.shift(1) - 1
    print('calculating factor...')
    part1 = sumF(panel.open, 5) * sumF(panel.returns, 5)
    part2 = delayF(sumF(panel.open, 5) * sumF(panel.returns, 5), 10)
    factor_df = -1 * rankF(part1 - part2).replace([np.inf, -np.inf], np.nan)
    print('done')

    return factor_df

def factor012(panel):

    print('calculating factor...')
    factor_df = signF(deltaF(panel.volume, 1)) * (-1 * deltaF(panel.close, 1))
    factor_df = factor_df.replace([np.inf, -np.inf], np.nan)
    print('done')

    return factor_df

def factor013(panel):

    print('calculating factor...')
    factor_df = -1 * rankF(covarianceF(rankF(panel.close), rankF(panel.volume), 5))
    factor_df = factor_df.replace([np.inf, -np.inf], np.nan)
    print('done')

    return factor_df

def factor014(panel):

    panel.returns = panel.close / panel.close.shift(1) - 1
    print('calculating factor...')
    part1 = -1 * rankF(deltaF(panel.returns, 3))
    part2 = correlationF(panel.open, panel.volume, 10)
    factor_df = (part1 * part2).replace([np.inf, -np.inf], np.nan)
    print('done')

    return factor_df

def factor015(panel):

    print('calculating factor...')
    factor_df = -1 * sumF(rankF(correlationF(rankF(panel.high), rankF(panel.volume), 3)), 3)
    factor_df = factor_df.replace([np.inf, -np.inf], np.nan)
    print('done')

    return factor_df

def factor016(panel):

    print('calculating factor...')
    factor_df = -1 * rankF(covarianceF(rankF(panel.high), rankF(panel.volume), 5))
    factor_df = factor_df.replace([np.inf, -np.inf], np.nan)
    print('done')

    return factor_df

def factor018(panel):

    print('calculating factor...')
    part1 = -1 * rankF(stddevF(absF(panel.close - panel.open), 5))
    part2 = correlationF(panel.close, panel.open, 10)
    factor_df = part1 + part2
    factor_df = factor_df.replace([np.inf, -np.inf], np.nan)
    print('done')

    return factor_df

def factor019(panel):

    panel.returns = panel.close / panel.close.shift(1) - 1
    print('calculating factor...')
    part1 = -1 * signF((panel.close - delayF(panel.close, 7)) + deltaF(panel.close, 7))
    part2 = 1 + sumF(panel.returns, 250)
    factor_df = (part1 * part2).replace([np.inf, -np.inf], np.nan)
    print('done')

    return factor_df

def factor020(panel):

    print('calculating factor...')
    part1 = -1 * rankF(panel.open - delayF(panel.high, 1))
    part2 = rankF(panel.open - delayF(panel.close, 1))
    part3 = rankF(panel.open - delayF(panel.low, 1))
    factor_df = (part1 * part2 * part3).replace([np.inf, -np.inf], np.nan)
    print('done')

    return factor_df

def factor022(panel):

    print('calculating factor...')
    factor_df = -1 * deltaF(correlationF(panel.high, panel.volume, 5), 5) * rankF(stddevF(panel.close, 20))
    factor_df = factor_df.replace([np.inf, -np.inf], np.nan)
    print('done')

    return factor_df

def factor025(panel):

    print('calculating factor...')
    factor_df = -1 * deltaF(correlationF(panel.high, panel.volume, 5), 5) * rankF(stddevF(panel.close, 20))
    factor_df = factor_df.replace([np.inf, -np.inf], np.nan)
    print('done')

    return factor_df

def factor026(panel):

    print('calculating factor...')
    factor_df = -1 * ts_maxF(correlationF(ts_rankF(panel.volume, 5), ts_rankF(panel.high, 5), 5), 3)
    factor_df = factor_df.replace([np.inf, -np.inf], np.nan)
    print('done')

    return factor_df

def factor030(panel):

    panel.returns = panel.close / panel.close.shift(1) - 1

    print('calculating factor...')
    inner1 = signF(panel.close - delayF(panel.close, 1)) + signF(delayF(panel.close, 1) - delayF(panel.close, 2)) \
            + signF(delayF(panel.close, 2) - delayF(panel.close, 3))
    part1 = (1 - rankF(inner1)) * sumF(panel.volume, 5)
    factor_df = (part1 / sumF(panel.volume, 20)).replace([np.inf, -np.inf], np.nan)
    print('done')

    return factor_df

def factor033(panel):

    print('calculating factor...')
    factor_df = rankF(-1 * (1 - panel.open / panel.close))
    factor_df = factor_df.replace([np.inf, -np.inf], np.nan)
    print('done')

    return factor_df

def factor034(panel):

    panel.returns = panel.close / panel.close.shift(1) - 1

    print('calculating factor...')
    part1 = 1 - rankF(stddevF(panel.returns, 2) / stddevF(panel.returns, 5))
    part2 = 1 - rankF(deltaF(panel.close, 1))
    factor_df = (part1 + part2).replace([np.inf, -np.inf], np.nan)
    print('done')

    return factor_df

def factor035(panel):

    panel.returns = panel.close / panel.close.shift(1) - 1

    print('calculating factor...')
    part1 = ts_rankF(panel.volume, 32) * (1 - ts_rankF(panel.close + panel.high - panel.low, 16))
    part2 = 1 - ts_rankF(panel.returns, 32)
    factor_df = (part1 * part2).replace([np.inf, -np.inf], np.nan)
    print('done')

    return factor_df

def factor037(panel):

    print('calculating factor...')
    part1 = rankF(correlationF(delayF(panel.open - panel.close, 1), panel.close, 200))
    part2 = rankF(panel.open - panel.close)
    factor_df = (part1 + part2).replace([np.inf, -np.inf], np.nan)
    print('done')

    return factor_df

def factor038(panel):

    print('calculating factor...')
    factor_df = -1 * rankF(ts_rankF(panel.close, 10)) * rankF(panel.close / panel.open)
    factor_df = factor_df.replace([np.inf, -np.inf], np.nan)
    print('done')

    return factor_df

def factor040(panel):

    print('calculating factor...')
    factor_df = -1 * rankF(stddevF(panel.high, 10)) * correlationF(panel.high, panel.volume, 10)
    factor_df = factor_df.replace([np.inf, -np.inf], np.nan)
    print('done')

    return factor_df

def factor041(panel):

    print('calculating factor...')
    vwap = ( 2*(panel.open + panel.close) + panel.high + panel.low ) / 6
    factor_df = (panel.high * panel.low) ** 0.5 - vwap
    factor_df = factor_df.replace([np.inf, -np.inf], np.nan)
    print('done')

    return factor_df


def factor042(panel):

    print('calculating factor...')
    vwap = ( 2 * (panel.open + panel.close) + panel.high + panel.low ) / 6
    factor_df = rankF(vwap - panel.close) / rankF(vwap + panel.close)
    factor_df = factor_df.replace([np.inf, -np.inf], np.nan)
    print('done')

    return factor_df


def factor044(panel):

    print('calculating factor...')
    factor_df = -1 * correlationF(panel.high, rankF(panel.volume), 5)
    factor_df = factor_df.replace([np.inf, -np.inf], np.nan)
    print('done')

    return factor_df


def factor045(panel):

    print('calculating factor...')
    part1 = rankF( sumF( delayF(panel.close, 5), 20 ) / 20 )
    part2 = correlationF(panel.close, panel.volume, 2)
    part3 = rankF( correlationF( sumF(panel.close, 5), sumF(panel.close, 20), 2 ) )
    factor_df = -1 * part1 * part2 * part3
    factor_df = factor_df.replace([np.inf, -np.inf], np.nan)
    print('done')

    return factor_df


def factor050(panel):

    h = panel.high
    l = panel.low
    o = panel.open
    c = panel.close
    v = panel.volume

    print('calculating factor...')
    vwap = (2 * (o + c) + h + l) / 6
    part1 = correlationF(rankF(v), rankF(vwap), 5)
    part2 = rankF(part1)
    factor_df = -1 * ts_maxF( part2, 5 )
    factor_df = factor_df.replace([np.inf, -np.inf], np.nan)
    print('done')

    return factor_df


def factor052(panel):

    l = panel.low
    r = panel.returns
    v = panel.volume

    print('calculating factor...')
    part1 = -1 * ts_minF(l, 5) + delayF( ts_minF(l, 5), 5 )
    part2 = rankF( (sumF(r, 240) - sumF(r, 20)) / 220 )
    part3 = ts_rankF(v, 5)
    factor_df = part1 * part2 * part3
    factor_df = factor_df.replace([np.inf, -np.inf], np.nan)
    print('done')

    return factor_df


def factor053(panel):

    c = panel.close
    l = panel.low
    h = panel.high

    print('calculating factor...')
    factor_df = -1 * deltaF( ((c-l) - (h-c)) / (c - l), 9 )
    factor_df = factor_df.replace([np.inf, -np.inf], np.nan)
    print('done')

    return factor_df


def factor054(panel):

    c = panel.close
    l = panel.low
    h = panel.high
    o = panel.open

    print('calculating factor...')
    factor_df = -1 * (l - c) * (o ** 5) / ( (l - h) * (c ** 5) )
    factor_df = factor_df.replace([np.inf, -np.inf], np.nan)
    print('done')

    return factor_df


def factor055(panel):

    c = panel.close
    l = panel.low
    h = panel.high
    v = panel.volume

    print('calculating factor...')
    part1 = (c - ts_minF(l, 12)) / ts_maxF(h, 12) - ts_minF(l, 12)
    part2 = rankF(part1)
    factor_df = -1 * correlationF(part2, rankF(v), 6)
    factor_df = factor_df.replace([np.inf, -np.inf], np.nan)
    print('done')

    return factor_df


def factor057(panel):

    c = panel.close
    l = panel.low
    h = panel.high
    o = panel.open
    v = panel.volume

    print('calculating factor...')
    vwap = (2 * (c + o) + h + l) / 6
    part1 = c - vwap
    part2 = decay_linearF( rankF(ts_argmaxF(c, 30)), 2 )
    factor_df = 0 - (1 * part1 / part2)
    factor_df = factor_df.replace([np.inf, -np.inf], np.nan)
    print('done')

    return factor_df


def factor066(panel):

    c = panel.close
    l = panel.low
    h = panel.high
    o = panel.open

    print('calculating factor...')
    vwap = (2 * (c + o) + h + l) / 6
    part1 = rankF( decay_linearF( deltaF(vwap, 4), 7 ) )
    part2 = (l + 0.96633) + (l * (1-0.96633) - vwap) / (o - (h + l)/2 )
    part3 = ts_rankF(decay_linearF(part2, 11), 7)
    factor_df = -1 * (part1 + part3)
    factor_df = factor_df.replace([np.inf, -np.inf], np.nan)
    print('done')

    return factor_df


def factor083(panel):

    c = panel.close
    l = panel.low
    h = panel.high
    o = panel.open
    v = panel.volume

    print('calculating factor...')
    vwap = (2 * (c + o) + h + l) / 6
    part1 = rankF( delayF( (h-l) / (sumF(c, 5) / 5), 2 ) ) * rankF(rankF(v))
    part2 = ( (h - l) / (sumF(c, 5) / 5) ) / (vwap - c)
    factor_df = -1 * part1 / part2
    factor_df = factor_df.replace([np.inf, -np.inf], np.nan)
    print('done')

    return factor_df


def factor084(panel):

    c = panel.close
    l = panel.low
    h = panel.high
    o = panel.open

    print('calculating factor...')
    vwap = (2 * (c + o) + h + l) / 6
    factor_df = -1 * signedpower( ts_rankF( vwap - ts_maxF(vwap, 15), 21 ), deltaF( c, 5 ) )
    factor_df = factor_df.replace([np.inf, -np.inf], np.nan)
    print('done')

    return factor_df


def factor101(panel):

    c = panel.close
    l = panel.low
    h = panel.high
    o = panel.open

    print('calculating factor...')
    factor_df = (c - o) / ( (h - l) + 0.001 )
    factor_df = factor_df.replace([np.inf, -np.inf], np.nan)
    print('done')

    return factor_df
