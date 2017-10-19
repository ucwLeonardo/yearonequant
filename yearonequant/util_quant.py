import sys
import random

import rqdatac
from rqdatac import *

# import talib

import datetime
import pandas as pd
import numpy as np
from pandas.tseries.offsets import *
import scipy
import scipy.stats

import plotly.plotly as py
from plotly import tools
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

# initialize plotly to enable offline mode
init_notebook_mode(connected=True)


# helper functions
# keep only year and moth(year-month) as string of datetime object
# return example: 2017-01-01
def date2ym_str(date):
    y = date.year
    m = date.month
    ym = '{}-{}'.format(y, m)
    return ym


def date2ymd_str(date):
    y = date.year
    m = date.month
    d = date.day
    ymd = '{}-{}-{}'.format(y, m, d)
    return ymd


def datetime2ymd_str(datetime):

    year = datetime.year
    month = datetime.month
    day = datetime.day

    # if month and day only has one character,
    # append '0' on head, that's the rule of cninfo, gosh
    if len(str(month)) == 1:
        month_str = '0' + str(month)
    else:
        month_str = str(month)
    if len(str(day)) == 1:
        day_str = '0' + str(day)
    else:
        day_str = str(day)

    return str(year) + '-' + month_str + '-' + day_str


def datetime2date(date_time):
    y = date_time.year
    m = date_time.month
    d = date_time.day
    return datetime.date(y, m, d)


def date2datetime(date):
    y = date.year
    m = date.month
    d = date.day
    return datetime.datetime(y, m, d)


def adjust_to_trading_date(date_time, trading_dates_list):
    """ trading_dates_list is a list of string indicate date
    """
    ymd_str = date2ymd_str(date_time)

    if ymd_str in trading_dates_list:  # this date is trading date
        if date_time.hour >= 15:  # event should be in next day
            return get_next_trading_date(ymd_str)
        else:  # return date as datetime.date() type
            return datetime2date(date_time)
    else:  # this date is not trading day, return next trading day
        return get_next_trading_date(ymd_str)


def complete_code(code):
    """
    Append stock code number with code type.
    :param code: code in digits as string
    :return: code in complete form
    """
    if len(code) < 6:  # code is empty or length smaller than 6
        return False
    # careful, code is string type
    elif code[0] == '6':  # 上证
        return code + '.XSHG'
    elif code[0] in ['0', '3']:  # 深证
        return code + '.XSHE'
    else:
        return False


# IO functions
def read_announce_csv(file_name):
    """
    Read announcement csv file into DataFrame
    :param file_name: file name
    :return: the DataFrame
    """
    df = pd.read_csv(file_name, dtype=str,
                     parse_dates=True,
                     index_col='Date',
                     usecols=['Code', 'Title', 'Link', 'Date'],
                     na_values=['nan'])
    return df


# Plot functions
# plot a time series and a band deviate by std_num of std
def plot_band(time_series, title_str, yaxis_str, std_num=1):
    # # sign in
    # py.sign_in('hyqLeonardo', 'aHHAi8RbFuit2fOfEizB')

    mean = time_series
    std = mean.std()
    upper = mean + std_num * std
    lower = mean - std_num * std

    upper_bound = go.Scatter(
        name='Upper Bound',
        x=mean.index,
        y=upper,
        mode='lines',
        marker=dict(color="444"),
        line=dict(width=0),
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty')

    trace = go.Scatter(
        name='Measurement',
        x=mean.index,
        y=mean,
        mode='lines',
        line=dict(color='rgb(31, 119, 180)'),
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty')

    lower_bound = go.Scatter(
        name='Lower Bound',
        x=mean.index,
        y=lower,
        marker=dict(color="444"),
        line=dict(width=0),
        mode='lines')

    data = [lower_bound, trace, upper_bound]

    layout = go.Layout(
        paper_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(title=yaxis_str),
        title=title_str,
        showlegend=False)

    fig = go.Figure(data=data, layout=layout)

    iplot(fig, filename=title_str)


def plot_series(series, title_str):
    """
    Plot series.
    :param series:      time series
    :param title_str:   title
    """
    data = [go.Scatter(x=series.index, y=series)]

    layout = go.Layout(
        paper_bgcolor='rgba(0,0,0,0)',
        title=title_str,
        showlegend=False
    )
    fig = go.Figure(data=data, layout=layout)

    iplot(fig, filename=title_str)

def plot_area(time_series, title_str):
    """
    Plot line and area beneath of a time series.
    :param time_series:
    :param title_str:
    :return:
    """
    trace = go.Scatter(
        x=time_series.index,
        y=time_series,
        fill='tonexty'
    )

    layout = go.Layout(
        paper_bgcolor='rgba(0,0,0,0)',
        title=title_str,
        showlegend=False
    )

    data = [trace]
    fig = go.Figure(data=data, layout=layout)

    iplot(fig, filename=title_str)


def plot_bar(time_series, title_str):
    """
    Plot time series as bar
    :param time_series:
    :param title_str: title of plot
    :return:
    """

    bar = [go.Bar(
        x=time_series.index,
        y=time_series
    )]

    data = bar

    layout = go.Layout(
        paper_bgcolor='rgba(0,0,0,0)',
        title=title_str,
        showlegend=False
    )

    fig = go.Figure(data=data, layout=layout)

    iplot(fig, filename=title_str)


def subplot_df_area(df, title_str):
    """
    Plot each column of df as a subplot.
    :param df: DataFrame to plot
    :param title_str: title of the whole plot
    """
    plot_num = df.shape[1]
    assert plot_num % 2 == 0
    row_num = 2
    col_num = int(plot_num / 2)

    title_str_tuple = tuple(df.columns)

    fig = tools.make_subplots(rows=row_num, cols=col_num,
                              subplot_titles=title_str_tuple)

    count = 0
    for i in range(1, row_num + 1):
        for j in range(1, col_num + 1):
            series = df.iloc[:, count]
            fig.append_trace(
                go.Scatter(
                    x=series.index,
                    y=series,
                    fill='tonexty'
                ), i, j)
            count += 1

    fig['layout'].update(
        paper_bgcolor='rgba(0,0,0,0)',
        title=title_str,
        showlegend=False
    )

    iplot(fig, filename=title_str)


def plot_df(df, title_str, plot_type='line'):
    """
    Plot each column of df as a subplot.
    :param df: DataFrame to plot
    :param title_str: title of the whole plot
    :param plot_type: line or area
    """
    valid_plot_type = ['line', 'area']
    if plot_type not in valid_plot_type:
        print("Invalid plot type! Feasible type: 'line', 'area'")
        return

    data = list()

    for i in range(df.shape[1]):

        series = df.iloc[:, i]
        if plot_type == 'line':
            data.append(
                go.Scatter(
                    x=series.index,
                    y=series,
                    name="set " + str(df.columns[i])
                ))
        if plot_type == 'area':
            data.append(
                go.Scatter(
                    x=series.index,
                    y=series,
                    fill='tonexty',
                    name="set " + str(df.columns[i])
                ))

    layout = go.Layout(
        paper_bgcolor='rgba(0,0,0,0)',
        title=title_str,
        showlegend=False
    )

    fig = go.Figure(data=data, layout=layout)

    iplot(fig, filename=title_str)


def plot_ohlc(df, title_str):
    
    trace = go.Ohlc(x=df.index,
                open=df.Open,
                high=df.High,
                low=df.Low,
                close=df.Close,
                increasing=dict(line=dict(color= 'red')),
                decreasing=dict(line=dict(color= 'green')))
    
    data = [trace]
    
    iplot(data, filename=title_str)
