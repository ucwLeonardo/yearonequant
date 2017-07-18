import sys
import random

import rqdatac
from rqdatac import *

import talib

import datetime
import pandas as pd
import numpy as np
from pandas.tseries.offsets import *

import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

############################## Initialize rqdatac and plotly environments ################################
# initialize rqdatac to enable online functions such as get_price() and get_next_trading_date()
rqdatac.init('xinjin', '123456', ('172.19.182.162', 16003))

# initialize plotly to enable offline mode
init_notebook_mode(connected=True)
############################## Initialize rqdatac and plotly environments ################################

## helper functions
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
    ''' trading_dates_list is a list of string indicate date
    '''
    ymd_str = date2ymd_str(date_time)

    if ymd_str in trading_dates_list:    # this date is trading date
        if date_time.hour >= 15:    # event should be in next day
            return get_next_trading_date(ymd_str)
        else: # return date as datetime.date() type
            return datetime2date(date_time)
    else: # this date is not trading day, return next trading day
        return get_next_trading_date(ymd_str)
    
def complete_code(code):

    if len(code) < 6: # code is empty or length smaller than 6
        return False
    # careful, code is string type
    elif code[0] == '6': # 上证
        return code + '.XSHG'
    elif code[0] in ['0', '3']: # 深证
        return code + '.XSHE'
    else:
        return False

## IO functions
def read_announce_csv(file_name):
    # read csv file into dataframe
    df = pd.read_csv(file_name, dtype=str,
                    parse_dates=True,
                    index_col='Date',
                    usecols=['Code', 'Title', 'Link', 'Date'],
                    na_values=['nan'])
    return df

    
## Plot functions
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
        fill='tonexty' )

    trace = go.Scatter(
        name='Measurement',
        x=mean.index,
        y=mean,
        mode='lines',
        line=dict(color='rgb(31, 119, 180)'),
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty' )

    lower_bound = go.Scatter(
        name='Lower Bound',
        x=mean.index,
        y=lower,
        marker=dict(color="444"),
        line=dict(width=0),
        mode='lines' )

    data = [lower_bound, trace, upper_bound]

    layout = go.Layout(
        paper_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(title=yaxis_str),
        title=title_str,
        showlegend = False)

    fig = go.Figure(data=data, layout=layout)

    iplot(fig, filename=title_str)

# plot line and area beneath of a time series
def plot_area(time_series, title_str):
    trace = go.Scatter(
        x=time_series.index,
        y=time_series,
        fill='tonexty'
    )

    layout = go.Layout(
        paper_bgcolor='rgba(0,0,0,0)',
        title=title_str,
        showlegend = False
    )

    data = [trace]
    fig = go.Figure(data=data, layout=layout)

    iplot(fig, filename=title_str)

# plot bar chart of a time sereis
def plot_bar(time_series, title_str):
    bar = [go.Bar(
        x=time_series.index,
        y=time_series
    )]

    layout = go.Layout(
        paper_bgcolor='rgba(0,0,0,0)',
        title=title_str,
        showlegend = False
    )

    data = bar
    fig = go.Figure(data=data, layout=layout)

    iplot(data, filename=title_str)