# -*- coding:utf-8 -*-
from yearonequant.util_quant import *

target_words = ['减持']
filter_words = ['不减持', '不存在减持', '终止', '限制', '购回', '回购', 
                        '完成', '更正', '更新', '完毕', '用于', '进展']

def filter_title(title, target_words, filter_words):
    
    # at least one word in target_words should be in title
    for w_positive in target_words:
        if w_positive in title:
            # words that should not be in title
            for w_negative in filter_words:
                if w_negative in title:
                    return False
                
            return True
        
    return False

def announce2event(file_name, backtest_start_date=None, verbose=False):
    '''
        @param file_name is the csv file of announcements to read from
        @param date is the start date of backtest
    '''

    # read csv file into dataframe
    df = pd.read_csv(file_name, dtype=str,
                    parse_dates=True,
                    index_col='Date',
                    usecols=['Code', 'Title', 'Link', 'Date'],
                    na_values=['nan'])
    
#     index = df.index
#     print(index)
#     new_index = index
#     for d in index:
#         print(d)
#         new_index.append(''.join(datetime.datetime.strptime(' '.join(str(index).split()), '%Y-%m-%d %H:%M:%S')))
#     df = df.reindex(new_index)
    #################################### check ###################################
    if verbose:
        print(df[-20:])
        print(type(df.index[0]))
        print(df.index[0])
        #print(df.index[0] > date2datetime(datetime.date(2017, 6, 12)))
        print(df.columns)
        # check index type
        for i in df.index:
            if i == '':
                print("There are null in df index!!!")
            elif not isinstance(i, type(datetime.datetime(2017,1,1))):
                print("There are {} type in df.index!!!".format(type(i)))
            else:
                pass
    #################################### check ###################################
    
    if backtest_start_date != None:
        try:
            # slice rows with date after backtest start date
            df = df[df.index > date2datetime(backtest_start_date)-BDay()]
        except:
            print('Something wrong when slice date of event df')

    # get date range
    start_date = date2ymd_str(df.index[-1])
    end_date = date2ymd_str(df.index[0])
    # get all valid trading dates
    trading_dates = get_trading_dates(start_date, end_date)
    # event df, no need to construct index and columns name, it's constructed on the fly
    event_df = pd.DataFrame(index=trading_dates)

    # loop over rows of df
    for date, row in df.iterrows():
        code = complete_code(str(row['Code']))
        # code has meaning and title pass the filter
        if date and code and filter_title(row['Title'], target_words, filter_words):
#             try:
                # keep only year-month-day, convert to datetime, index is list of trading dates 
                event_df.loc[adjust_to_trading_date(date, trading_dates), code] = 1
#             except:
#                 print('Set event error at :{}, code :{}'.format(date, code))
    
    # NOTICE: Remember to reverse the order of row because date index in 
    # csv file is in descending order, while date index in event_df should be
    # in ascending order.
    # event_df = event_df.iloc[::-1]

    return event_df

if __name__ == '__main__':
	announce2event('small.csv')
