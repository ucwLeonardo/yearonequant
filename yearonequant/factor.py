from yearonequant.util_quant import *

class Factor:
    """Compute indicators to evaluate single factor.

    :param factor_df: a dataframe of cross sectional factor values, stock in column
    :param price_df: a dataframe of cross sectional stock price
    NOTICE:
    factor_df and price_df should NOT have any NaN in it !
    """
    
    def __init__(self, factor_df, price_df):

        self.factor_df = factor_df
        self.price_df = price_df
        self.ret_df = price_df / price_df.shift(1)

    def get_rank_df(self, df, num_of_sets):
        """get rank row by row, group every row by sets,
            return a dataframe of rank number
        """ 

        rank_df = (df * np.array(0)).fillna(0)    # df of all 0
        rank_name = range(1, num_of_sets + 1)  # [1, 2, 3, ... ]

        # loop over row of df 
        for i in range(0, df.shape[0]):

            factor_series = df.iloc[i]   # the ith row
            # return a df of rank 
            rank_series = pd.qcut(factor_series, num_of_sets, labels=rank_name)
            rank_df.iloc[i] = rank_series

        return rank_df
    
    def get_ret_by_sets(self, ret_df, rank_df):
        """get cummulated return by set, column is set number, row is period
        """

        num_of_sets = int(max(rank_df.iloc[0])) # get set number
        ret_of_sets = pd.DataFrame()

        # loop over number of sets
        for i in range(1, num_of_sets + 1):

            # shift to ignore first row, which is NaN after computed ret
            shifted_rank_df = rank_df.shift(1)
            # select columns of price_df with same rank number (presented in rank_df)
            ret_of_set = ret_df[shifted_rank_df == i]
            ret_of_sets[i] = np.mean(ret_of_set, axis = 1) - 1 # set column with name "i"

        ret_of_sets = (ret_of_sets + 1).cumprod() - 1    # get cumulative return, last row is the cummulated return so far

        return ret_of_sets


    def get_ic(self, factor_df, ret_df, ic_type):
        """Return a series of IC value, respect to each time period
        """

        ic_list = []

        if ic_type == 'normal':

            for i in range(0, factor_df.shape[0] - 1):   # loop over rows

                factor_row = factor_df.iloc[i].dropna().values  # can't have NaN when using pearsonr
                ret_row = ret_df.iloc[i + 1].dropna().values
                # first retruned value is coefficient
                try:
                    ic = scipy.stats.pearsonr(factor_row, ret_row)[0]
                except:
                    print('failed at row {} of factor_df'.format(i))
                    print('failed factor row in normal')
                    print(factor_row)
                    print('failed return row in normal')
                    print(ret_row)
                ic_list.append(ic)

        if ic_type == 'rank':

            for i in range(0, factor_df.shape[0] - 1):   # loop over rows

                factor_row = factor_df.iloc[i].dropna().values  # can't have NaN when using pearsonr
                ret_row = ret_df.iloc[i + 1].dropna().values
                # first retruned value is coefficient
                ic = scipy.stats.spearmanr(factor_row, ret_row)[0]    
                ic_list.append(ic)

        ic = pd.Series(ic_list, index = factor_df.index[:-1])   # discard the last index

        return ic
   

    def get_ir(self, ic, window_size):
        """Return a series of IR value, respect to each time period's rolling value
        """

        ic_rolling_mean = ic.rolling(window = window_size).mean()
        ic_rolling_std = ic.rolling(window = window_size).std()

        ir = ic_rolling_mean / ic_rolling_std

        return ir


    def get_performance_by_sets(self, num_of_sets):
        """Return a dataframe of performance by sets.
        """

        rank_df = self.get_rank_df(self.factor_df, num_of_sets)

        ret_by_sets = self.get_ret_by_sets(self.ret_df, rank_df)
        cummulated_ret_by_sets = pd.Series(ret_by_sets.iloc[-1])

        ic_normal_by_sets = pd.Series(index=range(1, num_of_sets+1))
        ic_rank_by_sets = pd.Series(index=range(1, num_of_sets+1))

        for i in range(1, num_of_sets + 1):

            factor_of_set = self.factor_df[rank_df == i].dropna(axis=1, how='all')
            ret_of_set = self.ret_df[rank_df == i].dropna(axis=1, how='all')
            # print('factor of set {} is'.format(i))
            # print(factor_of_set)
            # print('return of set {} is'.format(i))
            # print(ret_of_set)
            ic_normal_by_sets[i] = np.mean(self.get_ic(factor_of_set, ret_of_set, 'normal'))
            ic_rank_by_sets[i] = np.mean(self.get_ic(factor_of_set, ret_of_set, 'rank'))
        
        performance = pd.DataFrame({'cummulated_ret_by_sets' : cummulated_ret_by_sets, 
                                'ic_normal_by_sets' : ic_normal_by_sets, 'ic_rank_by_sets' : ic_rank_by_sets})
        performance = performance.T

        self.performance_by_sets = performance

        return performance


    def get_performance_of_factor(self, window_size, ic_type='rank'):
        """Return a dataframe of factor indicators
        """

        ic = self.get_ic(self.factor_df, self.ret_df, ic_type)

        ic_ma = ic.rolling(window = window_size).mean()
        ic_std = ic.rolling(window = window_size).std()

        ir = ic_ma / ic_std

        ic_df = pd.DataFrame({'ic' : ic, 'ic_ma' : ic_ma, 'ic_std' : ic_std, 'ir' : ir})

        self.ic_indicator = ic_df

        return ic_df

    # def get_contribution_coefficient(number_of_sets):
    #     """(last set's return - first set's return) / (last set's return - first set's return),
    #         numerator is grouped by factor, denominator is grouped by return 
    #     """
    #     ret_of_sets_by_factor = self.get_ret_by_sets()

    #     # compute return of sets, where sets are grouped by return  
    #     rank_of_ret = self.get_rank_df(self.ret_df, number_of_sets)
    #     ret_of_sets_by_return = pd.DataFrame()

    #     # loop over set number to get mean of each set
    #     for i in range(1, self.num_of_sets + 1):

    #         shifted_rank_of_ret = rank_of_ret.shift(1)  # shift down one row, ignore first NaN row of ret
    #         ret_of_sets_by_return[i] = np.mean(ret[shifted_rank_of_ret == i], axis = 1) - 1

    #     numerator = abs(ret_of_sets_by_factor[self.num_of_sets] - ret_of_sets_by_factor[1])
    #     denominator = abs(ret_of_sets_by_return[self.num_of_sets] - ret_of_sets_by_return[1])

    #     return numerator / denominator
