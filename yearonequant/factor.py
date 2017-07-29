from yearonequant.util_quant import *


class Factor:
    """Compute indicators to evaluate single factor.

    :param factor_df: a DataFrame of cross sectional factor values, stock in column
    :param price_df: a DataFrame of cross sectional stock price
    """

    def __init__(self, factor_df, price_df):

        self.factor_df = factor_df
        self.price_df = price_df
        self.ret_df = price_df / price_df.shift(1)

        self.performance_by_sets = None
        self.ic_indicator = None

    @staticmethod
    def get_rank_df(df, num_of_sets):
        """get rank row by row, group every row by sets,
            return a DataFrame of rank number
        :param num_of_sets number of sets to be grouped
        """

        rank_df = (df * np.array(0)).fillna(0)  # df of all 0
        rank_name = range(1, num_of_sets + 1)  # [1, 2, 3, ... ]

        # loop over row of df 
        for i in range(0, df.shape[0]):
            factor_series = df.iloc[i]  # the ith row
            # return a df of rank 
            rank_series = pd.qcut(factor_series, num_of_sets, labels=rank_name)
            rank_df.iloc[i] = rank_series

        return rank_df

    @staticmethod
    def get_ret_by_sets(ret_df, rank_df):
        """get cumulate return by set, column is set number, row is period
        """

        num_of_sets = int(max(rank_df.iloc[0]))  # get set number
        ret_of_sets = pd.DataFrame()

        # loop over number of sets
        for i in range(1, num_of_sets + 1):
            # shift to ignore first row, which is NaN after computed ret
            shifted_rank_df = rank_df.shift(1)
            # select columns of price_df with same rank number (presented in rank_df)
            ret_of_set = ret_df[shifted_rank_df == i]
            ret_of_sets[i] = np.mean(ret_of_set, axis=1) - 1  # set column with name "i"

        # get cumulative return, last row is the cumulate return so far
        ret_of_sets = (ret_of_sets + 1).cumprod() - 1

        return ret_of_sets

    @staticmethod
    def get_ic(factor_df, ret_df, ic_type='rank'):
        """Return a series of IC value, respect to each time period
        :param factor_df DataFrame of factor values
        :param ret_df DataFrame of return values
        :param ic_type type of ic, rank for spearman, normal for pearson
        """

        ic_series = pd.Series(np.nan, factor_df.index[:-1])

        if ic_type == 'rank':

            for i in range(0, factor_df.shape[0] - 1):  # loop over rows

                current_factor = factor_df.iloc[i]
                future_ret = ret_df.iloc[i + 1]
                combined_data = pd.concat([current_factor, future_ret], axis=1).dropna()

                # first returned value is coefficient
                ic = scipy.stats.spearmanr(combined_data)[0]
                ic_series[i] = ic

        elif ic_type == 'normal':

            for i in range(0, factor_df.shape[0] - 1):  # loop over rows

                current_factor = factor_df.iloc[i]
                future_ret = ret_df.iloc[i + 1]
                combined_data = pd.concat([current_factor, future_ret], axis=1).dropna()

                # first returned value is coefficient
                ic = scipy.stats.pearsonr(combined_data.ix[:, 0], combined_data.ix[:, 1])[0]
                ic_series[i] = ic

        else:
            print("Invalid IC type! Feasible input: ['rank', 'normal']")

        return ic_series

    @staticmethod
    def get_ir(ic, window_size):
        """Return a series of IR value, respect to each time period's rolling value
        :param window_size window size of rolling MA
        """

        ic_rolling_mean = ic.rolling(window=window_size).mean()
        ic_rolling_std = ic.rolling(window=window_size).std()

        ir = ic_rolling_mean / ic_rolling_std

        return ir

    def get_performance_by_sets(self, num_of_sets):
        """Return a DataFrame of performance by sets.
        :param num_of_sets: number of sets to group
        :return: DataFrame of return performance by sets
        """

        rank_df = self.get_rank_df(self.factor_df, num_of_sets)

        ret_by_sets = self.get_ret_by_sets(self.ret_df, rank_df)
        cumulate_ret_by_sets = pd.Series(ret_by_sets.iloc[-1])

        ic_normal_by_sets = pd.Series(index=range(1, num_of_sets + 1))
        ic_rank_by_sets = pd.Series(index=range(1, num_of_sets + 1))

        for i in range(1, num_of_sets + 1):

            factor_of_set = self.factor_df[rank_df == i].dropna(axis=1, how='all')
            ret_of_set = self.ret_df[rank_df == i].dropna(axis=1, how='all')

            ic_normal_by_sets[i] = np.mean(self.get_ic(factor_of_set, ret_of_set, 'normal'))
            ic_rank_by_sets[i] = np.mean(self.get_ic(factor_of_set, ret_of_set, 'rank'))

        performance = pd.DataFrame({'cumulate_ret_by_sets': cumulate_ret_by_sets,
                                    'ic_normal_by_sets': ic_normal_by_sets,
                                    'ic_rank_by_sets': ic_rank_by_sets})
        performance = performance.T

        self.performance_by_sets = performance

        return performance

    def get_performance_of_factor(self, window_size, ic_type='rank'):
        """Return a DataFrame of factor indicators
        :param window_size: window size for moving average
        :param ic_type: can be normal or rank
        :return: DataFrame of factor's performance by set
        """

        ic = self.get_ic(self.factor_df, self.ret_df, ic_type)

        ic_ma = ic.rolling(window=window_size).mean()
        ic_std = ic.rolling(window=window_size).std()

        ir = ic_ma / ic_std

        ic_df = pd.DataFrame({'ic': ic, 'ic_ma': ic_ma, 'ic_std': ic_std, 'ir': ir})

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
