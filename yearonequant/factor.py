from yearonequant.util_quant import *


class Factor:
    """Compute indicators to evaluate single factor.

    :param factor_df:           a DataFrame of cross sectional factor values, stock in column
    :param price_df:            a DataFrame of cross sectional stock price
    :param leverage_ratio_df:   a DataFrame of dynamic leverage ratio for futures
    """

    def __init__(self, factor_df, price_df, leverage_ratio_df=None, days_required=60):

        self.factor_df = factor_df
        self.price_df = price_df
        self.leverage_ratio_df = leverage_ratio_df
        self.days_required = days_required

        self.preprocess()

        self.ret_df = price_df.pct_change()
        self.leveraged_ret_df = self.ret_df * self.leverage_ratio_df
        assert leverage_ratio_df.shape == self.ret_df.shape
        self.ret_of_sets = None

    def preprocess(self):
        """
        Preprocess input DataFrame.
        :return: void
        """

        # intersection of factor_df and price_df
        ind_fac = self.factor_df.index
        ind_price = self.price_df.index
        ind = ind_fac.join(ind_price, how='inner')
        self.factor_df = self.factor_df.ix[ind]
        self.price_df = self.price_df.ix[ind]
        # modify leverage_ratio_df accordingly if not None
        if self.leverage_ratio_df is not None:
            self.leverage_ratio_df = self.leverage_ratio_df.ix[ind]
            assert self.price_df.shape == self.leverage_ratio_df.shape

        # filter out first n days after going public
        for s in self.price_df.columns:
            price = self.price_df[s].dropna()
            filter_date_index = price.index[:self.days_required]
            self.price_df.ix[filter_date_index, s] = np.nan

    def get_ic(self, interval, ic_type='rank'):
        """Return a series of IC value, respect to each time period
        :param interval list of intervals in concern
        :param ic_type type of ic, rank for spearman, normal for pearson
        """
        valid_type = ['rank', 'normal']
        if ic_type not in valid_type:
            print("Invalid IC type! Feasible input: ['rank', 'normal']")
            return

        ic_df = pd.DataFrame(np.nan, index=self.factor_df.index, columns=interval)

        for n in range(0, len(interval)):

            ic_series = pd.Series(np.nan, self.factor_df.index)
            ret_df = self.price_df / self.price_df.shift(interval[n]) - 1

            if ic_type == 'rank':

                for i in range(interval[n], self.factor_df.shape[0]):  # loop over rows

                    previous_factor = self.factor_df.iloc[i - interval[n]]
                    current_ret = ret_df.iloc[i]
                    data = pd.concat([previous_factor, current_ret], axis=1).dropna()
                    # first returned value is coefficient
                    ic = scipy.stats.spearmanr(data)[0]
                    ic_series[i] = ic

            elif ic_type == 'normal':

                for i in range(interval[n], self.factor_df.shape[0]):  # loop over rows

                    previous_factor = self.factor_df.iloc[i - interval[n]]
                    current_ret = ret_df.iloc[i]
                    data = pd.concat([previous_factor, current_ret], axis=1).dropna()

                    ic = scipy.stats.pearsonr(data.ix[:, 0], data.ix[:, 1])[0]
                    ic_series[i] = ic

            ic_df.iloc[:, n] = ic_series

        return ic_df

    def get_performance_of_factor(self, interval, window_size, ic_type='rank', plot_graph=False):
        """Return a DataFrame of factor indicators
        :param: interval: a list of interval in concern
        :param: window_size: window size for moving average
        :param: ic_type: can be normal or rank
        :param: plot_graph:
        :return: panel of factor's performance, grouped by set
        """

        ic = self.get_ic(interval, ic_type)

        ic_ma = ic.rolling(window=window_size).mean()
        ic_std = ic.rolling(window=window_size).std()

        ir = ic_ma / ic_std

        ic_dp = pd.Panel({'information_coefficient': ic,
                          'rolling_mean_IC': ic_ma,
                          'rolling_std_IC': ic_std,
                          'information_ratio': ir})

        if plot_graph:
            # subplot
            for i in range(ic_dp.shape[2]):
                subplot_df_area(ic_dp.iloc[:, :, i],
                                title_str="Performance of Factor with Interval {}".format(interval[i]))

        return ic_dp

    def get_weighted_returns(self, use_leverage=False, plot_graph=False):
        """
        Factor weighted return
        :param use_leverage: include leverage ratio or not
        :param plot_graph:  plot the graph or not
        :return:
        """
        # sanity check
        if use_leverage and self.leverage_ratio_df is None:
            print('Please initialize Factor object with margin_rate_df provided')
            return

        weighted_factor_df = self.factor_df.div(self.factor_df.abs().sum(axis=1), axis=0)
        # if use_leverage is True, use leveraged return DataFrame
        return_df = self.leveraged_ret_df if use_leverage else self.ret_df
        # shift factor_df by 1, eliminate future function
        weighted_return_df = weighted_factor_df.shift(1) * return_df
        weighted_return = weighted_return_df.sum(axis=1)
        weighted_nv = (weighted_return + 1).cumprod()

        if plot_graph:
            plot_series(weighted_nv, 'Factor weighted nv')

        return weighted_nv

    def get_quantile_returns(self, num_of_sets, use_leverage=False, rebalance_period=1, top_bottom=False, plot_graph=False):
        """
        Return rate by set, column is set number, row is period
        :param num_of_sets: number of sets
        :param use_leverage: include leverage ratio or not
        :param rebalance_period: period to retain arrangement of sets
        :param top_bottom: if turned on, plot return of long first set, short last set
        :param plot_graph:
        :return:
        """
        if top_bottom:
            plot_graph = True
            
        ret_of_sets = pd.DataFrame(np.nan, index=self.factor_df.index, columns=range(1, num_of_sets + 1))

        # if use_leverage is True, use leveraged return DataFrame
        return_df = self.leveraged_ret_df if use_leverage else self.ret_df

        rebalanced_label = None
        for i in range(1, len(self.factor_df)):

            # get factor data at t-1
            previous_factor = self.factor_df[i - 1:i].dropna(axis=1)

            # tell if need to recompute set arrangement
            need_rebalance_label = i % rebalance_period == 0
            # use prev period's label
            if not need_rebalance_label and rebalanced_label is not None:
                # get realized returns at t
                current_ret = return_df[i:i + 1].dropna(axis=1)
                # eliminate the impact of external data, such as recent listed stocks
                labeled_data = pd.concat([current_ret.T, label], axis=1).dropna()

                # calculate returns for each set and update
                current_sets_ret = labeled_data.groupby(by='label').mean().T[:1]
                ret_of_sets.ix[current_sets_ret.index] = current_sets_ret
            elif previous_factor.dropna(axis=1).shape[1] > num_of_sets: # compute new label
                # corresponding ranks at t-1, ascending
                previous_rank = previous_factor.rank(axis=1)
                # transform (1, n) DataFrame to a (n, ) series,
                # therefore the output of pd.qcut() function will have indexes
                rank_series = pd.Series(previous_rank.values[0], index=previous_rank.columns)

                # label given data
                label = pd.qcut(x=rank_series.rank(method='first'), q=num_of_sets,
                                labels=range(1, num_of_sets + 1))
                label.name = 'label'
                # keep this arrangement
                rebalanced_label = label
                # get realized returns at t
                current_ret = return_df[i:i + 1].dropna(axis=1)
                # eliminate the impact of external data, such as recent listed stocks
                labeled_data = pd.concat([current_ret.T, label], axis=1).dropna()

                # calculate returns for each set and update
                current_sets_ret = labeled_data.groupby(by='label').mean().T[:1]
                ret_of_sets.ix[current_sets_ret.index] = current_sets_ret

        # plot
        if plot_graph:
            if top_bottom:
                return_of_bottom = ret_of_sets.iloc[:, 0]
                return_of_top = ret_of_sets.iloc[:, num_of_sets-1]
                spread = return_of_top - return_of_bottom
                nv_of_top_minus_bottom = (spread + 1).cumprod()
                plot_series(nv_of_top_minus_bottom, 'Long top short bottom nv')

            else:
                nv_of_sets = (ret_of_sets + 1).cumprod()
                plot_df(nv_of_sets, "Net Value of Sets")

        self.ret_of_sets = ret_of_sets

        return ret_of_sets

    def get_quantile_performance(self, num_of_sets, rebalance_period=1, plot_graph=False):
        """
        Get performance of each set
        :param num_of_sets:
        :param rebalance_period: period to retain arrangement of sets
        :param plot_graph:
        :return: DataFrame with set number in column, indicator in row
        """

        if self.ret_of_sets is not None:
            quantile_ret = self.ret_of_sets
        else:
            quantile_ret = self.get_quantile_returns(num_of_sets, rebalance_period)

        nv = pd.Series((quantile_ret + 1).cumprod()[-1:].values[0], index=quantile_ret.columns)
        mu = quantile_ret.mean()
        sigma = quantile_ret.std()
        sharpe = mu / sigma

        performance_df = pd.DataFrame({'Ending_NV': nv,
                                       'Mean_Return': mu,
                                       'Stdev': sigma,
                                       'Sharpe_Ratio': sharpe})

        performance_df = performance_df.T

        if plot_graph:
            # plot each set's sharpe ratio
            plot_bar(sharpe, 'Sharpe Ratio by Sets')

        return performance_df


# def get_contribution_coefficient(number_of_sets):
#     """(last set's return - first set's return) / (last set's return - first set's return),
#         numerator is grouped by factor, denominator is grouped by return
#     """
#     ret_of_sets_by_factor = self.quantile_returns()

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
