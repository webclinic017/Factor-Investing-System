import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import scipy.stats as stats
import pandas as pd
import numpy as np

import seaborn as sns
sns.set_style('darkgrid')
pd.plotting.register_matplotlib_converters()

from datetime import timedelta

class FactorAnalysis:
    
    def __init__(self, qb, tickers, startDate, endDate, resolution):
        
        # add symbols
        symbols = [qb.AddEquity(ticker, resolution).Symbol for ticker in tickers]
        
        # get historical data at initialization ----------------------------------------------------------
        ohlcvDf = qb.History(symbols, startDate, endDate, resolution)
        # when using daily resolution, QuantConnect uses the date at midnight after the trading day
        # hence skipping Mondays and showing Saturdays. We avoid this by subtracting one day from the index
        ohlcvDf.index = ohlcvDf.index.set_levels(ohlcvDf.index.levels[1] - timedelta(1), level = 'time')
        
        self.ohlcvDf = ohlcvDf.dropna()
        
    def GetFactorsDf(self, fct = None):

        '''
        Description:
            Apply a function to a MultiIndex Dataframe of historical data
            Group on symbol first to get a ohlcv series per symbol, and apply a custom function to it
            in order to get a factor value per symbol and day
        Args:
            fct: Function to calculate the custom factor
        Returns:
            MultiIndex Dataframe (symbol/time indexes) with the factor values
        '''
        
        if fct is None:
            raise ValueError('fct arguments needs to be provided to calculate factors')
        
        # group by symbol to get a timeseries of historical data per symbol and apply CustomFactor function
        factorsDf = self.ohlcvDf.groupby('symbol', group_keys = False).apply(lambda x: fct(x)).dropna()
        factorsDf.columns = ['Factor_' + str(i + 1) for i in range(len(factorsDf.columns))]
        # sort indexes
        factorsDf = factorsDf.sort_index(level = ['symbol', 'time'])

        return factorsDf
    
    def GetStandardizedFactorsDf(self, factorsDf):
        
        '''
        Description:
            Winsorize and standardize factors
        Args:
            factorsDf: MultiIndex Dataframe (symbol/time indexes) with the factor values
        Returns:
            MultiIndex Dataframe (symbol/time indexes) with standardized factor values
        '''
        
        # winsorization
        winsorizedFactorsDf = factorsDf.apply(stats.mstats.winsorize, limits = [0.025, 0.025])
        # zscore standardization
        standardizedFactorsDf = winsorizedFactorsDf.apply(stats.zscore)
        
        return standardizedFactorsDf
    
    def GetCombinedFactorsDf(self, factorsDf, combinedFactorWeightsDict = None):
        
        '''
        Description:
            Create a combined factor as a linear combination of individual factors
        Args:
            factorsDf: MultiIndex Dataframe (symbol/time indexes) with the factor values
            combinedFactorWeightsDict: Dictionary with factor names and weights to calculate a combined factor
        Returns:
            MultiIndex Dataframe (symbol/time indexes) with the individual factors and the combined factor
        '''
        
        # make a deep copy of the DataFrame
        combinedFactorsDf = factorsDf.copy(deep = True)
        
        # calculate a combined factor
        if combinedFactorWeightsDict is None:
            return combinedFactorsDf
        elif not combinedFactorWeightsDict:
            combinedFactorsDf['Combined_Factor'] = combinedFactorsDf.sum(axis = 1)
        else:
            combinedFactorsDf['Combined_Factor'] = sum(combinedFactorsDf[key] * value
                                                       for key, value in combinedFactorWeightsDict.items())
        
        return combinedFactorsDf
    
    def GetFinalFactorsDf(self, fct = None, combinedFactorWeightsDict = None, standardize = True):

        '''
        Description:
            - Apply a function to a MultiIndex Dataframe of historical data
              Group on symbol first to get a ohlcv series per symbol, and apply a custom function to it
                  in order to get a factor value per symbol and day
            - If required, standardize the factors and remove potential outliers
            - If required, add a combined factor as a linear combination of individual factors
        Args:
            fct: Function to calculate the custom factor
            standardize: Boolean to standardize data
            combinedFactorWeightsDict: Dictionary with factor names and weights to calculate a combined factor
        Returns:
            MultiIndex Dataframe (symbol/time indexes) with the factor values
        '''
        
        # get factorsDf
        factorsDf = self.GetFactorsDf(fct)
        
        # standardize
        if standardize:
            factorsDf = self.GetStandardizedFactorsDf(factorsDf)
        
        # add combined factor
        if combinedFactorWeightsDict is not None:
            factorsDf = self.GetCombinedFactorsDf(factorsDf, combinedFactorWeightsDict)

        return factorsDf
    
    def GetPricesDf(self, field = 'close'):
    
        '''
        Description:
            Get a MultiIndex Dataframe of chosen field
        Args:
            field: open, high, low, close or volume
        Returns:
            MultiIndex Dataframe (symbol/time indexes) with the chosen field
        '''
        
        # select only chose field and turn into a dataframe
        pricesDf = self.ohlcvDf[field].to_frame()
        pricesDf.columns = ['price']
        # forward fill nas and after that drop rows with some nas left
        pricesDf = pricesDf.sort_index(level = ['symbol', 'time'])
        pricesDf = pricesDf.groupby('symbol').fillna(method = 'ffill').dropna()

        return pricesDf
    
    def GetFactorsPricesDf(self, factorsDf, field = 'close'):
    
        '''
        Description:
            Get a MultiIndex Dataframe (symbol/time indexes) with all the factors and chosen prices
        Args:
            factorsDf: MultiIndex Dataframe (symbol/time indexes) with the factor values
            field: open, high, low, close or volume
        Returns:
            MultiIndex Dataframe (symbol/time indexes) with all the factors and chosen prices
        '''

        # get the pricesDf
        pricesDf = self.GetPricesDf(field)
        
        # merge factorsDf and pricesDf and fill forward nans by symbol
        factorsPricesDf = pd.merge(factorsDf, pricesDf, how = 'right', left_index = True, right_index = True)
        factorsPricesDf = factorsPricesDf.sort_index(level = ['symbol', 'time'])
        factorsPricesDf = factorsPricesDf.groupby('symbol').fillna(method = 'ffill').dropna()

        return factorsPricesDf
    
    def GetFactorsForwardReturnsDf(self, factorsPricesDf, forwardPeriods = [1, 5, 21]):
    
        '''
        Description:
            Generate a MultiIndex Dataframe (symbol/time indexes) with all previous info plus forward returns
        Args:
            factorsPricesDf:  MultiIndex Dataframe (symbol/time indexes) with all the factors and chosen prices
            forwardPeriods: List of integers defining the different periods for forward returns
        Returns:
            MultiIndex Dataframe (symbol/time indexes) with the factor values and forward returns
        '''

        # make sure 1 day forward returns are calculated even if not provided by user
        if 1 not in forwardPeriods:
            forwardPeriods.append(1)
        
        # calculate forward returns per period
        for period in forwardPeriods:
            factorsPricesDf[str(period) + 'D'] = (factorsPricesDf.groupby('symbol', group_keys = False)
                                                    .apply(lambda x: x['price'].pct_change(period).shift(-period)))
        
        # drop column price
        factorsForwardReturnsDf = factorsPricesDf.dropna().drop('price', axis = 1)

        return factorsForwardReturnsDf
    
    def GetFactorQuantilesForwardReturnsDf(self, factorsDf, field = 'close',
                                            forwardPeriods = [1, 5, 21],
                                            factor = 'Factor_1', q = 5):
        
        '''
        Description:
            Create a MultiIndex Dataframe (symbol/time indexes) with the factor values,
            forward returns and the quantile groups
        Args:
            factorsDf: MultiIndex Dataframe (symbol/time indexes) with the factor values
            field: open, high, low, close or volume
            forwardPeriods: List of integers defining the different periods for forward returns
            factor: Chosen factor to create quantiles for
            q: Number of quantile groups
        Returns:
            MultiIndex Dataframe (symbol/time indexes) with the factor values, forward returns and the quantile groups
        '''
        
        # get factorsForwardReturnsDf
        factorsPricesDf = self.GetFactorsPricesDf(factorsDf, field)
        factorsForwardReturnsDf = self.GetFactorsForwardReturnsDf(factorsPricesDf, forwardPeriods)
        
        # reorder index levels to have time and then symbols so we can then create quantiles per day
        factorsForwardReturnsDf = factorsForwardReturnsDf.reorder_levels(['time', 'symbol'])
        factorsForwardReturnsDf = factorsForwardReturnsDf.sort_index(level = ['time', 'symbol'])
        
        # calculate quintiles given the chosen factor and rename columns
        factorsForwardReturnsDf['Quantile'] = factorsForwardReturnsDf[factor].groupby('time').apply(lambda x: pd.qcut(x, q, labels = False, duplicates = 'drop')).add(1)
        factorsForwardReturnsDf['Quantile'] = 'Group_' + factorsForwardReturnsDf['Quantile'].astype(str)
        
        # remove the other factor columns
        factorCols = [x for x in factorsForwardReturnsDf.columns if 'Factor' not in x or x == factor]
        factorQuantilesForwardReturnsDf = factorsForwardReturnsDf[factorCols]
        
        return factorQuantilesForwardReturnsDf
    
    def GetReturnsByQuantileDf(self, factorQuantilesForwardReturnsDf, forwardPeriod = 1, weighting = 'mean'):
    
        '''
        Description:
            Generate a SingleIndex Dataframe with period forward returns by quantile and time
        Args:
            factorQuantilesForwardReturnsDf: MultiIndex Dataframe (symbol/time indexes) with the factor values,
                                             forward returns and the quantile groups
            forwardPeriod: The period of forward returns
            weighting: The weighting to apply to the returns in each quantile after grouping:
                        - mean: Take the average of all the stock returns within each quantile
                        - factor: Take a factor-weighted return within each quantile
        Returns:
            SingleIndex Dataframe with period forward returns by quantile and time
        '''

        # we drop the symbols and convert to a MultiIndex Dataframe with Quantile and time as indexes and forward returns
        df = factorQuantilesForwardReturnsDf.droplevel(['symbol'])
        df.set_index('Quantile', append = True, inplace = True)
        df = df.reorder_levels(['Quantile', 'time'])
        df = df.sort_index(level = ['Quantile', 'time'])
        
        # get the column name for the factor and period
        factorCol = [x for x in df.columns if 'Factor' in x][0]
        periodCol = [str(forwardPeriod) + 'D'][0]
        
        if weighting == 'mean':
            df = df[[periodCol]]
            # group by Quantile and time and get the mean returns (equal weight across all stocks within each quantiles)
            returnsByQuantileDf = df.groupby(['Quantile', 'time']).mean()
        elif weighting == 'factor':
            relevantCols = [factorCol, periodCol]
            df = df[relevantCols]
            # group by Quantile and time and create a column with weights based on factor values
            df['Factor_Weights'] = (df.groupby(['Quantile', 'time'], group_keys = False)
                                    .apply(lambda x: x[factorCol].abs() / x[factorCol].abs().sum()))
            # group by Quantile and time and calculate the factor weighted average returns
            returnsByQuantileDf = (df.groupby(['Quantile', 'time'], group_keys = False)
                                    .apply(lambda x: (x['Factor_Weights'] * x[periodCol]).sum())).to_frame()

        # unstack to convert to SingleIndex Dataframe
        returnsByQuantileDf = returnsByQuantileDf.unstack(0).fillna(0)
        returnsByQuantileDf.columns = returnsByQuantileDf.columns.droplevel(0)
        returnsByQuantileDf.columns.name = None
        
        # finally keep every nth row to match with the forward period returns
        returnsByQuantileDf = returnsByQuantileDf.iloc[::forwardPeriod, :]

        return returnsByQuantileDf
    
    def GetMeanReturnsByQuantileDf(self, factorQuantilesForwardReturnsDf):
            
        '''
        Description:
            Generate a SingleIndex Dataframe with mean returns by quantile and time
        Args:
            factorQuantilesForwardReturnsDf: MultiIndex Dataframe (symbol/time indexes) with the factor values,
                                             forward returns and the quantile groups
        Returns:
            SingleIndex Dataframe with mean returns by quantile and time
        '''
    
        # remove factor columns, group by quantile and take the average return
        factorCol = [x for x in factorQuantilesForwardReturnsDf.columns if 'Factor' in x]
        quantileMeanReturn = factorQuantilesForwardReturnsDf.drop(factorCol, axis = 1).groupby('Quantile').mean()
        
        return quantileMeanReturn
    
    def GetPortfolioLongShortReturnsDf(self, returnsByQuantileDf, portfolioWeightsDict = None):
        
        '''
        Description:
            Generate a SingleIndex Dataframe with the returns of a Long-Short portfolio
        Args:
            returnsByQuantileDf: SingleIndex Dataframe with period forward returns by quantile and time
            portfolioWeightsDict: Dictionary with quantiles and weights to create a portfolio of returns
        Returns:
            SingleIndex Dataframe with the returns of Long-Short portfolio
        '''
        
        # if no portfolioWeightsDict are provided, create a default one
        # going 100% long top quintile and 100% short bottom quintile
        if portfolioWeightsDict is None:
            quantileGroups = sorted(list(returnsByQuantileDf.columns))
            topQuantile = quantileGroups[-1]
            bottomQuantile = quantileGroups[0]
            portfolioWeightsDict = {topQuantile: 1, bottomQuantile: -1}
        
        # we calculate the weighted average portfolio returns based on given weights for each quintile
        col = list(portfolioWeightsDict.keys())
        portfolioLongShortReturnsDf = returnsByQuantileDf.loc[: , col]
        portfolioLongShortReturnsDf[col[0]] = portfolioLongShortReturnsDf[col[0]] * portfolioWeightsDict[col[0]]
        portfolioLongShortReturnsDf[col[1]] = portfolioLongShortReturnsDf[col[1]] * portfolioWeightsDict[col[1]]
        portfolioLongShortReturnsDf['Strategy'] = portfolioLongShortReturnsDf.sum(axis = 1)
        portfolioLongShortReturnsDf = portfolioLongShortReturnsDf[['Strategy']]

        return portfolioLongShortReturnsDf
    
    def GetCumulativeReturnsDf(self, returnsDf):
        
        '''
        Description:
            Convert a DataFrame of returns into a DataFrame of cumulative returns
        Args:
            returnsDf: SingleIndex Dataframe with returns
        Returns:
            SingleIndex Dataframe with cumulative returns
        '''
        
        cumulativeReturnsDf = returnsDf.add(1).cumprod().add(-1)

        return cumulativeReturnsDf

    # ploting functions -----------------------------------------------------------------------------------------
    
    def PlotFactorsCorrMatrix(self, factorsDf):
        
        '''
        Description:
            Plot the factors correlation matrix
        Args:
            factorsDf: MultiIndex Dataframe (symbol/time indexes) with the factor values
        Returns:
            Plot the factors correlation matrix
        '''

        corrMatrix = round(factorsDf.corr(), 2)
        
        nCol = len(list(factorsDf.columns))
        plt.subplots(figsize = (nCol, nCol))
        sns.heatmap(corrMatrix, annot = True)
        
        plt.show()
        
    def PlotHistograms(self, factorsDf):
        
        '''
        Description:
            Plot the histogram for each factor
        Args:
            factorsDf: MultiIndex Dataframe (symbol/time indexes) with the factor values
        Returns:
            Plot the histogram for each factor
        '''
        
        nCol = len(list(factorsDf.columns))
        factorsDf.hist(figsize = (nCol * 3, nCol * 2), bins = 50)
        
        plt.show()
        
    def PlotBoxPlotQuantilesCount(self, factorQuantilesForwardReturnsDf):
                
        '''
        Description:
            Plot a box plot with the distributions of number of stocks in each quintile.
            The objective is to make sure each quintile has an almost equal number of stocks most of the time
        Args:
            factorQuantilesForwardReturnsDf: MultiIndex Dataframe (symbol/time indexes) with the factor values,
                                             forward returns and the quantile groups
        Returns:
            Plot a box plot with the distributions of number of stocks in each quintile
        '''
        
        factorCol = [x for x in factorQuantilesForwardReturnsDf.columns if 'Factor' in x]
        df = factorQuantilesForwardReturnsDf.groupby(['Quantile', 'time'])[factorCol].count()
        df = df.unstack(0)
        df.columns = df.columns.droplevel(0)
        df.name = None
        
        ax = sns.boxplot(data = df, width = 0.5, palette = "colorblind", orient = 'h')
        ax.set_title('Distribution Of Number Of Assets Within Quintiles')
        
        plt.show()
    
    def PlotMeanReturnsByQuantile(self, factorQuantilesForwardReturnsDf):
        
        '''
        Description:
            Plot the mean return for each quantile group and forward return period
        Args:
            factorQuantilesForwardReturnsDf: MultiIndex Dataframe (symbol/time indexes) with the factor values,
                                             forward returns and the quantile groups
        Returns:
            Plot with the mean return for each quantile group and forward return period
        '''
        
        meanReturnsByQuantileDf = self.GetMeanReturnsByQuantileDf(factorQuantilesForwardReturnsDf)
        # plot
        ax = meanReturnsByQuantileDf.plot(kind = 'bar', figsize = (12, 5))
        ax.set_title('Mean Returns By Quantile Group And Forward Period Return', fontdict = {'fontsize': 15})
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        
        plt.show()    
        
    def PlotCumulativeReturnsByQuantile(self, factorQuantilesForwardReturnsDf,
                                        forwardPeriod = 1, weighting = 'mean'):
        
        '''
        Description:
            Plot cumulative returns per quantile group
        Args:
            factorQuantilesForwardReturnsDf: MultiIndex Dataframe (symbol/time indexes) with the factor values,
                                             forward returns and the quantile groups
            forwardPeriod: The period of forward returns
            weighting: The weighting to apply to the returns in each quantile after grouping:
                        - mean: Take the average of all the stock returns within each quantile
                        - factor: Take a factor-weighted return within each quantile
        Returns:
            Plot with the cumulative returns per quantile group
        '''
        
        # get returns by quantile
        returnsByQuantileDf = self.GetReturnsByQuantileDf(factorQuantilesForwardReturnsDf, forwardPeriod, weighting)
        cumulativeReturnsByQuantileDf = self.GetCumulativeReturnsDf(returnsByQuantileDf)
        
        # take logarithm for better visualization
        cumulativeReturnsByQuantileDf = np.log(1 + cumulativeReturnsByQuantileDf)
        
        # get the relevant columns
        colTop = cumulativeReturnsByQuantileDf.iloc[:, [-1]].columns[0]
        colBottom = cumulativeReturnsByQuantileDf.iloc[:, [0]].columns[0]
        colMiddle = cumulativeReturnsByQuantileDf.drop([colTop, colBottom], axis = 1).columns
        
        # plot
        fig, ax = plt.subplots(figsize = (12, 5))
        ax.plot(cumulativeReturnsByQuantileDf[colBottom], color = 'red', linewidth = 2)
        ax.plot(cumulativeReturnsByQuantileDf[colMiddle], alpha = 0.3)
        ax.plot(cumulativeReturnsByQuantileDf[colTop], color = 'green', linewidth = 2)
        # formatting
        ax.axhline(y = 0, color = 'black', linestyle = '--', linewidth = 0.5)
        ax.set_title('Cumulative Log-Returns By Quantile Group', fontdict = {'fontsize': 15})
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.legend(cumulativeReturnsByQuantileDf.columns, loc = 'best')
        
        plt.show()
        
    def PlotPortfolioLongShortCumulativeReturns(self, factorQuantilesForwardReturnsDf,
                                                forwardPeriod = 1, weighting = 'mean',
                                                portfolioWeightsDict = None):

        '''
        Description:
            Plot cumulative returns for a long-short portfolio
        Args:
            factorQuantilesForwardReturnsDf: MultiIndex Dataframe (symbol/time indexes) with the factor values,
                                             forward returns and the quantile groups
            forwardPeriod: The period of forward returns
            weighting: The weighting to apply to the returns in each quantile after grouping:
                        - mean: Take the average of all the stock returns within each quantile
                        - factor: Take a factor-weighted return within each quantile
        Returns:
            Plot cumulative returns for a long-short portfolio
        '''
        
        # get returns by quantile
        returnsByQuantileDf = self.GetReturnsByQuantileDf(factorQuantilesForwardReturnsDf, forwardPeriod, weighting)
        # calculate returns for a long-short portolio
        portfolioLongShortReturnsDf = self.GetPortfolioLongShortReturnsDf(returnsByQuantileDf, portfolioWeightsDict)
        portfolioLongShortCumulativeReturnsDf = self.GetCumulativeReturnsDf(portfolioLongShortReturnsDf)
        
        # prepare plot
        fig, ax = plt.subplots(figsize = (12, 5))
        
        # plot portfolio
        colPortfolio = portfolioLongShortCumulativeReturnsDf.iloc[:, [0]].columns[0]
        ax.plot(portfolioLongShortCumulativeReturnsDf[colPortfolio], color = 'black', linewidth = 2)
            
        if len(portfolioLongShortCumulativeReturnsDf.columns) > 1:
            colFactors = portfolioLongShortCumulativeReturnsDf.iloc[:, 1:].columns
            # plot factors
            ax.plot(portfolioLongShortCumulativeReturnsDf[colFactors], alpha = 0.3)
            
        # formatting
        ax.axhline(y = 0, color = 'black', linestyle = '--', linewidth = 0.5)
        ax.set_title('Cumulative Returns Long-Short Portfolio', fontdict = {'fontsize': 15})
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.legend(portfolioLongShortCumulativeReturnsDf.columns, loc = 'best')
        
        plt.show()
        
    def PlotIC(self, factorQuantilesForwardReturnsDf):
        
        '''
        Description:
            Plot the Information Coefficient (Spearman Rank Correlation) for different periods along with a moving average
        Args:
            factorQuantilesForwardReturnsDf: MultiIndex Dataframe (symbol/time indexes) with the factor values,
                                             forward returns and the quantile groups
        Returns:
            Plot of the Information Coefficient (Spearman Rank Correlation) for different periods along with a moving average
        '''
        
        # get the forward periods and factor columns
        forwardPeriods = [int(x.split('D', 1)[0]) for x in factorQuantilesForwardReturnsDf.columns if 'D' in x]
        factorCol = [x for x in factorQuantilesForwardReturnsDf.columns if 'Factor' in x]
        
        # iterate over the periods
        for period in forwardPeriods:
            col = str(period) + 'D'
            # calculate the spearman rank coefficient for each day between the factor values and forward returns
            icDf = (factorQuantilesForwardReturnsDf.groupby('time')
                    .apply(lambda x: stats.spearmanr(x[factorCol], x[col])[0]).to_frame().dropna())
            icDf.columns = ['IC']
            # apply a moving average for smoothing
            icDf['21D Moving Average'] = icDf.rolling(21).apply(lambda x: np.mean(x))
            
            # plot
            fig, ax = plt.subplots(figsize = (12, 5))
            ax.plot(icDf['IC'], alpha = 0.5)
            ax.plot(icDf['21D Moving Average'])
            ax.axhline(y = 0, color = 'black', linestyle = '--', linewidth = 0.5)

            mu = icDf['IC'].mean()
            sigma = icDf['IC'].std()
            textstr = '\n'.join((
                        r'$\mu=%.2f$' % (mu, ),
                        r'$\sigma=%.2f$' % (sigma, )))
            props = dict(boxstyle = 'round', facecolor = 'white', alpha = 0.5)
            ax.text(0.05, 0.95, textstr, transform = ax.transAxes, fontsize = 14,
                    verticalalignment = 'top', bbox = props)

            ax.set_title(col + ' Forward Return Information Coefficient (IC)', fontdict = {'fontsize': 15})
            ax.legend(icDf.columns, loc = 'upper right')
            
            plt.show()
            
    # run full factor analysis --------------------------------------------------------------------------------------
    
    def RunFactorAnalysis(self, factorQuantilesForwardReturnsDf, forwardPeriod = 1,
                        weighting = 'mean', portfolioWeightsDict = None, makePlots = True):
                                
        '''
        Description:
            Run all needed functions and generate relevant DataFrames and plots for analysis
        Args:
            factorQuantilesForwardReturnsDf: MultiIndex Dataframe (symbol/time indexes) with the factor values,
                                             forward returns and the quantile groups
            forwardPeriod: The period of forward returns
            weighting: The weighting to apply to the returns in each quantile after grouping:
                        - mean: Take the average of all the stock returns within each quantile
                        - factor: Take a factor-weighted return within each quantile
            portfolioWeightsDict: Dictionary with quantiles and weights to create a portfolio of returns
        Returns:
            Plots for factor analysis
        '''
        
        # plotting
        if makePlots:
            self.PlotMeanReturnsByQuantile(factorQuantilesForwardReturnsDf)
            self.PlotCumulativeReturnsByQuantile(factorQuantilesForwardReturnsDf)
            self.PlotPortfolioLongShortCumulativeReturns(factorQuantilesForwardReturnsDf)
            self.PlotIC(factorQuantilesForwardReturnsDf)
        
        # keep DataFrames
        self.returnsByQuantileDf = self.GetReturnsByQuantileDf(factorQuantilesForwardReturnsDf, forwardPeriod, weighting)
        self.cumulativeReturnsByQuantileDf = self.GetCumulativeReturnsDf(self.returnsByQuantileDf)
        self.portfolioLongShortReturnsDf = self.GetPortfolioLongShortReturnsDf(self.returnsByQuantileDf, portfolioWeightsDict)
        self.portfolioLongShortCumulativeReturnsDf = self.GetCumulativeReturnsDf(self.portfolioLongShortReturnsDf)