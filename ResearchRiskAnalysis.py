import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import statsmodels.api as sm
import pandas as pd
import numpy as np

import seaborn as sns
sns.set_style('darkgrid')
pd.plotting.register_matplotlib_converters()

from statsmodels.regression.rolling import RollingOLS
from io import StringIO

class RiskAnalysis:
    
    def __init__(self, qb):
        
        # get Fama-French and industry factors
        industryFactorsUrl = 'https://www.dropbox.com/s/24bjtztzglo3eyf/12_Industry_Portfolios_Daily.CSV?dl=1'
        ffFiveFactorsUrl = 'https://www.dropbox.com/s/88m1nohi597et20/F-F_Research_Data_5_Factors_2x3_daily.CSV?dl=1'
        self.industryFactorsDf = self.GetExternalFactorsDf(qb, industryFactorsUrl)
        self.ffFiveFactorsDf = self.GetExternalFactorsDf(qb, ffFiveFactorsUrl)
        
    def GetExternalFactorsDf(self, qb, url):
        
        '''
        Description:
            Download a DataFrame with data from external sources
        Args:
            qb: QuantBook
            url: URL for the data source
        Returns:
            SingleIndex Dataframe
        '''
    
        strFile = qb.Download(url)
        df = pd.read_csv(StringIO(strFile), sep = ',')
        df['Date'] = pd.to_datetime(df['Date'], format = '%Y%m%d')
        df.set_index('Date', inplace = True)
        df = df.div(100)
        df.drop('RF', axis = 1, errors = 'ignore', inplace = True)

        return df
    
    def GetCombinedReturnsDf(self, returnsDf, externalFactorsDf = None):
        
        '''
        Description:
            Merge two DataFrames
        Args:
            returnsDf: SingleIndex Dataframe with returns from our strategy
            externalFactorsDf: SingleIndex Dataframe with returns from external factors
        Returns:
            SingleIndex Dataframe with returns
        '''
        
        # if no externalFactorsDf is provided, use the default Fama-French Five Factors
        if externalFactorsDf is None:
            externalFactorsDf = self.ffFiveFactorsDf
        
        # merge returnsDf with externalFactorsDf
        combinedReturnsDf = pd.merge(returnsDf, externalFactorsDf, left_index = True, right_index = True)
        
        return combinedReturnsDf
    
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
        
    def RunRegression(self, returnsDf, dependentColumn = 'Strategy'):
        
        '''
        Description:
            Run Regression using the dependentColumn against the rest of the columns
        Args:
            returnsDf: SingleIndex Dataframe with returns
            dependentColumn: Name for the column to be used as dependent variable
        Returns:
            Summary of the model
        '''
        
        # create variables
        Y = returnsDf[[dependentColumn]]
        X = returnsDf[[x for x in returnsDf.columns if x != dependentColumn]]
        # adding a constant
        X = sm.add_constant(X)

        # fit regression model
        model = sm.OLS(Y, X).fit()
        
        # show summary from the model
        print(model.summary())
        
        return model
        
    def RunRollingRegression(self, returnsDf, dependentColumn = 'Strategy', lookback = 126):
        
        '''
        Description:
            Run Rolling Regression using the dependentColumn against the rest of the columns
        Args:
            returnsDf: SingleIndex Dataframe with returns
            dependentColumn: Name for the column to be used as dependent variable
            lookback: Number of observations for the lookback window
        Returns:
            Rolling Regression Model
        '''
        
        endog = returnsDf[[dependentColumn]]
        exogVariables = [x for x in returnsDf.columns if x != dependentColumn]
        exog = sm.add_constant(returnsDf[exogVariables])
        rollingModel = RollingOLS(endog, exog, window = lookback).fit()
        
        return rollingModel

    # ploting functions -----------------------------------------------------------------------------------------
        
    def PlotCumulativeReturns(self, returnsDf):

        '''
        Description:
            Plot cumulative returns
        Args:
            returnsDf: SingleIndex Dataframe with returns
        Returns:
            Plot cumulative returns
        '''
        
        # calculate cumulative returns
        cumulativeReturnsDf = self.GetCumulativeReturnsDf(returnsDf)
        # take logarithm for better visualization
        cumulativeReturnsDf = np.log(1 + cumulativeReturnsDf)
        
        # prepare plot
        fig, ax = plt.subplots(figsize = (12, 5))
        
        # plot portfolio
        colPortfolio = cumulativeReturnsDf.iloc[:, [0]].columns[0]
        ax.plot(cumulativeReturnsDf[colPortfolio], color = 'black', linewidth = 2)
            
        if len(cumulativeReturnsDf.columns) > 1:
            colFactors = cumulativeReturnsDf.iloc[:, 1:].columns
            # plot factors
            ax.plot(cumulativeReturnsDf[colFactors], alpha = 0.5)
            
        # formatting
        ax.axhline(y = 0, color = 'black', linestyle = '--', linewidth = 0.5)
        ax.set_title('Cumulative Log-Returns', fontdict = {'fontsize': 15})
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.legend(cumulativeReturnsDf.columns, loc = 'best')
        
        plt.show()
        
    def PlotRegressionModel(self, returnsDf, dependentColumn = 'Strategy'):
        
        '''
        Description:
            Run Regression and plot partial regression
        Args:
            returnsDf: SingleIndex Dataframe with returns
            dependentColumn: Name for the column to be used as dependent variable
        Returns:
            Summary of the regression model and partial regression plots
        '''
        
        # run regression
        model = self.RunRegression(returnsDf, dependentColumn)

        # plot partial regression
        exogVariables = [x for x in returnsDf.columns if x != dependentColumn]
        figsize = (10, len(exogVariables) * 2)
        fig = plt.figure(figsize = figsize)
        fig = sm.graphics.plot_partregress_grid(model, fig = fig)
        
        plt.show()
        
    def PlotRollingRegressionCoefficients(self, returnsDf, dependentColumn = 'Strategy', lookback = 126):
        
        '''
        Description:
            Run Rolling Regression and plot the time series of estimated coefficients for each predictor
        Args:
            returnsDf: SingleIndex Dataframe with returns
            dependentColumn: Name for the column to be used as dependent variable
            lookback: Number of observations for the lookback window
        Returns:
            Plot of time series of estimated coefficients for each predictor
        '''
        
        # run rolling regression
        rollingModel = self.RunRollingRegression(returnsDf, dependentColumn, lookback)
        exogVariables = [x for x in returnsDf.columns if x != dependentColumn]
        
        # plot
        figsize = (10, len(exogVariables) * 3)
        fig = rollingModel.plot_recursive_coefficient(variables = exogVariables, figsize = figsize)
        
        plt.show()
        
    def PlotBoxPlotRollingFactorExposure(self, returnsDf, dependentColumn = 'Strategy', lookback = 126):
        
        '''
        Description:
            Run Rolling Regression and make a box plot with the distributions of the estimated coefficients
        Args:
            returnsDf: SingleIndex Dataframe with returns
            dependentColumn: Name for the column to be used as dependent variable
            lookback: Number of observations for the lookback window
        Returns:
            Box plot with distributions of estimated coefficients during the rolling regression
        '''
        
        # run rolling regression
        rollingModel = self.RunRollingRegression(returnsDf, dependentColumn, lookback)
        
        fig, ax = plt.subplots(figsize = (10, 8))
        ax = sns.boxplot(data = rollingModel.params.dropna().drop('const', axis = 1), 
                        width = 0.5,
                        palette = "colorblind",
                        orient = 'h')
        ax.axvline(x = 0, color = 'black', linestyle = '--', linewidth = 0.5)
        ax.set_title('Distribution of Risk Factor Rolling Exposures', fontdict = {'fontsize': 15})
        
        plt.show()
    
    # run full risk analysis --------------------------------------------------------------------------------------
    
    def RunRiskAnalysis(self, returnsDf, externalFactorsDf = None, dependentColumn = 'Strategy', lookback = 126):
        
        # if no externalFactorsDf is provided, use the default Fama-French Five Factors
        if externalFactorsDf is None:
            externalFactorsDf = self.ffFiveFactorsDf
        
        # merge returnsDf with externalFactorsDf
        combinedReturnsDf = pd.merge(returnsDf, externalFactorsDf, left_index = True, right_index = True)
        
        # plot
        self.PlotCumulativeReturns(combinedReturnsDf)
        print('---------------------------------------------------------------------------------------------')
        print('---- Regression Analysis --------------------------------------------------------------------')
        print('---------------------------------------------------------------------------------------------')
        self.PlotRegressionModel(combinedReturnsDf, dependentColumn)
        print('---------------------------------------------------------------------------------------------')
        print('---- Rolling Regression Analysis (Rolling Coefficients) -------------------------------------')
        print('---------------------------------------------------------------------------------------------')
        self.PlotRollingRegressionCoefficients(combinedReturnsDf, dependentColumn, lookback)
        self.PlotBoxPlotRollingFactorExposure(combinedReturnsDf, dependentColumn, lookback)