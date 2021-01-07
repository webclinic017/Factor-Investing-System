import pandas as pd
from scipy.stats import zscore
from classSymbolData import SymbolData

def MakeCalculations(algorithm, symbols, lookback, resolution, fundamentalDataBySymbolDict):
    
    '''
    Description:
        Make required calculations using historical data for each symbol
    Args:
        symbols: The symbols to make calculations for
        lookback: Lookback period for historical data
        resolution: Resolution for historical data
        fundamentalDataBySymbolDict: Dictionary of symbols containing factors and the direction of the factor (for sorting)
    Return:
        calculations: Dictionary containing the calculations per symbol
    '''
    
    # store calculations
    calculations = {}

    if len(symbols) > 0:
        # get historical prices for new symbols
        history = GetHistory(algorithm, symbols,
                            lookbackPeriod = lookback,
                            resolution = resolution)
            
        for symbol in symbols:
            # if symbol has no historical data continue the loop
            if (symbol not in history.index
            or len(history.loc[symbol]['close']) < lookback
            or history.loc[symbol].get('close') is None
            or history.loc[symbol].get('close').isna().any()):
                algorithm.Log('no history found for: ' + str(symbol.Value))
                continue

            else:
                # add symbol to calculations
                calculations[symbol] = SymbolData(symbol)
                
                try:
                    calculations[symbol].CalculateFactors(history, fundamentalDataBySymbolDict)
                except Exception as e:
                    algorithm.Log('removing from calculations due to ' + str(e))
                    calculations.pop(symbol)
                    continue
                
    return calculations
    
def GetFundamentalDataDict(algorithm, securitiesData, module = 'universe'):
    
    ''' Create a dictionary of symbols and fundamental factors ready for sorting '''

    fundamentalDataBySymbolDict = {}
    
    # loop through data and get fundamental data
    for x in securitiesData:
        if module == 'alpha':
            if not x.Symbol in algorithm.ActiveSecurities.Keys:
                continue
            fundamental = algorithm.ActiveSecurities[x.Symbol].Fundamentals
        elif module == 'universe':
            fundamental = x
        else:
            raise ValueError('module argument must be either universe or alpha')
            
        # dictionary of symbols containing factors and the direction of the factor (1 for sorting descending and -1 for sorting ascending)
        fundamentalDataBySymbolDict[x.Symbol] = {
                                                    #fundamental.ValuationRatios.BookValuePerShare: 1,
                                                    #fundamental.FinancialStatements.BalanceSheet.TotalEquity.Value: -1,
                                                    #fundamental.OperationRatios.OperationMargin.Value: 1,
                                                    #fundamental.OperationRatios.ROE.Value: 1,
                                                    #fundamental.OperationRatios.TotalAssetsGrowth.Value: 1,
                                                    #fundamental.ValuationRatios.PERatio: 1
                                                }
                                                    
        # check validity of data
        if None in list(fundamentalDataBySymbolDict[x.Symbol].keys()):
            fundamentalDataBySymbolDict.pop(x.Symbol)
                                                    
    return fundamentalDataBySymbolDict
    
def GetLongShortLists(self, algorithm, calculations):
    
    ''' Create lists of long/short stocks '''
            
    # get factors
    factorsDict = { symbol: symbolData.factorsList for symbol, symbolData in calculations.items() if symbolData.factorsList is not None }
    factorsDf = pd.DataFrame.from_dict(factorsDict, orient = 'index')
    
    # normalize factor
    normFactorsDf = factorsDf.apply(zscore)
    normFactorsDf.columns = ['Factor_' + str(x + 1) for x in normFactorsDf.columns]
    
    # combine factors using equal weighting
    #normFactorsDf['combinedFactor'] = normFactorsDf.sum(axis = 1)
    normFactorsDf['combinedFactor'] = normFactorsDf['Factor_1'] * 1 + normFactorsDf['Factor_2'] * 1
        
    # sort descending
    sortedNormFactorsDf = normFactorsDf.sort_values(by = 'combinedFactor', ascending = False) # descending
    
    # create long/short lists
    positionsEachSide = int(self.maxNumberOfPositions / 2)
    longs = list(sortedNormFactorsDf[:positionsEachSide].index)
    shorts = list(sortedNormFactorsDf[-positionsEachSide:].index)
    shorts = [x for x in shorts if x not in longs]
    
    return longs, shorts

def GetHistory(algorithm, symbols, lookbackPeriod, resolution):
    
    ''' Pull historical data in batches '''
    
    total = len(symbols)
    batchsize = 50
    
    if total <= batchsize:
        history = algorithm.History(symbols, lookbackPeriod, resolution)
    else:
        history = algorithm.History(symbols[0:batchsize], lookbackPeriod, resolution)
        for i in range(batchsize, total + 1, batchsize):
            batch = symbols[i:(i + batchsize)]
            historyTemp = algorithm.History(batch, lookbackPeriod, resolution)
            history = pd.concat([history, historyTemp])
            
    return history
    
def UpdateBenchmarkValue(self, algorithm):
        
    ''' Simulate buy and hold the Benchmark '''
    
    if self.initBenchmarkPrice == 0:
        self.initBenchmarkCash = algorithm.Portfolio.Cash
        self.initBenchmarkPrice = algorithm.Benchmark.Evaluate(algorithm.Time)
        self.benchmarkValue = self.initBenchmarkCash
    else:
        currentBenchmarkPrice = algorithm.Benchmark.Evaluate(algorithm.Time)
        self.benchmarkValue = (currentBenchmarkPrice / self.initBenchmarkPrice) * self.initBenchmarkCash
        
def UpdatePlots(self, algorithm):
    
    ''' Update Portfolio Exposure and Drawdown plots '''
    
    # simulate buy and hold the benchmark and plot its daily value --------------
    UpdateBenchmarkValue(self, algorithm)
    algorithm.Plot('Strategy Equity', self.benchmark, self.benchmarkValue)

    # get current portfolio value
    currentTotalPortfolioValue = algorithm.Portfolio.TotalPortfolioValue
    
    # plot the daily total portfolio exposure % --------------------------------
    longHoldings = sum([x.HoldingsValue for x in algorithm.Portfolio.Values if x.IsLong])
    shortHoldings = sum([x.HoldingsValue for x in algorithm.Portfolio.Values if x.IsShort])
    totalHoldings = longHoldings + shortHoldings
    totalPortfolioExposure = (totalHoldings / currentTotalPortfolioValue) * 100
    algorithm.Plot('Chart Total Portfolio Exposure %', 'Daily Portfolio Exposure %', totalPortfolioExposure)
    
    # plot the daily number of longs and shorts --------------------------------
    nLongs = sum(x.IsLong for x in algorithm.Portfolio.Values)
    nShorts = sum(x.IsShort for x in algorithm.Portfolio.Values)
    algorithm.Plot('Chart Number Of Longs/Shorts', 'Daily N Longs', nLongs)
    algorithm.Plot('Chart Number Of Longs/Shorts', 'Daily N Shorts', nShorts)
    
    # plot the drawdown % from the most recent high ---------------------------
    if not self.portfolioValueHighInitialized:
        self.portfolioHigh = currentTotalPortfolioValue # set initial portfolio value
        self.portfolioValueHighInitialized = True
        
    # update trailing high value of the portfolio
    if self.portfolioValueHigh < currentTotalPortfolioValue:
        self.portfolioValueHigh = currentTotalPortfolioValue

    currentDrawdownPercent = ((float(currentTotalPortfolioValue) / float(self.portfolioValueHigh)) - 1.0) * 100
    algorithm.Plot('Chart Drawdown %', 'Drawdown %', currentDrawdownPercent)