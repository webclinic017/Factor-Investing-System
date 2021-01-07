from clr import AddReference
AddReference("System")
AddReference("QuantConnect.Common")
AddReference("QuantConnect.Indicators")
AddReference("QuantConnect.Algorithm.Framework")

from QuantConnect.Data.UniverseSelection import *
from Selection.FundamentalUniverseSelectionModel import FundamentalUniverseSelectionModel

from HelperFunctions import GetFundamentalDataDict, MakeCalculations, GetLongShortLists, UpdatePlots
import pandas as pd
import numpy as np

class FactorModelUniverseSelectionModel(FundamentalUniverseSelectionModel):

    def __init__(self,
                benchmark = 'SPY',
                nStocks = 500,
                lookback = 252,
                maxNumberOfPositions = 20,
                rebalancingFunc = Expiry.EndOfMonth,
                filterFineData = True,
                universeSettings = None,
                securityInitializer = None):
        
        self.benchmark = benchmark
        
        self.nStocks = nStocks
        self.lookback = lookback
        self.maxNumberOfPositions = maxNumberOfPositions

        self.rebalancingFunc = rebalancingFunc
        self.nextRebalance = None
        
        self.initBenchmarkPrice = 0
        self.portfolioValueHigh = 0 # initialize portfolioValueHigh for drawdown calculation
        self.portfolioValueHighInitialized = False # initialize portfolioValueHighInitialized for drawdown calculation
        
        super().__init__(filterFineData, universeSettings, securityInitializer)

    def SelectCoarse(self, algorithm, coarse):
        
        ''' Perform Universe selection based on price and volume '''
        
        # update plots -----------------------------------------------------------------------------------------------
        UpdatePlots(self, algorithm)
        
        # rebalancing logic -------------------------------------------------------------------------------------------
        if self.nextRebalance is not None and algorithm.Time < self.nextRebalance:
            return Universe.Unchanged
        self.nextRebalance = self.rebalancingFunc(algorithm.Time)
        
        # get new coarse candidates -----------------------------------------------------------------------------------

        # filtered by price and select the top dollar volume stocks
        filteredCoarse = [x for x in coarse if x.HasFundamentalData]
        sortedDollarVolume = sorted(filteredCoarse, key = lambda x: x.DollarVolume, reverse = True)
        coarseSymbols = [x.Symbol for x in sortedDollarVolume][:(self.nStocks * 2)]
        
        return coarseSymbols
        
    def SelectFine(self, algorithm, fine):
        
        ''' Select securities based on fundamental factor modelling '''
        
        sortedMarketCap = sorted(fine, key = lambda x: x.MarketCap, reverse = True)[:self.nStocks]

        # generate dictionary with factors -----------------------------------------------------------------------------
        fundamentalDataBySymbolDict = GetFundamentalDataDict(algorithm, sortedMarketCap, module = 'universe')
                    
        # make calculations to create long/short lists -----------------------------------------------------------------
        fineSymbols = list(fundamentalDataBySymbolDict.keys())
        calculations = MakeCalculations(algorithm, fineSymbols, self.lookback, Resolution.Daily, fundamentalDataBySymbolDict)
        
        # get long/short lists of symbols
        longs, shorts = GetLongShortLists(self, algorithm, calculations)
        finalSymbols = longs + shorts

        return finalSymbols