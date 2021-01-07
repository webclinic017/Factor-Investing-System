import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

class SymbolData:
    
    ''' Perform calculations '''
    
    def __init__(self, symbol):
        self.Symbol = symbol
        
        self.fundamentalDataDict = {}
        
        self.momentum = None
        self.volatility = None
        self.skewness = None
        self.kurt = None
        self.positionVsHL = None
        self.meanOvernightReturns = None
    
    def CalculateFactors(self, history, fundamentalDataBySymbolDict):
        
        self.fundamentalDataDict = fundamentalDataBySymbolDict[self.Symbol]
        self.momentum = self.CalculateMomentum(history)
        self.volatility = self.CalculateVolatility(history)
        #self.skewness = self.CalculateSkewness(history)
        #self.kurt = self.CalculateKurtosis(history)
        #self.distanceVsHL = self.CalculateDistanceVsHL(history)
        #self.meanOvernightReturns = self.CalculateMeanOvernightReturns(history)
    
    def CalculateMomentum(self, history):
        
        closePrices = history.loc[self.Symbol]['close']
        momentum = (closePrices[-1] / closePrices[-252]) - 1
        
        return momentum
        
    def CalculateVolatility(self, history):
        
        closePrices = history.loc[self.Symbol]['close']
        returns = closePrices.pct_change().dropna()
        volatility = np.nanstd(returns, axis = 0)
        
        return volatility
        
    def CalculateSkewness(self, history):
        
        closePrices = history.loc[self.Symbol]['close']
        returns = closePrices.pct_change().dropna()
        skewness = skew(returns)
        
        return skewness
        
    def CalculateKurtosis(self, history):
        
        closePrices = history.loc[self.Symbol]['close']
        returns = closePrices.pct_change().dropna()
        kurt = kurtosis(returns)
        
        return kurt
        
    def CalculateDistanceVsHL(self, history):
        
        closePrices = history.loc[self.Symbol]['close']
        annualHigh = max(closePrices)
        annualLow = min(closePrices)
        distanceVsHL = (closePrices[-1] - annualLow) / (annualHigh - annualLow)
        
        return distanceVsHL
        
    def CalculateMeanOvernightReturns(self, history):
        
        overnnightReturns = (history.loc[self.Symbol]['open'] / history.loc[self.Symbol]['close'].shift(1)) - 1
        meanOvernightReturns = np.nanmean(overnnightReturns, axis = 0)
        return meanOvernightReturns
            
    @property
    def factorsList(self):
        technicalFactors = [self.momentum, self.volatility]
        fundamentalFactors = [float(key) * value for key, value in self.fundamentalDataDict.items()]
        
        if all(v is not None for v in technicalFactors):
            return technicalFactors + fundamentalFactors
        else:
            return None