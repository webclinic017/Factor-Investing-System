from clr import AddReference
AddReference("QuantConnect.Common")
AddReference("QuantConnect.Algorithm")
AddReference("QuantConnect.Algorithm.Framework")

from QuantConnect import *
from QuantConnect.Algorithm import *
from QuantConnect.Algorithm.Framework import *
from QuantConnect.Algorithm.Framework.Alphas import AlphaModel, Insight, InsightType, InsightDirection

from HelperFunctions import GetFundamentalDataDict, MakeCalculations, GetLongShortLists
from datetime import timedelta, datetime
import pandas as pd
import numpy as np

class LongShortAlphaCreationModel(AlphaModel):

    def __init__(self, maxNumberOfPositions = 10, lookback = 252):
        
        self.maxNumberOfPositions = maxNumberOfPositions
        self.lookback = lookback
        
        self.securities = []
        self.day = 0

    def Update(self, algorithm, data):
        
        insights = [] # list to store the new insights to be created
        
        if algorithm.Time.day != self.day and algorithm.Time.hour > 9:
            for symbol, direction in self.insightsDict.items():
                if data.ContainsKey(symbol) and symbol in algorithm.ActiveSecurities.Keys and algorithm.ActiveSecurities[symbol].Price > 0:
                    insights.append(Insight.Price(symbol, Expiry.EndOfDay, direction))
                    
            self.day = algorithm.Time.day

        return insights
            
    def OnSecuritiesChanged(self, algorithm, changes):
        
        '''
        Description:
            Event fired each time the we add/remove securities from the data feed
        Args:
            algorithm: The algorithm instance that experienced the change in securities
            changes: The security additions and removals from the algorithm
        '''
            
        # check current securities in our self.securities list
        securitiesList = [x.Symbol.Value for x in self.securities]
        algorithm.Log('(Alpha module) securities in self.securities before OnSecuritiesChanged: ' + str(securitiesList))
            
        # add new securities
        addedSecurities = [x for x in changes.AddedSecurities if x not in self.securities]
        for added in addedSecurities:
            self.securities.append(added)
            
        newSecuritiesList = [x.Symbol.Value for x in addedSecurities]
        algorithm.Log('(Alpha module) new securities added to self.securities:'+ str(newSecuritiesList))

        # remove securities
        removedSecurities = [x for x in changes.RemovedSecurities if x in self.securities]
        for removed in removedSecurities:
            self.securities.remove(removed)
                
        removedList = [x.Symbol.Value for x in removedSecurities]
        algorithm.Log('(Alpha module) securities removed from self.securities: ' + str(removedList))
        
        # print the final securities in self.securities for today
        securitiesList = [x.Symbol.Value for x in self.securities]
        algorithm.Log('(Alpha module) final securities in self.securities after OnSecuritiesChanged: ' + str(securitiesList))
        
        # generate dictionary with factors -------------------------------------------------------
        fundamentalDataBySymbolDict = GetFundamentalDataDict(algorithm, self.securities, module = 'alpha')
                    
        # make calculations to create long/short lists -------------------------------------------
        currentSymbols = list(fundamentalDataBySymbolDict.keys())
        calculations = MakeCalculations(algorithm, currentSymbols, self.lookback, Resolution.Daily, fundamentalDataBySymbolDict)
        
        # get long/short lists
        longs, shorts = GetLongShortLists(self, algorithm, calculations)
        finalSymbols = longs + shorts
        
        # update the insightsDict dictionary with long/short signals
        self.insightsDict = {}
        for symbol in finalSymbols:
            if symbol in longs:
                direction = 1
            else:
                direction = -1
                
            self.insightsDict[symbol] = direction