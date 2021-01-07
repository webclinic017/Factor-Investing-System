from clr import AddReference
AddReference("QuantConnect.Common")
AddReference("QuantConnect.Algorithm.Framework")

from QuantConnect import Resolution, Extensions
from QuantConnect.Algorithm.Framework.Alphas import *
from QuantConnect.Algorithm.Framework.Portfolio import *
from itertools import groupby
from datetime import datetime, timedelta

class CustomEqualWeightingPortfolioConstructionModel(PortfolioConstructionModel):
    
    '''
    Description:
        Provide a custom implementation of IPortfolioConstructionModel that gives equal weighting to all active securities
    Details:
        - The target percent holdings of each security is 1/N where N is the number of securities with active Up/Down insights
        - For InsightDirection.Up, long targets are returned
        - For InsightDirection.Down, short targets are returned
        - For InsightDirection.Flat, closing position targets are returned
    '''

    def __init__(self, initialAllocationPerSecurity = 0.1, rebalancingFunc = Expiry.EndOfMonth):
        
        '''
        Description:
            Initialize a new instance of CustomEqualWeightingPortfolioConstructionModel
        Args:
            initialAllocationPerSecurity: Portfolio exposure per security (as a % of total equity)
        '''
        
        # portfolio exposure per security (as a % of total equity)
        self.initialAllocationPerSecurity = initialAllocationPerSecurity
        self.rebalancingFunc = rebalancingFunc
        
        self.insightCollection = InsightCollection()
        self.removedSymbols = []
        
        self.nextRebalance = None

    def CreateTargets(self, algorithm, insights):

        '''
        Description:
            Create portfolio targets from the specified insights
        Args:
            algorithm: The algorithm instance
            insights: The insights to create portfolio targets from
        Returns:
            An enumerable of portfolio targets to be sent to the execution model
        '''

        targets = []
            
        if len(insights) == 0:
            return targets
        
        # apply rebalancing logic
        if self.nextRebalance is not None and algorithm.Time < self.nextRebalance and len(self.removedSymbols) == 0:
            return targets
        self.nextRebalance = self.rebalancingFunc(algorithm.Time)
        
        # here we get the new insights and add them to our insight collection
        for insight in insights:
            self.insightCollection.Add(insight)
            
        # create flatten target for each security that was removed from the universe
        if len(self.removedSymbols) > 0:
            universeDeselectionTargets = [ PortfolioTarget(symbol, 0) for symbol in self.removedSymbols ]
            targets.extend(universeDeselectionTargets)
            algorithm.Log('(Portfolio module) liquidating: ' + str([x.Value for x in self.removedSymbols]))
            self.removedSymbols = []

        # get insight that have not expired of each symbol that is still in the universe
        activeInsights = self.insightCollection.GetActiveInsights(algorithm.UtcTime)

        # get the last generated active insight for each symbol
        lastActiveInsights = []
        for symbol, g in groupby(activeInsights, lambda x: x.Symbol):
            lastActiveInsights.append(sorted(g, key = lambda x: x.GeneratedTimeUtc)[-1])
        
        # determine target percent for the given insights
        for insight in lastActiveInsights:
            allocationPercent = self.initialAllocationPerSecurity * insight.Direction
            target = PortfolioTarget.Percent(algorithm, insight.Symbol, allocationPercent)
            targets.append(target)
            
        return targets
        
    def OnSecuritiesChanged(self, algorithm, changes):
        
        '''
        Description:
            Event fired each time the we add/remove securities from the data feed
        Args:
            algorithm: The algorithm instance that experienced the change in securities
            changes: The security additions and removals from the algorithm
        '''
        
        newRemovedSymbols = [x.Symbol for x in changes.RemovedSecurities if x.Symbol not in self.removedSymbols]
        
        # get removed symbol and invalidate them in the insight collection
        self.removedSymbols.extend(newRemovedSymbols)
        self.insightCollection.Clear(self.removedSymbols)
            
        removedList = [x.Value for x in self.removedSymbols]
        algorithm.Log('(Portfolio module) securities removed from Universe: ' + str(removedList))