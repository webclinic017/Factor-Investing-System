### PRODUCT INFORMATION --------------------------------------------------------------------------------
# Copyright Emilio Freire Bauzano
# Use entirely at your own risk.
# This algorithm contains open source code from other sources and no claim is being made to such code.
# Do not remove this copyright notice.
### ----------------------------------------------------------------------------------------------------

from FactorModelUniverseSelection import FactorModelUniverseSelectionModel
from LongShortAlphaCreation import LongShortAlphaCreationModel
from CustomEqualWeightingPortfolioConstruction import CustomEqualWeightingPortfolioConstructionModel

class LongShortEquityFrameworkAlgorithm(QCAlgorithmFramework):
    
    '''
    Trading Logic:
        Long-Short Equity Strategy using factor modelling
    Modules:
        Universe:
            - Final selection based on factor modelling:
                Combination of technical and fundamental factors
            - Long the Top N stocks
            - Short the Bottom N stocks
        Alpha: Creation of Up/Down Insights at the Market Open:
            - Up Insights (to go Long)
            - Down Insights (to go Short)
        Portfolio:
            Equal-Weighting Portfolio with monthly rebalancing
        Execution:
            Immediate Execution with Market Orders
        Risk:
            Null
    '''

    def Initialize(self):
        
        ### user-defined inputs ---------------------------------------------------------------------------

        self.SetStartDate(2018, 1, 1)   # set start date
        self.SetEndDate(2020, 10, 1)    # set end date
        self.SetCash(1000000)           # set strategy cash
        
        # select benchmark ticker
        benchmark = 'SPY'
        
        # date rule for rebalancing our portfolio by updating long-short positions based on factor values
        rebalancingFunc = Expiry.EndOfMonth
        
        # number of stocks to keep for factor modelling calculations
        nStocks = 100
        
        # number of positions to hold on each side (long/short)
        positionsOnEachSide = 20
        
        # lookback for historical data to calculate factors
        lookback = 252
        
        # select the leverage factor
        leverageFactor = 1
        
        ### --------------------------------------------------------------------------------------------------
        
        # calculate initialAllocationPerSecurity and maxNumberOfPositions
        initialAllocationPerSecurity = (1 / positionsOnEachSide) * leverageFactor
        maxNumberOfPositions = positionsOnEachSide * 2
        
        # set requested data resolution
        self.UniverseSettings.Resolution = Resolution.Hour
        # add leverage to new securities (this does not add leverage to current holdings in the account)
        leverageNeeded = max(1, maxNumberOfPositions * initialAllocationPerSecurity * leverageFactor)
        self.UniverseSettings.Leverage = leverageNeeded + 1
        
        # let's plot the series of daily total portfolio exposure %
        portfolioExposurePlot = Chart('Chart Total Portfolio Exposure %')
        portfolioExposurePlot.AddSeries(Series('Daily Portfolio Exposure %', SeriesType.Line, ''))
        self.AddChart(portfolioExposurePlot)
        
        # let's plot the series of daily number of open longs and shorts
        nLongShortPlot = Chart('Chart Number Of Longs/Shorts')
        nLongShortPlot.AddSeries(Series('Daily N Longs', SeriesType.Line, ''))
        nLongShortPlot.AddSeries(Series('Daily N Shorts', SeriesType.Line, ''))
        self.AddChart(nLongShortPlot)
        
        # let's plot the series of drawdown % from the most recent high
        drawdownPlot = Chart('Chart Drawdown %')
        drawdownPlot.AddSeries(Series('Drawdown %', SeriesType.Line, '%'))
        self.AddChart(drawdownPlot)
        
        # add benchmark
        self.SetBenchmark(benchmark)
        
        # select modules                               
        self.SetUniverseSelection(FactorModelUniverseSelectionModel(benchmark = benchmark,
                                                                    nStocks = nStocks,
                                                                    lookback = lookback,
                                                                    maxNumberOfPositions = maxNumberOfPositions,
                                                                    rebalancingFunc = rebalancingFunc))
        
        self.SetAlpha(LongShortAlphaCreationModel(maxNumberOfPositions = maxNumberOfPositions, lookback = lookback))
        
        self.SetPortfolioConstruction(CustomEqualWeightingPortfolioConstructionModel(initialAllocationPerSecurity = initialAllocationPerSecurity,
                                                                                    rebalancingFunc = rebalancingFunc))
                                                                                    
        self.SetExecution(ImmediateExecutionModel())
        
        self.SetRiskManagement(NullRiskManagementModel())