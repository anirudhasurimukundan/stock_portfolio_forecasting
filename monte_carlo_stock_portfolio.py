# Python script to perform Monte-Carlo simulation of Stock portfolio

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader.data as pdr
import yfinance as yf
yf.pdr_override()
import datetime as dt

# Import the data
def get_data(stocks, start, end):
    # Get the stocks data between start date and end date
    stockData = pdr.get_data_yahoo(stocks, start, end)

    # Get the closing prices of the stock at each date
    stockData = stockData['Close']

    # Get the "percentage change in stock value" = "returns"
    returns = stockData.pct_change()
    
    # Mean of the returns
    meanReturns = returns.mean()

    # Covariance matrix of the returns
    covMatrix = returns.cov()

    return meanReturns, covMatrix


stockList = ['CBA', 'BHP', 'TLS', 'NAB', 'WBC', 'STO']
stocks = [stock + '.AX' for stock in stockList] # Add .AX suffix for yahoo data
enddate = dt.datetime.now()
startdate = enddate - dt.timedelta(days=300)
meanReturns, covMatrix = get_data(stocks, startdate, enddate)

# Define weights
weights = np.random.random(len(meanReturns))

# Normalize the weights
weights /= np.sum(weights)

