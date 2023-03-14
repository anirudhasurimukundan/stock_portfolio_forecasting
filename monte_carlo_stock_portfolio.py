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


if __name__ == "__main__":
    stockList = ['CBA', 'BHP', 'TLS', 'NAB', 'WBC', 'STO']
    stocks = [stock + '.AX' for stock in stockList] # Add .AX suffix for yahoo data
    enddate = dt.datetime.now()
    startdate = enddate - dt.timedelta(days=300)
    meanReturns, covMatrix = get_data(stocks, startdate, enddate)

    # Define weights
    weights = np.random.random(len(meanReturns))

    # Normalize the weights
    weights /= np.sum(weights)

    #### Monte-Carlo method
    # Number of simulations
    num_monte_carlo_simulations = 100
    # Number of days
    T = 10

    # Define the mean matrix
    meanMatrix = np.full(shape=(T, len(weights)), fill_value=meanReturns)
    meanMatrix = meanMatrix.T # size = number of stocks \times number of days

    # Define the portfolio matrix as empty array
    portfolio_simulations = np.full(shape=(T, num_monte_carlo_simulations), fill_value=0.0)

    initial_portfolio_value = 10000.; # equivalent to $10000 or 10000 Rupees etc.,

    # Monte-Carlo loop
    # We compute daily returns assuming it follows a Multivariate Normal Distribution (MVN)
    # $R_t ~ MVN(\mu, \Sigma)$
    # where the $\Sigma$ is determined from the Cholesky decomposition using which
    # the lower triangular matrix L is determined as 
    # L $\in$ LL' = $\Sigma$
    # with
    # $R_t = \mu + L \times Z_t$
    # Daily returns = Mean matrix + dot_product(L, Z)
    # where
    # $Z_t ~ N(0, I)$, i.e., $Z_t$ are samples from a normal distribution with
    # I representing the identity matrix
    for m in range(0, num_monte_carlo_simulations):
        # sample the random variables
        Z = np.random.normal(size=(T, len(weights))) # number of days \times number of stocks
        L = np.linalg.cholesky(covMatrix) # number of stocks \times number of stocks
        dailyReturns = meanMatrix + np.inner(L, Z)
        
        # save the daily returns into portfolio simulations matrix
        portfolio_simulations[:, m] = np.cumprod(np.inner(weights, dailyReturns.T)+1)*initial_portfolio_value


    # Plot the simulated portfolio values
    plt.figure(1)
    plt.xlabel('Days')
    plt.ylabel('Portfolio value (currency unit)')
    plt.title('Monte-Carlo simulation of a stock portfolio')
    plt.plot(portfolio_simulations)
    plt.show()
