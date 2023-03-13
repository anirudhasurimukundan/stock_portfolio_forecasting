# Python script to perform Monte-Carlo simulation of Stock portfolio

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from pandas_datareader import pdr

# Import the data
def get_data(stocks, start, end):
    stockData = pdr.get_data(stocks, start, end)

