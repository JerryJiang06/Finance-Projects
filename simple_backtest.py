import yfinance as yf

import pandas as pd
import numpy as np

def cagr(prices): 
    return float((prices.iloc[-1]/prices.iloc[0])**(252/len(prices))-1)

def annualized_vol(daily_returns): 
    return float(daily_returns.std() * np.sqrt(252))

def sharpe_ratio(daily_returns, rf=0.0):
    excess = daily_returns - rf/252
    return float((excess.mean()*252)/(daily_returns.std()*np.sqrt(252)))

def sortino_ratio(daily_returns, rf=0.0):
    excess = daily_returns - rf/252
    downside_std = excess[excess<0].std() * np.sqrt(252)
    return float((excess.mean()*252)/downside_std)

def max_drawdown(prices):
    roll_max = prices.cummax()
    drawdown = prices / roll_max - 1
    return float(drawdown.min())

def ulcer_index(prices):
    roll_max = prices.cummax()
    drawdown_squared = (prices / roll_max - 1)**2
    return float(np.sqrt(np.mean(drawdown_squared)))

def time_underwater(prices):
    peaks = prices.cummax()
    tuw = (prices<peaks).astype(int) #0s and 1s
    return int(tuw.sum()) #total days underwater

def var(returns, level = 0.05):
    var = np.percentile(returns, level*100)
    return float(var)

def cvar(returns, level = 0.05):
    var = np.percentile(returns, level*100)
    cvar = returns[returns <= var].mean()
    return float(cvar)



etfs = ["SPY", "SSO", "UPRO", "^VIX"]
#data: pd.DataFrame = yf.download(etfs, start="2009-06-25")["Close"] 
data = pd.read_csv("etf_09_06_25.csv", index_col=0, parse_dates = True)
daily_returns: pd.DataFrame = data.pct_change().dropna()
#data.to_csv("etf_09_06_25.csv")



di = {}
for etf in etfs:
    d = {}
    s1 = data[etf]
    s2 = daily_returns[etf]
    d["CAGR"] = cagr(s1)
    d["An Vol"] = annualized_vol(s2)
    d["Sharpe"] = sharpe_ratio(s2)
    d["Sortino"] = sortino_ratio(s2)
    d["Max Drawdown"] = max_drawdown(s1)
    d["Ulcer Index"] = ulcer_index(s1)
    d["Time Underwater"] = time_underwater(s1)
    d["VaR"] = var(s2)
    d["CVaR"] = cvar(s2)
    di[etf] = d

df = pd.DataFrame(di)
print(df)