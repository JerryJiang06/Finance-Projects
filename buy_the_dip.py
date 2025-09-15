import yfinance as yf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm



def backfill_leveraged(df: pd.DataFrame, base_col: str, lev_col: str, leverage: int):
    df = df.copy()
    base_returns = df[base_col].pct_change()
    lev_returns = leverage * base_returns

    first_valid = df[lev_col].first_valid_index()
    if first_valid is None:
        raise ValueError("No valid data in leveraged ETF column to anchor reconstruction.")

    anchor_price = df.loc[first_valid, lev_col]

    # Rebuild synthetic price series
    synthetic_prices = (1 + lev_returns).cumprod() * anchor_price / ((1 + lev_returns.loc[:first_valid]).cumprod().iloc[-1])

    # Fill in missing values with synthetic
    df[lev_col] = df[lev_col].combine_first(synthetic_prices)

    return df



etfs = ["SPY", "SSO", "UPRO", "^VIX"]
#data: pd.DataFrame = yf.download(etfs, start="2006-06-21")["Close"] #2009-06-25 2006-06-21
data = pd.read_csv("etf_06_06_21.csv", index_col=0, parse_dates = True)
#data = backfill_leveraged(data, "SPY", "UPRO", 3)
#data.to_csv("etf_06_06_21.csv")



def rolling_drawdown(prices): 
    roll_max = prices.cummax()
    drawdown = prices / roll_max - 1
    return drawdown

def forward_returns_etf(prices, buy_index, hold_days):
    if buy_index + hold_days < len(prices):
        return round(prices.iloc[buy_index + hold_days] / prices.iloc[buy_index] - 1,3)
    return np.nan

def forward_returns_leap(prices, buy_index, hold_days, leverage, IV, r):
    if buy_index + hold_days < len(prices):
        share_price_i = prices.iloc[buy_index]
        share_price_f = prices.iloc[buy_index+hold_days]
        strike_price = share_price_i - share_price_i / leverage
        option_price_i = bs_call_price(share_price_i, strike_price, 2, r, IV) #buying a 2 year LEAP and selling it after a year
        option_price_f = bs_call_price(share_price_f, strike_price, 1-hold_days/252, r, IV) 
        return round(option_price_f/option_price_i-1,3)
    return np.nan

def time_to_new_high(prices, buy_index):
    buy_price = prices.iloc[buy_index]
    for i in range(buy_index, len(prices)):
        if prices.iloc[i] > buy_price:
            return i - buy_index
    return None  # never recovered in dataset

def ATH(prices, current_index):
    roll_max = prices.cummax()
    if current_index == 0:
        return True  # first day is always ATH
    return roll_max.iloc[current_index] > roll_max.iloc[current_index - 1]



def dip_backtest_etf(prices, drawdown_threshold, vix_threshold, hold_days=[21, 63, 126, 252]): #prices is list of leveraged etfs
    d = drawdown_threshold
    drawdowns = rolling_drawdown(data["SPY"])
    results = []
    for i in range(len(prices)):
        if ATH(data["SPY"], i): #resets drawdown threshold
            drawdown_threshold = d
        if drawdowns.iloc[i] <= drawdown_threshold and data["^VIX"].iloc[i]>vix_threshold:
            drawdown_threshold = drawdowns.iloc[i] * 1.1 #update threshold to current drawdown
            fwd_ret = []
            for n in hold_days:
                fwd_ret.append(forward_returns_etf(prices, i, n))
            rec_time = time_to_new_high(prices, i)
            results.append({
                "Date": data["SPY"].index[i],
                "Drawdown": drawdowns.iloc[i],
                "1M": fwd_ret[0], "3M": fwd_ret[1], "6M": fwd_ret[2], "12M": fwd_ret[3],
                "RecoveryTime": rec_time
            })
    return pd.DataFrame(results)



def bs_call_price(S, K, T, r, sigma):
    if T <= 0: 
        return max(S-K, 0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def dip_backtest_leap(leverage, drawdown_threshold, vix_threshold, hold_days=[21, 63, 126, 252]): #prices is list of leveraged etfs
    d = drawdown_threshold
    drawdowns = rolling_drawdown(data["SPY"])
    results = []
    for i in range(len(data["SPY"])): #SPY price history
        if ATH(data["SPY"], i): #resets drawdown threshold
            drawdown_threshold = d
        if drawdowns.iloc[i] <= drawdown_threshold and data["^VIX"].iloc[i]>vix_threshold:
            drawdown_threshold = drawdowns.iloc[i] * 1.1 #update threshold to current drawdown
            fwd_ret = []
            for n in hold_days:
                fwd_ret.append(forward_returns_leap(data["SPY"], i, n, leverage, 0.2, 0.03))
            rec_time = time_to_new_high(data["SPY"], i) #approximate for LEAPs
            results.append({
                "Date": data["SPY"].index[i],
                "Drawdown": drawdowns.iloc[i],
                "1M": fwd_ret[0], "3M": fwd_ret[1], "6M": fwd_ret[2], "12M": fwd_ret[3],
                "RecoveryTime": rec_time
            })
    return pd.DataFrame(results)



def get_stats(data,x):
    #what I want for each time interval: generate % success for buying dip, average return, stdev, worst return
    m = data[x].mean()
    std = data[x].std()
    p = max(1, len((data[x])[data[x]>0]))
    n = len((data[x])[data[x]<=0])
    mi = data[x].min()
    return [m, std, round(p/(n+p), 3), mi] #mean return, standard deviation, probability, worst return


# print(dip_backtest_etf(data["SSO"], -0.2, 30))
# print("---------")
# print(dip_backtest_leap(2, -0.2, 30))


intervals = ["1M", "3M", "6M", "12M"]

dfs = []

for i in range(3):
    df = dip_backtest_etf(data[etfs[i]], -0.1-0.05*i, 30) #increase drawdown threshold for higher leverage buys
    dfs.append(df)

    # print(len(df))
    # d2 = {}
    # for n in intervals:
    #     d2[n] = get_stats(df, n)
    # df2 = pd.DataFrame(d2)
    # print(etfs[i])
    # print(df2)

for i in range(2):
    df = dip_backtest_leap(2+i, -0.1-0.05*i, 30)
    dfs.append(df)

    # print(len(df))
    # d2 = {}
    # for n in intervals:
    #     d2[n] = get_stats(df, n)
    # df2 = pd.DataFrame(d2)
    # print(str(i+1)+"x leverage")
    # print(df2)



def plot_returns(dfs): #plot bar graphs showing return distribution for each method
    n1 = len(dfs)
    n2 = len(intervals)
    fig, axs = plt.subplots(n1, n2)
    names = ["SPY", "SSO", "UPRO", "2x lev", "3x lev"]
    for i in range(n1):
        for j in range(n2):
            etf = dfs[i][intervals[j]]
            etf.hist(bins = [-0.6+0.1*i for i in range(40)], ax = axs[i][j], edgecolor = "black")
            axs[i][j].set_title(names[i] + " " + intervals[j] + " return")
    plt.show()

plot_returns(dfs)


