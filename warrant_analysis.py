import yfinance as yf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq

import datetime as dt
import pytz



# ---------- Black-Scholes for call ----------
def bs_call_price(S, K, T, r, sigma): 
    """Black-Scholes call price"""
    if T <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def implied_volatility(price, S, K, T, r): #warrant price, share price, strike price, time, rate
    """Solve for implied volatility using Brent's method"""
    try:
        f = lambda sigma: bs_call_price(S, K, T, r, sigma) - price
        return brentq(f, 1e-6, 5.0, maxiter=500)
    except ValueError:
        return np.nan



# ---------- Inputs ----------
ticker_stock = "PGY"
ticker_warrant = "PGYWW"
start_date = "2024-01-01"
r = 0.05 
K = 138
expiry = dt.datetime(2027, 6, 30, tzinfo=pytz.UTC)



# ---------- Download 4H Data ----------
data_stock = yf.download(ticker_stock, start=start_date, interval="4h")["Close"]
data_warrant = yf.download(ticker_warrant, start=start_date, interval="4h")["Close"]

df = pd.concat([data_stock, data_warrant], axis=1)
df.columns = ["Stock", "Warrant"]
df.dropna(inplace=True)



# ---------- Calculate Implied Vol ----------
ivs = []
for date, row in df.iterrows():
    S = row["Stock"]
    W = row["Warrant"]
    
    # Time to expiry in YEARS including intraday fraction
    T = (expiry - date).total_seconds() / (365 * 24 * 3600)
    
    if W > 0 and S > 0 and T > 0:
        iv = implied_volatility(W, S, K, T, r)
    else:
        iv = np.nan
    ivs.append(iv)

df["ImpliedVol"] = ivs



# ---------- Plot ----------

fig, ax1 = plt.subplots()

ax1.set_xlabel("Date")
ax1.set_ylabel("Warrant Implied Volatility", color = "tab:red")
ax1.plot(df.index, df["ImpliedVol"], label="Warrant Implied Volatility", color = "tab:red")
ax1.tick_params(axis="y", labelcolor = "tab:red")

ax2 = ax1.twinx()
ax2.set_ylabel("Share Price", color = "tab:blue")
ax2.plot(df.index, df["Stock"], label="Share Price", color = "tab:blue")
ax2.plot(df.index, df["Warrant"], label="Warrant Price", color = "tab:green")
ax2.tick_params(axis="y", labelcolor = "tab:blue")

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()


#below is for analyzing correlation between stock and warrant price. we want to see a concave up shape
# df.plot.scatter(x="Stock", y="Warrant")
# plt.show()



#bonus function for creating payout table, good for direct comparison with warrant payout table
def create_price_table(times, prices, iv, strike): #time in years, stock price 
    ll=[]
    for time in times:
        l = []
        for price in prices:
            option_price = bs_call_price(price, strike, time, 0.05, iv)
            l.append(round(option_price,2))
        ll.append(l)
    df = pd.DataFrame(ll, columns=prices)
    return df

print(create_price_table([1.75, 1.5, 1.25, 1, 0.75, 0.5, 0.25, 0.01], [3.3, 10, 11, 12, 13, 14, 15, 16, 17, 18], 0.9, 11.5))