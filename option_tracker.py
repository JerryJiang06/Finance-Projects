import yfinance as yf

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.stats import norm
from scipy.optimize import brentq

from datetime import datetime
import datetime as dt
import pytz



stock = "QS"
ticker = yf.Ticker(stock)
expiry = dt.datetime(2026, 1, 16, tzinfo=pytz.UTC)
sp = ticker.history(period="1d", interval="1m")["Close"].iloc[-1]
now = datetime.now(pytz.timezone("US/Eastern"))
time = (expiry-now).total_seconds()/(365*24*3600)
r=0.05



def bs_call_price(S, K, T, r, sigma): 
    """Black-Scholes call price"""
    if T <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def implied_volatility(price, S, K, T, r): #option price, share price, strike price, time, rate
    """Solve for implied volatility using Brent's method"""
    try:
        f = lambda sigma: bs_call_price(S, K, T, r, sigma) - price
        return brentq(f, 1e-6, 5.0, maxiter=500)
    except ValueError:
        return np.nan

def get_delta(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return float(round(scipy.stats.norm.cdf(d1),3))

def get_gamma(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return float(round(scipy.stats.norm.pdf(d1)/(sp*sigma*np.sqrt(time)),3))



def get_call_option_table(expiry): #returns a df of the call table with delta and gamma added in
    opt = ticker.option_chain(expiry)
    df = opt.calls 
    delta = []
    gamma = []
    for index, row in df.iterrows():
        strike = row["strike"]
        iv = row["impliedVolatility"]
        delta.append(get_delta(sp, strike, time, r, iv))
        gamma.append(get_gamma(sp, strike, time, r, iv))
    df["delta"] = delta
    df["gamma"] = gamma
    return df

call_table = get_call_option_table("2026-01-16")



def calculate_delta_gamma_old(df, strike): #dataframe and strike price, returns list of [delta, gamma] and plots
    l=[]
    for index, row in df.iterrows():
        if row["strike"]==strike:
            strike = row["strike"]
            iv = row["impliedVolatility"]
            l.append(float(get_delta(sp, strike, time, r, iv)))
            l.append(float(get_gamma(sp, strike, time, r, iv)))
            d = {"sharePrice":[], "optionPrice":[]}
            for i in range(-4,5,1): #plots price graph using second degree taylor approximation (not too accurate)
                interval = 0.2
                delta_sp = interval*i
                option_price = row["lastPrice"] + delta_sp * l[0] + 0.5 * l[1] * delta_sp * delta_sp
                d["sharePrice"].append(sp+delta_sp)
                d["optionPrice"].append(option_price)
            new_df = pd.DataFrame(d)
            new_df.plot(x = "sharePrice", y = "optionPrice", kind = "line")
            plt.title("$"+str(strike)+" "+stock+" call")
            plt.xlabel("stock price")
            plt.ylabel("option price")
            plt.show()
    return l

#print(calculate_delta_gamma_old(df,12))



def predict_option_prices(strike, last_price): #manually input option price and recalculates IV cuz yfinance is wonky
            l=[]    
            iv = implied_volatility(last_price, sp, strike, time, r)
            l.append(float(get_delta(sp, strike, time, r, iv)))
            l.append(float(get_gamma(sp, strike, time, r, iv)))
            l.append(iv)
            print(l)

            d = {"sharePrice":[], "optionPrice":[], "timeValue":[]}
            for i in range(-3,8,1): #plots price graph using more accuate calculation, assuming constant IV
                interval = 1
                delta_sp = interval*i
                option_price = bs_call_price(sp+delta_sp, strike, time, r, iv)
                d["sharePrice"].append(sp+delta_sp)
                d["optionPrice"].append(option_price)
                d["timeValue"].append(option_price-max(0, delta_sp+sp-strike))  
            new_df = pd.DataFrame(d)
            new_df.plot(x = "sharePrice", y = ["optionPrice", "timeValue"], kind = "line")
            plt.title("$"+str(strike)+" "+stock+" call, IV: "+str(round(iv, 2)))
            plt.xlabel("stock price")
            plt.ylabel("option price")
            plt.show()

#predict_option_prices(12, 1.42)



def predict_combination(strike, last_price, num_shares, num_calls): #manually input option price and recalculates IV cuz yfinance is wonky
            l=[]    
            iv = implied_volatility(last_price, sp, strike, time, r)
            l.append(float(get_delta(sp, strike, time, r, iv)))
            l.append(float(get_gamma(sp, strike, time, r, iv)))
            l.append(iv)
            print(l)

            d = {"sharePrice":[], "tradeValue":[]}
            for i in range(-3,8,1): #plots price graph using more accuate calculation, assuming constant IV
                interval = 1
                delta_sp = interval*i
                option_price = bs_call_price(sp+delta_sp, strike, time, r, iv)
                d["sharePrice"].append(sp+delta_sp)
                d["tradeValue"].append(option_price*num_calls*100+sp*num_shares)
            new_df = pd.DataFrame(d)
            new_df.plot(x = "sharePrice", y = "tradeValue", kind = "line")
            plt.title("$"+str(strike)+" "+stock+" call + stock positions, IV: "+str(round(iv, 2)))
            plt.xlabel("stock price")
            plt.ylabel("position value")
            plt.show()

#predict_combination(12, 1.42, 200, 1)



def find_best_strike(df, price_target): #returns a strike that maximizes return, or plot the thing
    d = {"strike":[], "return":[]}
    for index, row in df.iterrows():
         strike = row["strike"]
         price = row["lastPrice"]
         iv = row["impliedVolatility"]
         final_price = bs_call_price(price_target, strike, time/2, r, iv)
         d["strike"].append(strike)
         d["return"].append(final_price/price-1)
    df2 = pd.DataFrame(d)
    df2.plot(x="strike", y="return", kind="line")
    plt.xlabel("Strike Price")
    plt.ylabel("Expected Return (with uncertainty)")
    plt.show()

#find_best_strike(call_table, 25)



def find_best_strike2(df, price_target, std):
    d = {"strike":[], "return":[]}
    # grid of possible future prices around the target
    price_grid = np.linspace(price_target - 3*std, price_target + 3*std, 50)
    weights = norm.pdf(price_grid, loc=price_target, scale=std)
    weights /= weights.sum()  # normalize to sum=1
    
    for index, row in df.iterrows():
        strike = row["strike"]
        premium = row["lastPrice"]
        iv = row["impliedVolatility"]

        # Expected option price = weighted avg across distribution
        exp_price = 0
        for S, w in zip(price_grid, weights):
            call_val = bs_call_price(S, strike, time/2, r, iv)
            exp_price += w * call_val

        # Expected return relative to premium paid
        exp_return = exp_price / premium - 1

        d["strike"].append(strike)
        d["return"].append(exp_return)

    df2 = pd.DataFrame(d)
    df2.plot(x="strike", y="return", kind="line")
    plt.xlabel("Strike Price")
    plt.ylabel("Expected Return (with uncertainty)")
    plt.show()

find_best_strike2(call_table, 20, 5)