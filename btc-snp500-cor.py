import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the ticker symbols
sp500_ticker = "^GSPC"
bitcoin_ticker = "BTC-USD"

# Define the date range
start_date = "2023-03-18"
end_date = "2025-03-18"

# Download data for S&P 500
print("Downloading S&P 500 data...")
sp500_data = yf.download(sp500_ticker, start=start_date, end=end_date, interval="1d", auto_adjust=True)
sp500_data = sp500_data[["Close"]].rename(columns={"Close": "S&P 500"})

# Download data for Bitcoin
print("Downloading Bitcoin data...")
bitcoin_data = yf.download(bitcoin_ticker, start=start_date, end=end_date, interval="1d", auto_adjust=True)
bitcoin_data = bitcoin_data[["Close"]].rename(columns={"Close": "Bitcoin"})

# Merge data into a single DataFrame
print("Merging data...")
data = sp500_data.join(bitcoin_data, how="inner")
data["S&P 500 Return"] = data["S&P 500"].pct_change()
data["Bitcoin Return"] = data["Bitcoin"].pct_change()

# Create lagged returns for S&P 500
lags = range(1, 6)  # Test for 1 to 5-day lags
for lag in lags:
    data[f"S&P 500 Lag {lag}"] = data["S&P 500 Return"].shift(lag)

# Ensure data has no NaN values before correlation calculations
data = data.dropna()

# Calculate lagged correlations
print("Calculating lagged correlations...")
lagged_correlations = {}
for lag in lags:
    correlation = data[f"S&P 500 Lag {lag}"].corr(data["Bitcoin Return"])
    lagged_correlations[lag] = correlation
    print(f"Lag {lag} days: Correlation = {correlation:.4f}")

# Find the lag with the highest absolute correlation
best_lag = max(lagged_correlations, key=lambda x: abs(lagged_correlations[x]))
print(f"\nBest lag: {best_lag} days with correlation {lagged_correlations[best_lag]:.4f}")

# Plot the data
plt.figure(figsize=(12, 6))
plt.plot(data.index, data["S&P 500"], label="S&P 500", color="blue")
plt.plot(data.index, data["Bitcoin"], label="Bitcoin", color="orange")
plt.title("S&P 500 and Bitcoin Prices")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid()
plt.show()

"""Lag: 10

This indicates that the returns of the S&P 500 are shifted 10 days forward relative to Bitcoin returns.

In simpler terms, the calculation is checking if changes in the S&P 500 today affect Bitcoin 10 days later.

Correlation: -0.0321

The Pearson correlation coefficient measures the strength and direction of the linear relationship between two variables.

Range: -1 to 1.

-1: Perfect negative correlation (as one increases, the other decreases).

0: No correlation (no relationship between the two variables).

1: Perfect positive correlation (both move together in the same direction).

In this case: The correlation is -0.0321, which is very close to 0. This suggests no significant relationship between the S&P 500 returns (10 days earlier) and Bitcoin returns.

P-value: 0.8345

The p-value tests the statistical significance of the correlation:

A low p-value (typically < 0.05) means the correlation is statistically significant, i.e., the observed correlation is unlikely to have occurred by chance.

A high p-value (≥ 0.05) means the correlation is not statistically significant.

In this case: The p-value is 0.8345, which is very high, meaning the result is not significant. This implies that the observed correlation could easily have occurred due to random chance.
"""