"""
A simple script to test the Yahoo finance API for a single ticker
"""

import sys

try:
    import yfinance as yf
except ImportError:
    print("yfinanace is not installed. Run pip install yfinance")
    sys.exit(1)

TICKER = "AAPL"

def main():
    print(f"Fetching {TICKER} from yahoo finance..")
    t = yf.Ticker(TICKER)
    hist = t.history(period = "5d", interval = "1d")
    if hist.empty:
        print("Connection failure, no data returned")
        sys.exit(1)
    last = hist["Close"].iloc[-1]
    last_date = hist.index[-1]
    print(f"Ticker {TICKER}")
    print(f"Last close: {last:.2f} (as of {last_date})")
    print("yahoo API OK")
    return 0

if __name__ == "__main__":
    sys.exit(main())
