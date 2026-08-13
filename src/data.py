import logging
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def download_sp500_data(years: int = 3) -> pd.DataFrame:
    """
    Downloads historical daily data for S&P 500 (^GSPC).
    
    Parameters:
        years (int): Number of years of history to download.
        
    Returns:
        pd.DataFrame: S&P 500 daily data.
    """
    ticker = "^GSPC"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    logger.info(f"Downloading {ticker} data from {start_str} to {end_str}...")
    try:
        df = yf.download(ticker, start=start_str, end=end_str)
        if df.empty:
            raise ValueError("Downloaded DataFrame is empty. Check internet connection or ticker.")
        logger.info(f"Successfully downloaded {len(df)} rows.")
        return df
    except Exception as e:
        logger.error(f"Error downloading data from yfinance: {e}")
        raise

def compute_forward_returns(df: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """
    Computes the forward returns for a given horizon.
    
    Parameters:
        df (pd.DataFrame): Dataframe with 'Adj Close' column.
        horizon (int): Prediction horizon in business days.
        
    Returns:
        pd.Series: Forward returns series.
    """
    if 'Adj Close' not in df.columns:
        # yfinance can return multi-index or single index depending on download shape.
        # Let's ensure we get the column properly.
        if ('Adj Close', '^GSPC') in df.columns:
            adj_close = df[('Adj Close', '^GSPC')]
        elif 'Close' in df.columns:
            adj_close = df['Close']
        else:
            raise KeyError("Could not find 'Adj Close' or 'Close' in DataFrame columns.")
    else:
        adj_close = df['Adj Close']
        
    # Squeeze if series is still a DataFrame (due to multi-index)
    if isinstance(adj_close, pd.DataFrame):
        adj_close = adj_close.iloc[:, 0]
        
    # R_t = Price_{t+H} / Price_t - 1
    # We shift by -horizon to align R_t with features at time t.
    forward_prices = adj_close.shift(-horizon)
    forward_returns = (forward_prices / adj_close) - 1.0
    return forward_returns
