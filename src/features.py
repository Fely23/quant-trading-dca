import numpy as np
import pandas as pd

def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes technical indicators vectorially using Pandas/NumPy.
    
    Parameters:
        df (pd.DataFrame): Dataframe with OHLCV data.
        
    Returns:
        pd.DataFrame: Dataframe with added feature columns.
    """
    features = pd.DataFrame(index=df.index)
    
    if ('Adj Close', '^GSPC') in df.columns:
        close = df[('Adj Close', '^GSPC')]
    elif 'Adj Close' in df.columns:
        close = df['Adj Close']
    elif 'Close' in df.columns:
        close = df['Close']
    else:
        raise KeyError("Could not find Close price column.")
        
    # Squeeze if series is a DataFrame
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
        
    close = close.astype(float)
    
    # 1. Historical Returns (Lagged returns)
    features['ret_1d'] = close.pct_change(1)
    features['ret_5d'] = close.pct_change(5)
    features['ret_21d'] = close.pct_change(21)
    
    # 2. Historical Volatility (Standard deviation of daily log returns)
    log_ret = np.log(close / close.shift(1))
    features['vol_5d'] = log_ret.rolling(window=5).std() * np.sqrt(252)
    features['vol_21d'] = log_ret.rolling(window=21).std() * np.sqrt(252)
    
    # 3. Relative Strength Index (RSI) - 14 days
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # Use Wilder's EMA smoothing: alpha = 1/N
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    features['rsi_14'] = 100 - (100 / (1 + rs))
    
    # 4. MACD (Moving Average Convergence Divergence)
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    features['macd'] = ema_12 - ema_26
    features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
    features['macd_hist'] = features['macd'] - features['macd_signal']
    
    # 5. Bollinger Bands (20 days)
    bb_mean = close.rolling(window=20).mean()
    bb_std = close.rolling(window=20).std()
    bb_upper = bb_mean + 2 * bb_std
    bb_lower = bb_mean - 2 * bb_std
    
    features['bb_width'] = (bb_upper - bb_lower) / (bb_mean + 1e-10)
    features['bb_pctb'] = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)
    
    # 6. Moving Average Crossover Ratio
    features['ema_ratio'] = ema_12 / ema_26
    
    # Drop rows with NaN values resulting from rolling indicators (max window is 21)
    # We will let the pipeline handle alignment, but return full features here.
    return features
