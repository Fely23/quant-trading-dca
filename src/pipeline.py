import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from typing import Dict, Any, List, Tuple
from src.validation import PurgedTimeSeriesSplit

logger = logging.getLogger(__name__)

def run_cv_experiment(
    X: pd.DataFrame, 
    close: pd.Series,
    forward_returns: pd.Series,
    n_splits: int = 5,
    horizon: int = 5,
    embargo_pct: float = 0.01,
    percentile: float = 0.70,
    tx_cost: float = 0.0005
) -> Dict[str, Any]:
    """
    Runs the purged cross-validation pipeline, training models on each fold
    without look-ahead bias and evaluating performance.
    
    Parameters:
        X (pd.DataFrame): Features.
        close (pd.Series): Close prices of S&P 500.
        forward_returns (pd.Series): Forward 5-day returns.
        n_splits (int): Number of splits.
        horizon (int): Prediction horizon.
        embargo_pct (float): Embargo percentage.
        percentile (float): Percentile threshold for target classification.
        tx_cost (float): Transaction cost per trade (slippage + commissions).
        
    Returns:
        Dict[str, Any]: Dictionary containing test predictions, metrics, and equity curves.
    """
    # Align data (remove rows where features or targets are NaN)
    # The last 'horizon' rows of forward_returns will be NaN (unknown future price)
    aligned_data = pd.concat([X, close, forward_returns], axis=1, keys=['features', 'close', 'forward_ret']).dropna()
    
    X_aligned = aligned_data['features']
    close_aligned = aligned_data['close']
    forward_ret_aligned = aligned_data['forward_ret']
    
    if isinstance(close_aligned, pd.DataFrame):
        close_aligned = close_aligned.iloc[:, 0]
    if isinstance(forward_ret_aligned, pd.DataFrame):
        forward_ret_aligned = forward_ret_aligned.iloc[:, 0]
        
    n_samples = len(X_aligned)
    logger.info(f"Aligned dataset has {n_samples} rows.")
    
    cv = PurgedTimeSeriesSplit(n_splits=n_splits, horizon=horizon, embargo_pct=embargo_pct)
    
    # Storage for out-of-fold results
    oof_predictions = np.zeros(n_samples, dtype=int)
    oof_probabilities = np.zeros(n_samples)
    oof_targets = np.zeros(n_samples, dtype=int)
    
    # We will track which indices were in a test fold
    test_indices_all = []
    
    fold_metrics = []
    
    # Compute 1-day returns for backtesting
    ret_1d = close_aligned.pct_change(1).shift(-1) # return from t to t+1, aligned at t
    
    for fold, (train_idx, test_idx) in enumerate(cv.split(X_aligned)):
        logger.info(f"--- Fold {fold+1}/{n_splits} ---")
        logger.info(f"Train size: {len(train_idx)}, Test size: {len(test_idx)}")
        
        # Get raw data
        X_train, X_test = X_aligned.iloc[train_idx], X_aligned.iloc[test_idx]
        r_train, r_test = forward_ret_aligned.iloc[train_idx], forward_ret_aligned.iloc[test_idx]
        
        # Compute threshold strictly on train set
        threshold = r_train.quantile(percentile)
        
        # Create targets
        y_train = (r_train > threshold).astype(int).values
        y_test = (r_test > threshold).astype(int).values
        
        oof_targets[test_idx] = y_test
        test_indices_all.extend(test_idx)
        
        # Check distribution
        train_pos_pct = np.mean(y_train)
        test_pos_pct = np.mean(y_test)
        logger.info(f"Train threshold: {threshold:.4f} (Positives: {train_pos_pct:.2%})")
        logger.info(f"Test Positives: {test_pos_pct:.2%}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Fit classifier
        # We use a Random Forest with restricted depth to prevent overfitting on 3 years of daily data
        model = RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=10, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Predict
        preds = model.predict(X_test_scaled)
        probs = model.predict_proba(X_test_scaled)[:, 1]
        
        oof_predictions[test_idx] = preds
        oof_probabilities[test_idx] = probs
        
        # Calculate fold classification metrics
        fold_auc = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else 0.5
        fold_metrics.append({
            'fold': fold + 1,
            'auc': fold_auc,
            'precision': np.mean(preds[y_test == 1] == 1) if np.sum(preds == 1) > 0 else 0.0,
            'accuracy': np.mean(preds == y_test)
        })
        logger.info(f"Fold {fold+1} AUC: {fold_auc:.4f}, Accuracy: {fold_metrics[-1]['accuracy']:.4f}")
        
    # Combine test indices and evaluate OOF
    test_indices_all = np.array(sorted(test_indices_all))
    
    oof_targets_test = oof_targets[test_indices_all]
    oof_predictions_test = oof_predictions[test_indices_all]
    oof_probabilities_test = oof_probabilities[test_indices_all]
    
    # Overall classification report
    report = classification_report(oof_targets_test, oof_predictions_test, output_dict=True)
    auc_total = roc_auc_score(oof_targets_test, oof_probabilities_test)
    cm = confusion_matrix(oof_targets_test, oof_predictions_test)
    
    # Backtest simulation
    # Align predictions with dates
    df_backtest = pd.DataFrame(index=X_aligned.index[test_indices_all])
    df_backtest['close'] = close_aligned.iloc[test_indices_all]
    df_backtest['ret_1d'] = ret_1d.iloc[test_indices_all]
    df_backtest['pred'] = oof_predictions[test_indices_all]
    df_backtest['prob'] = oof_probabilities[test_indices_all]
    
    # Backtest logic:
    # Position at day t (close) is determined by prediction at day t.
    # We hold this position from t to t+1. The return is position * ret_1d.
    # Note: ret_1d is the return from t to t+1, which is aligned at t.
    df_backtest['position'] = df_backtest['pred']
    
    # Trade execution costs:
    # Position change: |position_t - position_{t-1}|
    # Initial trade: compare position with 0.
    df_backtest['prev_position'] = df_backtest['position'].shift(1).fillna(0)
    df_backtest['trade'] = (df_backtest['position'] != df_backtest['prev_position']).astype(int)
    
    # Strategy daily returns before costs
    df_backtest['strat_ret_raw'] = df_backtest['position'] * df_backtest['ret_1d']
    # Strategy daily returns after costs
    df_backtest['strat_ret_net'] = df_backtest['strat_ret_raw'] - df_backtest['trade'] * tx_cost
    
    # Benchmark returns: Buy and Hold (position = 1 always)
    df_backtest['bh_ret'] = df_backtest['ret_1d']
    # Transaction cost for initiating B&H is 1 trade at start
    df_backtest.loc[df_backtest.index[0], 'bh_ret'] -= tx_cost
    
    # Fill any NaNs at the end because ret_1d shift(-1) leaves last row as NaN
    df_backtest = df_backtest.fillna(0)
    
    # Cumulative returns
    df_backtest['cum_strat_raw'] = (1 + df_backtest['strat_ret_raw']).cumprod() - 1
    df_backtest['cum_strat_net'] = (1 + df_backtest['strat_ret_net']).cumprod() - 1
    df_backtest['cum_bh'] = (1 + df_backtest['bh_ret']).cumprod() - 1
    
    # Calculate performance metrics
    metrics = calculate_backtest_metrics(df_backtest, tx_cost)
    
    return {
        'classification_report': report,
        'auc': auc_total,
        'confusion_matrix': cm,
        'fold_metrics': fold_metrics,
        'backtest_df': df_backtest,
        'backtest_metrics': metrics
    }

def calculate_backtest_metrics(df: pd.DataFrame, tx_cost: float) -> Dict[str, float]:
    """
    Computes key trading metrics (Sharpe, Sortino, Max Drawdown).
    """
    n_days = len(df)
    years = n_days / 252.0
    
    def get_stats(ret_col):
        daily_ret = df[ret_col]
        # Annualized return
        cum_ret = (1 + daily_ret).cumprod().iloc[-1] - 1 if len(daily_ret) > 0 else 0.0
        ann_ret = (1 + cum_ret) ** (1 / years) - 1 if years > 0 else 0.0
        
        # Annualized volatility
        ann_vol = daily_ret.std() * np.sqrt(252)
        
        # Sharpe Ratio (assumes risk-free rate = 0)
        sharpe = (daily_ret.mean() / (daily_ret.std() + 1e-10)) * np.sqrt(252)
        
        # Sortino Ratio
        downside_std = daily_ret[daily_ret < 0].std()
        sortino = (daily_ret.mean() / (downside_std + 1e-10)) * np.sqrt(252)
        
        # Max Drawdown
        cum_prod = (1 + daily_ret).cumprod()
        running_max = cum_prod.cummax()
        drawdown = (cum_prod - running_max) / running_max
        max_dd = drawdown.min()
        
        return {
            'cum_ret': cum_ret,
            'ann_ret': ann_ret,
            'ann_vol': ann_vol,
            'sharpe': sharpe,
            'sortino': sortino,
            'max_dd': max_dd
        }
        
    stats_net = get_stats('strat_ret_net')
    stats_raw = get_stats('strat_ret_raw')
    stats_bh = get_stats('bh_ret')
    
    # Calculate trades count
    total_trades = df['trade'].sum()
    
    # Probabilistic Sharpe Ratio (PSR) estimation against 0 Sharpe
    # Skewness and Kurtosis of daily net returns
    net_ret = df['strat_ret_net']
    skew = net_ret.skew()
    kurt = net_ret.kurtosis()
    # If standard deviation is 0, PSR is 0
    if net_ret.std() > 0:
        # Standard deviation of the estimated Sharpe Ratio
        # variance of Sharpe = (1 + (1 + 0.5 * skew^2) * Sharpe^2 / 252 - skew * Sharpe * kurtosis?)
        # Let's use Lopez de Prado's formula:
        # var(SR) = (1 - skew * SR + (kurt - 1)/4 * SR^2) / (T - 1)
        sr = stats_net['sharpe'] / np.sqrt(252) # daily Sharpe
        T = len(net_ret)
        var_sr = (1 - skew * sr + (kurt + 2)/4 * sr**2) / (T - 1)
        # Probabilistic Sharpe (against null hypothesis SR = 0)
        # Z = SR / sqrt(var_sr)
        # Convert daily variance back to annualized
        sr_ann = stats_net['sharpe']
        var_sr_ann = var_sr * 252
        from scipy.stats import norm
        z_stat = sr_ann / (np.sqrt(var_sr_ann) + 1e-10)
        psr = norm.cdf(z_stat)
    else:
        psr = 0.5
        
    return {
        'strat_cum_ret_net': stats_net['cum_ret'],
        'strat_ann_ret_net': stats_net['ann_ret'],
        'strat_ann_vol_net': stats_net['ann_vol'],
        'strat_sharpe_net': stats_net['sharpe'],
        'strat_sortino_net': stats_net['sortino'],
        'strat_max_dd_net': stats_net['max_dd'],
        
        'strat_cum_ret_raw': stats_raw['cum_ret'],
        'strat_sharpe_raw': stats_raw['sharpe'],
        
        'bh_cum_ret': stats_bh['cum_ret'],
        'bh_ann_ret': stats_bh['ann_ret'],
        'bh_ann_vol': stats_bh['ann_vol'],
        'bh_sharpe': stats_bh['sharpe'],
        'bh_max_dd': stats_bh['max_dd'],
        
        'total_trades': total_trades,
        'psr': psr
    }
