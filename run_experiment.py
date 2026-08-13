import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
from src.data import download_sp500_data, compute_forward_returns
from src.features import compute_technical_indicators
from src.pipeline import run_cv_experiment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Define parameters
    years = 3
    horizon = 5
    n_splits = 5
    embargo_pct = 0.01
    percentile = 0.70
    tx_cost = 0.0005 # 5 bps
    
    # 1. Download Data
    df = download_sp500_data(years=years)
    
    # yfinance output may contain multi-index headers depending on version.
    # We retrieve the Close/Adj Close properly in our data/features modules.
    
    # 2. Compute Target (Forward Returns)
    logger.info("Computing 5-day forward returns...")
    forward_returns = compute_forward_returns(df, horizon=horizon)
    
    # 3. Compute Features
    logger.info("Computing technical indicators (features)...")
    X = compute_technical_indicators(df)
    
    # Retrieve Close Series for the backtest
    if ('Adj Close', '^GSPC') in df.columns:
        close = df[('Adj Close', '^GSPC')]
    elif 'Adj Close' in df.columns:
        close = df['Adj Close']
    elif 'Close' in df.columns:
        close = df['Close']
    else:
        close = df.iloc[:, 0]
        
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
        
    # 4. Run Cross-Validation Experiment
    logger.info("Starting purged cross-validation experiment...")
    results = run_cv_experiment(
        X=X,
        close=close,
        forward_returns=forward_returns,
        n_splits=n_splits,
        horizon=horizon,
        embargo_pct=embargo_pct,
        percentile=percentile,
        tx_cost=tx_cost
    )
    
    # 5. Print Results
    logger.info("Experiment completed. Displaying metrics...")
    
    print("\n" + "="*50)
    print("CLASSIFICATION METRICS (Out-of-Fold Test Set)")
    print("="*50)
    print(f"ROC-AUC Score: {results['auc']:.4f}")
    print("\nConfusion Matrix:")
    print(results['confusion_matrix'])
    
    report = results['classification_report']
    print("\nClassification Report:")
    for label in ['0', '1', '0.0', '1.0']:
        if label in report:
            print(f"Class {label} -> Precision: {report[label]['precision']:.4f}, Recall: {report[label]['recall']:.4f}, F1-Score: {report[label]['f1-score']:.4f}")
    print(f"Overall Accuracy: {report['accuracy']:.4f}")
    
    print("\n" + "="*50)
    print("BACKTESTING METRICS")
    print("="*50)
    bm = results['backtest_metrics']
    print(f"Total Trading Days: {len(results['backtest_df'])}")
    print(f"Total Strategy Trades: {bm['total_trades']}")
    print(f"Strategy Cumulative Return (Raw): {bm['strat_cum_ret_raw']:.2%}")
    print(f"Strategy Cumulative Return (Net): {bm['strat_cum_ret_net']:.2%}")
    print(f"Buy & Hold Cumulative Return: {bm['bh_cum_ret']:.2%}")
    print("-"*50)
    print(f"Strategy Ann. Return (Net): {bm['strat_ann_ret_net']:.2%}")
    print(f"Strategy Ann. Volatility (Net): {bm['strat_ann_vol_net']:.2%}")
    print(f"Strategy Sharpe Ratio (Net): {bm['strat_sharpe_net']:.4f}")
    print(f"Strategy Sortino Ratio (Net): {bm['strat_sortino_net']:.4f}")
    print(f"Strategy Max Drawdown (Net): {bm['strat_max_dd_net']:.2%}")
    print("-"*50)
    print(f"Buy & Hold Ann. Return: {bm['bh_ann_ret']:.2%}")
    print(f"Buy & Hold Ann. Volatility: {bm['bh_ann_vol']:.2%}")
    print(f"Buy & Hold Sharpe Ratio: {bm['bh_sharpe']:.4f}")
    print(f"Buy & Hold Max Drawdown: {bm['bh_max_dd']:.2%}")
    print("-"*50)
    print(f"Probabilistic Sharpe Ratio (PSR): {bm['psr']:.2%}")
    print("="*50)
    
    # 6. Plot Equity Curves
    df_plot = results['backtest_df']
    plt.figure(figsize=(12, 6))
    plt.plot(df_plot.index, df_plot['cum_strat_net'] * 100, label='Strategy (Net of Costs)', color='#1f77b4', linewidth=2)
    plt.plot(df_plot.index, df_plot['cum_strat_raw'] * 100, label='Strategy (Raw / No Costs)', color='#aec7e8', linestyle='--')
    plt.plot(df_plot.index, df_plot['cum_bh'] * 100, label='Buy & Hold (Benchmark)', color='#ff7f0e', linewidth=1.5)
    
    plt.title('S&P 500 Classification Strategy - Cumulative Performance', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (%)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper left')
    
    # Ensure directory exists and save plot
    artifact_dir = "/Users/andresfelipebonilla/.gemini/antigravity/brain/1f23c500-f5b4-4333-981f-257069c16549"
    os.makedirs(artifact_dir, exist_ok=True)
    plot_path = os.path.join(artifact_dir, "equity_curve.png")
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info(f"Performance plot saved to {plot_path}")

if __name__ == "__main__":
    main()
