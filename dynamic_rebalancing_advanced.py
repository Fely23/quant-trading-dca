import os
import logging
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, List, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Bisection method for IRR calculation
def calculate_irr(cash_flows: List[float], dates: List[datetime]) -> float:
    t0 = dates[0]
    years_from_start = np.array([(d - t0).days / 365.25 for d in dates])
    
    def net_present_value(r: float) -> float:
        return np.sum(np.array(cash_flows) / ((1 + r) ** years_from_start))
    
    low, high = -0.99, 5.0
    for _ in range(100):
        mid = (low + high) / 2
        npv = net_present_value(mid)
        if abs(npv) < 1e-6:
            return mid
        if npv > 0:
            low = mid
        else:
            high = mid
    return (low + high) / 2

# Scipy optimizer for GMV using Ledoit-Wolf covariance
def optimize_min_variance(cov_matrix: np.ndarray) -> np.ndarray:
    n_assets = cov_matrix.shape[0]
    init_weights = np.array([1.0 / n_assets] * n_assets)
    
    def portfolio_variance(weights):
        return np.dot(weights.T, np.dot(cov_matrix, weights))
        
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    bounds = tuple((0.0, 1.0) for _ in range(n_assets))
    
    res = minimize(
        portfolio_variance, 
        init_weights, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints,
        options={'ftol': 1e-9, 'disp': False}
    )
    if not res.success:
        return init_weights
    return res.x

class AdvancedDCASimulator:
    def __init__(self, tickers: List[str], tx_cost_base: float = 0.0005, lookback: int = 60):
        self.tickers = tickers
        self.tx_cost_base = tx_cost_base
        self.lookback = lookback
        
    def download_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Downloads data and caches to CSV locally to save tokens/requests if run multiple times.
        """
        cache_file = "prices_cache.csv"
        
        # Check if cache is valid (exists and created/modified today)
        if os.path.exists(cache_file):
            mtime = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if mtime.date() == datetime.now().date():
                logger.info("Loading asset prices from local CSV cache...")
                df_cache = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                # Check if all tickers are present
                if all(t in df_cache.columns for t in self.tickers):
                    return df_cache
                    
        logger.info("Cache invalid or missing. Downloading from yfinance...")
        df_list = []
        for ticker in self.tickers:
            data = yf.download(ticker, start=start_date, end=end_date)
            if data.empty:
                raise ValueError(f"No data downloaded for {ticker}")
                
            if ('Adj Close', ticker) in data.columns:
                close = data[('Adj Close', ticker)]
            elif 'Adj Close' in data.columns:
                close = data['Adj Close']
            elif 'Close' in data.columns:
                close = data['Close']
            else:
                close = data.iloc[:, 0]
                
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
                
            close = close.rename(ticker)
            df_list.append(close)
            
        master_data = yf.download("SPY", start=start_date, end=end_date)
        master_index = master_data.index
        
        df_prices = pd.concat(df_list, axis=1).reindex(master_index).ffill().bfill()
        # Save to cache
        df_prices.to_csv(cache_file)
        return df_prices

    def simulate(
        self, 
        prices: pd.DataFrame, 
        strategy_type: str, 
        static_weights: Dict[str, float] = None,
        monthly_investment: float = 500.0, 
        total_months: int = 24
    ) -> Dict[str, Any]:
        dates = prices.index
        daily_returns = prices.pct_change().fillna(0)
        
        # 1. Financial: Calculate Rolling 20-day volatility of SPY to determine dynamic tx costs
        spy_rets = daily_returns['SPY']
        spy_vol_20d = spy_rets.rolling(window=20).std() * np.sqrt(252)
        # Volatility median over a rolling 60-day window
        spy_vol_median = spy_vol_20d.rolling(window=60).median()
        
        # 2. Slice to the evaluation period (excluding lookback buffer)
        eval_prices = prices.iloc[self.lookback:]
        eval_dates = eval_prices.index
        
        ym = eval_dates.to_period('M')
        first_dates_all = eval_dates[~ym.duplicated()]
        investment_dates = first_dates_all[:total_months]
        
        start_sim_idx = prices.index.get_loc(investment_dates[0])
        sim_prices = prices.iloc[start_sim_idx:]
        sim_dates = sim_prices.index
        
        shares = {t: 0.0 for t in self.tickers}
        portfolio_values = []
        nav_values = []
        injected_cash_flow = []
        cash_flow_dates = []
        
        total_injected = 0.0
        current_nav = 1.0
        prev_value = 0.0
        
        # Meta-Labeling state variables
        # We store monthly historical data to train the Meta-Labeler online
        meta_features_history = [] # List of dicts
        meta_targets_history = []   # List of 0 or 1
        
        # Track previous month details to compute targets
        prev_investment_date = None
        prev_selected_weights = None
        prev_portfolio_value_before_rebalance = None
        prev_feature_row = None
        
        for t, date in enumerate(sim_dates):
            # Price change return calculation (NAV)
            if t > 0:
                value_pre_purchase = sum(shares[ticker] * sim_prices.loc[date, ticker] for ticker in self.tickers)
                if prev_value > 0:
                    daily_ret = (value_pre_purchase / prev_value) - 1.0
                else:
                    daily_ret = 0.0
                current_nav = current_nav * (1.0 + daily_ret)
            else:
                daily_ret = 0.0
                current_nav = 1.0
                
            # Process monthly investment & rebalance
            is_invest_day = date in investment_dates
            if is_invest_day:
                total_injected += monthly_investment
                injected_cash_flow.append(-monthly_investment)
                cash_flow_dates.append(date)
                
                # Fetch current portfolio value before injection
                current_portfolio_value = sum(shares[ticker] * sim_prices.loc[date, ticker] for ticker in self.tickers)
                
                # A. Update target of the previous month's meta-labeler sample
                # If we had a pending investment from last month, check its performance
                if prev_investment_date is not None:
                    # Return of the chosen portfolio during the past month
                    past_month_ret = (current_portfolio_value / (prev_portfolio_value_before_rebalance + monthly_investment)) - 1.0
                    target = 1 if past_month_ret > 0 else 0
                    
                    meta_features_history.append(prev_feature_row)
                    meta_targets_history.append(target)
                
                # B. Determine Dynamic Transaction Costs based on SPY volatility
                current_spy_vol = spy_vol_20d.loc[date]
                median_spy_vol = spy_vol_median.loc[date]
                
                # If current volatility is above median, double transaction cost
                if current_spy_vol > median_spy_vol:
                    current_tx_cost = self.tx_cost_base * 2.0
                else:
                    current_tx_cost = self.tx_cost_base
                
                total_to_allocate = current_portfolio_value + monthly_investment
                current_holdings_val = {ticker: shares[ticker] * sim_prices.loc[date, ticker] for ticker in self.tickers}
                
                # C. Compute Target Weights
                if strategy_type == 'static':
                    target_weights = {ticker: static_weights.get(ticker, 0.0) for ticker in self.tickers}
                    
                elif strategy_type == 'min_variance':
                    # Ledoit-Wolf Covariance regularisation
                    loc_idx = prices.index.get_loc(date)
                    ret_slice = daily_returns.iloc[loc_idx - self.lookback : loc_idx]
                    
                    lw = LedoitWolf()
                    lw.fit(ret_slice)
                    # Annualize cov
                    cov_matrix = lw.covariance_ * 252
                    
                    optimal_w = optimize_min_variance(cov_matrix)
                    target_weights = {self.tickers[i]: optimal_w[i] for i in range(len(self.tickers))}
                    
                elif strategy_type in ['momentum', 'momentum_meta']:
                    # Risk-Adjusted Momentum Calculation: 60d Return / 60d Volatility
                    loc_idx = prices.index.get_loc(date)
                    price_slice = prices.iloc[loc_idx - self.lookback : loc_idx + 1]
                    ret_slice = daily_returns.iloc[loc_idx - self.lookback : loc_idx]
                    
                    raw_rets = (price_slice.iloc[-1] / price_slice.iloc[0]) - 1.0
                    vols = ret_slice.std() * np.sqrt(252)
                    risk_adj_mom = raw_rets / (vols + 1e-10)
                    
                    sorted_assets = risk_adj_mom.sort_values(ascending=False).index.tolist()
                    top_3 = sorted_assets[:3]
                    
                    target_weights = {ticker: 0.0 for ticker in self.tickers}
                    for asset in top_3:
                        if raw_rets[asset] > 0:
                            target_weights[asset] += 1.0 / 3.0
                        else:
                            target_weights['SHV'] += 1.0 / 3.0
                    
                    # D. AI Suggestion: Meta-Labeler Override
                    if strategy_type == 'momentum_meta':
                        # Feature extraction for current month:
                        # 1. SPY Volatility
                        # 2. GLD vs SPY 60d return ratio (fear proxy)
                        # 3. Average momentum of top 3 selected assets
                        gld_ret = raw_rets['GLD']
                        spy_ret = raw_rets['SPY']
                        fear_ratio = gld_ret / (spy_ret + 1e-10)
                        avg_mom = np.mean([risk_adj_mom[a] for a in top_3])
                        
                        feature_row = [current_spy_vol, fear_ratio, avg_mom]
                        
                        # Train and predict if we have enough historical data (at least 6 months)
                        if len(meta_targets_history) >= 6:
                            X_train = np.array(meta_features_history)
                            y_train = np.array(meta_targets_history)
                            
                            # Train meta classifier
                            meta_model = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
                            meta_model.fit(X_train, y_train)
                            
                            # Predict probability of success (next month return > 0)
                            prob_success = meta_model.predict_proba([feature_row])[0, 1]
                            
                            # If probability of success is low (< 50%), override and allocate 100% to SHV
                            if prob_success < 0.50:
                                # Override target weights to 100% SHV (defensive cash)
                                target_weights = {ticker: 0.0 for ticker in self.tickers}
                                target_weights['SHV'] = 1.0
                                
                        # Save state for the next month's evaluation
                        prev_investment_date = date
                        prev_selected_weights = target_weights.copy()
                        prev_portfolio_value_before_rebalance = current_portfolio_value
                        prev_feature_row = feature_row
                
                # Rebalance execution math
                v_net = total_to_allocate
                for _ in range(3):
                    tc_est = 0.0
                    for ticker in self.tickers:
                        target_val = v_net * target_weights[ticker]
                        trade = target_val - current_holdings_val[ticker]
                        tc_est += abs(trade) * current_tx_cost
                    v_net = total_to_allocate - tc_est
                    
                # Update shares
                for ticker in self.tickers:
                    net_alloc_val = v_net * target_weights[ticker]
                    price = sim_prices.loc[date, ticker]
                    shares[ticker] = net_alloc_val / price
                    
            # Calculate daily value
            current_value = sum(shares[ticker] * sim_prices.loc[date, ticker] for ticker in self.tickers)
            portfolio_values.append(current_value)
            nav_values.append(current_nav)
            prev_value = current_value
            
        df_sim = pd.DataFrame(index=sim_dates)
        df_sim['portfolio_value'] = portfolio_values
        df_sim['nav'] = nav_values
        
        # Add final value for IRR
        injected_cash_flow.append(current_value)
        cash_flow_dates.append(sim_dates[-1])
        
        # Metrics
        net_profit = current_value - total_injected
        roi = net_profit / total_injected
        irr = calculate_irr(injected_cash_flow, cash_flow_dates)
        
        nav_series = df_sim['nav']
        running_max = nav_series.cummax()
        drawdowns = (nav_series - running_max) / running_max
        max_dd = drawdowns.min()
        
        nav_rets = nav_series.pct_change().dropna()
        sharpe = (nav_rets.mean() / (nav_rets.std() + 1e-10)) * np.sqrt(252)
        
        return {
            'df': df_sim,
            'total_injected': total_injected,
            'final_value': current_value,
            'net_profit': net_profit,
            'roi': roi,
            'irr': irr,
            'max_dd': max_dd,
            'sharpe': sharpe
        }

def main():
    tickers = ['QQQ', 'SMH', 'BTC-USD', 'SPY', 'GLD', 'IEF', 'USMV', 'XLV', 'SHV']
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2 * 365 + 150)
    
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    simulator = AdvancedDCASimulator(tickers=tickers, tx_cost_base=0.0005, lookback=60)
    prices = simulator.download_data(start_str, end_str)
    
    # 1. Moderado Estático (Benchmark)
    static_mod_weights = {t: 0.0 for t in tickers}
    static_mod_weights['SPY'] = 0.40
    static_mod_weights['QQQ'] = 0.30
    static_mod_weights['GLD'] = 0.15
    static_mod_weights['IEF'] = 0.15
    
    logger.info("Simulating Benchmark: Moderado Estático...")
    res_static = simulator.simulate(prices, strategy_type='static', static_weights=static_mod_weights)
    
    # 2. Mínima Varianza Regularizada con Ledoit-Wolf
    logger.info("Simulating Mínima Varianza Regularizada (Ledoit-Wolf)...")
    res_min_var = simulator.simulate(prices, strategy_type='min_variance')
    
    # 3. Momentum Dinámico con Momentum Ajustado y Costos Dinámicos
    logger.info("Simulating Momentum Dinámico Ajustado...")
    res_momentum = simulator.simulate(prices, strategy_type='momentum')
    
    # 4. Momentum Dinámico + Meta-Labeling (IA)
    logger.info("Simulating Momentum Dinámico + Meta-Labeling (IA)...")
    res_meta = simulator.simulate(prices, strategy_type='momentum_meta')
    
    results = {
        '1. Moderado Estático (Benchmark)': res_static,
        '2. Mínima Varianza (Ledoit-Wolf)': res_min_var,
        '3. Momentum Ajustado': res_momentum,
        '4. Momentum + Meta-Labeling (IA)': res_meta
    }
    
    # Print results table
    print("\n" + "="*110)
    print("COMPARATIVA DE PORTAFOLIOS AVANZADOS: ANÁLISIS DE REBALANCEO E IA (500 USD/mes - 2 Años)")
    print("="*110)
    print(f"{'Estrategia':<35} | {'Invertido':<10} | {'Valor Final':<12} | {'Ganancia':<10} | {'ROI':<8} | {'TIR (IRR)':<10} | {'Max DD':<8} | {'Sharpe':<6}")
    print("-"*120)
    for name, res in results.items():
        print(f"{name:<35} | ${res['total_injected']:<9.2f} | ${res['final_value']:<11.2f} | ${res['net_profit']:<9.2f} | {res['roi']:<7.2%} | {res['irr']:<9.2%} | {res['max_dd']:<7.2%} | {res['sharpe']:<6.3f}")
    print("="*110)
    
    # Plot equity curves
    plt.figure(figsize=(12, 6))
    
    # Plot capital injected
    sample_df = res_static['df']
    eval_dates = sample_df.index
    ym = eval_dates.to_period('M')
    first_dates_all = eval_dates[~ym.duplicated()]
    investment_dates = first_dates_all[:24]
    
    accumulated_invested = []
    current_inv = 0.0
    for date in eval_dates:
        if date in investment_dates:
            current_inv += 500.0
        accumulated_invested.append(current_inv)
        
    plt.fill_between(eval_dates, accumulated_invested, label='Capital Invertido Acum.', color='#d3d3d3', alpha=0.5)
    
    colors = {
        '1. Moderado Estático (Benchmark)': '#7f7f7f', # Grey
        '2. Mínima Varianza (Ledoit-Wolf)': '#2ca02c', # Green
        '3. Momentum Ajustado': '#d62728',             # Red
        '4. Momentum + Meta-Labeling (IA)': '#9467bd'  # Purple
    }
    
    for name, res in results.items():
        plt.plot(res['df'].index, res['df']['portfolio_value'], label=name, color=colors[name], linewidth=2)
        
    plt.title('Estrategias de Rebalanceo Avanzadas (Ledoit-Wolf & Meta-Labeling)', fontsize=13, fontweight='bold')
    plt.xlabel('Fecha')
    plt.ylabel('Valor del Portafolio (USD)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper left')
    
    artifact_dir = "/Users/andresfelipebonilla/.gemini/antigravity/brain/1f23c500-f5b4-4333-981f-257069c16549"
    os.makedirs(artifact_dir, exist_ok=True)
    plot_path = os.path.join(artifact_dir, "advanced_portfolio_simulation.png")
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info(f"Advanced rebalancing plot saved to {plot_path}")

if __name__ == "__main__":
    main()
