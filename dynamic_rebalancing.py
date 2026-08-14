import os
import logging
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.optimize import minimize
from typing import Dict, List, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Bisection method to find the IRR (Internal Rate of Return) for monthly cash flows
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

# Scipy optimizer for Global Minimum Variance (GMV) portfolio
def optimize_min_variance(cov_matrix: pd.DataFrame) -> np.ndarray:
    n_assets = cov_matrix.shape[0]
    init_weights = np.array([1.0 / n_assets] * n_assets)
    
    def portfolio_variance(weights):
        return np.dot(weights.T, np.dot(cov_matrix.values, weights))
        
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
        # Fallback to equal weight if optimization fails
        return init_weights
    return res.x

class DynamicDCASimulator:
    def __init__(self, tickers: List[str], tx_cost: float = 0.0005, lookback: int = 60):
        self.tickers = tickers
        self.tx_cost = tx_cost
        self.lookback = lookback
        
    def download_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        logger.info(f"Downloading data for all tickers: {self.tickers}...")
        df_list = []
        for ticker in self.tickers:
            data = yf.download(ticker, start=start_date, end=end_date)
            if data.empty:
                raise ValueError(f"No data downloaded for ticker {ticker}")
                
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
            
        # Consolidate on QQQ/SPY index
        master_data = yf.download("QQQ", start=start_date, end=end_date)
        master_index = master_data.index
        
        df_prices = pd.concat(df_list, axis=1, sort=False).reindex(master_index).ffill().bfill()
        return df_prices

    def simulate(
        self, 
        prices: pd.DataFrame, 
        strategy_type: str, 
        static_weights: Dict[str, float] = None,
        monthly_investment: float = 500.0, 
        total_months: int = 24
    ) -> Dict[str, Any]:
        """
        Simulates the monthly DCA investment with dynamic or static rebalancing.
        """
        dates = prices.index
        # We need daily returns to calculate covariance and momentum
        daily_returns = prices.pct_change().fillna(0)
        
        # Determine investment dates (first trading day of each month in the evaluation window)
        # To leave the first 'lookback' days for initial calculation, we find the date after the lookback.
        eval_prices = prices.iloc[self.lookback:]
        eval_dates = eval_prices.index
        
        ym = eval_dates.to_period('M')
        first_dates_all = eval_dates[~ym.duplicated()]
        investment_dates = first_dates_all[:total_months]
        
        # Truncate simulation period to start on the first trading day after lookback, and end on last day of total_months
        start_sim_idx = prices.index.get_loc(investment_dates[0])
        sim_prices = prices.iloc[start_sim_idx:]
        sim_dates = sim_prices.index
        
        shares = {t: 0.0 for t in self.tickers}
        portfolio_values = []
        nav_values = []
        allocation_history = []
        injected_cash_flow = []
        cash_flow_dates = []
        
        total_injected = 0.0
        current_nav = 1.0
        prev_value = 0.0
        
        for t, date in enumerate(sim_dates):
            # 1. Price change return calculation (NAV)
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
                
            # 2. Check for investment and rebalance
            is_invest_day = date in investment_dates
            if is_invest_day:
                total_injected += monthly_investment
                injected_cash_flow.append(-monthly_investment)
                cash_flow_dates.append(date)
                
                # Get current portfolio value before injection
                current_portfolio_value = sum(shares[ticker] * sim_prices.loc[date, ticker] for ticker in self.tickers)
                total_to_allocate = current_portfolio_value + monthly_investment
                
                # Get current holdings value for each asset
                current_holdings_val = {ticker: shares[ticker] * sim_prices.loc[date, ticker] for ticker in self.tickers}
                
                # Compute Target Weights
                if strategy_type == 'static':
                    if static_weights is None:
                        raise ValueError("static_weights is required when strategy_type='static'")
                    target_weights = {ticker: static_weights.get(ticker, 0.0) for ticker in self.tickers}
                    
                elif strategy_type == 'momentum':
                    # Calculate 60-day historical returns of all assets
                    # Look back from 'date' in the prices dataframe
                    loc_idx = prices.index.get_loc(date)
                    price_slice = prices.iloc[loc_idx - self.lookback : loc_idx + 1]
                    momentum_rets = (price_slice.iloc[-1] / price_slice.iloc[0]) - 1.0
                    
                    # Rank assets by returns
                    sorted_assets = momentum_rets.sort_values(ascending=False).index.tolist()
                    top_3 = sorted_assets[:3]
                    
                    # If momentum is negative or zero, replace with SHV
                    target_weights = {ticker: 0.0 for ticker in self.tickers}
                    for asset in top_3:
                        if momentum_rets[asset] > 0:
                            target_weights[asset] += 1.0 / 3.0
                        else:
                            target_weights['SHV'] += 1.0 / 3.0
                            
                elif strategy_type == 'min_variance':
                    # Calculate rolling covariance matrix of past 60 days daily returns
                    loc_idx = prices.index.get_loc(date)
                    ret_slice = daily_returns.iloc[loc_idx - self.lookback : loc_idx]
                    cov_matrix = ret_slice.cov() * 252 # Annualized covariance
                    
                    # Optimize
                    optimal_w = optimize_min_variance(cov_matrix)
                    target_weights = {self.tickers[i]: optimal_w[i] for i in range(len(self.tickers))}
                    
                else:
                    raise ValueError(f"Unknown strategy type: {strategy_type}")

                total_weight = sum(target_weights.values())
                if not np.isclose(total_weight, 1.0):
                    raise ValueError(
                        f"Target weights must sum to 1.0; received {total_weight:.6f}"
                    )

                allocation_history.append({'date': date, **target_weights})
                    
                # Rebalance math taking transaction costs into account:
                # Target allocation in USD: target_alloc = W * V_net
                # But V_net = V_total - TC
                # TC = sum( |W * V_net - CurrentVal| * tx_cost )
                # We approximate V_net using an iterative feedback loop (3 iterations is highly accurate)
                v_net = total_to_allocate
                for _ in range(3):
                    tc_est = 0.0
                    for ticker in self.tickers:
                        target_val = v_net * target_weights[ticker]
                        trade = target_val - current_holdings_val[ticker]
                        tc_est += abs(trade) * self.tx_cost
                    v_net = total_to_allocate - tc_est
                    
                # Execute trades: update shares
                for ticker in self.tickers:
                    net_alloc_val = v_net * target_weights[ticker]
                    price = sim_prices.loc[date, ticker]
                    shares[ticker] = net_alloc_val / price
                    
            # 3. Calculate portfolio value after rebalance
            current_value = sum(shares[ticker] * sim_prices.loc[date, ticker] for ticker in self.tickers)
            portfolio_values.append(current_value)
            nav_values.append(current_nav)
            prev_value = current_value
            
        df_sim = pd.DataFrame(index=sim_dates)
        df_sim['portfolio_value'] = portfolio_values
        df_sim['nav'] = nav_values
        weights_df = pd.DataFrame(allocation_history).set_index('date')
        weights_df.index.name = 'date'
        
        # Add final positive cash flow for IRR
        injected_cash_flow.append(current_value)
        cash_flow_dates.append(sim_dates[-1])
        
        # Calculate metrics
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
            'weights': weights_df,
            'total_injected': total_injected,
            'final_value': current_value,
            'net_profit': net_profit,
            'roi': roi,
            'irr': irr,
            'max_dd': max_dd,
            'sharpe': sharpe
        }

def main():
    # Assets universe
    tickers = ['QQQ', 'SMH', 'BTC-USD', 'SPY', 'GLD', 'IEF', 'USMV', 'XLV', 'SHV']
    
    # 2 years simulation window + lookback (60 trading days ~ 90 calendar days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2 * 365 + 150)
    
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    simulator = DynamicDCASimulator(tickers=tickers, tx_cost=0.0005, lookback=60)
    prices = simulator.download_data(start_str, end_str)
    
    # 1. Simulate Static Moderado Portfolio as Benchmark
    # Weights for Moderado: SPY: 40%, QQQ: 30%, GLD: 15%, IEF: 15%
    # The rest are 0.0
    static_mod_weights = {t: 0.0 for t in tickers}
    static_mod_weights['SPY'] = 0.40
    static_mod_weights['QQQ'] = 0.30
    static_mod_weights['GLD'] = 0.15
    static_mod_weights['IEF'] = 0.15
    
    logger.info("Simulating Benchmark: Moderado Estático...")
    res_static = simulator.simulate(prices, strategy_type='static', static_weights=static_mod_weights)
    
    # 2. Simulate Dynamic Momentum Portfolio
    logger.info("Simulating Dynamic Momentum Portfolio...")
    res_momentum = simulator.simulate(prices, strategy_type='momentum')
    
    # 3. Simulate Minimum Variance Portfolio
    logger.info("Simulating Minimum Variance Portfolio...")
    res_min_var = simulator.simulate(prices, strategy_type='min_variance')
    
    results = {
        'Moderado Estático (Benchmark)': res_static,
        'Momentum Dinámico (DMP)': res_momentum,
        'Mínima Varianza (MVP)': res_min_var
    }
    
    # Print comparison table
    print("\n" + "="*90)
    print("COMPARATIVA DE REBALANCEO: ESTÁTICO VS DINÁMICO (500 USD/mes - 2 Años)")
    print("="*90)
    print(f"{'Estrategia':<30} | {'Invertido':<10} | {'Valor Final':<12} | {'Ganancia':<10} | {'ROI':<8} | {'TIR (IRR)':<10} | {'Max DD':<8} | {'Sharpe':<6}")
    print("-"*120)
    for name, res in results.items():
        print(f"{name:<30} | ${res['total_injected']:<9.2f} | ${res['final_value']:<11.2f} | ${res['net_profit']:<9.2f} | {res['roi']:<7.2%} | {res['irr']:<9.2%} | {res['max_dd']:<7.2%} | {res['sharpe']:<6.3f}")
    print("="*90)

    # Make the monthly target allocations auditable.
    for name, res in results.items():
        print(f"\nPESOS OBJETIVO MENSUALES — {name}")
        print(res['weights'].round(4).to_string())
    
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
        'Moderado Estático (Benchmark)': '#7f7f7f', # Grey
        'Momentum Dinámico (DMP)': '#d62728', # Red
        'Mínima Varianza (MVP)': '#1f77b4' # Blue
    }
    
    for name, res in results.items():
        plt.plot(res['df'].index, res['df']['portfolio_value'], label=name, color=colors[name], linewidth=2)
        
    plt.title('Comparación de Estrategias: Rebalanceo Estático vs Dinámico (DCA)', fontsize=14, fontweight='bold')
    plt.xlabel('Fecha')
    plt.ylabel('Valor del Portafolio (USD)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper left')
    
    artifact_dir = "/Users/andresfelipebonilla/.gemini/antigravity/brain/1f23c500-f5b4-4333-981f-257069c16549"
    os.makedirs(artifact_dir, exist_ok=True)
    plot_path = os.path.join(artifact_dir, "dynamic_portfolio_simulation.png")
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info(f"Dynamic rebalancing comparison plot saved to {plot_path}")

if __name__ == "__main__":
    main()
