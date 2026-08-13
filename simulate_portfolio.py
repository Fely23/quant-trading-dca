import os
import logging
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Bisection method to find the IRR (Internal Rate of Return) for monthly cash flows
def calculate_irr(cash_flows: List[float], dates: List[datetime]) -> float:
    """
    Calculates the annualized Internal Rate of Return (IRR) for irregular cash flows.
    """
    t0 = dates[0]
    # Time in years from t0 for each cash flow
    years_from_start = np.array([(d - t0).days / 365.25 for d in dates])
    
    def net_present_value(r: float) -> float:
        return np.sum(np.array(cash_flows) / ((1 + r) ** years_from_start))
    
    # Bisection method bounds
    low, high = -0.99, 5.0
    for _ in range(100):
        mid = (low + high) / 2
        npv = net_present_value(mid)
        if abs(npv) < 1e-6:
            return mid
        if npv > 0:
            low = mid if net_present_value(low) * npv < 0 else low
            # wait, if rate increases, NPV decreases (for standard project, but here cash flows are negative and final value is positive)
            # Let's check: NPV(r) = -500 - 500/(1+r)^t1 + ... + V_final/(1+r)^T
            # derivative dNPV/dr: 500*t1/(1+r)^(t1+1) ... - T*V_final/(1+r)^(T+1).
            # Usually dNPV/dr is negative. So if NPV > 0, we need to increase rate (r) to reduce NPV to 0.
            # So low = mid.
            low = mid
        else:
            high = mid
    return (low + high) / 2

class DCASimulator:
    def __init__(self, tickers: List[str], weights: Dict[str, float], tx_cost: float = 0.0005):
        self.tickers = tickers
        self.weights = weights
        self.tx_cost = tx_cost
        
    def download_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Downloads data and returns a consolidated DataFrame of adjusted close prices.
        """
        logger.info(f"Downloading data for tickers: {self.tickers}...")
        df_list = []
        for ticker in self.tickers:
            data = yf.download(ticker, start=start_date, end=end_date)
            if data.empty:
                raise ValueError(f"No data downloaded for ticker {ticker}")
            
            # yfinance returns multi-index or single index depending on columns.
            # We extract Close or Adj Close column.
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
            
        # Consolidate on common index (inner join/forward fill to align ETF/BTC trading days)
        # Note: Bitcoin trades 24/7, ETFs trade Mon-Fri. We align to ETF trading days.
        # We find an ETF ticker or use SPY/QQQ if present as master index
        etf_tickers = [t for t in self.tickers if t != "BTC-USD"]
        master_ticker = etf_tickers[0] if etf_tickers else self.tickers[0]
        
        # Download master to get trading index
        master_data = yf.download(master_ticker, start=start_date, end=end_date)
        master_index = master_data.index
        
        df_prices = pd.concat(df_list, axis=1).reindex(master_index).ffill().bfill()
        return df_prices

    def simulate(self, prices: pd.DataFrame, monthly_investment: float = 500.0, total_months: int = 24) -> Dict[str, Any]:
        """
        Simulates the monthly DCA investment.
        """
        dates = prices.index
        
        # Group by Year-Month to find the first trading date of each month
        ym = dates.to_period('M')
        first_dates_all = dates[~ym.duplicated()]
        
        # Limit to the first `total_months` investment dates
        investment_dates = first_dates_all[:total_months]
        
        # Portfolios variables
        shares = {t: 0.0 for t in self.tickers}
        portfolio_values = []
        nav_values = []
        injected_cash_flow = []
        cash_flow_dates = []
        
        total_injected = 0.0
        current_nav = 1.0
        
        # Track daily metrics
        prev_value = 0.0
        
        for t, date in enumerate(dates):
            # 1. Price change return calculation (NAV)
            if t > 0:
                # Value of yesterday's shares at today's prices
                value_pre_purchase = sum(shares[ticker] * prices.loc[date, ticker] for ticker in self.tickers)
                if prev_value > 0:
                    daily_ret = (value_pre_purchase / prev_value) - 1.0
                else:
                    daily_ret = 0.0
                current_nav = current_nav * (1.0 + daily_ret)
            else:
                daily_ret = 0.0
                current_nav = 1.0
            
            # 2. Process investment if today is an investment date
            is_invest_day = date in investment_dates
            if is_invest_day:
                total_injected += monthly_investment
                injected_cash_flow.append(-monthly_investment)
                cash_flow_dates.append(date)
                
                # Buy shares
                for ticker in self.tickers:
                    alloc = monthly_investment * self.weights[ticker]
                    net_alloc = alloc * (1.0 - self.tx_cost)
                    price = prices.loc[date, ticker]
                    shares[ticker] += net_alloc / price
            
            # 3. Calculate portfolio value after investment
            current_value = sum(shares[ticker] * prices.loc[date, ticker] for ticker in self.tickers)
            
            portfolio_values.append(current_value)
            nav_values.append(current_nav)
            
            prev_value = current_value
            
        # Create output series
        df_sim = pd.DataFrame(index=dates)
        df_sim['portfolio_value'] = portfolio_values
        df_sim['nav'] = nav_values
        
        # Add the final value as a positive cash flow at the end date for IRR calculation
        final_date = dates[-1]
        injected_cash_flow.append(current_value)
        cash_flow_dates.append(final_date)
        
        # Calculate metrics
        net_profit = current_value - total_injected
        roi = net_profit / total_injected
        
        # Annualized IRR
        irr = calculate_irr(injected_cash_flow, cash_flow_dates)
        
        # Max Drawdown on NAV
        nav_series = df_sim['nav']
        running_max = nav_series.cummax()
        drawdowns = (nav_series - running_max) / running_max
        max_dd = drawdowns.min()
        
        # Sharpe Ratio of NAV daily returns
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
    # Setup date range (2 years)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2 * 365 + 60) # add 2 months buffer to get full 24 monthly periods
    
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    # Portfolio Scenarios
    scenarios = {
        'Agresivo (Momentum)': {
            'tickers': ['QQQ', 'SMH', 'BTC-USD'],
            'weights': {'QQQ': 0.50, 'SMH': 0.30, 'BTC-USD': 0.20}
        },
        'Moderado (Adaptable)': {
            'tickers': ['SPY', 'QQQ', 'GLD', 'IEF'],
            'weights': {'SPY': 0.40, 'QQQ': 0.30, 'GLD': 0.15, 'IEF': 0.15}
        },
        'Conservador (Defensivo)': {
            'tickers': ['USMV', 'XLV', 'SHV'],
            'weights': {'USMV': 0.40, 'XLV': 0.30, 'SHV': 0.30}
        }
    }
    
    # Consolidation of all unique tickers
    unique_tickers = list(set([t for s in scenarios.values() for t in s['tickers']]))
    
    # Download consolidated data
    # Create simulator with dummy values just for download
    dummy_sim = DCASimulator(unique_tickers, {})
    prices = dummy_sim.download_data(start_str, end_str)
    
    # Run simulations
    results = {}
    for name, config in scenarios.items():
        logger.info(f"Simulating scenario: {name}...")
        sim = DCASimulator(config['tickers'], config['weights'])
        # Filter prices to config tickers
        prices_subset = prices[config['tickers']]
        
        # We slice prices to contain exactly 24 calendar months of data for the DCA.
        # Let's find the start index which begins the first month in the last 2 years.
        # We can group by year-month and take the last 24 months.
        yms = prices_subset.index.to_period('M')
        unique_yms = sorted(yms.unique())
        if len(unique_yms) > 24:
            # Take the last 24 completed months
            target_yms = unique_yms[-24:]
            prices_subset = prices_subset[yms.isin(target_yms)]
            
        res = sim.simulate(prices_subset, monthly_investment=500.0, total_months=24)
        results[name] = res
        
    # Print table of results
    print("\n" + "="*80)
    print("SIMULACIÓN DE PORTAFOLIOS MULTIESCENARIO (Inyección Mensual 500 USD - 2 Años)")
    print("="*80)
    print(f"{'Escenario':<25} | {'Invertido':<10} | {'Valor Final':<12} | {'Ganancia':<10} | {'ROI':<8} | {'TIR (IRR)':<10} | {'Max DD':<8} | {'Sharpe':<6}")
    print("-"*110)
    for name, res in results.items():
        print(f"{name:<25} | ${res['total_injected']:<9.2f} | ${res['final_value']:<11.2f} | ${res['net_profit']:<9.2f} | {res['roi']:<7.2%} | {res['irr']:<9.2%} | {res['max_dd']:<7.2%} | {res['sharpe']:<6.3f}")
    print("="*80)
    
    # Plot equity curves of portfolio value
    plt.figure(figsize=(12, 6))
    
    colors = {
        'Agresivo (Momentum)': '#d62728', # Red
        'Moderado (Adaptable)': '#1f77b4', # Blue
        'Conservador (Defensivo)': '#2ca02c' # Green
    }
    
    # We will plot the portfolio cumulative injected capital as a grey shaded area
    # and the portfolios equity curves on top.
    # Take index from one of the results
    sample_name = list(results.keys())[0]
    sample_df = results[sample_name]['df']
    
    # Calculate accumulated injection step curve
    ym = sample_df.index.to_period('M')
    first_dates_all = sample_df.index[~ym.duplicated()]
    investment_dates = first_dates_all[:24]
    
    accumulated_invested = []
    current_inv = 0.0
    for date in sample_df.index:
        if date in investment_dates:
            current_inv += 500.0
        accumulated_invested.append(current_inv)
        
    plt.fill_between(sample_df.index, accumulated_invested, label='Capital Invertido Acum.', color='#d3d3d3', alpha=0.5)
    
    for name, res in results.items():
        plt.plot(res['df'].index, res['df']['portfolio_value'], label=name, color=colors[name], linewidth=2)
        
    plt.title('Simulación de Inversión Mensual DCA (500 USD/mes - 2 Años)', fontsize=14, fontweight='bold')
    plt.xlabel('Fecha')
    plt.ylabel('Valor del Portafolio (USD)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper left')
    
    artifact_dir = "/Users/andresfelipebonilla/.gemini/antigravity/brain/1f23c500-f5b4-4333-981f-257069c16549"
    os.makedirs(artifact_dir, exist_ok=True)
    plot_path = os.path.join(artifact_dir, "portfolio_simulation.png")
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info(f"Portfolio simulation plot saved to {plot_path}")

if __name__ == "__main__":
    main()
