"""
Correlation-Based Trading Strategies
Inspired by the Match Finder indicator - trades based on asset correlations.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from strategy.gold_line_strategy import Signal


class CorrelationBreakoutStrategy:
    """
    Correlation Breakout Strategy
    Based on the Match Finder concept - when correlation with a lead asset breaks,
    it signals a potential trend change.
    
    Logic:
    - Calculate rolling correlation with a reference asset (e.g., BTC for alts)
    - When correlation drops significantly, asset may be diverging (opportunity)
    - When correlation rises back, asset is returning to normal behavior
    """
    
    def __init__(self, lookback=30, threshold=0.7, divergence_threshold=0.3):
        self.lookback = lookback
        self.threshold = threshold  # High correlation threshold
        self.divergence_threshold = divergence_threshold  # Low correlation = divergence
        self.name = f"Corr Breakout {lookback}"
    
    def _calculate_correlation(self, series1: pd.Series, series2: pd.Series, window: int) -> pd.Series:
        """Calculate rolling Pearson correlation"""
        return series1.rolling(window=window).corr(series2)
    
    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Signal]]:
        df = df.copy()
        df.columns = df.columns.str.lower()
        
        # For crypto, we use BTC as the reference (simulate by using lagged/shifted data)
        # In reality, you'd fetch BTC data separately
        # Here we'll use a proxy: correlation with the SMA trend
        sma = df['close'].rolling(window=self.lookback).mean()
        
        # Calculate how "in-sync" price is with its SMA (proxy for market correlation)
        df['price_normalized'] = (df['close'] - df['close'].rolling(self.lookback).mean()) / df['close'].rolling(self.lookback).std()
        df['sma_normalized'] = (sma - sma.rolling(self.lookback).mean()) / sma.rolling(self.lookback).std()
        
        df['correlation'] = df['price_normalized'].rolling(window=self.lookback).corr(df['sma_normalized'])
        
        signals = []
        position = None
        
        for i in range(self.lookback * 2, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            if pd.isna(row['correlation']):
                continue
            
            # Buy when correlation drops (divergence) then starts recovering
            if (prev_row['correlation'] < self.divergence_threshold and 
                row['correlation'] >= self.divergence_threshold and 
                row['close'] > prev_row['close']):
                if position != 'LONG':
                    if position == 'SHORT':
                        signals.append(Signal(df.index[i], 'BUY', row['close'], 0.8, 
                                              {'corr': row['correlation']}, 'Correlation close short'))
                    signals.append(Signal(df.index[i], 'BUY', row['close'], 0.8,
                                          {'corr': row['correlation']}, 'Bullish divergence recovery'))
                    position = 'LONG'
            
            # Sell when highly correlated but price weakening
            elif (row['correlation'] > self.threshold and 
                  row['close'] < prev_row['close'] and
                  prev_row['close'] < df['close'].iloc[i-2]):
                if position != 'SHORT':
                    if position == 'LONG':
                        signals.append(Signal(df.index[i], 'SELL', row['close'], 0.8,
                                              {'corr': row['correlation']}, 'Correlation close long'))
                    signals.append(Signal(df.index[i], 'SELL', row['close'], 0.8,
                                          {'corr': row['correlation']}, 'Bearish weakness'))
                    position = 'SHORT'
        
        return df, signals


class LeadLagStrategy:
    """
    Lead-Lag Strategy
    Trades based on detecting when one asset leads another.
    Uses cross-correlation to find lead/lag relationships.
    """
    
    def __init__(self, lookback=20, lead_periods=3):
        self.lookback = lookback
        self.lead_periods = lead_periods
        self.name = f"Lead-Lag {lookback}"
    
    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Signal]]:
        df = df.copy()
        df.columns = df.columns.str.lower()
        
        # Calculate returns
        df['returns'] = df['close'].pct_change()
        
        # Lagged returns (what price did N periods ago)
        df['lagged_returns'] = df['returns'].shift(self.lead_periods)
        
        # Correlation between current returns and lagged returns
        # High correlation means past predicts future
        df['lead_lag_corr'] = df['returns'].rolling(window=self.lookback).corr(df['lagged_returns'])
        
        # Cumulative lagged returns signal
        df['cum_lagged_ret'] = df['lagged_returns'].rolling(window=self.lead_periods).sum()
        
        signals = []
        position = None
        
        for i in range(self.lookback + self.lead_periods + 1, len(df)):
            row = df.iloc[i]
            
            if pd.isna(row['lead_lag_corr']) or pd.isna(row['cum_lagged_ret']):
                continue
            
            # If past returns predict future AND past was positive, go long
            if row['lead_lag_corr'] > 0.3 and row['cum_lagged_ret'] > 0.02:
                if position != 'LONG':
                    if position == 'SHORT':
                        signals.append(Signal(df.index[i], 'BUY', row['close'], 0.7, {}, 'Lead-lag close short'))
                    signals.append(Signal(df.index[i], 'BUY', row['close'], 0.7, {}, 'Lead-lag bullish'))
                    position = 'LONG'
            
            # If past predicts future AND past was negative, go short
            elif row['lead_lag_corr'] > 0.3 and row['cum_lagged_ret'] < -0.02:
                if position != 'SHORT':
                    if position == 'LONG':
                        signals.append(Signal(df.index[i], 'SELL', row['close'], 0.7, {}, 'Lead-lag close long'))
                    signals.append(Signal(df.index[i], 'SELL', row['close'], 0.7, {}, 'Lead-lag bearish'))
                    position = 'SHORT'
        
        return df, signals


class MeanReversionStrategy:
    """
    Mean Reversion Strategy
    Trades when price deviates significantly from its mean.
    Based on the idea that correlations tend to revert.
    """
    
    def __init__(self, lookback=20, zscore_threshold=2.0):
        self.lookback = lookback
        self.zscore_threshold = zscore_threshold
        self.name = f"Mean Rev {lookback}"
    
    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Signal]]:
        df = df.copy()
        df.columns = df.columns.str.lower()
        
        # Calculate Z-score
        df['sma'] = df['close'].rolling(window=self.lookback).mean()
        df['std'] = df['close'].rolling(window=self.lookback).std()
        df['zscore'] = (df['close'] - df['sma']) / df['std']
        
        signals = []
        position = None
        
        for i in range(self.lookback + 1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            if pd.isna(row['zscore']):
                continue
            
            # Buy when oversold (z-score crosses back above -threshold)
            if prev_row['zscore'] < -self.zscore_threshold and row['zscore'] >= -self.zscore_threshold:
                if position != 'LONG':
                    if position == 'SHORT':
                        signals.append(Signal(df.index[i], 'BUY', row['close'], 0.8, 
                                              {'zscore': row['zscore']}, 'Mean rev close short'))
                    signals.append(Signal(df.index[i], 'BUY', row['close'], 0.8,
                                          {'zscore': row['zscore']}, 'Oversold bounce'))
                    position = 'LONG'
            
            # Sell when overbought (z-score crosses back below +threshold)
            elif prev_row['zscore'] > self.zscore_threshold and row['zscore'] <= self.zscore_threshold:
                if position != 'SHORT':
                    if position == 'LONG':
                        signals.append(Signal(df.index[i], 'SELL', row['close'], 0.8,
                                              {'zscore': row['zscore']}, 'Mean rev close long'))
                    signals.append(Signal(df.index[i], 'SELL', row['close'], 0.8,
                                          {'zscore': row['zscore']}, 'Overbought pullback'))
                    position = 'SHORT'
        
        return df, signals


class MomentumCorrelationStrategy:
    """
    Momentum + Correlation Strategy
    Combines momentum with correlation analysis.
    """
    
    def __init__(self, mom_period=10, corr_period=20):
        self.mom_period = mom_period
        self.corr_period = corr_period
        self.name = f"Mom-Corr {mom_period}/{corr_period}"
    
    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Signal]]:
        df = df.copy()
        df.columns = df.columns.str.lower()
        
        # Momentum
        df['momentum'] = df['close'] / df['close'].shift(self.mom_period) - 1
        
        # Price vs Volume correlation (high corr = healthy trend)
        df['pv_corr'] = df['close'].rolling(window=self.corr_period).corr(df['volume'])
        
        # Rolling volatility
        df['volatility'] = df['close'].pct_change().rolling(window=self.corr_period).std()
        
        signals = []
        position = None
        
        for i in range(self.corr_period + self.mom_period, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            if pd.isna(row['momentum']) or pd.isna(row['pv_corr']):
                continue
            
            # Buy: Positive momentum + Volume confirming (positive P-V correlation)
            if row['momentum'] > 0.02 and row['pv_corr'] > 0.3 and prev_row['momentum'] <= 0.02:
                if position != 'LONG':
                    if position == 'SHORT':
                        signals.append(Signal(df.index[i], 'BUY', row['close'], 0.85, {}, 'Mom-Corr close short'))
                    signals.append(Signal(df.index[i], 'BUY', row['close'], 0.85, {}, 'Momentum + Volume confirm'))
                    position = 'LONG'
            
            # Sell: Negative momentum + Volume confirming
            elif row['momentum'] < -0.02 and row['pv_corr'] > 0.3 and prev_row['momentum'] >= -0.02:
                if position != 'SHORT':
                    if position == 'LONG':
                        signals.append(Signal(df.index[i], 'SELL', row['close'], 0.85, {}, 'Mom-Corr close long'))
                    signals.append(Signal(df.index[i], 'SELL', row['close'], 0.85, {}, 'Momentum down + Volume confirm'))
                    position = 'SHORT'
        
        return df, signals


def compare_correlation_strategies(symbol='BTCUSDT', timeframe='720', days=730):
    """Compare all correlation-based strategies"""
    import yaml
    from data.bybit_connector import BybitConnector
    from backtesting.backtest_engine import BacktestEngine
    
    with open('config/strategy_params.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 80)
    print(f"CORRELATION-BASED STRATEGIES COMPARISON")
    print(f"Symbol: {symbol} | Timeframe: {timeframe}min | History: {days} days")
    print("=" * 80)
    
    connector = BybitConnector(cache_dir='data/cache')
    df = connector.get_historical_data(symbol, timeframe, days, use_cache=True)
    
    print(f"📊 Loaded {len(df)} candles\n")
    
    strategies = [
        CorrelationBreakoutStrategy(30, 0.7, 0.3),
        CorrelationBreakoutStrategy(20, 0.6, 0.2),
        LeadLagStrategy(20, 3),
        LeadLagStrategy(10, 2),
        MeanReversionStrategy(20, 2.0),
        MeanReversionStrategy(30, 1.5),
        MomentumCorrelationStrategy(10, 20),
        MomentumCorrelationStrategy(5, 10),
    ]
    
    results = []
    
    for strategy in strategies:
        try:
            df_copy = df.copy()
            _, signals = strategy.run(df_copy)
            
            if not signals or len(signals) < 2:
                print(f"  {strategy.name}: No signals")
                continue
            
            backtester = BacktestEngine(config)
            result = backtester.run(df_copy, signals, symbol, timeframe)
            
            results.append({
                'name': strategy.name,
                'trades': result.total_trades,
                'win_rate': result.win_rate,
                'pnl_pct': result.total_pnl_pct,
                'profit_factor': result.profit_factor,
            })
            
            print(f"  ✅ {strategy.name:<20} {result.total_trades:>4} trades  {result.win_rate:>5.1f}% win  {result.total_pnl_pct:>+7.2f}%")
            
        except Exception as e:
            print(f"  ❌ {strategy.name}: {e}")
    
    results.sort(key=lambda x: x['pnl_pct'], reverse=True)
    
    print("\n" + "=" * 80)
    print("📈 CORRELATION STRATEGIES RANKED BY PnL")
    print("=" * 80)
    print(f"{'Strategy':<25} {'Trades':<8} {'Win %':<8} {'PnL %':<10} {'PF':<8}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['name']:<25} {r['trades']:<8} {r['win_rate']:.1f}%{'':<3} {r['pnl_pct']:+.2f}%{'':<4} {r['profit_factor']:.2f}")
    
    print("=" * 80)
    
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Correlation Strategies Comparison')
    parser.add_argument('--symbol', type=str, default='BTCUSDT')
    parser.add_argument('--timeframe', type=str, default='720')
    parser.add_argument('--days', type=int, default=730)
    
    args = parser.parse_args()
    compare_correlation_strategies(args.symbol, args.timeframe, args.days)
