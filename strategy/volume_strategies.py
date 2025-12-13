"""
Volume-Based Strategies
Implements volume-related trading strategies based on popular TradingView concepts.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from strategy.gold_line_strategy import Signal


class VolumeImbalanceStrategy:
    """
    Volume Imbalance Strategy
    Based on the concept of buy/sell volume imbalance.
    Entry when volume delta (buy-sell) shows strong imbalance.
    """
    
    def __init__(self, lookback=20, delta_threshold=0.3):
        self.lookback = lookback
        self.delta_threshold = delta_threshold  # 30% imbalance required
        self.name = f"Vol Imbalance {lookback}"
    
    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Signal]]:
        df = df.copy()
        df.columns = df.columns.str.lower()
        
        # Classify volume as buying or selling based on candle direction
        df['bull_vol'] = np.where(df['close'] > df['open'], df['volume'], 0)
        df['bear_vol'] = np.where(df['close'] <= df['open'], df['volume'], 0)
        
        # Rolling sum volume
        df['bull_sum'] = df['bull_vol'].rolling(window=self.lookback).sum()
        df['bear_sum'] = df['bear_vol'].rolling(window=self.lookback).sum()
        df['total_vol'] = df['bull_sum'] + df['bear_sum']
        
        # Delta as percentage
        df['vol_delta'] = (df['bull_sum'] - df['bear_sum']) / df['total_vol']
        
        signals = []
        position = None
        
        for i in range(self.lookback + 1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            if pd.isna(row['vol_delta']):
                continue
            
            # Buy: Strong buying volume imbalance
            if row['vol_delta'] > self.delta_threshold and prev_row['vol_delta'] <= self.delta_threshold:
                if position != 'LONG':
                    if position == 'SHORT':
                        signals.append(Signal(df.index[i], 'BUY', row['close'], 0.8, 
                                              {'delta': row['vol_delta']}, 'Vol imbalance close short'))
                    signals.append(Signal(df.index[i], 'BUY', row['close'], 0.8,
                                          {'delta': row['vol_delta']}, 'Bullish volume imbalance'))
                    position = 'LONG'
            
            # Sell: Strong selling volume imbalance
            elif row['vol_delta'] < -self.delta_threshold and prev_row['vol_delta'] >= -self.delta_threshold:
                if position != 'SHORT':
                    if position == 'LONG':
                        signals.append(Signal(df.index[i], 'SELL', row['close'], 0.8,
                                              {'delta': row['vol_delta']}, 'Vol imbalance close long'))
                    signals.append(Signal(df.index[i], 'SELL', row['close'], 0.8,
                                          {'delta': row['vol_delta']}, 'Bearish volume imbalance'))
                    position = 'SHORT'
        
        return df, signals


class OBVStrategy:
    """
    On-Balance Volume (OBV) Strategy
    Classic volume-based momentum indicator.
    """
    
    def __init__(self, signal_period=20):
        self.signal_period = signal_period
        self.name = f"OBV {signal_period}"
    
    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Signal]]:
        df = df.copy()
        df.columns = df.columns.str.lower()
        
        # Calculate OBV
        df['obv'] = 0.0
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                df.loc[df.index[i], 'obv'] = df['obv'].iloc[i-1] + df['volume'].iloc[i]
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                df.loc[df.index[i], 'obv'] = df['obv'].iloc[i-1] - df['volume'].iloc[i]
            else:
                df.loc[df.index[i], 'obv'] = df['obv'].iloc[i-1]
        
        # OBV signal line
        df['obv_signal'] = df['obv'].rolling(window=self.signal_period).mean()
        
        signals = []
        position = None
        
        for i in range(self.signal_period + 1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            if pd.isna(row['obv_signal']):
                continue
            
            # Buy: OBV crosses above signal
            if prev_row['obv'] < prev_row['obv_signal'] and row['obv'] >= row['obv_signal']:
                if position != 'LONG':
                    if position == 'SHORT':
                        signals.append(Signal(df.index[i], 'BUY', row['close'], 0.7, {}, 'OBV close short'))
                    signals.append(Signal(df.index[i], 'BUY', row['close'], 0.7, {}, 'OBV bullish'))
                    position = 'LONG'
            
            # Sell: OBV crosses below signal
            elif prev_row['obv'] > prev_row['obv_signal'] and row['obv'] <= row['obv_signal']:
                if position != 'SHORT':
                    if position == 'LONG':
                        signals.append(Signal(df.index[i], 'SELL', row['close'], 0.7, {}, 'OBV close long'))
                    signals.append(Signal(df.index[i], 'SELL', row['close'], 0.7, {}, 'OBV bearish'))
                    position = 'SHORT'
        
        return df, signals


class VWAPBandStrategy:
    """
    VWAP with Standard Deviation Bands Strategy
    Buy at lower band, sell at upper band.
    """
    
    def __init__(self, std_mult=2.0):
        self.std_mult = std_mult
        self.name = f"VWAP Bands {std_mult}"
    
    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Signal]]:
        df = df.copy()
        df.columns = df.columns.str.lower()
        
        # VWAP calculation
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['tp_vol'] = df['typical_price'] * df['volume']
        df['cum_vol'] = df['volume'].cumsum()
        df['cum_tp_vol'] = df['tp_vol'].cumsum()
        df['vwap'] = df['cum_tp_vol'] / df['cum_vol']
        
        # Standard deviation bands
        df['sq_diff'] = ((df['typical_price'] - df['vwap']) ** 2) * df['volume']
        df['cum_sq_diff'] = df['sq_diff'].cumsum()
        df['std'] = np.sqrt(df['cum_sq_diff'] / df['cum_vol'])
        
        df['upper_band'] = df['vwap'] + (self.std_mult * df['std'])
        df['lower_band'] = df['vwap'] - (self.std_mult * df['std'])
        
        signals = []
        position = None
        
        for i in range(50, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            if pd.isna(row['lower_band']):
                continue
            
            # Buy: Price crosses above lower band (bounce)
            if prev_row['close'] < prev_row['lower_band'] and row['close'] >= row['lower_band']:
                if position != 'LONG':
                    if position == 'SHORT':
                        signals.append(Signal(df.index[i], 'BUY', row['close'], 0.8, {}, 'VWAP close short'))
                    signals.append(Signal(df.index[i], 'BUY', row['close'], 0.8, {}, 'VWAP lower band bounce'))
                    position = 'LONG'
            
            # Sell: Price crosses below upper band
            elif prev_row['close'] > prev_row['upper_band'] and row['close'] <= row['upper_band']:
                if position != 'SHORT':
                    if position == 'LONG':
                        signals.append(Signal(df.index[i], 'SELL', row['close'], 0.8, {}, 'VWAP close long'))
                    signals.append(Signal(df.index[i], 'SELL', row['close'], 0.8, {}, 'VWAP upper band rejection'))
                    position = 'SHORT'
        
        return df, signals


class VolumePriceConfirmationStrategy:
    """
    Volume-Price Confirmation Strategy
    Combines price breakout with volume confirmation.
    """
    
    def __init__(self, lookback=20, vol_mult=1.5):
        self.lookback = lookback
        self.vol_mult = vol_mult  # Volume must be 1.5x average
        self.name = f"Vol-Price {lookback}"
    
    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Signal]]:
        df = df.copy()
        df.columns = df.columns.str.lower()
        
        # Price breakout levels
        df['highest_high'] = df['high'].rolling(window=self.lookback).max()
        df['lowest_low'] = df['low'].rolling(window=self.lookback).min()
        
        # Volume average
        df['avg_vol'] = df['volume'].rolling(window=self.lookback).mean()
        
        signals = []
        position = None
        
        for i in range(self.lookback + 1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            high_vol = row['volume'] > (row['avg_vol'] * self.vol_mult)
            
            # Buy: Breakout above high with high volume
            if row['close'] > prev_row['highest_high'] and high_vol and position != 'LONG':
                if position == 'SHORT':
                    signals.append(Signal(df.index[i], 'BUY', row['close'], 0.9, 
                                          {'volume': row['volume']}, 'Vol-Price close short'))
                signals.append(Signal(df.index[i], 'BUY', row['close'], 0.9,
                                      {'volume': row['volume']}, 'Breakout with volume'))
                position = 'LONG'
            
            # Sell: Breakdown below low with high volume
            elif row['close'] < prev_row['lowest_low'] and high_vol and position != 'SHORT':
                if position == 'LONG':
                    signals.append(Signal(df.index[i], 'SELL', row['close'], 0.9,
                                          {'volume': row['volume']}, 'Vol-Price close long'))
                signals.append(Signal(df.index[i], 'SELL', row['close'], 0.9,
                                      {'volume': row['volume']}, 'Breakdown with volume'))
                position = 'SHORT'
        
        return df, signals


def compare_volume_strategies(symbol='BTCUSDT', timeframe='720', days=730):
    """Compare all volume-based strategies"""
    import yaml
    from data.bybit_connector import BybitConnector
    from backtesting.backtest_engine import BacktestEngine
    
    with open('config/strategy_params.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 80)
    print(f"VOLUME-BASED STRATEGIES COMPARISON")
    print(f"Symbol: {symbol} | Timeframe: {timeframe}min | History: {days} days")
    print("=" * 80)
    
    connector = BybitConnector(cache_dir='data/cache')
    df = connector.get_historical_data(symbol, timeframe, days, use_cache=True)
    
    print(f"📊 Loaded {len(df)} candles\n")
    
    strategies = [
        VolumeImbalanceStrategy(20, 0.3),
        VolumeImbalanceStrategy(10, 0.2),
        OBVStrategy(20),
        OBVStrategy(10),
        VWAPBandStrategy(2.0),
        VWAPBandStrategy(1.5),
        VolumePriceConfirmationStrategy(20, 1.5),
        VolumePriceConfirmationStrategy(10, 2.0),
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
                'sharpe': result.sharpe_ratio,
            })
            
            print(f"  ✅ {strategy.name:<20} {result.total_trades:>4} trades  {result.win_rate:>5.1f}% win  {result.total_pnl_pct:>+7.2f}%")
            
        except Exception as e:
            print(f"  ❌ {strategy.name}: {e}")
    
    results.sort(key=lambda x: x['pnl_pct'], reverse=True)
    
    print("\n" + "=" * 80)
    print("📈 VOLUME STRATEGIES RANKED BY PnL")
    print("=" * 80)
    print(f"{'Strategy':<25} {'Trades':<8} {'Win %':<8} {'PnL %':<10} {'PF':<8}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['name']:<25} {r['trades']:<8} {r['win_rate']:.1f}%{'':<3} {r['pnl_pct']:+.2f}%{'':<4} {r['profit_factor']:.2f}")
    
    print("=" * 80)
    
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Volume Strategies Comparison')
    parser.add_argument('--symbol', type=str, default='BTCUSDT')
    parser.add_argument('--timeframe', type=str, default='720')
    parser.add_argument('--days', type=int, default=730)
    
    args = parser.parse_args()
    compare_volume_strategies(args.symbol, args.timeframe, args.days)
