"""
Multi-Strategy Comparison
Implements and compares multiple trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
from dataclasses import dataclass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from strategy.gold_line_strategy import Signal


class RSIOverboughtOversold:
    """RSI Overbought/Oversold Strategy"""
    
    def __init__(self, length=14, overbought=70, oversold=30):
        self.length = length
        self.overbought = overbought
        self.oversold = oversold
        self.name = "RSI O/O"
    
    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Signal]]:
        df = df.copy()
        df.columns = df.columns.str.lower()
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.length).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.length).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        signals = []
        in_position = False
        
        for i in range(1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            if pd.isna(row['rsi']):
                continue
            
            # Buy when RSI crosses above oversold
            if prev_row['rsi'] < self.oversold and row['rsi'] >= self.oversold and not in_position:
                signals.append(Signal(
                    timestamp=df.index[i],
                    type='BUY',
                    price=row['close'],
                    confidence=0.7,
                    indicators={'rsi': row['rsi']},
                    context='RSI oversold bounce'
                ))
                in_position = True
            
            # Sell when RSI crosses below overbought
            elif prev_row['rsi'] > self.overbought and row['rsi'] <= self.overbought and in_position:
                signals.append(Signal(
                    timestamp=df.index[i],
                    type='SELL',
                    price=row['close'],
                    confidence=0.7,
                    indicators={'rsi': row['rsi']},
                    context='RSI overbought exit'
                ))
                in_position = False
        
        return df, signals


class MACDCrossover:
    """MACD Crossover Strategy"""
    
    def __init__(self, fast=12, slow=26, signal=9):
        self.fast = fast
        self.slow = slow
        self.signal_len = signal
        self.name = "MACD Cross"
    
    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Signal]]:
        df = df.copy()
        df.columns = df.columns.str.lower()
        
        # Calculate MACD
        exp1 = df['close'].ewm(span=self.fast, adjust=False).mean()
        exp2 = df['close'].ewm(span=self.slow, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=self.signal_len, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        signals = []
        in_position = False
        
        for i in range(1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            if pd.isna(row['macd_signal']):
                continue
            
            # Buy when MACD crosses above signal
            if prev_row['macd'] < prev_row['macd_signal'] and row['macd'] >= row['macd_signal'] and not in_position:
                signals.append(Signal(
                    timestamp=df.index[i],
                    type='BUY',
                    price=row['close'],
                    confidence=0.7,
                    indicators={'macd': row['macd'], 'signal': row['macd_signal']},
                    context='MACD bullish crossover'
                ))
                in_position = True
            
            # Sell when MACD crosses below signal
            elif prev_row['macd'] > prev_row['macd_signal'] and row['macd'] <= row['macd_signal'] and in_position:
                signals.append(Signal(
                    timestamp=df.index[i],
                    type='SELL',
                    price=row['close'],
                    confidence=0.7,
                    indicators={'macd': row['macd'], 'signal': row['macd_signal']},
                    context='MACD bearish crossover'
                ))
                in_position = False
        
        return df, signals


class EMACrossover:
    """EMA Crossover Strategy (Golden/Death Cross)"""
    
    def __init__(self, fast=9, slow=21):
        self.fast = fast
        self.slow = slow
        self.name = f"EMA {fast}/{slow}"
    
    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Signal]]:
        df = df.copy()
        df.columns = df.columns.str.lower()
        
        df['ema_fast'] = df['close'].ewm(span=self.fast, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=self.slow, adjust=False).mean()
        
        signals = []
        in_position = False
        
        for i in range(1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            if pd.isna(row['ema_slow']):
                continue
            
            # Buy when fast EMA crosses above slow EMA
            if prev_row['ema_fast'] < prev_row['ema_slow'] and row['ema_fast'] >= row['ema_slow'] and not in_position:
                signals.append(Signal(
                    timestamp=df.index[i],
                    type='BUY',
                    price=row['close'],
                    confidence=0.7,
                    indicators={'ema_fast': row['ema_fast'], 'ema_slow': row['ema_slow']},
                    context='Golden cross'
                ))
                in_position = True
            
            # Sell when fast EMA crosses below slow EMA
            elif prev_row['ema_fast'] > prev_row['ema_slow'] and row['ema_fast'] <= row['ema_slow'] and in_position:
                signals.append(Signal(
                    timestamp=df.index[i],
                    type='SELL',
                    price=row['close'],
                    confidence=0.7,
                    indicators={'ema_fast': row['ema_fast'], 'ema_slow': row['ema_slow']},
                    context='Death cross'
                ))
                in_position = False
        
        return df, signals


class BreakoutStrategy:
    """Support/Resistance Breakout Strategy"""
    
    def __init__(self, lookback=20):
        self.lookback = lookback
        self.name = f"Breakout {lookback}"
    
    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Signal]]:
        df = df.copy()
        df.columns = df.columns.str.lower()
        
        df['highest'] = df['high'].rolling(window=self.lookback).max()
        df['lowest'] = df['low'].rolling(window=self.lookback).min()
        
        signals = []
        in_position = False
        
        for i in range(self.lookback + 1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            # Buy on breakout above previous high
            if row['close'] > prev_row['highest'] and not in_position:
                signals.append(Signal(
                    timestamp=df.index[i],
                    type='BUY',
                    price=row['close'],
                    confidence=0.7,
                    indicators={'resistance': prev_row['highest']},
                    context='Resistance breakout'
                ))
                in_position = True
            
            # Sell on breakdown below previous low
            elif row['close'] < prev_row['lowest'] and in_position:
                signals.append(Signal(
                    timestamp=df.index[i],
                    type='SELL',
                    price=row['close'],
                    confidence=0.7,
                    indicators={'support': prev_row['lowest']},
                    context='Support breakdown'
                ))
                in_position = False
        
        return df, signals


def compare_strategies(symbol='BTCUSDT', timeframe='720', days=730):
    """Compare all strategies"""
    import yaml
    from data.bybit_connector import BybitConnector
    from backtesting.backtest_engine import BacktestEngine
    from strategy.gold_line_strategy import GoldLineStrategy
    from strategy.bollinger_strategy import BollingerBandsStrategy
    
    # Load config
    with open('config/strategy_params.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 80)
    print(f"MULTI-STRATEGY COMPARISON")
    print(f"Symbol: {symbol} | Timeframe: {timeframe}min | History: {days} days")
    print("=" * 80)
    
    # Fetch data
    connector = BybitConnector(cache_dir='data/cache')
    df = connector.get_historical_data(symbol, timeframe, days, use_cache=True)
    
    if df.empty:
        print("No data")
        return
    
    print(f"📊 Loaded {len(df)} candles\n")
    
    # Strategies to test
    strategies = [
        ('Gold Line', GoldLineStrategy(config)),
        ('Bollinger Bands', BollingerBandsStrategy(config)),
        ('RSI 14/70/30', RSIOverboughtOversold(14, 70, 30)),
        ('RSI 7/80/20', RSIOverboughtOversold(7, 80, 20)),
        ('MACD 12/26/9', MACDCrossover(12, 26, 9)),
        ('MACD 8/17/9', MACDCrossover(8, 17, 9)),
        ('EMA 9/21', EMACrossover(9, 21)),
        ('EMA 20/50', EMACrossover(20, 50)),
        ('Breakout 20', BreakoutStrategy(20)),
        ('Breakout 50', BreakoutStrategy(50)),
    ]
    
    results = []
    
    for name, strategy in strategies:
        try:
            df_copy = df.copy()
            _, signals = strategy.run(df_copy)
            
            if not signals:
                print(f"  {name}: No signals generated")
                continue
            
            backtester = BacktestEngine(config)
            result = backtester.run(df_copy, signals, symbol, timeframe)
            
            results.append({
                'name': name,
                'trades': result.total_trades,
                'win_rate': result.win_rate,
                'pnl_pct': result.total_pnl_pct,
                'profit_factor': result.profit_factor,
                'sharpe': result.sharpe_ratio,
                'max_dd': result.max_drawdown_pct
            })
            
            print(f"  ✅ {name}: {result.total_trades} trades, {result.win_rate:.1f}% win, {result.total_pnl_pct:+.2f}%")
            
        except Exception as e:
            print(f"  ❌ {name}: {e}")
    
    # Sort by PnL
    results.sort(key=lambda x: x['pnl_pct'], reverse=True)
    
    print("\n" + "=" * 80)
    print("📈 STRATEGY COMPARISON RESULTS (sorted by PnL)")
    print("=" * 80)
    print(f"{'Strategy':<20} {'Trades':<8} {'Win %':<8} {'PnL %':<10} {'PF':<6} {'Sharpe':<8} {'Max DD':<8}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['name']:<20} {r['trades']:<8} {r['win_rate']:.1f}%{'':<3} {r['pnl_pct']:+.2f}%{'':<4} {r['profit_factor']:.2f}{'':<3} {r['sharpe']:.2f}{'':<4} {r['max_dd']:.2f}%")
    
    print("=" * 80)
    
    if results:
        best = results[0]
        print(f"\n🏆 BEST STRATEGY: {best['name']}")
        print(f"   PnL: {best['pnl_pct']:+.2f}% | Win Rate: {best['win_rate']:.1f}% | Profit Factor: {best['profit_factor']:.2f}")
    
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Multi-Strategy Comparison')
    parser.add_argument('--symbol', type=str, default='BTCUSDT')
    parser.add_argument('--timeframe', type=str, default='720')
    parser.add_argument('--days', type=int, default=730)
    
    args = parser.parse_args()
    compare_strategies(args.symbol, args.timeframe, args.days)
