"""
Microstructure Trading Strategies
Based on Holographic Market Microstructure indicator concepts.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from strategy.gold_line_strategy import Signal


class OrderFlowImbalanceStrategy:
    """
    Order Flow Imbalance Strategy
    Based on HMS indicator's buy/sell volume imbalance.
    """
    
    def __init__(self, period=20, threshold=2.0):
        self.period = period
        self.threshold = threshold
        self.name = f"Order Flow {period}"
    
    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Signal]]:
        df = df.copy()
        df.columns = df.columns.str.lower()
        
        # Calculate buy/sell volume
        df['buy_vol'] = df['volume'] * (df['close'] - df['low']) / (df['high'] - df['low'])
        df['sell_vol'] = df['volume'] * (df['high'] - df['close']) / (df['high'] - df['low'])
        df['flow_imbalance'] = df['buy_vol'] - df['sell_vol']
        
        # Cumulative order flow
        df['ofi'] = df['flow_imbalance'].cumsum()
        df['ofi_ma'] = df['ofi'].rolling(window=self.period).mean()
        df['ofi_std'] = (df['ofi'] - df['ofi_ma']).rolling(window=self.period).std()
        df['ofi_normalized'] = (df['ofi'] - df['ofi_ma']) / df['ofi_std']
        
        signals = []
        position = None
        
        for i in range(self.period + 1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            if pd.isna(row['ofi_normalized']):
                continue
            
            # Strong buying flow
            if row['ofi_normalized'] > self.threshold and prev_row['ofi_normalized'] <= self.threshold:
                if position != 'LONG':
                    if position == 'SHORT':
                        signals.append(Signal(df.index[i], 'BUY', row['close'], 0.85, {}, 'OFI close short'))
                    signals.append(Signal(df.index[i], 'BUY', row['close'], 0.85, 
                                          {'ofi': row['ofi_normalized']}, 'Strong buying flow'))
                    position = 'LONG'
            
            # Strong selling flow
            elif row['ofi_normalized'] < -self.threshold and prev_row['ofi_normalized'] >= -self.threshold:
                if position != 'SHORT':
                    if position == 'LONG':
                        signals.append(Signal(df.index[i], 'SELL', row['close'], 0.85, {}, 'OFI close long'))
                    signals.append(Signal(df.index[i], 'SELL', row['close'], 0.85,
                                          {'ofi': row['ofi_normalized']}, 'Strong selling flow'))
                    position = 'SHORT'
        
        return df, signals


class SmartMoneyStrategy:
    """
    Smart Money Strategy
    Based on HMS indicator's accumulation/distribution detection.
    """
    
    def __init__(self, period=20, threshold=70):
        self.period = period
        self.threshold = threshold
        self.name = f"Smart Money {period}"
    
    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Signal]]:
        df = df.copy()
        df.columns = df.columns.str.lower()
        
        # Price and volume rate of change
        df['price_roc'] = df['close'].pct_change(periods=self.period) * 100
        df['vol_roc'] = df['volume'].pct_change(periods=self.period) * 100
        
        # Smart money index
        df['smi'] = 0
        df.loc[(df['price_roc'] > 0) & (df['vol_roc'] > self.threshold), 'smi'] = 1
        df.loc[(df['price_roc'] < 0) & (df['vol_roc'] > self.threshold), 'smi'] = -1
        
        signals = []
        position = None
        
        for i in range(self.period + 1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            # Accumulation detected
            if row['smi'] == 1 and prev_row['smi'] != 1:
                if position != 'LONG':
                    if position == 'SHORT':
                        signals.append(Signal(df.index[i], 'BUY', row['close'], 0.9, {}, 'SM close short'))
                    signals.append(Signal(df.index[i], 'BUY', row['close'], 0.9, {}, 'Smart money accumulation'))
                    position = 'LONG'
            
            # Distribution detected
            elif row['smi'] == -1 and prev_row['smi'] != -1:
                if position != 'SHORT':
                    if position == 'LONG':
                        signals.append(Signal(df.index[i], 'SELL', row['close'], 0.9, {}, 'SM close long'))
                    signals.append(Signal(df.index[i], 'SELL', row['close'], 0.9, {}, 'Smart money distribution'))
                    position = 'SHORT'
        
        return df, signals


class FractalStructureStrategy:
    """
    Fractal Structure Strategy
    Based on HMS indicator's fractal highs/lows and microtrend.
    """
    
    def __init__(self, fractal_period=5, structure_depth=3):
        self.fractal_period = fractal_period
        self.structure_depth = structure_depth
        self.name = f"Fractal {fractal_period}"
    
    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Signal]]:
        df = df.copy()
        df.columns = df.columns.str.lower()
        
        # Detect fractals
        fp = self.fractal_period
        df['fractal_high'] = df['high'] == df['high'].rolling(window=fp*2+1, center=True).max()
        df['fractal_low'] = df['low'] == df['low'].rolling(window=fp*2+1, center=True).min()
        
        signals = []
        position = None
        
        structure_highs = []
        structure_lows = []
        
        for i in range(fp*2+1, len(df)):
            row = df.iloc[i]
            
            # Collect fractals
            if df['fractal_high'].iloc[i-fp]:
                structure_highs.append(df['high'].iloc[i-fp])
                if len(structure_highs) > self.structure_depth:
                    structure_highs.pop(0)
            
            if df['fractal_low'].iloc[i-fp]:
                structure_lows.append(df['low'].iloc[i-fp])
                if len(structure_lows) > self.structure_depth:
                    structure_lows.pop(0)
            
            # Calculate microtrend
            if len(structure_highs) >= 2 and len(structure_lows) >= 2:
                high_trend = 1 if structure_highs[-1] > structure_highs[-2] else -1
                low_trend = 1 if structure_lows[-1] > structure_lows[-2] else -1
                microtrend = high_trend + low_trend
                
                # Bullish structure
                if microtrend >= 2 and position != 'LONG':
                    if position == 'SHORT':
                        signals.append(Signal(df.index[i], 'BUY', row['close'], 0.8, {}, 'Fractal close short'))
                    signals.append(Signal(df.index[i], 'BUY', row['close'], 0.8, {}, 'Bullish fractal structure'))
                    position = 'LONG'
                
                # Bearish structure
                elif microtrend <= -2 and position != 'SHORT':
                    if position == 'LONG':
                        signals.append(Signal(df.index[i], 'SELL', row['close'], 0.8, {}, 'Fractal close long'))
                    signals.append(Signal(df.index[i], 'SELL', row['close'], 0.8, {}, 'Bearish fractal structure'))
                    position = 'SHORT'
        
        return df, signals


class LiquidityVoidStrategy:
    """
    Liquidity Void Strategy
    Based on HMS indicator's liquidity void detection.
    Trades reversals after liquidity voids.
    """
    
    def __init__(self, atr_mult=1.5, vol_threshold=0.7):
        self.atr_mult = atr_mult
        self.vol_threshold = vol_threshold
        self.name = f"Liquidity Void {atr_mult}"
    
    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Signal]]:
        df = df.copy()
        df.columns = df.columns.str.lower()
        
        # ATR for void threshold
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=14).mean()
        
        df['avg_vol'] = df['volume'].rolling(window=20).mean()
        df['range'] = df['high'] - df['low']
        
        # Liquidity void: Large range + low volume
        df['is_void'] = (df['range'] > df['atr'] * self.atr_mult) & (df['volume'] < df['avg_vol'] * self.vol_threshold)
        
        signals = []
        position = None
        
        for i in range(25, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            # Void followed by bullish candle
            if prev_row['is_void'] and row['close'] > row['open']:
                if position != 'LONG':
                    if position == 'SHORT':
                        signals.append(Signal(df.index[i], 'BUY', row['close'], 0.75, {}, 'Void close short'))
                    signals.append(Signal(df.index[i], 'BUY', row['close'], 0.75, {}, 'Void reversal up'))
                    position = 'LONG'
            
            # Void followed by bearish candle
            elif prev_row['is_void'] and row['close'] < row['open']:
                if position != 'SHORT':
                    if position == 'LONG':
                        signals.append(Signal(df.index[i], 'SELL', row['close'], 0.75, {}, 'Void close long'))
                    signals.append(Signal(df.index[i], 'SELL', row['close'], 0.75, {}, 'Void reversal down'))
                    position = 'SHORT'
        
        return df, signals


class DivergenceStrategy:
    """
    Price/Volume Divergence Strategy
    Based on HMS indicator's divergence detection.
    """
    
    def __init__(self, period=20):
        self.period = period
        self.name = f"Divergence {period}"
    
    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Signal]]:
        df = df.copy()
        df.columns = df.columns.str.lower()
        
        # Linear regression trends
        from scipy import stats
        
        df['price_trend'] = np.nan
        df['vol_trend'] = np.nan
        
        for i in range(self.period, len(df)):
            x = np.arange(self.period)
            price_slope, _, _, _, _ = stats.linregress(x, df['close'].iloc[i-self.period:i].values)
            vol_slope, _, _, _, _ = stats.linregress(x, df['volume'].iloc[i-self.period:i].values)
            df.loc[df.index[i], 'price_trend'] = price_slope
            df.loc[df.index[i], 'vol_trend'] = vol_slope
        
        # Divergence: Price up + Volume down, or Price down + Volume up
        df['divergence'] = ((df['price_trend'] > 0) & (df['vol_trend'] < 0)) | \
                           ((df['price_trend'] < 0) & (df['vol_trend'] > 0))
        
        signals = []
        position = None
        
        for i in range(self.period + 1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            if row['divergence'] and not prev_row['divergence']:
                # Bearish divergence: price up, volume down
                if row['price_trend'] > 0 and row['vol_trend'] < 0:
                    if position != 'SHORT':
                        if position == 'LONG':
                            signals.append(Signal(df.index[i], 'SELL', row['close'], 0.8, {}, 'Div close long'))
                        signals.append(Signal(df.index[i], 'SELL', row['close'], 0.8, {}, 'Bearish divergence'))
                        position = 'SHORT'
                
                # Bullish divergence: price down, volume up
                elif row['price_trend'] < 0 and row['vol_trend'] > 0:
                    if position != 'LONG':
                        if position == 'SHORT':
                            signals.append(Signal(df.index[i], 'BUY', row['close'], 0.8, {}, 'Div close short'))
                        signals.append(Signal(df.index[i], 'BUY', row['close'], 0.8, {}, 'Bullish divergence'))
                        position = 'LONG'
        
        return df, signals


class RangeOscillatorStrategy:
    """
    Range Oscillator Strategy (Zeiierman)
    Trades based on range-weighted oscillator breakouts.
    """
    
    def __init__(self, length=50, atr_mult=2.0):
        self.length = length
        self.atr_mult = atr_mult
        self.name = f"Range Osc {length}"
    
    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Signal]]:
        df = df.copy()
        df.columns = df.columns.str.lower()
        
        # ATR for range
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=200).mean()
        df['range_atr'] = df['atr'] * self.atr_mult
        
        # Weighted MA
        df['ma'] = np.nan
        for i in range(self.length, len(df)):
            sum_wc = 0.0
            sum_w = 0.0
            for j in range(self.length):
                delta = abs(df['close'].iloc[i-j] - df['close'].iloc[i-j-1])
                w = delta / df['close'].iloc[i-j-1] if df['close'].iloc[i-j-1] != 0 else 0
                sum_wc += df['close'].iloc[i-j] * w
                sum_w += w
            df.loc[df.index[i], 'ma'] = sum_wc / sum_w if sum_w != 0 else df['close'].iloc[i]
        
        # Range Oscillator
        df['osc'] = 100 * (df['close'] - df['ma']) / df['range_atr']
        
        signals = []
        position = None
        
        for i in range(self.length + 1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            if pd.isna(row['osc']) or pd.isna(row['ma']):
                continue
            
            # Breakout above upper range
            if row['close'] > row['ma'] + row['range_atr'] and prev_row['close'] <= prev_row['ma'] + prev_row['range_atr']:
                if position != 'LONG':
                    if position == 'SHORT':
                        signals.append(Signal(df.index[i], 'BUY', row['close'], 0.85, {}, 'RO close short'))
                    signals.append(Signal(df.index[i], 'BUY', row['close'], 0.85, {'osc': row['osc']}, 'Range breakout up'))
                    position = 'LONG'
            
            # Breakout below lower range
            elif row['close'] < row['ma'] - row['range_atr'] and prev_row['close'] >= prev_row['ma'] - prev_row['range_atr']:
                if position != 'SHORT':
                    if position == 'LONG':
                        signals.append(Signal(df.index[i], 'SELL', row['close'], 0.85, {}, 'RO close long'))
                    signals.append(Signal(df.index[i], 'SELL', row['close'], 0.85, {'osc': row['osc']}, 'Range breakout down'))
                    position = 'SHORT'
        
        return df, signals


def compare_microstructure_strategies(symbol='BTCUSDT', timeframe='720', days=730):
    """Compare all microstructure strategies"""
    import yaml
    from data.bybit_connector import BybitConnector
    from backtesting.backtest_engine import BacktestEngine
    
    with open('config/strategy_params.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 80)
    print(f"MICROSTRUCTURE STRATEGIES (HMS)")
    print(f"Symbol: {symbol} | Timeframe: {timeframe}min | History: {days} days")
    print("=" * 80)
    
    connector = BybitConnector(cache_dir='data/cache')
    df = connector.get_historical_data(symbol, timeframe, days, use_cache=True)
    
    print(f"📊 Loaded {len(df)} candles\n")
    
    strategies = [
        OrderFlowImbalanceStrategy(20, 2.0),
        OrderFlowImbalanceStrategy(10, 1.5),
        SmartMoneyStrategy(20, 70),
        SmartMoneyStrategy(10, 50),
        FractalStructureStrategy(5, 3),
        FractalStructureStrategy(3, 2),
        LiquidityVoidStrategy(1.5, 0.7),
        LiquidityVoidStrategy(2.0, 0.5),
        DivergenceStrategy(20),
        DivergenceStrategy(10),
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
    print("📈 MICROSTRUCTURE STRATEGIES RANKED BY PnL")
    print("=" * 80)
    
    for r in results:
        print(f"{r['name']:<25} {r['trades']:<8} {r['win_rate']:.1f}%{'':<3} {r['pnl_pct']:+.2f}%")
    
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Microstructure Strategies')
    parser.add_argument('--symbol', type=str, default='BTCUSDT')
    parser.add_argument('--timeframe', type=str, default='720')
    parser.add_argument('--days', type=int, default=730)
    
    args = parser.parse_args()
    compare_microstructure_strategies(args.symbol, args.timeframe, args.days)
