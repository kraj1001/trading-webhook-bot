"""
TradingView Editor's Choice & Popular Community Strategies
These are the most popular and well-known strategies from TradingView's community.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from strategy.gold_line_strategy import Signal


class UTBotAlertStrategy:
    """
    UT Bot Alert Strategy (by QuantNomad)
    One of the most popular TradingView strategies.
    Uses ATR trailing stop for signals.
    """
    
    def __init__(self, key_value=1, atr_period=10):
        self.key_value = key_value
        self.atr_period = atr_period
        self.name = f"UT Bot {key_value}/{atr_period}"
    
    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Signal]]:
        df = df.copy()
        df.columns = df.columns.str.lower()
        
        # ATR calculation
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_period).mean()
        
        # Trailing stop
        n_loss = self.key_value * atr
        
        xatr_trailing_stop = pd.Series(index=df.index, dtype=float)
        xatr_trailing_stop.iloc[0] = df['close'].iloc[0]
        
        for i in range(1, len(df)):
            if df['close'].iloc[i] > xatr_trailing_stop.iloc[i-1] and df['close'].iloc[i-1] > xatr_trailing_stop.iloc[i-1]:
                xatr_trailing_stop.iloc[i] = max(xatr_trailing_stop.iloc[i-1], df['close'].iloc[i] - n_loss.iloc[i])
            elif df['close'].iloc[i] < xatr_trailing_stop.iloc[i-1] and df['close'].iloc[i-1] < xatr_trailing_stop.iloc[i-1]:
                xatr_trailing_stop.iloc[i] = min(xatr_trailing_stop.iloc[i-1], df['close'].iloc[i] + n_loss.iloc[i])
            elif df['close'].iloc[i] > xatr_trailing_stop.iloc[i-1]:
                xatr_trailing_stop.iloc[i] = df['close'].iloc[i] - n_loss.iloc[i]
            else:
                xatr_trailing_stop.iloc[i] = df['close'].iloc[i] + n_loss.iloc[i]
        
        df['trailing_stop'] = xatr_trailing_stop
        
        signals = []
        position = None
        
        for i in range(self.atr_period + 1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            # Buy signal: Close crosses above trailing stop
            if prev_row['close'] <= prev_row['trailing_stop'] and row['close'] > row['trailing_stop']:
                if position != 'LONG':
                    if position == 'SHORT':
                        signals.append(Signal(df.index[i], 'BUY', row['close'], 0.85, {}, 'UT Bot close short'))
                    signals.append(Signal(df.index[i], 'BUY', row['close'], 0.85, {}, 'UT Bot buy'))
                    position = 'LONG'
            
            # Sell signal: Close crosses below trailing stop
            elif prev_row['close'] >= prev_row['trailing_stop'] and row['close'] < row['trailing_stop']:
                if position != 'SHORT':
                    if position == 'LONG':
                        signals.append(Signal(df.index[i], 'SELL', row['close'], 0.85, {}, 'UT Bot close long'))
                    signals.append(Signal(df.index[i], 'SELL', row['close'], 0.85, {}, 'UT Bot sell'))
                    position = 'SHORT'
        
        return df, signals


class HalftrendStrategy:
    """
    Halftrend Strategy (by Alex Orekhov)
    Popular trend-following indicator.
    """
    
    def __init__(self, amplitude=2, channel_deviation=2):
        self.amplitude = amplitude
        self.channel_deviation = channel_deviation
        self.name = f"Halftrend {amplitude}"
    
    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Signal]]:
        df = df.copy()
        df.columns = df.columns.str.lower()
        
        # ATR for channel
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=100).mean()
        
        # High/Low with amplitude
        high_ma = df['high'].rolling(window=self.amplitude).max()
        low_ma = df['low'].rolling(window=self.amplitude).min()
        
        halftrend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=int)
        
        direction.iloc[0] = 1
        halftrend.iloc[0] = (high_ma.iloc[0] + low_ma.iloc[0]) / 2
        
        for i in range(1, len(df)):
            if pd.isna(high_ma.iloc[i]) or pd.isna(low_ma.iloc[i]):
                halftrend.iloc[i] = halftrend.iloc[i-1]
                direction.iloc[i] = direction.iloc[i-1]
                continue
            
            if direction.iloc[i-1] == 1:
                max_low = max(low_ma.iloc[i], halftrend.iloc[i-1])
                if df['close'].iloc[i] < max_low - (atr.iloc[i] * self.channel_deviation if not pd.isna(atr.iloc[i]) else 0):
                    direction.iloc[i] = -1
                    halftrend.iloc[i] = high_ma.iloc[i]
                else:
                    direction.iloc[i] = 1
                    halftrend.iloc[i] = max_low
            else:
                min_high = min(high_ma.iloc[i], halftrend.iloc[i-1])
                if df['close'].iloc[i] > min_high + (atr.iloc[i] * self.channel_deviation if not pd.isna(atr.iloc[i]) else 0):
                    direction.iloc[i] = 1
                    halftrend.iloc[i] = low_ma.iloc[i]
                else:
                    direction.iloc[i] = -1
                    halftrend.iloc[i] = min_high
        
        df['halftrend'] = halftrend
        df['direction'] = direction
        
        signals = []
        position = None
        
        for i in range(self.amplitude + 1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            if row['direction'] == 1 and prev_row['direction'] == -1:
                if position != 'LONG':
                    if position == 'SHORT':
                        signals.append(Signal(df.index[i], 'BUY', row['close'], 0.8, {}, 'Halftrend close short'))
                    signals.append(Signal(df.index[i], 'BUY', row['close'], 0.8, {}, 'Halftrend bullish'))
                    position = 'LONG'
            
            elif row['direction'] == -1 and prev_row['direction'] == 1:
                if position != 'SHORT':
                    if position == 'LONG':
                        signals.append(Signal(df.index[i], 'SELL', row['close'], 0.8, {}, 'Halftrend close long'))
                    signals.append(Signal(df.index[i], 'SELL', row['close'], 0.8, {}, 'Halftrend bearish'))
                    position = 'SHORT'
        
        return df, signals


class ChandelierExitStrategy:
    """
    Chandelier Exit Strategy
    Classic ATR-based trailing stop strategy.
    """
    
    def __init__(self, atr_period=22, atr_mult=3.0):
        self.atr_period = atr_period
        self.atr_mult = atr_mult
        self.name = f"Chandelier {atr_period}"
    
    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Signal]]:
        df = df.copy()
        df.columns = df.columns.str.lower()
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_period).mean()
        
        # Chandelier Exit
        highest_high = df['high'].rolling(window=self.atr_period).max()
        lowest_low = df['low'].rolling(window=self.atr_period).min()
        
        df['chandelier_long'] = highest_high - atr * self.atr_mult
        df['chandelier_short'] = lowest_low + atr * self.atr_mult
        
        signals = []
        position = None
        
        for i in range(self.atr_period + 1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            if pd.isna(row['chandelier_long']):
                continue
            
            # Buy: Price crosses above chandelier short (uptrend)
            if prev_row['close'] <= prev_row['chandelier_short'] and row['close'] > row['chandelier_short']:
                if position != 'LONG':
                    if position == 'SHORT':
                        signals.append(Signal(df.index[i], 'BUY', row['close'], 0.85, {}, 'Chandelier close short'))
                    signals.append(Signal(df.index[i], 'BUY', row['close'], 0.85, {}, 'Chandelier buy'))
                    position = 'LONG'
            
            # Sell: Price crosses below chandelier long (downtrend)
            elif prev_row['close'] >= prev_row['chandelier_long'] and row['close'] < row['chandelier_long']:
                if position != 'SHORT':
                    if position == 'LONG':
                        signals.append(Signal(df.index[i], 'SELL', row['close'], 0.85, {}, 'Chandelier close long'))
                    signals.append(Signal(df.index[i], 'SELL', row['close'], 0.85, {}, 'Chandelier sell'))
                    position = 'SHORT'
        
        return df, signals


class SSLChannelStrategy:
    """
    SSL Channel Strategy
    Very popular indicator on TradingView.
    Uses SMA crossover with high/low filtering.
    """
    
    def __init__(self, period=10):
        self.period = period
        self.name = f"SSL Channel {period}"
    
    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Signal]]:
        df = df.copy()
        df.columns = df.columns.str.lower()
        
        # SSL Channel
        sma_high = df['high'].rolling(window=self.period).mean()
        sma_low = df['low'].rolling(window=self.period).mean()
        
        hlv = pd.Series(index=df.index, dtype=int)
        hlv.iloc[0] = 0
        
        for i in range(1, len(df)):
            if df['close'].iloc[i] > sma_high.iloc[i]:
                hlv.iloc[i] = 1
            elif df['close'].iloc[i] < sma_low.iloc[i]:
                hlv.iloc[i] = -1
            else:
                hlv.iloc[i] = hlv.iloc[i-1]
        
        df['ssl_down'] = np.where(hlv < 0, sma_high, sma_low)
        df['ssl_up'] = np.where(hlv < 0, sma_low, sma_high)
        df['ssl_hlv'] = hlv
        
        signals = []
        position = None
        
        for i in range(self.period + 1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            # Buy: SSL crosses up
            if prev_row['ssl_hlv'] <= 0 and row['ssl_hlv'] > 0:
                if position != 'LONG':
                    if position == 'SHORT':
                        signals.append(Signal(df.index[i], 'BUY', row['close'], 0.8, {}, 'SSL close short'))
                    signals.append(Signal(df.index[i], 'BUY', row['close'], 0.8, {}, 'SSL bullish'))
                    position = 'LONG'
            
            # Sell: SSL crosses down
            elif prev_row['ssl_hlv'] >= 0 and row['ssl_hlv'] < 0:
                if position != 'SHORT':
                    if position == 'LONG':
                        signals.append(Signal(df.index[i], 'SELL', row['close'], 0.8, {}, 'SSL close long'))
                    signals.append(Signal(df.index[i], 'SELL', row['close'], 0.8, {}, 'SSL bearish'))
                    position = 'SHORT'
        
        return df, signals


class AroonStrategy:
    """
    Aroon Indicator Strategy
    Measures time since highs/lows.
    """
    
    def __init__(self, period=25):
        self.period = period
        self.name = f"Aroon {period}"
    
    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Signal]]:
        df = df.copy()
        df.columns = df.columns.str.lower()
        
        # Aroon Up: Number of periods since highest high
        # Aroon Down: Number of periods since lowest low
        
        aroon_up = pd.Series(index=df.index, dtype=float)
        aroon_down = pd.Series(index=df.index, dtype=float)
        
        for i in range(self.period, len(df)):
            window = df['high'].iloc[i-self.period:i+1]
            periods_since_high = self.period - window.values.argmax()
            aroon_up.iloc[i] = ((self.period - periods_since_high) / self.period) * 100
            
            window = df['low'].iloc[i-self.period:i+1]
            periods_since_low = self.period - window.values.argmin()
            aroon_down.iloc[i] = ((self.period - periods_since_low) / self.period) * 100
        
        df['aroon_up'] = aroon_up
        df['aroon_down'] = aroon_down
        
        signals = []
        position = None
        
        for i in range(self.period + 1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            if pd.isna(row['aroon_up']) or pd.isna(row['aroon_down']):
                continue
            
            # Buy: Aroon Up crosses above Aroon Down
            if prev_row['aroon_up'] <= prev_row['aroon_down'] and row['aroon_up'] > row['aroon_down']:
                if position != 'LONG':
                    if position == 'SHORT':
                        signals.append(Signal(df.index[i], 'BUY', row['close'], 0.75, {}, 'Aroon close short'))
                    signals.append(Signal(df.index[i], 'BUY', row['close'], 0.75, {}, 'Aroon bullish'))
                    position = 'LONG'
            
            # Sell: Aroon Down crosses above Aroon Up
            elif prev_row['aroon_down'] <= prev_row['aroon_up'] and row['aroon_down'] > row['aroon_up']:
                if position != 'SHORT':
                    if position == 'LONG':
                        signals.append(Signal(df.index[i], 'SELL', row['close'], 0.75, {}, 'Aroon close long'))
                    signals.append(Signal(df.index[i], 'SELL', row['close'], 0.75, {}, 'Aroon bearish'))
                    position = 'SHORT'
        
        return df, signals


class CMFStrategy:
    """
    Chaikin Money Flow Strategy
    Measures buying/selling pressure.
    """
    
    def __init__(self, period=20):
        self.period = period
        self.name = f"CMF {period}"
    
    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Signal]]:
        df = df.copy()
        df.columns = df.columns.str.lower()
        
        # Money Flow Multiplier
        mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        mfm = mfm.replace([np.inf, -np.inf], 0).fillna(0)
        
        # Money Flow Volume
        mfv = mfm * df['volume']
        
        # CMF
        df['cmf'] = mfv.rolling(window=self.period).sum() / df['volume'].rolling(window=self.period).sum()
        
        signals = []
        position = None
        
        for i in range(self.period + 1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            if pd.isna(row['cmf']):
                continue
            
            # Buy: CMF crosses above zero
            if prev_row['cmf'] < 0 and row['cmf'] >= 0:
                if position != 'LONG':
                    if position == 'SHORT':
                        signals.append(Signal(df.index[i], 'BUY', row['close'], 0.8, {'cmf': row['cmf']}, 'CMF close short'))
                    signals.append(Signal(df.index[i], 'BUY', row['close'], 0.8, {'cmf': row['cmf']}, 'CMF bullish'))
                    position = 'LONG'
            
            # Sell: CMF crosses below zero
            elif prev_row['cmf'] > 0 and row['cmf'] <= 0:
                if position != 'SHORT':
                    if position == 'LONG':
                        signals.append(Signal(df.index[i], 'SELL', row['close'], 0.8, {'cmf': row['cmf']}, 'CMF close long'))
                    signals.append(Signal(df.index[i], 'SELL', row['close'], 0.8, {'cmf': row['cmf']}, 'CMF bearish'))
                    position = 'SHORT'
        
        return df, signals


class VortexStrategy:
    """
    Vortex Indicator Strategy
    Measures positive/negative trend movement.
    """
    
    def __init__(self, period=14):
        self.period = period
        self.name = f"Vortex {period}"
    
    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Signal]]:
        df = df.copy()
        df.columns = df.columns.str.lower()
        
        # True Range
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Vortex Movement
        vm_plus = np.abs(df['high'] - df['low'].shift())
        vm_minus = np.abs(df['low'] - df['high'].shift())
        
        # Vortex Indicator
        sum_tr = tr.rolling(window=self.period).sum()
        df['vi_plus'] = vm_plus.rolling(window=self.period).sum() / sum_tr
        df['vi_minus'] = vm_minus.rolling(window=self.period).sum() / sum_tr
        
        signals = []
        position = None
        
        for i in range(self.period + 1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            if pd.isna(row['vi_plus']) or pd.isna(row['vi_minus']):
                continue
            
            # Buy: VI+ crosses above VI-
            if prev_row['vi_plus'] <= prev_row['vi_minus'] and row['vi_plus'] > row['vi_minus']:
                if position != 'LONG':
                    if position == 'SHORT':
                        signals.append(Signal(df.index[i], 'BUY', row['close'], 0.75, {}, 'Vortex close short'))
                    signals.append(Signal(df.index[i], 'BUY', row['close'], 0.75, {}, 'Vortex bullish'))
                    position = 'LONG'
            
            # Sell: VI- crosses above VI+
            elif prev_row['vi_minus'] <= prev_row['vi_plus'] and row['vi_minus'] > row['vi_plus']:
                if position != 'SHORT':
                    if position == 'LONG':
                        signals.append(Signal(df.index[i], 'SELL', row['close'], 0.75, {}, 'Vortex close long'))
                    signals.append(Signal(df.index[i], 'SELL', row['close'], 0.75, {}, 'Vortex bearish'))
                    position = 'SHORT'
        
        return df, signals


class MFIStrategy:
    """
    Money Flow Index Strategy
    Volume-weighted RSI.
    """
    
    def __init__(self, period=14, overbought=80, oversold=20):
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        self.name = f"MFI {period}"
    
    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Signal]]:
        df = df.copy()
        df.columns = df.columns.str.lower()
        
        # Typical Price
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        
        # Money Flow
        money_flow = typical_price * df['volume']
        
        # Positive/Negative Money Flow
        pos_flow = pd.Series(index=df.index, dtype=float)
        neg_flow = pd.Series(index=df.index, dtype=float)
        
        for i in range(1, len(df)):
            if typical_price.iloc[i] > typical_price.iloc[i-1]:
                pos_flow.iloc[i] = money_flow.iloc[i]
                neg_flow.iloc[i] = 0
            else:
                pos_flow.iloc[i] = 0
                neg_flow.iloc[i] = money_flow.iloc[i]
        
        # Money Ratio and MFI
        pos_sum = pos_flow.rolling(window=self.period).sum()
        neg_sum = neg_flow.rolling(window=self.period).sum()
        money_ratio = pos_sum / neg_sum
        df['mfi'] = 100 - (100 / (1 + money_ratio))
        
        signals = []
        position = None
        
        for i in range(self.period + 1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            if pd.isna(row['mfi']):
                continue
            
            # Buy: MFI crosses above oversold
            if prev_row['mfi'] < self.oversold and row['mfi'] >= self.oversold:
                if position != 'LONG':
                    if position == 'SHORT':
                        signals.append(Signal(df.index[i], 'BUY', row['close'], 0.8, {'mfi': row['mfi']}, 'MFI close short'))
                    signals.append(Signal(df.index[i], 'BUY', row['close'], 0.8, {'mfi': row['mfi']}, 'MFI oversold'))
                    position = 'LONG'
            
            # Sell: MFI crosses below overbought
            elif prev_row['mfi'] > self.overbought and row['mfi'] <= self.overbought:
                if position != 'SHORT':
                    if position == 'LONG':
                        signals.append(Signal(df.index[i], 'SELL', row['close'], 0.8, {'mfi': row['mfi']}, 'MFI close long'))
                    signals.append(Signal(df.index[i], 'SELL', row['close'], 0.8, {'mfi': row['mfi']}, 'MFI overbought'))
                    position = 'SHORT'
        
        return df, signals


def compare_editors_choice_strategies(symbol='BTCUSDT', timeframe='720', days=730):
    """Compare all Editor's Choice strategies"""
    import yaml
    from data.bybit_connector import BybitConnector
    from backtesting.backtest_engine import BacktestEngine
    
    with open('config/strategy_params.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 80)
    print(f"TRADINGVIEW EDITOR'S CHOICE STRATEGIES")
    print(f"Symbol: {symbol} | Timeframe: {timeframe}min | History: {days} days")
    print("=" * 80)
    
    connector = BybitConnector(cache_dir='data/cache')
    df = connector.get_historical_data(symbol, timeframe, days, use_cache=True)
    
    print(f"📊 Loaded {len(df)} candles\n")
    
    strategies = [
        UTBotAlertStrategy(1, 10),
        UTBotAlertStrategy(2, 10),
        HalftrendStrategy(2, 2),
        HalftrendStrategy(3, 2),
        ChandelierExitStrategy(22, 3.0),
        ChandelierExitStrategy(14, 2.0),
        SSLChannelStrategy(10),
        SSLChannelStrategy(14),
        AroonStrategy(25),
        AroonStrategy(14),
        CMFStrategy(20),
        CMFStrategy(10),
        VortexStrategy(14),
        VortexStrategy(21),
        MFIStrategy(14, 80, 20),
        MFIStrategy(10, 70, 30),
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
    print("📈 EDITOR'S CHOICE STRATEGIES RANKED BY PnL")
    print("=" * 80)
    print(f"{'Strategy':<25} {'Trades':<8} {'Win %':<8} {'PnL %':<10} {'PF':<8}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['name']:<25} {r['trades']:<8} {r['win_rate']:.1f}%{'':<3} {r['pnl_pct']:+.2f}%{'':<4} {r['profit_factor']:.2f}")
    
    print("=" * 80)
    
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Editor's Choice Strategies Comparison")
    parser.add_argument('--symbol', type=str, default='BTCUSDT')
    parser.add_argument('--timeframe', type=str, default='720')
    parser.add_argument('--days', type=int, default=730)
    
    args = parser.parse_args()
    compare_editors_choice_strategies(args.symbol, args.timeframe, args.days)
