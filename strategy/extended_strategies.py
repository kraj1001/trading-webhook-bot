"""
Additional TradingView Popular Strategies
More strategies from TradingView's most popular scripts.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from strategy.gold_line_strategy import Signal


class DonchianChannelStrategy:
    """
    Donchian Channel Strategy (Turtle Trading Basis)
    Buy: Price breaks above highest high of N periods
    Sell: Price breaks below lowest low of N periods
    """
    
    def __init__(self, entry_period=20, exit_period=10):
        self.entry_period = entry_period
        self.exit_period = exit_period
        self.name = f"Donchian {entry_period}/{exit_period}"
    
    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Signal]]:
        df = df.copy()
        df.columns = df.columns.str.lower()
        
        df['entry_high'] = df['high'].rolling(window=self.entry_period).max()
        df['entry_low'] = df['low'].rolling(window=self.entry_period).min()
        df['exit_high'] = df['high'].rolling(window=self.exit_period).max()
        df['exit_low'] = df['low'].rolling(window=self.exit_period).min()
        
        signals = []
        position = None
        
        for i in range(self.entry_period + 1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            # Long entry: Break above entry channel
            if row['close'] > prev_row['entry_high'] and position != 'LONG':
                if position == 'SHORT':
                    signals.append(Signal(df.index[i], 'BUY', row['close'], 0.8, {}, 'Donchian close short'))
                signals.append(Signal(df.index[i], 'BUY', row['close'], 0.8, {}, 'Donchian breakout up'))
                position = 'LONG'
            
            # Short entry: Break below entry channel
            elif row['close'] < prev_row['entry_low'] and position != 'SHORT':
                if position == 'LONG':
                    signals.append(Signal(df.index[i], 'SELL', row['close'], 0.8, {}, 'Donchian close long'))
                signals.append(Signal(df.index[i], 'SELL', row['close'], 0.8, {}, 'Donchian breakout down'))
                position = 'SHORT'
        
        return df, signals


class KeltnerChannelStrategy:
    """
    Keltner Channel Strategy
    Uses EMA and ATR for channel bands
    """
    
    def __init__(self, ema_period=20, atr_period=10, atr_mult=2.0):
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.atr_mult = atr_mult
        self.name = f"Keltner {ema_period}"
    
    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Signal]]:
        df = df.copy()
        df.columns = df.columns.str.lower()
        
        # EMA
        df['kc_mid'] = df['close'].ewm(span=self.ema_period, adjust=False).mean()
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_period).mean()
        
        df['kc_upper'] = df['kc_mid'] + (self.atr_mult * atr)
        df['kc_lower'] = df['kc_mid'] - (self.atr_mult * atr)
        
        signals = []
        position = None
        
        for i in range(self.ema_period + 1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            if pd.isna(row['kc_upper']):
                continue
            
            # Buy: Close above upper channel
            if row['close'] > row['kc_upper'] and prev_row['close'] <= prev_row['kc_upper'] and position != 'LONG':
                if position == 'SHORT':
                    signals.append(Signal(df.index[i], 'BUY', row['close'], 0.8, {}, 'Keltner close short'))
                signals.append(Signal(df.index[i], 'BUY', row['close'], 0.8, {}, 'Keltner break up'))
                position = 'LONG'
            
            # Sell: Close below lower channel
            elif row['close'] < row['kc_lower'] and prev_row['close'] >= prev_row['kc_lower'] and position != 'SHORT':
                if position == 'LONG':
                    signals.append(Signal(df.index[i], 'SELL', row['close'], 0.8, {}, 'Keltner close long'))
                signals.append(Signal(df.index[i], 'SELL', row['close'], 0.8, {}, 'Keltner break down'))
                position = 'SHORT'
        
        return df, signals


class HullMAStrategy:
    """
    Hull Moving Average Strategy
    Faster MA with less lag
    """
    
    def __init__(self, period=16):
        self.period = period
        self.name = f"Hull MA {period}"
    
    def _wma(self, series, period):
        weights = np.arange(1, period + 1)
        return series.rolling(window=period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    
    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Signal]]:
        df = df.copy()
        df.columns = df.columns.str.lower()
        
        half_period = int(self.period / 2)
        sqrt_period = int(np.sqrt(self.period))
        
        wma_half = self._wma(df['close'], half_period)
        wma_full = self._wma(df['close'], self.period)
        
        raw_hull = 2 * wma_half - wma_full
        df['hull_ma'] = self._wma(raw_hull, sqrt_period)
        
        signals = []
        position = None
        
        for i in range(self.period + sqrt_period + 1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            if pd.isna(row['hull_ma']) or pd.isna(prev_row['hull_ma']):
                continue
            
            # Hull MA turning up
            if row['hull_ma'] > prev_row['hull_ma'] and position != 'LONG':
                if position == 'SHORT':
                    signals.append(Signal(df.index[i], 'BUY', row['close'], 0.8, {}, 'Hull close short'))
                signals.append(Signal(df.index[i], 'BUY', row['close'], 0.8, {}, 'Hull turn up'))
                position = 'LONG'
            
            # Hull MA turning down
            elif row['hull_ma'] < prev_row['hull_ma'] and position != 'SHORT':
                if position == 'LONG':
                    signals.append(Signal(df.index[i], 'SELL', row['close'], 0.8, {}, 'Hull close long'))
                signals.append(Signal(df.index[i], 'SELL', row['close'], 0.8, {}, 'Hull turn down'))
                position = 'SHORT'
        
        return df, signals


class SqueezeMomentumStrategy:
    """
    Squeeze Momentum Indicator (LazyBear's famous indicator)
    Combines Bollinger Bands and Keltner Channel
    """
    
    def __init__(self, bb_length=20, bb_mult=2.0, kc_length=20, kc_mult=1.5):
        self.bb_length = bb_length
        self.bb_mult = bb_mult
        self.kc_length = kc_length
        self.kc_mult = kc_mult
        self.name = "Squeeze Momentum"
    
    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Signal]]:
        df = df.copy()
        df.columns = df.columns.str.lower()
        
        # Bollinger Bands
        df['bb_basis'] = df['close'].rolling(window=self.bb_length).mean()
        df['bb_std'] = df['close'].rolling(window=self.bb_length).std()
        df['bb_upper'] = df['bb_basis'] + (self.bb_mult * df['bb_std'])
        df['bb_lower'] = df['bb_basis'] - (self.bb_mult * df['bb_std'])
        
        # Keltner Channel
        df['kc_mid'] = df['close'].ewm(span=self.kc_length, adjust=False).mean()
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=self.kc_length).mean()
        df['kc_upper'] = df['kc_mid'] + (self.kc_mult * atr)
        df['kc_lower'] = df['kc_mid'] - (self.kc_mult * atr)
        
        # Squeeze: BB inside KC
        df['squeeze_on'] = (df['bb_lower'] > df['kc_lower']) & (df['bb_upper'] < df['kc_upper'])
        
        # Momentum
        highest = df['high'].rolling(window=self.kc_length).max()
        lowest = df['low'].rolling(window=self.kc_length).min()
        avg = (highest + lowest) / 2
        df['mom'] = df['close'] - ((avg + df['kc_mid']) / 2)
        
        signals = []
        position = None
        
        for i in range(self.kc_length + 1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            if pd.isna(row['mom']):
                continue
            
            # Squeeze release + momentum up = Buy
            if prev_row['squeeze_on'] and not row['squeeze_on'] and row['mom'] > 0 and position != 'LONG':
                if position == 'SHORT':
                    signals.append(Signal(df.index[i], 'BUY', row['close'], 0.9, {}, 'Squeeze close short'))
                signals.append(Signal(df.index[i], 'BUY', row['close'], 0.9, {}, 'Squeeze fire up'))
                position = 'LONG'
            
            # Squeeze release + momentum down = Sell
            elif prev_row['squeeze_on'] and not row['squeeze_on'] and row['mom'] < 0 and position != 'SHORT':
                if position == 'LONG':
                    signals.append(Signal(df.index[i], 'SELL', row['close'], 0.9, {}, 'Squeeze close long'))
                signals.append(Signal(df.index[i], 'SELL', row['close'], 0.9, {}, 'Squeeze fire down'))
                position = 'SHORT'
        
        return df, signals


class VWAPStrategy:
    """
    VWAP Strategy (Intraday)
    Buy above VWAP, Sell below VWAP
    """
    
    def __init__(self):
        self.name = "VWAP"
    
    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Signal]]:
        df = df.copy()
        df.columns = df.columns.str.lower()
        
        # Typical price * volume
        df['tp_vol'] = ((df['high'] + df['low'] + df['close']) / 3) * df['volume']
        df['cum_vol'] = df['volume'].cumsum()
        df['cum_tp_vol'] = df['tp_vol'].cumsum()
        df['vwap'] = df['cum_tp_vol'] / df['cum_vol']
        
        signals = []
        position = None
        
        for i in range(20, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            # Price crosses above VWAP
            if prev_row['close'] < prev_row['vwap'] and row['close'] >= row['vwap'] and position != 'LONG':
                if position == 'SHORT':
                    signals.append(Signal(df.index[i], 'BUY', row['close'], 0.7, {}, 'VWAP close short'))
                signals.append(Signal(df.index[i], 'BUY', row['close'], 0.7, {}, 'VWAP cross up'))
                position = 'LONG'
            
            # Price crosses below VWAP
            elif prev_row['close'] > prev_row['vwap'] and row['close'] <= row['vwap'] and position != 'SHORT':
                if position == 'LONG':
                    signals.append(Signal(df.index[i], 'SELL', row['close'], 0.7, {}, 'VWAP close long'))
                signals.append(Signal(df.index[i], 'SELL', row['close'], 0.7, {}, 'VWAP cross down'))
                position = 'SHORT'
        
        return df, signals


class RSIDivergenceStrategy:
    """
    RSI Divergence Strategy
    Looks for bullish/bearish divergences
    """
    
    def __init__(self, period=14, lookback=5):
        self.period = period
        self.lookback = lookback
        self.name = f"RSI Divergence"
    
    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Signal]]:
        df = df.copy()
        df.columns = df.columns.str.lower()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Local highs/lows
        df['price_high'] = df['close'].rolling(window=self.lookback, center=True).max()
        df['price_low'] = df['close'].rolling(window=self.lookback, center=True).min()
        df['rsi_high'] = df['rsi'].rolling(window=self.lookback, center=True).max()
        df['rsi_low'] = df['rsi'].rolling(window=self.lookback, center=True).min()
        
        signals = []
        position = None
        
        for i in range(self.period + self.lookback, len(df) - self.lookback):
            row = df.iloc[i]
            
            if pd.isna(row['rsi']):
                continue
            
            # Bullish divergence: Price makes lower low, RSI makes higher low
            if i > self.lookback * 2:
                prev_price_low = df['close'].iloc[i-self.lookback*2:i-self.lookback].min()
                prev_rsi_low = df['rsi'].iloc[i-self.lookback*2:i-self.lookback].min()
                
                if row['close'] < prev_price_low and row['rsi'] > prev_rsi_low and row['rsi'] < 40:
                    if position != 'LONG':
                        if position == 'SHORT':
                            signals.append(Signal(df.index[i], 'BUY', row['close'], 0.85, {}, 'RSI div close short'))
                        signals.append(Signal(df.index[i], 'BUY', row['close'], 0.85, {}, 'Bullish RSI divergence'))
                        position = 'LONG'
                
                # Bearish divergence: Price makes higher high, RSI makes lower high
                prev_price_high = df['close'].iloc[i-self.lookback*2:i-self.lookback].max()
                prev_rsi_high = df['rsi'].iloc[i-self.lookback*2:i-self.lookback].max()
                
                if row['close'] > prev_price_high and row['rsi'] < prev_rsi_high and row['rsi'] > 60:
                    if position != 'SHORT':
                        if position == 'LONG':
                            signals.append(Signal(df.index[i], 'SELL', row['close'], 0.85, {}, 'RSI div close long'))
                        signals.append(Signal(df.index[i], 'SELL', row['close'], 0.85, {}, 'Bearish RSI divergence'))
                        position = 'SHORT'
        
        return df, signals


class MomentumStrategy:
    """
    Simple Momentum Strategy
    Buy when momentum is positive and increasing
    """
    
    def __init__(self, period=10):
        self.period = period
        self.name = f"Momentum {period}"
    
    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Signal]]:
        df = df.copy()
        df.columns = df.columns.str.lower()
        
        df['momentum'] = df['close'] - df['close'].shift(self.period)
        df['mom_ma'] = df['momentum'].rolling(window=5).mean()
        
        signals = []
        position = None
        
        for i in range(self.period + 5, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            if pd.isna(row['mom_ma']):
                continue
            
            # Momentum crosses above zero
            if prev_row['momentum'] < 0 and row['momentum'] >= 0 and row['mom_ma'] > prev_row['mom_ma']:
                if position != 'LONG':
                    if position == 'SHORT':
                        signals.append(Signal(df.index[i], 'BUY', row['close'], 0.7, {}, 'Momentum close short'))
                    signals.append(Signal(df.index[i], 'BUY', row['close'], 0.7, {}, 'Momentum up'))
                    position = 'LONG'
            
            # Momentum crosses below zero
            elif prev_row['momentum'] > 0 and row['momentum'] <= 0 and row['mom_ma'] < prev_row['mom_ma']:
                if position != 'SHORT':
                    if position == 'LONG':
                        signals.append(Signal(df.index[i], 'SELL', row['close'], 0.7, {}, 'Momentum close long'))
                    signals.append(Signal(df.index[i], 'SELL', row['close'], 0.7, {}, 'Momentum down'))
                    position = 'SHORT'
        
        return df, signals


def compare_all_strategies(symbol='BTCUSDT', timeframe='720', days=730):
    """Compare all TradingView strategies"""
    import yaml
    from data.bybit_connector import BybitConnector
    from backtesting.backtest_engine import BacktestEngine
    from strategy.gold_line_strategy import GoldLineStrategy
    from strategy.bollinger_strategy import BollingerBandsStrategy
    from compare_strategies import RSIOverboughtOversold, MACDCrossover, EMACrossover
    from strategy.tradingview_strategies import (
        SuperTrendStrategy, StochasticStrategy, ADXStrategy, 
        ParabolicSARStrategy, WilliamsRStrategy, IchimokuStrategy
    )
    
    with open('config/strategy_params.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 90)
    print(f"COMPREHENSIVE TRADINGVIEW STRATEGIES COMPARISON")
    print(f"Symbol: {symbol} | Timeframe: {timeframe}min | History: {days} days")
    print("=" * 90)
    
    connector = BybitConnector(cache_dir='data/cache')
    df = connector.get_historical_data(symbol, timeframe, days, use_cache=True)
    
    print(f"📊 Loaded {len(df)} candles\n")
    
    strategies = [
        # Original strategies
        ('Gold Line', GoldLineStrategy(config)),
        ('Bollinger Bands', BollingerBandsStrategy(config)),
        ('RSI 14', RSIOverboughtOversold(14, 70, 30)),
        ('MACD 8/17/9', MACDCrossover(8, 17, 9)),
        ('EMA 9/21', EMACrossover(9, 21)),
        ('SuperTrend', SuperTrendStrategy(10, 3.0)),
        ('Stochastic', StochasticStrategy(14, 3, 80, 20)),
        ('Williams %R', WilliamsRStrategy(14)),
        ('ADX', ADXStrategy(14, 25)),
        ('Ichimoku', IchimokuStrategy()),
        ('Parabolic SAR', ParabolicSARStrategy()),
        # New strategies
        ('Donchian 20/10', DonchianChannelStrategy(20, 10)),
        ('Donchian 55/20', DonchianChannelStrategy(55, 20)),
        ('Keltner', KeltnerChannelStrategy(20, 10, 2.0)),
        ('Hull MA', HullMAStrategy(16)),
        ('Squeeze Momentum', SqueezeMomentumStrategy()),
        ('VWAP', VWAPStrategy()),
        ('RSI Divergence', RSIDivergenceStrategy()),
        ('Momentum', MomentumStrategy(10)),
    ]
    
    results = []
    
    for name, strategy in strategies:
        try:
            df_copy = df.copy()
            _, signals = strategy.run(df_copy)
            
            if not signals or len(signals) < 2:
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
            })
            
            print(f"  ✅ {name:<18} {result.total_trades:>4} trades  {result.win_rate:>5.1f}% win  {result.total_pnl_pct:>+7.2f}%  PF {result.profit_factor:.2f}")
            
        except Exception as e:
            print(f"  ❌ {name}: {e}")
    
    results.sort(key=lambda x: x['pnl_pct'], reverse=True)
    
    print("\n" + "=" * 90)
    print("📈 ALL STRATEGIES RANKED BY PnL")
    print("=" * 90)
    print(f"{'Rank':<5} {'Strategy':<20} {'Trades':<8} {'Win %':<8} {'PnL %':<10} {'PF':<8}")
    print("-" * 90)
    
    for i, r in enumerate(results, 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i:>2}"
        print(f"{emoji:<5} {r['name']:<20} {r['trades']:<8} {r['win_rate']:.1f}%{'':<3} {r['pnl_pct']:+.2f}%{'':<4} {r['profit_factor']:.2f}")
    
    print("=" * 90)
    
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Extended TradingView Strategies')
    parser.add_argument('--symbol', type=str, default='BTCUSDT')
    parser.add_argument('--timeframe', type=str, default='720')
    parser.add_argument('--days', type=int, default=730)
    
    args = parser.parse_args()
    compare_all_strategies(args.symbol, args.timeframe, args.days)
