"""
Popular TradingView Strategies
Implements the most popular strategies from TradingView for comparison testing.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from strategy.gold_line_strategy import Signal


class SuperTrendStrategy:
    """
    SuperTrend Strategy - One of the most popular TradingView strategies.
    Uses ATR to create dynamic support/resistance bands.
    """
    
    def __init__(self, period=10, multiplier=3.0):
        self.period = period
        self.multiplier = multiplier
        self.name = f"SuperTrend {period}/{multiplier}"
    
    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Signal]]:
        df = df.copy()
        df.columns = df.columns.str.lower()
        
        # Calculate ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=self.period).mean()
        
        # Calculate SuperTrend bands
        hl2 = (df['high'] + df['low']) / 2
        upper_band = hl2 + (self.multiplier * atr)
        lower_band = hl2 - (self.multiplier * atr)
        
        # Initialize SuperTrend
        supertrend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=int)
        
        for i in range(self.period, len(df)):
            if df['close'].iloc[i] > upper_band.iloc[i-1]:
                direction.iloc[i] = 1
            elif df['close'].iloc[i] < lower_band.iloc[i-1]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = direction.iloc[i-1] if pd.notna(direction.iloc[i-1]) else 1
            
            if direction.iloc[i] == 1:
                supertrend.iloc[i] = lower_band.iloc[i]
            else:
                supertrend.iloc[i] = upper_band.iloc[i]
        
        df['supertrend'] = supertrend
        df['st_direction'] = direction
        
        signals = []
        in_position = False
        
        for i in range(self.period + 1, len(df)):
            if pd.isna(df['st_direction'].iloc[i]):
                continue
            
            # Buy when direction changes to 1 (bullish)
            if df['st_direction'].iloc[i] == 1 and df['st_direction'].iloc[i-1] == -1 and not in_position:
                signals.append(Signal(
                    timestamp=df.index[i], type='BUY', price=df['close'].iloc[i],
                    confidence=0.8, indicators={'supertrend': df['supertrend'].iloc[i]},
                    context='SuperTrend bullish'
                ))
                in_position = True
            
            # Sell when direction changes to -1 (bearish)
            elif df['st_direction'].iloc[i] == -1 and df['st_direction'].iloc[i-1] == 1 and in_position:
                signals.append(Signal(
                    timestamp=df.index[i], type='SELL', price=df['close'].iloc[i],
                    confidence=0.8, indicators={'supertrend': df['supertrend'].iloc[i]},
                    context='SuperTrend bearish'
                ))
                in_position = False
        
        return df, signals


class StochasticStrategy:
    """Stochastic Oscillator Strategy"""
    
    def __init__(self, k_period=14, d_period=3, overbought=80, oversold=20):
        self.k_period = k_period
        self.d_period = d_period
        self.overbought = overbought
        self.oversold = oversold
        self.name = f"Stoch {k_period}/{d_period}"
    
    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Signal]]:
        df = df.copy()
        df.columns = df.columns.str.lower()
        
        # Calculate Stochastic
        lowest_low = df['low'].rolling(window=self.k_period).min()
        highest_high = df['high'].rolling(window=self.k_period).max()
        df['stoch_k'] = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low)
        df['stoch_d'] = df['stoch_k'].rolling(window=self.d_period).mean()
        
        signals = []
        in_position = False
        
        for i in range(self.k_period + self.d_period, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            if pd.isna(row['stoch_d']):
                continue
            
            # Buy: K crosses above D from oversold
            if (prev_row['stoch_k'] < prev_row['stoch_d'] and row['stoch_k'] >= row['stoch_d'] 
                and row['stoch_k'] < 50 and not in_position):
                signals.append(Signal(
                    timestamp=df.index[i], type='BUY', price=row['close'],
                    confidence=0.7, indicators={'stoch_k': row['stoch_k'], 'stoch_d': row['stoch_d']},
                    context='Stochastic bullish crossover'
                ))
                in_position = True
            
            # Sell: K crosses below D from overbought
            elif (prev_row['stoch_k'] > prev_row['stoch_d'] and row['stoch_k'] <= row['stoch_d']
                  and row['stoch_k'] > 50 and in_position):
                signals.append(Signal(
                    timestamp=df.index[i], type='SELL', price=row['close'],
                    confidence=0.7, indicators={'stoch_k': row['stoch_k'], 'stoch_d': row['stoch_d']},
                    context='Stochastic bearish crossover'
                ))
                in_position = False
        
        return df, signals


class WilliamsRStrategy:
    """Williams %R Strategy"""
    
    def __init__(self, period=14, overbought=-20, oversold=-80):
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        self.name = f"Williams %R {period}"
    
    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Signal]]:
        df = df.copy()
        df.columns = df.columns.str.lower()
        
        highest_high = df['high'].rolling(window=self.period).max()
        lowest_low = df['low'].rolling(window=self.period).min()
        df['williams_r'] = -100 * (highest_high - df['close']) / (highest_high - lowest_low)
        
        signals = []
        in_position = False
        
        for i in range(self.period + 1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            if pd.isna(row['williams_r']):
                continue
            
            # Buy: Crosses above oversold
            if prev_row['williams_r'] < self.oversold and row['williams_r'] >= self.oversold and not in_position:
                signals.append(Signal(
                    timestamp=df.index[i], type='BUY', price=row['close'],
                    confidence=0.7, indicators={'williams_r': row['williams_r']},
                    context='Williams %R oversold exit'
                ))
                in_position = True
            
            # Sell: Crosses below overbought
            elif prev_row['williams_r'] > self.overbought and row['williams_r'] <= self.overbought and in_position:
                signals.append(Signal(
                    timestamp=df.index[i], type='SELL', price=row['close'],
                    confidence=0.7, indicators={'williams_r': row['williams_r']},
                    context='Williams %R overbought exit'
                ))
                in_position = False
        
        return df, signals


class ADXStrategy:
    """ADX (Average Directional Index) Strategy - Trend Strength"""
    
    def __init__(self, period=14, adx_threshold=25):
        self.period = period
        self.adx_threshold = adx_threshold
        self.name = f"ADX {period}"
    
    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Signal]]:
        df = df.copy()
        df.columns = df.columns.str.lower()
        
        # Calculate +DM and -DM
        df['up_move'] = df['high'].diff()
        df['down_move'] = -df['low'].diff()
        
        df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
        df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.ewm(span=self.period, adjust=False).mean()
        
        # +DI and -DI
        plus_di = 100 * (df['plus_dm'].ewm(span=self.period, adjust=False).mean() / atr)
        minus_di = 100 * (df['minus_dm'].ewm(span=self.period, adjust=False).mean() / atr)
        
        # ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        df['adx'] = dx.ewm(span=self.period, adjust=False).mean()
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        
        signals = []
        in_position = False
        
        for i in range(self.period * 2, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            if pd.isna(row['adx']):
                continue
            
            # Buy: +DI crosses above -DI with strong trend
            if (prev_row['plus_di'] < prev_row['minus_di'] and row['plus_di'] >= row['minus_di']
                and row['adx'] > self.adx_threshold and not in_position):
                signals.append(Signal(
                    timestamp=df.index[i], type='BUY', price=row['close'],
                    confidence=0.8, indicators={'adx': row['adx'], 'plus_di': row['plus_di'], 'minus_di': row['minus_di']},
                    context='ADX bullish crossover'
                ))
                in_position = True
            
            # Sell: -DI crosses above +DI
            elif (prev_row['minus_di'] < prev_row['plus_di'] and row['minus_di'] >= row['plus_di'] and in_position):
                signals.append(Signal(
                    timestamp=df.index[i], type='SELL', price=row['close'],
                    confidence=0.8, indicators={'adx': row['adx'], 'plus_di': row['plus_di'], 'minus_di': row['minus_di']},
                    context='ADX bearish crossover'
                ))
                in_position = False
        
        return df, signals


class IchimokuStrategy:
    """Ichimoku Cloud Strategy - Popular Japanese indicator"""
    
    def __init__(self, tenkan=9, kijun=26, senkou_b=52):
        self.tenkan = tenkan
        self.kijun = kijun
        self.senkou_b = senkou_b
        self.name = "Ichimoku"
    
    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Signal]]:
        df = df.copy()
        df.columns = df.columns.str.lower()
        
        # Tenkan-sen (Conversion Line)
        high_tenkan = df['high'].rolling(window=self.tenkan).max()
        low_tenkan = df['low'].rolling(window=self.tenkan).min()
        df['tenkan_sen'] = (high_tenkan + low_tenkan) / 2
        
        # Kijun-sen (Base Line)
        high_kijun = df['high'].rolling(window=self.kijun).max()
        low_kijun = df['low'].rolling(window=self.kijun).min()
        df['kijun_sen'] = (high_kijun + low_kijun) / 2
        
        # Senkou Span A
        df['senkou_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(self.kijun)
        
        # Senkou Span B
        high_senkou = df['high'].rolling(window=self.senkou_b).max()
        low_senkou = df['low'].rolling(window=self.senkou_b).min()
        df['senkou_b'] = ((high_senkou + low_senkou) / 2).shift(self.kijun)
        
        signals = []
        in_position = False
        
        for i in range(self.senkou_b + self.kijun, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            if pd.isna(row['senkou_a']) or pd.isna(row['senkou_b']):
                continue
            
            cloud_top = max(row['senkou_a'], row['senkou_b'])
            cloud_bottom = min(row['senkou_a'], row['senkou_b'])
            
            # Buy: Tenkan crosses above Kijun AND price above cloud
            if (prev_row['tenkan_sen'] < prev_row['kijun_sen'] and row['tenkan_sen'] >= row['kijun_sen']
                and row['close'] > cloud_top and not in_position):
                signals.append(Signal(
                    timestamp=df.index[i], type='BUY', price=row['close'],
                    confidence=0.85, indicators={'tenkan': row['tenkan_sen'], 'kijun': row['kijun_sen']},
                    context='Ichimoku bullish'
                ))
                in_position = True
            
            # Sell: Tenkan crosses below Kijun OR price below cloud
            elif ((prev_row['tenkan_sen'] > prev_row['kijun_sen'] and row['tenkan_sen'] <= row['kijun_sen'])
                  or row['close'] < cloud_bottom) and in_position:
                signals.append(Signal(
                    timestamp=df.index[i], type='SELL', price=row['close'],
                    confidence=0.85, indicators={'tenkan': row['tenkan_sen'], 'kijun': row['kijun_sen']},
                    context='Ichimoku bearish'
                ))
                in_position = False
        
        return df, signals


class ParabolicSARStrategy:
    """Parabolic SAR Strategy"""
    
    def __init__(self, af_start=0.02, af_step=0.02, af_max=0.2):
        self.af_start = af_start
        self.af_step = af_step
        self.af_max = af_max
        self.name = "PSAR"
    
    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Signal]]:
        df = df.copy()
        df.columns = df.columns.str.lower()
        
        # Simplified PSAR calculation
        psar = df['close'].copy()
        af = self.af_start
        trend = 1
        ep = df['high'].iloc[0]
        
        for i in range(1, len(df)):
            if trend == 1:
                psar.iloc[i] = psar.iloc[i-1] + af * (ep - psar.iloc[i-1])
                if df['low'].iloc[i] < psar.iloc[i]:
                    trend = -1
                    psar.iloc[i] = ep
                    ep = df['low'].iloc[i]
                    af = self.af_start
                else:
                    if df['high'].iloc[i] > ep:
                        ep = df['high'].iloc[i]
                        af = min(af + self.af_step, self.af_max)
            else:
                psar.iloc[i] = psar.iloc[i-1] - af * (psar.iloc[i-1] - ep)
                if df['high'].iloc[i] > psar.iloc[i]:
                    trend = 1
                    psar.iloc[i] = ep
                    ep = df['high'].iloc[i]
                    af = self.af_start
                else:
                    if df['low'].iloc[i] < ep:
                        ep = df['low'].iloc[i]
                        af = min(af + self.af_step, self.af_max)
        
        df['psar'] = psar
        
        signals = []
        in_position = False
        
        for i in range(2, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            # Buy: Price crosses above PSAR
            if prev_row['close'] < prev_row['psar'] and row['close'] >= row['psar'] and not in_position:
                signals.append(Signal(
                    timestamp=df.index[i], type='BUY', price=row['close'],
                    confidence=0.7, indicators={'psar': row['psar']},
                    context='PSAR bullish flip'
                ))
                in_position = True
            
            # Sell: Price crosses below PSAR
            elif prev_row['close'] > prev_row['psar'] and row['close'] <= row['psar'] and in_position:
                signals.append(Signal(
                    timestamp=df.index[i], type='SELL', price=row['close'],
                    confidence=0.7, indicators={'psar': row['psar']},
                    context='PSAR bearish flip'
                ))
                in_position = False
        
        return df, signals


def compare_tradingview_strategies(symbol='BTCUSDT', timeframe='720', days=730):
    """Compare all TradingView strategies"""
    import yaml
    from data.bybit_connector import BybitConnector
    from backtesting.backtest_engine import BacktestEngine
    from strategy.gold_line_strategy import GoldLineStrategy
    from strategy.bollinger_strategy import BollingerBandsStrategy
    from compare_strategies import RSIOverboughtOversold, MACDCrossover, EMACrossover
    
    with open('config/strategy_params.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 80)
    print(f"TRADINGVIEW STRATEGIES COMPARISON")
    print(f"Symbol: {symbol} | Timeframe: {timeframe}min | History: {days} days")
    print("=" * 80)
    
    connector = BybitConnector(cache_dir='data/cache')
    df = connector.get_historical_data(symbol, timeframe, days, use_cache=True)
    
    if df.empty:
        print("No data")
        return
    
    print(f"📊 Loaded {len(df)} candles\n")
    
    strategies = [
        # Our strategies
        ('Gold Line', GoldLineStrategy(config)),
        ('Bollinger Bands', BollingerBandsStrategy(config)),
        ('RSI', RSIOverboughtOversold(14, 70, 30)),
        ('MACD', MACDCrossover(8, 17, 9)),
        ('EMA 9/21', EMACrossover(9, 21)),
        # TradingView popular strategies
        ('SuperTrend', SuperTrendStrategy(10, 3.0)),
        ('SuperTrend 7/2', SuperTrendStrategy(7, 2.0)),
        ('Stochastic', StochasticStrategy(14, 3, 80, 20)),
        ('Williams %R', WilliamsRStrategy(14, -20, -80)),
        ('ADX', ADXStrategy(14, 25)),
        ('ADX 20', ADXStrategy(20, 20)),
        ('Ichimoku', IchimokuStrategy(9, 26, 52)),
        ('Parabolic SAR', ParabolicSARStrategy(0.02, 0.02, 0.2)),
    ]
    
    results = []
    
    for name, strategy in strategies:
        try:
            df_copy = df.copy()
            _, signals = strategy.run(df_copy)
            
            if not signals or len(signals) < 2:
                print(f"  {name}: No signals")
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
    
    results.sort(key=lambda x: x['pnl_pct'], reverse=True)
    
    print("\n" + "=" * 80)
    print("📈 ALL STRATEGIES RANKED BY PnL")
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
    parser = argparse.ArgumentParser(description='TradingView Strategies Comparison')
    parser.add_argument('--symbol', type=str, default='BTCUSDT')
    parser.add_argument('--timeframe', type=str, default='720')
    parser.add_argument('--days', type=int, default=730)
    
    args = parser.parse_args()
    compare_tradingview_strategies(args.symbol, args.timeframe, args.days)
