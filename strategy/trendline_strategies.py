"""
Trendline Channel Strategies
Based on the Trend Line Methods indicator - pivot-based trendlines.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from strategy.gold_line_strategy import Signal


class PivotTrendlineStrategy:
    """
    Pivot Trendline Channel Strategy
    Based on Trend Line Methods indicator.
    Uses pivot highs/lows to create dynamic trendlines and trade breakouts.
    """
    
    def __init__(self, pivot_left=5, pivot_right=5, pivot_count=5):
        self.pivot_left = pivot_left
        self.pivot_right = pivot_right
        self.pivot_count = pivot_count
        self.name = f"Pivot TL {pivot_left}/{pivot_right}"
    
    def _find_pivots(self, df: pd.DataFrame):
        """Find pivot highs and lows"""
        pivot_highs = []
        pivot_lows = []
        
        for i in range(self.pivot_left + self.pivot_right, len(df)):
            idx = i - self.pivot_right
            
            # Check for pivot high
            is_pivot_high = True
            high_val = df['high'].iloc[idx]
            for j in range(idx - self.pivot_left, idx + self.pivot_right + 1):
                if j != idx and j >= 0 and j < len(df):
                    if df['high'].iloc[j] >= high_val:
                        is_pivot_high = False
                        break
            if is_pivot_high:
                pivot_highs.append((idx, high_val))
            
            # Check for pivot low
            is_pivot_low = True
            low_val = df['low'].iloc[idx]
            for j in range(idx - self.pivot_left, idx + self.pivot_right + 1):
                if j != idx and j >= 0 and j < len(df):
                    if df['low'].iloc[j] <= low_val:
                        is_pivot_low = False
                        break
            if is_pivot_low:
                pivot_lows.append((idx, low_val))
        
        return pivot_highs, pivot_lows
    
    def _calculate_trendline(self, pivots, current_bar):
        """Calculate trendline from pivots using linear regression"""
        if len(pivots) < 2:
            return None, None
        
        # Use last N pivots
        recent_pivots = pivots[-self.pivot_count:] if len(pivots) > self.pivot_count else pivots
        
        if len(recent_pivots) < 2:
            return None, None
        
        # Simple linear regression
        x_vals = [p[0] for p in recent_pivots]
        y_vals = [p[1] for p in recent_pivots]
        
        n = len(x_vals)
        sum_x = sum(x_vals)
        sum_y = sum(y_vals)
        sum_xy = sum(x * y for x, y in zip(x_vals, y_vals))
        sum_x2 = sum(x * x for x in x_vals)
        
        denom = n * sum_x2 - sum_x * sum_x
        if denom == 0:
            return None, None
        
        slope = (n * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / n
        
        return slope, intercept
    
    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Signal]]:
        df = df.copy()
        df.columns = df.columns.str.lower()
        
        pivot_highs, pivot_lows = self._find_pivots(df)
        
        df['high_trendline'] = np.nan
        df['low_trendline'] = np.nan
        
        signals = []
        position = None
        
        lookback = self.pivot_left + self.pivot_right + 10
        
        for i in range(lookback, len(df)):
            # Get pivots up to this point
            ph_filtered = [(idx, val) for idx, val in pivot_highs if idx < i]
            pl_filtered = [(idx, val) for idx, val in pivot_lows if idx < i]
            
            slope_hi, int_hi = self._calculate_trendline(ph_filtered, i)
            slope_lo, int_lo = self._calculate_trendline(pl_filtered, i)
            
            if slope_hi is not None:
                df.loc[df.index[i], 'high_trendline'] = slope_hi * i + int_hi
            if slope_lo is not None:
                df.loc[df.index[i], 'low_trendline'] = slope_lo * i + int_lo
            
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            # Buy: Price breaks above high trendline
            if (not pd.isna(row['high_trendline']) and 
                not pd.isna(prev_row['high_trendline']) and
                prev_row['close'] <= prev_row['high_trendline'] and 
                row['close'] > row['high_trendline']):
                if position != 'LONG':
                    if position == 'SHORT':
                        signals.append(Signal(df.index[i], 'BUY', row['close'], 0.85, {}, 'Pivot TL close short'))
                    signals.append(Signal(df.index[i], 'BUY', row['close'], 0.85, {}, 'Pivot TL breakout up'))
                    position = 'LONG'
            
            # Sell: Price breaks below low trendline
            elif (not pd.isna(row['low_trendline']) and 
                  not pd.isna(prev_row['low_trendline']) and
                  prev_row['close'] >= prev_row['low_trendline'] and 
                  row['close'] < row['low_trendline']):
                if position != 'SHORT':
                    if position == 'LONG':
                        signals.append(Signal(df.index[i], 'SELL', row['close'], 0.85, {}, 'Pivot TL close long'))
                    signals.append(Signal(df.index[i], 'SELL', row['close'], 0.85, {}, 'Pivot TL breakout down'))
                    position = 'SHORT'
        
        return df, signals


class FivePointChannelStrategy:
    """
    5-Point Straight Channel Strategy
    Based on Method 2 from Trend Line Methods.
    Uses 5 highest highs and 5 lowest lows for channel.
    """
    
    def __init__(self, length=100):
        self.length = length
        self.name = f"5-Point Channel {length}"
    
    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Signal]]:
        df = df.copy()
        df.columns = df.columns.str.lower()
        
        df['channel_high'] = np.nan
        df['channel_low'] = np.nan
        
        seg_len = max(1, self.length // 5)
        
        signals = []
        position = None
        
        for i in range(self.length, len(df)):
            # Find 5 representative highs and lows
            hi_points = []
            lo_points = []
            
            for k in range(5):
                start = i - self.length + k * seg_len
                end = min(start + seg_len, i)
                
                if start >= 0 and end <= len(df):
                    segment = df.iloc[start:end]
                    if len(segment) > 0:
                        max_idx = segment['high'].idxmax()
                        min_idx = segment['low'].idxmin()
                        hi_points.append((df.index.get_loc(max_idx), df.loc[max_idx, 'high']))
                        lo_points.append((df.index.get_loc(min_idx), df.loc[min_idx, 'low']))
            
            # Linear regression for channel lines
            if len(hi_points) >= 2:
                x_hi = [p[0] for p in hi_points]
                y_hi = [p[1] for p in hi_points]
                n = len(x_hi)
                slope_hi = (n * sum(x * y for x, y in zip(x_hi, y_hi)) - sum(x_hi) * sum(y_hi)) / \
                           (n * sum(x * x for x in x_hi) - sum(x_hi) ** 2) if (n * sum(x * x for x in x_hi) - sum(x_hi) ** 2) != 0 else 0
                int_hi = (sum(y_hi) - slope_hi * sum(x_hi)) / n
                df.loc[df.index[i], 'channel_high'] = slope_hi * i + int_hi
            
            if len(lo_points) >= 2:
                x_lo = [p[0] for p in lo_points]
                y_lo = [p[1] for p in lo_points]
                n = len(x_lo)
                slope_lo = (n * sum(x * y for x, y in zip(x_lo, y_lo)) - sum(x_lo) * sum(y_lo)) / \
                           (n * sum(x * x for x in x_lo) - sum(x_lo) ** 2) if (n * sum(x * x for x in x_lo) - sum(x_lo) ** 2) != 0 else 0
                int_lo = (sum(y_lo) - slope_lo * sum(x_lo)) / n
                df.loc[df.index[i], 'channel_low'] = slope_lo * i + int_lo
            
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            # Buy: Price breaks above channel high
            if (not pd.isna(row['channel_high']) and
                not pd.isna(prev_row['channel_high']) and
                prev_row['close'] <= prev_row['channel_high'] and
                row['close'] > row['channel_high']):
                if position != 'LONG':
                    if position == 'SHORT':
                        signals.append(Signal(df.index[i], 'BUY', row['close'], 0.85, {}, '5-Point close short'))
                    signals.append(Signal(df.index[i], 'BUY', row['close'], 0.85, {}, '5-Point breakout up'))
                    position = 'LONG'
            
            # Sell: Price breaks below channel low
            elif (not pd.isna(row['channel_low']) and
                  not pd.isna(prev_row['channel_low']) and
                  prev_row['close'] >= prev_row['channel_low'] and
                  row['close'] < row['channel_low']):
                if position != 'SHORT':
                    if position == 'LONG':
                        signals.append(Signal(df.index[i], 'SELL', row['close'], 0.85, {}, '5-Point close long'))
                    signals.append(Signal(df.index[i], 'SELL', row['close'], 0.85, {}, '5-Point breakout down'))
                    position = 'SHORT'
        
        return df, signals


class ChannelBounceStrategy:
    """
    Channel Bounce Strategy
    Trades bounces off channel lines instead of breakouts.
    """
    
    def __init__(self, length=50, bounce_threshold=0.02):
        self.length = length
        self.bounce_threshold = bounce_threshold
        self.name = f"Channel Bounce {length}"
    
    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Signal]]:
        df = df.copy()
        df.columns = df.columns.str.lower()
        
        # Simple channel using rolling high/low
        df['channel_high'] = df['high'].rolling(window=self.length).max()
        df['channel_low'] = df['low'].rolling(window=self.length).min()
        df['channel_mid'] = (df['channel_high'] + df['channel_low']) / 2
        
        signals = []
        position = None
        
        for i in range(self.length + 5, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            if pd.isna(row['channel_high']) or pd.isna(row['channel_low']):
                continue
            
            channel_range = row['channel_high'] - row['channel_low']
            
            # Buy: Near channel low, bouncing up
            near_low = (row['close'] - row['channel_low']) / channel_range < self.bounce_threshold
            bouncing_up = row['close'] > prev_row['close']
            
            if near_low and bouncing_up:
                if position != 'LONG':
                    if position == 'SHORT':
                        signals.append(Signal(df.index[i], 'BUY', row['close'], 0.8, {}, 'Channel bounce close short'))
                    signals.append(Signal(df.index[i], 'BUY', row['close'], 0.8, {}, 'Channel bounce up'))
                    position = 'LONG'
            
            # Sell: Near channel high, bouncing down
            near_high = (row['channel_high'] - row['close']) / channel_range < self.bounce_threshold
            bouncing_down = row['close'] < prev_row['close']
            
            if near_high and bouncing_down:
                if position != 'SHORT':
                    if position == 'LONG':
                        signals.append(Signal(df.index[i], 'SELL', row['close'], 0.8, {}, 'Channel bounce close long'))
                    signals.append(Signal(df.index[i], 'SELL', row['close'], 0.8, {}, 'Channel bounce down'))
                    position = 'SHORT'
        
        return df, signals


class ATHBreakoutStrategy:
    """
    ATH Breakout Strategy (Trendoscope)
    Based on Breakouts & Pullbacks indicator.
    Buys on new All-Time High breakouts after a minimum gap.
    """
    
    def __init__(self, min_gap=30, pullback_entry=True):
        self.min_gap = min_gap
        self.pullback_entry = pullback_entry
        self.name = f"ATH Breakout {min_gap}"
    
    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Signal]]:
        df = df.copy()
        df.columns = df.columns.str.lower()
        
        # Track ATH
        df['ath'] = df['high'].cummax()
        df['new_ath'] = df['high'] == df['ath']
        
        # Track local lows since last ATH
        signals = []
        position = None
        
        last_ath_idx = 0
        last_ath_price = df['high'].iloc[0]
        local_low = df['low'].iloc[0]
        local_low_idx = 0
        
        for i in range(1, len(df)):
            row = df.iloc[i]
            
            # Update local low
            if row['low'] < local_low:
                local_low = row['low']
                local_low_idx = i
            
            # Check for new ATH with minimum gap
            if row['high'] > last_ath_price:
                gap = i - last_ath_idx
                
                if gap >= self.min_gap:
                    # ATH Breakout detected!
                    if position != 'LONG':
                        if position == 'SHORT':
                            signals.append(Signal(df.index[i], 'BUY', row['close'], 0.9, 
                                                  {'gap': gap}, 'ATH close short'))
                        signals.append(Signal(df.index[i], 'BUY', row['close'], 0.9,
                                              {'gap': gap}, 'ATH Breakout'))
                        position = 'LONG'
                
                # Reset tracking
                last_ath_idx = i
                last_ath_price = row['high']
                local_low = row['low']
                local_low_idx = i
            
            # Exit on significant pullback (e.g., 5% from ATH)
            if position == 'LONG':
                pullback_pct = (last_ath_price - row['close']) / last_ath_price * 100
                if pullback_pct > 5:
                    signals.append(Signal(df.index[i], 'SELL', row['close'], 0.8,
                                          {'pullback': pullback_pct}, 'Pullback exit'))
                    position = None
        
        return df, signals


class PullbackBuyStrategy:
    """
    Pullback Buy Strategy
    Based on Trendoscope's pullback analysis.
    Buys on pullbacks after ATH breakouts.
    """
    
    def __init__(self, min_gap=30, pullback_pct=3.0, recovery_confirm=True):
        self.min_gap = min_gap
        self.pullback_pct = pullback_pct
        self.recovery_confirm = recovery_confirm
        self.name = f"Pullback Buy {pullback_pct}%"
    
    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Signal]]:
        df = df.copy()
        df.columns = df.columns.str.lower()
        
        signals = []
        position = None
        
        last_ath_price = df['high'].iloc[0]
        last_ath_idx = 0
        in_pullback = False
        pullback_low = None
        pullback_low_idx = None
        
        for i in range(1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            # New ATH
            if row['high'] > last_ath_price:
                gap = i - last_ath_idx
                if gap >= self.min_gap:
                    in_pullback = False  # Reset on new ATH breakout
                last_ath_price = row['high']
                last_ath_idx = i
                pullback_low = row['low']
                pullback_low_idx = i
            else:
                # Track lowest point since ATH
                if pullback_low is None or row['low'] < pullback_low:
                    pullback_low = row['low']
                    pullback_low_idx = i
                
                # Calculate pullback percentage
                pullback = (last_ath_price - row['low']) / last_ath_price * 100
                
                if pullback >= self.pullback_pct:
                    in_pullback = True
                
                # Buy on recovery from pullback
                if in_pullback and row['close'] > prev_row['close']:
                    recovery = (row['close'] - pullback_low) / pullback_low * 100
                    
                    if recovery > 1.0:  # Some recovery
                        if position != 'LONG':
                            if position == 'SHORT':
                                signals.append(Signal(df.index[i], 'BUY', row['close'], 0.85, {},
                                                      'Pullback close short'))
                            signals.append(Signal(df.index[i], 'BUY', row['close'], 0.85,
                                                  {'pullback': pullback, 'recovery': recovery},
                                                  'Pullback recovery'))
                            position = 'LONG'
                            in_pullback = False
            
            # Take profit at new highs
            if position == 'LONG' and row['high'] >= last_ath_price * 0.99:
                signals.append(Signal(df.index[i], 'SELL', row['close'], 0.8, {},
                                      'Take profit near ATH'))
                position = None
        
        return df, signals


def compare_trendline_strategies(symbol='BTCUSDT', timeframe='720', days=730):
    """Compare all trendline and breakout strategies"""
    import yaml
    from data.bybit_connector import BybitConnector
    from backtesting.backtest_engine import BacktestEngine
    
    with open('config/strategy_params.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 80)
    print(f"TRENDLINE & BREAKOUT STRATEGIES")
    print(f"Symbol: {symbol} | Timeframe: {timeframe}min | History: {days} days")
    print("=" * 80)
    
    connector = BybitConnector(cache_dir='data/cache')
    df = connector.get_historical_data(symbol, timeframe, days, use_cache=True)
    
    print(f"📊 Loaded {len(df)} candles\n")
    
    strategies = [
        # Trendline strategies
        PivotTrendlineStrategy(5, 5, 5),
        PivotTrendlineStrategy(10, 10, 3),
        FivePointChannelStrategy(100),
        FivePointChannelStrategy(50),
        ChannelBounceStrategy(50, 0.02),
        ChannelBounceStrategy(30, 0.03),
        # ATH Breakout strategies (new)
        ATHBreakoutStrategy(30, True),
        ATHBreakoutStrategy(20, True),
        ATHBreakoutStrategy(50, True),
        PullbackBuyStrategy(30, 3.0),
        PullbackBuyStrategy(20, 5.0),
        PullbackBuyStrategy(30, 2.0),
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
    print("📈 TRENDLINE STRATEGIES RANKED BY PnL")
    print("=" * 80)
    print(f"{'Strategy':<25} {'Trades':<8} {'Win %':<8} {'PnL %':<10} {'PF':<8}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['name']:<25} {r['trades']:<8} {r['win_rate']:.1f}%{'':<3} {r['pnl_pct']:+.2f}%{'':<4} {r['profit_factor']:.2f}")
    
    print("=" * 80)
    
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Trendline Strategies Comparison')
    parser.add_argument('--symbol', type=str, default='BTCUSDT')
    parser.add_argument('--timeframe', type=str, default='720')
    parser.add_argument('--days', type=int, default=730)
    
    args = parser.parse_args()
    compare_trendline_strategies(args.symbol, args.timeframe, args.days)
