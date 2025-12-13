"""
Multi-Indicator Combination Strategy
Tests all combinations of indicators to find the best performing setup.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
from dataclasses import dataclass
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from strategy.gold_line_strategy import Signal


class MultiIndicatorStrategy:
    """
    Combines multiple indicators for entry/exit decisions.
    Entry requires ALL selected indicators to agree.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        
        # Indicator settings
        self.rsi_length = config.get('rsi_length', 14)
        self.rsi_overbought = config.get('rsi_overbought', 70)
        self.rsi_oversold = config.get('rsi_oversold', 30)
        
        self.macd_fast = config.get('macd_fast', 8)
        self.macd_slow = config.get('macd_slow', 17)
        self.macd_signal = config.get('macd_signal', 9)
        
        self.bb_length = config.get('bb_length', 20)
        self.bb_mult = config.get('bb_mult', 2.0)
        
        self.ema_fast = config.get('ema_fast', 9)
        self.ema_slow = config.get('ema_slow', 21)
        
        self.cci_length = config.get('cci_length', 14)
        self.cci_upper = config.get('cci_upper', 100)
        self.cci_lower = config.get('cci_lower', -100)
        
        # Which indicators to use (combine)
        self.use_rsi = config.get('use_rsi', True)
        self.use_macd = config.get('use_macd', True)
        self.use_bb = config.get('use_bb', False)
        self.use_ema = config.get('use_ema', False)
        self.use_cci = config.get('use_cci', False)
        
        self.name = self._get_name()
    
    def _get_name(self):
        parts = []
        if self.use_macd: parts.append('MACD')
        if self.use_rsi: parts.append('RSI')
        if self.use_bb: parts.append('BB')
        if self.use_ema: parts.append('EMA')
        if self.use_cci: parts.append('CCI')
        return '+'.join(parts) if parts else 'None'
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all indicators"""
        df = df.copy()
        df.columns = df.columns.str.lower()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_length).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_length).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=self.macd_fast, adjust=False).mean()
        exp2 = df['close'].ewm(span=self.macd_slow, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=self.macd_signal, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_basis'] = df['close'].rolling(window=self.bb_length).mean()
        df['bb_std'] = df['close'].rolling(window=self.bb_length).std()
        df['bb_upper'] = df['bb_basis'] + (self.bb_mult * df['bb_std'])
        df['bb_lower'] = df['bb_basis'] - (self.bb_mult * df['bb_std'])
        
        # EMA
        df['ema_fast'] = df['close'].ewm(span=self.ema_fast, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=self.ema_slow, adjust=False).mean()
        
        # CCI
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma = typical_price.rolling(window=self.cci_length).mean()
        mad = typical_price.rolling(window=self.cci_length).apply(lambda x: np.abs(x - x.mean()).mean())
        df['cci'] = (typical_price - sma) / (0.015 * mad)
        
        return df
    
    def check_buy_conditions(self, row, prev_row) -> Dict[str, bool]:
        """Check individual buy conditions for each indicator"""
        conditions = {}
        
        if self.use_rsi:
            # RSI oversold bounce
            conditions['rsi'] = (prev_row['rsi'] < self.rsi_oversold and row['rsi'] >= self.rsi_oversold) or row['rsi'] < 50
        
        if self.use_macd:
            # MACD bullish crossover or positive histogram
            conditions['macd'] = (prev_row['macd'] < prev_row['macd_signal'] and row['macd'] >= row['macd_signal']) or row['macd'] > 0
        
        if self.use_bb:
            # Price bouncing from lower band
            conditions['bb'] = row['close'] > row['bb_lower'] and prev_row['close'] <= prev_row['bb_lower']
        
        if self.use_ema:
            # Fast EMA above slow EMA (uptrend)
            conditions['ema'] = row['ema_fast'] > row['ema_slow']
        
        if self.use_cci:
            # CCI oversold bounce
            conditions['cci'] = (prev_row['cci'] < self.cci_lower and row['cci'] >= self.cci_lower) or row['cci'] > 0
        
        return conditions
    
    def check_sell_conditions(self, row, prev_row) -> Dict[str, bool]:
        """Check individual sell conditions for each indicator"""
        conditions = {}
        
        if self.use_rsi:
            # RSI overbought reversal
            conditions['rsi'] = (prev_row['rsi'] > self.rsi_overbought and row['rsi'] <= self.rsi_overbought) or row['rsi'] > 50
        
        if self.use_macd:
            # MACD bearish crossover or negative histogram
            conditions['macd'] = (prev_row['macd'] > prev_row['macd_signal'] and row['macd'] <= row['macd_signal']) or row['macd'] < 0
        
        if self.use_bb:
            # Price breaking below upper band
            conditions['bb'] = row['close'] < row['bb_upper'] and prev_row['close'] >= prev_row['bb_upper']
        
        if self.use_ema:
            # Fast EMA below slow EMA (downtrend)
            conditions['ema'] = row['ema_fast'] < row['ema_slow']
        
        if self.use_cci:
            # CCI overbought reversal
            conditions['cci'] = (prev_row['cci'] > self.cci_upper and row['cci'] <= self.cci_upper) or row['cci'] < 0
        
        return conditions
    
    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Signal]]:
        """Run strategy"""
        df = self.calculate_indicators(df)
        
        signals = []
        in_position = False
        
        for i in range(1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            # Skip if indicators not ready
            if pd.isna(row['macd_signal']) or pd.isna(row['rsi']):
                continue
            
            indicators = {
                'rsi': row['rsi'],
                'macd': row['macd'],
                'cci': row['cci'],
                'ema_fast': row['ema_fast'],
                'ema_slow': row['ema_slow']
            }
            
            buy_conditions = self.check_buy_conditions(row, prev_row)
            sell_conditions = self.check_sell_conditions(row, prev_row)
            
            # BUY: All enabled indicators must agree
            if not in_position and buy_conditions:
                if all(buy_conditions.values()):
                    signals.append(Signal(
                        timestamp=df.index[i],
                        type='BUY',
                        price=row['close'],
                        confidence=len(buy_conditions) / 5,  # More indicators = higher confidence
                        indicators=indicators,
                        context=f"Combined: {', '.join(buy_conditions.keys())}"
                    ))
                    in_position = True
            
            # SELL: All enabled indicators must agree
            elif in_position and sell_conditions:
                if all(sell_conditions.values()):
                    signals.append(Signal(
                        timestamp=df.index[i],
                        type='SELL',
                        price=row['close'],
                        confidence=len(sell_conditions) / 5,
                        indicators=indicators,
                        context=f"Exit: {', '.join(sell_conditions.keys())}"
                    ))
                    in_position = False
        
        return df, signals


def test_indicator_combinations(symbol='BTCUSDT', timeframe='720', days=730):
    """Test all indicator combinations"""
    import yaml
    from data.bybit_connector import BybitConnector
    from backtesting.backtest_engine import BacktestEngine
    
    # Load config
    with open('config/strategy_params.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 80)
    print(f"MULTI-INDICATOR COMBINATION TEST")
    print(f"Symbol: {symbol} | Timeframe: {timeframe}min | History: {days} days")
    print("=" * 80)
    
    # Fetch data
    connector = BybitConnector(cache_dir='data/cache')
    df = connector.get_historical_data(symbol, timeframe, days, use_cache=True)
    
    if df.empty:
        print("No data")
        return
    
    print(f"📊 Loaded {len(df)} candles\n")
    
    # Test all combinations of indicators (2^5 - 1 = 31 combinations, excluding empty)
    indicators = ['macd', 'rsi', 'bb', 'ema', 'cci']
    results = []
    
    # Generate all non-empty combinations
    for r in range(1, len(indicators) + 1):
        for combo in itertools.combinations(indicators, r):
            indicator_config = {
                'use_macd': 'macd' in combo,
                'use_rsi': 'rsi' in combo,
                'use_bb': 'bb' in combo,
                'use_ema': 'ema' in combo,
                'use_cci': 'cci' in combo,
                # Best settings from previous tests
                'macd_fast': 8,
                'macd_slow': 17,
                'macd_signal': 9,
                'rsi_length': 14,
                'rsi_overbought': 70,
                'rsi_oversold': 30,
                'ema_fast': 9,
                'ema_slow': 21
            }
            
            try:
                strategy = MultiIndicatorStrategy(indicator_config)
                df_copy = df.copy()
                _, signals = strategy.run(df_copy)
                
                if not signals or len(signals) < 2:
                    continue
                
                backtester = BacktestEngine(config)
                result = backtester.run(df_copy, signals, symbol, timeframe)
                
                results.append({
                    'name': strategy.name,
                    'num_indicators': len(combo),
                    'trades': result.total_trades,
                    'win_rate': result.win_rate,
                    'pnl_pct': result.total_pnl_pct,
                    'profit_factor': result.profit_factor,
                    'sharpe': result.sharpe_ratio,
                    'max_dd': result.max_drawdown_pct
                })
                
                print(f"  {strategy.name}: {result.total_trades} trades, {result.win_rate:.1f}% win, {result.total_pnl_pct:+.2f}%")
                
            except Exception as e:
                pass
    
    # Sort by PnL
    results.sort(key=lambda x: x['pnl_pct'], reverse=True)
    
    print("\n" + "=" * 80)
    print("📈 INDICATOR COMBINATION RESULTS (Top 15 by PnL)")
    print("=" * 80)
    print(f"{'Combination':<25} {'#Ind':<5} {'Trades':<8} {'Win %':<8} {'PnL %':<10} {'PF':<6} {'Sharpe':<8}")
    print("-" * 80)
    
    for r in results[:15]:
        print(f"{r['name']:<25} {r['num_indicators']:<5} {r['trades']:<8} {r['win_rate']:.1f}%{'':<3} {r['pnl_pct']:+.2f}%{'':<4} {r['profit_factor']:.2f}{'':<3} {r['sharpe']:.2f}")
    
    print("=" * 80)
    
    # Analysis
    print("\n📊 ANALYSIS BY NUMBER OF INDICATORS:")
    for n in range(1, 6):
        n_results = [r for r in results if r['num_indicators'] == n]
        if n_results:
            avg_pnl = sum(r['pnl_pct'] for r in n_results) / len(n_results)
            best = max(n_results, key=lambda x: x['pnl_pct'])
            print(f"  {n} indicator(s): Avg PnL {avg_pnl:+.2f}% | Best: {best['name']} ({best['pnl_pct']:+.2f}%)")
    
    if results:
        best = results[0]
        print(f"\n🏆 BEST COMBINATION: {best['name']}")
        print(f"   PnL: {best['pnl_pct']:+.2f}% | Win Rate: {best['win_rate']:.1f}% | Profit Factor: {best['profit_factor']:.2f}")
    
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Multi-Indicator Combination Test')
    parser.add_argument('--symbol', type=str, default='BTCUSDT')
    parser.add_argument('--timeframe', type=str, default='720')
    parser.add_argument('--days', type=int, default=730)
    
    args = parser.parse_args()
    test_indicator_combinations(args.symbol, args.timeframe, args.days)
