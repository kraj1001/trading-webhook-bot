#!/usr/bin/env python3
"""
Stop-Loss Optimization
Tests different stop-loss and take-profit configurations including trailing stops.
"""

import yaml
import sys
import copy
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from data.bybit_connector import BybitConnector
from strategy.gold_line_strategy import GoldLineStrategy, Signal
from backtesting.backtest_engine import BacktestEngine


def load_config(config_path: str = 'config/strategy_params.yaml') -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class TrailingStopBacktester(BacktestEngine):
    """Extended backtester with trailing stop support"""
    
    def __init__(self, config, trailing_stop_pct=None, trailing_activation_pct=None):
        super().__init__(config)
        self.trailing_stop_pct = trailing_stop_pct  # e.g., 0.01 = 1%
        self.trailing_activation_pct = trailing_activation_pct  # Activate after X% profit
    
    def run_with_trailing(self, df, signals, symbol='BTCUSDT', timeframe='240'):
        """Run backtest with optional trailing stop"""
        capital = self.initial_capital
        equity_curve = [capital]
        trades = []
        
        in_position = False
        position_direction = None
        entry_price = 0
        entry_time = None
        entry_signal = None
        position_size = 0
        highest_price = 0  # For trailing stop (longs)
        lowest_price = float('inf')  # For trailing stop (shorts)
        
        buy_signals = {s.timestamp: s for s in signals if s.type == 'BUY'}
        sell_signals = {s.timestamp: s for s in signals if s.type == 'SELL'}
        
        for i in range(len(df)):
            row = df.iloc[i]
            timestamp = df.index[i]
            current_price = row['close']
            high_price = row['high']
            low_price = row['low']
            
            if in_position:
                # Update trailing stop levels
                if position_direction == 'LONG':
                    highest_price = max(highest_price, high_price)
                else:
                    lowest_price = min(lowest_price, low_price)
                
                exit_reason = None
                
                if position_direction == 'LONG':
                    pnl_pct = (current_price - entry_price) / entry_price
                    
                    # Check trailing stop
                    if self.trailing_stop_pct and self.trailing_activation_pct:
                        if (highest_price - entry_price) / entry_price >= self.trailing_activation_pct:
                            trailing_stop_price = highest_price * (1 - self.trailing_stop_pct)
                            if low_price <= trailing_stop_price:
                                exit_reason = 'TRAILING_STOP'
                                current_price = trailing_stop_price
                    
                    # Regular stop loss
                    if not exit_reason and pnl_pct <= -self.stop_loss_pct:
                        exit_reason = 'STOP_LOSS'
                    elif not exit_reason and pnl_pct >= self.take_profit_pct:
                        exit_reason = 'TAKE_PROFIT'
                    elif not exit_reason and timestamp in sell_signals:
                        exit_reason = 'SIGNAL'
                
                else:  # SHORT
                    pnl_pct = (entry_price - current_price) / entry_price
                    
                    # Check trailing stop
                    if self.trailing_stop_pct and self.trailing_activation_pct:
                        if (entry_price - lowest_price) / entry_price >= self.trailing_activation_pct:
                            trailing_stop_price = lowest_price * (1 + self.trailing_stop_pct)
                            if high_price >= trailing_stop_price:
                                exit_reason = 'TRAILING_STOP'
                                current_price = trailing_stop_price
                    
                    if not exit_reason and pnl_pct <= -self.stop_loss_pct:
                        exit_reason = 'STOP_LOSS'
                    elif not exit_reason and pnl_pct >= self.take_profit_pct:
                        exit_reason = 'TAKE_PROFIT'
                    elif not exit_reason and timestamp in buy_signals:
                        exit_reason = 'SIGNAL'
                
                if exit_reason:
                    exit_price = current_price * (1 - self.slippage_pct if position_direction == 'LONG' else 1 + self.slippage_pct)
                    
                    if position_direction == 'LONG':
                        final_pnl_pct = (exit_price - entry_price) / entry_price
                    else:
                        final_pnl_pct = (entry_price - exit_price) / entry_price
                    
                    final_pnl_pct -= self.commission_pct * 2
                    pnl = position_size * final_pnl_pct
                    capital += pnl
                    
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': timestamp,
                        'direction': position_direction,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'pnl_pct': final_pnl_pct * 100,
                        'exit_reason': exit_reason
                    })
                    
                    in_position = False
                    position_direction = None
            
            if not in_position:
                if timestamp in buy_signals:
                    in_position = True
                    position_direction = 'LONG'
                    entry_price = current_price * (1 + self.slippage_pct)
                    entry_time = timestamp
                    position_size = capital * self.position_size_pct
                    highest_price = current_price
                    lowest_price = float('inf')
                
                elif timestamp in sell_signals:
                    in_position = True
                    position_direction = 'SHORT'
                    entry_price = current_price * (1 - self.slippage_pct)
                    entry_time = timestamp
                    position_size = capital * self.position_size_pct
                    highest_price = 0
                    lowest_price = current_price
            
            equity_curve.append(capital)
        
        # Calculate metrics
        if not trades:
            return {'trades': 0, 'win_rate': 0, 'pnl_pct': 0, 'profit_factor': 0}
        
        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] <= 0]
        total_wins = sum(t['pnl'] for t in wins)
        total_losses = abs(sum(t['pnl'] for t in losses))
        
        return {
            'trades': len(trades),
            'win_rate': len(wins) / len(trades) * 100,
            'pnl_pct': (capital - self.initial_capital) / self.initial_capital * 100,
            'profit_factor': total_wins / total_losses if total_losses > 0 else float('inf'),
            'trailing_stops': len([t for t in trades if t['exit_reason'] == 'TRAILING_STOP']),
            'stop_losses': len([t for t in trades if t['exit_reason'] == 'STOP_LOSS']),
            'take_profits': len([t for t in trades if t['exit_reason'] == 'TAKE_PROFIT'])
        }


def run_stoploss_optimization(symbol='BTCUSDT', timeframe='240', days=730):
    """Test different stop-loss configurations"""
    
    config = load_config()
    
    print("=" * 80)
    print(f"Stop-Loss Optimization: {symbol} {timeframe}min ({days} days)")
    print("=" * 80)
    
    # Fetch data
    connector = BybitConnector(cache_dir='data/cache')
    df = connector.get_historical_data(symbol, timeframe, days, use_cache=True)
    
    if df.empty:
        print("Error: No data")
        return
    
    # Run strategy
    strategy = GoldLineStrategy(config)
    df_with_indicators, signals = strategy.run(df)
    
    print(f"\n📊 Data: {len(df)} candles, {len(signals)} signals")
    
    # Test configurations
    configs = [
        # (stop_loss%, take_profit%, trailing_stop%, trailing_activation%)
        {'name': 'SL 1% / TP 2%', 'sl': 0.01, 'tp': 0.02, 'trail': None, 'trail_act': None},
        {'name': 'SL 1.5% / TP 3%', 'sl': 0.015, 'tp': 0.03, 'trail': None, 'trail_act': None},
        {'name': 'SL 2% / TP 4%', 'sl': 0.02, 'tp': 0.04, 'trail': None, 'trail_act': None},
        {'name': 'SL 2.5% / TP 5%', 'sl': 0.025, 'tp': 0.05, 'trail': None, 'trail_act': None},
        {'name': 'SL 3% / TP 6%', 'sl': 0.03, 'tp': 0.06, 'trail': None, 'trail_act': None},
        # Trailing stops
        {'name': 'SL 2% + Trail 1% @1%', 'sl': 0.02, 'tp': 0.10, 'trail': 0.01, 'trail_act': 0.01},
        {'name': 'SL 2% + Trail 1.5% @1.5%', 'sl': 0.02, 'tp': 0.10, 'trail': 0.015, 'trail_act': 0.015},
        {'name': 'SL 2% + Trail 2% @2%', 'sl': 0.02, 'tp': 0.10, 'trail': 0.02, 'trail_act': 0.02},
        {'name': 'SL 1.5% + Trail 1% @1%', 'sl': 0.015, 'tp': 0.10, 'trail': 0.01, 'trail_act': 0.01},
        {'name': 'SL 1% + Trail 0.5% @0.5%', 'sl': 0.01, 'tp': 0.10, 'trail': 0.005, 'trail_act': 0.005},
    ]
    
    results = []
    
    print("\n🔍 Testing configurations...\n")
    
    for cfg in configs:
        test_config = copy.deepcopy(config)
        test_config['trading']['stop_loss_pct'] = cfg['sl'] * 100
        test_config['trading']['take_profit_pct'] = cfg['tp'] * 100
        
        backtester = TrailingStopBacktester(
            test_config, 
            trailing_stop_pct=cfg['trail'],
            trailing_activation_pct=cfg['trail_act']
        )
        
        result = backtester.run_with_trailing(df_with_indicators, signals, symbol, timeframe)
        result['name'] = cfg['name']
        results.append(result)
        
        print(f"  {cfg['name']:<30} | Trades: {result['trades']:<4} | Win: {result['win_rate']:.1f}% | PnL: {result['pnl_pct']:+.2f}%")
    
    # Print comparison table
    print("\n" + "=" * 80)
    print("📈 STOP-LOSS COMPARISON RESULTS")
    print("=" * 80)
    print(f"{'Configuration':<32} {'Trades':<7} {'Win %':<8} {'PnL %':<10} {'PF':<6} {'Trail':<6} {'SL':<6} {'TP':<6}")
    print("-" * 80)
    
    results.sort(key=lambda x: x['pnl_pct'], reverse=True)
    
    for r in results:
        print(f"{r['name']:<32} {r['trades']:<7} {r['win_rate']:.1f}%{'':<3} {r['pnl_pct']:+.2f}%{'':<4} {r['profit_factor']:.2f}{'':<3} {r.get('trailing_stops', 0):<6} {r.get('stop_losses', 0):<6} {r.get('take_profits', 0):<6}")
    
    print("=" * 80)
    
    if results:
        best = results[0]
        print(f"\n🏆 Best Configuration: {best['name']}")
        print(f"   PnL: {best['pnl_pct']:+.2f}% | Win Rate: {best['win_rate']:.1f}% | Profit Factor: {best['profit_factor']:.2f}")
    
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Stop-Loss Optimization')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading pair')
    parser.add_argument('--timeframe', type=str, default='240', help='Timeframe')
    parser.add_argument('--days', type=int, default=730, help='Days of history')
    
    args = parser.parse_args()
    run_stoploss_optimization(args.symbol, args.timeframe, args.days)
