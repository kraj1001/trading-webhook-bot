#!/usr/bin/env python3
"""
Multi-Timeframe Backtest Comparison
Runs backtests across multiple timeframes and compares results.
"""

import yaml
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from data.bybit_connector import BybitConnector
from strategy.gold_line_strategy import GoldLineStrategy
from backtesting.backtest_engine import BacktestEngine


def load_config(config_path: str = 'config/strategy_params.yaml') -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_comparison(symbol: str = 'BTCUSDT', days: int = 365):
    """Run backtests on multiple timeframes and compare"""
    
    timeframes = ['5', '15', '30', '60', '240']  # 5m, 15m, 30m, 1h, 4h
    config = load_config()
    
    print("=" * 70)
    print(f"Multi-Timeframe Comparison: {symbol}")
    print(f"Period: {days} days")
    print("=" * 70)
    
    results = []
    
    for tf in timeframes:
        tf_label = {
            '5': '5min',
            '15': '15min', 
            '30': '30min',
            '60': '1hour',
            '240': '4hour'
        }.get(tf, f'{tf}m')
        
        print(f"\n📊 Testing {tf_label}...")
        
        try:
            # Fetch data
            connector = BybitConnector(cache_dir='data/cache')
            df = connector.get_historical_data(
                symbol=symbol,
                interval=tf,
                days=days,
                use_cache=True
            )
            
            if df.empty or len(df) < 100:
                print(f"   ⚠️ Insufficient data for {tf_label}")
                continue
            
            # Run strategy
            strategy = GoldLineStrategy(config)
            df_with_indicators, signals = strategy.run(df)
            
            # Run backtest
            backtester = BacktestEngine(config)
            result = backtester.run(df_with_indicators, signals, symbol, tf)
            
            results.append({
                'timeframe': tf_label,
                'candles': len(df),
                'trades': result.total_trades,
                'win_rate': result.win_rate,
                'pnl_pct': result.total_pnl_pct,
                'profit_factor': result.profit_factor,
                'max_dd': result.max_drawdown_pct,
                'sharpe': result.sharpe_ratio
            })
            
            print(f"   ✅ {result.total_trades} trades | Win Rate: {result.win_rate:.1f}% | PnL: {result.total_pnl_pct:+.2f}%")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Print comparison table
    print("\n" + "=" * 70)
    print("📈 TIMEFRAME COMPARISON RESULTS")
    print("=" * 70)
    print(f"{'Timeframe':<10} {'Trades':<8} {'Win Rate':<10} {'PnL %':<10} {'Profit Factor':<14} {'Max DD %':<10}")
    print("-" * 70)
    
    # Sort by PnL
    results.sort(key=lambda x: x['pnl_pct'], reverse=True)
    
    for r in results:
        print(f"{r['timeframe']:<10} {r['trades']:<8} {r['win_rate']:.1f}%{'':<5} {r['pnl_pct']:+.2f}%{'':<4} {r['profit_factor']:.2f}{'':<10} {r['max_dd']:.2f}%")
    
    print("=" * 70)
    
    if results:
        best = results[0]
        print(f"\n🏆 Best Timeframe: {best['timeframe']} with {best['pnl_pct']:+.2f}% PnL and {best['win_rate']:.1f}% win rate")
    
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Multi-Timeframe Comparison')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading pair')
    parser.add_argument('--days', type=int, default=365, help='Days of history')
    
    args = parser.parse_args()
    run_comparison(args.symbol, args.days)
