#!/usr/bin/env python3
"""
Comprehensive Strategy Optimizer
Tests all indicator combinations and timeframes to find optimal settings.
"""

import yaml
import sys
import copy
import itertools
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from data.bybit_connector import BybitConnector
from strategy.gold_line_strategy import GoldLineStrategy
from backtesting.backtest_engine import BacktestEngine


def load_config(config_path: str = 'config/strategy_params.yaml') -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class StrategyOptimizer:
    """Comprehensive strategy optimizer"""
    
    def __init__(self, symbol='BTCUSDT', days=365):
        self.symbol = symbol
        self.days = days
        self.base_config = load_config()
        self.results = []
        
    def fetch_data(self, timeframe: str):
        """Fetch data for a given timeframe"""
        connector = BybitConnector(cache_dir='data/cache')
        return connector.get_historical_data(
            symbol=self.symbol,
            interval=timeframe,
            days=self.days,
            use_cache=True
        )
    
    def run_backtest(self, df, config):
        """Run a single backtest with given config"""
        strategy = GoldLineStrategy(config)
        df_with_indicators, signals = strategy.run(df)
        
        backtester = BacktestEngine(config)
        result = backtester.run(df_with_indicators, signals, self.symbol, '15')
        
        return {
            'trades': result.total_trades,
            'win_rate': result.win_rate,
            'pnl_pct': result.total_pnl_pct,
            'profit_factor': result.profit_factor,
            'sharpe': result.sharpe_ratio,
            'max_dd': result.max_drawdown_pct
        }
    
    def test_timeframes(self):
        """Test all timeframes"""
        timeframes = {
            '5': '5min',
            '15': '15min',
            '30': '30min',
            '60': '1hour',
            '120': '2hour',
            '240': '4hour',
            '360': '6hour',
            '720': '12hour',
            'D': '1day'
        }
        
        print("\n" + "="*80)
        print("📊 TIMEFRAME COMPARISON")
        print("="*80)
        
        results = []
        for tf, label in timeframes.items():
            print(f"\n Testing {label}...", end=" ", flush=True)
            try:
                df = self.fetch_data(tf)
                if df.empty or len(df) < 100:
                    print("⚠️ Insufficient data")
                    continue
                    
                result = self.run_backtest(df, self.base_config)
                result['timeframe'] = label
                result['candles'] = len(df)
                results.append(result)
                print(f"✅ {result['trades']} trades, {result['win_rate']:.1f}% win, {result['pnl_pct']:+.2f}%")
            except Exception as e:
                print(f"❌ {e}")
        
        # Sort by PnL
        results.sort(key=lambda x: x['pnl_pct'], reverse=True)
        
        print("\n" + "="*80)
        print(f"{'Timeframe':<12} {'Candles':<10} {'Trades':<8} {'Win %':<8} {'PnL %':<10} {'PF':<6} {'Sharpe':<8}")
        print("-"*80)
        for r in results:
            print(f"{r['timeframe']:<12} {r['candles']:<10} {r['trades']:<8} {r['win_rate']:.1f}%{'':<3} {r['pnl_pct']:+.2f}%{'':<4} {r['profit_factor']:.2f}{'':<3} {r['sharpe']:.2f}")
        
        return results
    
    def test_indicator_combinations(self, timeframe='240'):
        """Test different indicator parameter combinations"""
        
        print("\n" + "="*80)
        print("📊 INDICATOR PARAMETER OPTIMIZATION")
        print(f"Timeframe: {timeframe}min")
        print("="*80)
        
        df = self.fetch_data(timeframe)
        if df.empty:
            print("No data available")
            return []
        
        # Parameter ranges to test
        cci_lengths = [10, 14, 20, 30]
        cci_levels = [50, 75, 100]
        macd_configs = [
            (8, 17, 9),   # Fast
            (12, 26, 9),  # Standard
            (12, 17, 8),  # Our current
            (19, 39, 9)   # Slow
        ]
        rsi_lengths = [7, 14, 21]
        rsi_levels = [(30, 70), (40, 60), (35, 65)]
        
        results = []
        total_tests = len(cci_lengths) * len(cci_levels) * len(macd_configs) * len(rsi_lengths) * len(rsi_levels)
        
        print(f"\nTesting {total_tests} combinations...")
        test_num = 0
        
        for cci_len in cci_lengths:
            for cci_lvl in cci_levels:
                for macd_cfg in macd_configs:
                    for rsi_len in rsi_lengths:
                        for rsi_lvl in rsi_levels:
                            test_num += 1
                            if test_num % 20 == 0:
                                print(f"  Progress: {test_num}/{total_tests} ({test_num/total_tests*100:.0f}%)")
                            
                            config = copy.deepcopy(self.base_config)
                            config['cci']['length'] = cci_len
                            config['cci']['upper_level'] = cci_lvl
                            config['cci']['lower_level'] = -cci_lvl
                            config['macd']['fast_length'] = macd_cfg[0]
                            config['macd']['slow_length'] = macd_cfg[1]
                            config['macd']['signal_length'] = macd_cfg[2]
                            config['rsi']['length'] = rsi_len
                            config['rsi']['lower_limit'] = rsi_lvl[0]
                            config['rsi']['upper_limit'] = rsi_lvl[1]
                            
                            try:
                                result = self.run_backtest(df, config)
                                result['params'] = {
                                    'cci': f"{cci_len}/{cci_lvl}",
                                    'macd': f"{macd_cfg[0]}/{macd_cfg[1]}/{macd_cfg[2]}",
                                    'rsi': f"{rsi_len}/{rsi_lvl[0]}-{rsi_lvl[1]}"
                                }
                                results.append(result)
                            except:
                                pass
        
        # Sort by PnL
        results.sort(key=lambda x: x['pnl_pct'], reverse=True)
        
        # Show top 10
        print("\n" + "="*80)
        print("🏆 TOP 10 INDICATOR COMBINATIONS")
        print("="*80)
        print(f"{'CCI':<10} {'MACD':<12} {'RSI':<12} {'Trades':<8} {'Win %':<8} {'PnL %':<10} {'PF':<6}")
        print("-"*80)
        
        for r in results[:10]:
            p = r['params']
            print(f"{p['cci']:<10} {p['macd']:<12} {p['rsi']:<12} {r['trades']:<8} {r['win_rate']:.1f}%{'':<3} {r['pnl_pct']:+.2f}%{'':<4} {r['profit_factor']:.2f}")
        
        return results
    
    def test_filter_combinations(self, timeframe='240'):
        """Test different filter on/off combinations"""
        
        print("\n" + "="*80)
        print("📊 FILTER COMBINATION TEST")
        print("="*80)
        
        df = self.fetch_data(timeframe)
        if df.empty:
            return []
        
        filters = ['use_macd_filter', 'use_rsi_filter', 'use_trend_filter', 'use_macd_direction_filter']
        
        results = []
        
        # Test all combinations (2^4 = 16)
        for combo in itertools.product([True, False], repeat=4):
            config = copy.deepcopy(self.base_config)
            for i, f in enumerate(filters):
                config['filters'][f] = combo[i]
            
            filter_str = ''.join(['1' if c else '0' for c in combo])
            
            result = self.run_backtest(df, config)
            result['filters'] = filter_str
            result['filter_names'] = [filters[i].replace('use_', '').replace('_filter', '') for i in range(4) if combo[i]]
            results.append(result)
        
        results.sort(key=lambda x: x['pnl_pct'], reverse=True)
        
        print(f"\n{'Filters':<40} {'Trades':<8} {'Win %':<8} {'PnL %':<10} {'PF':<6}")
        print("-"*80)
        
        for r in results:
            filter_names = ', '.join(r['filter_names']) if r['filter_names'] else 'None'
            print(f"{filter_names:<40} {r['trades']:<8} {r['win_rate']:.1f}%{'':<3} {r['pnl_pct']:+.2f}%{'':<4} {r['profit_factor']:.2f}")
        
        return results
    
    def run_full_optimization(self):
        """Run all optimization tests"""
        print("="*80)
        print(f"🚀 COMPREHENSIVE STRATEGY OPTIMIZATION")
        print(f"Symbol: {self.symbol} | History: {self.days} days")
        print("="*80)
        
        # 1. Timeframe comparison
        tf_results = self.test_timeframes()
        best_tf = tf_results[0]['timeframe'] if tf_results else '240'
        
        # Convert timeframe label back to code
        tf_map = {'5min': '5', '15min': '15', '30min': '30', '1hour': '60', '2hour': '120', '4hour': '240', '6hour': '360', '12hour': '720', '1day': 'D'}
        best_tf_code = [k for k, v in tf_map.items() if v == best_tf][0] if best_tf in tf_map.values() else '240'
        
        print(f"\n🏆 Best Timeframe: {best_tf}")
        
        # 2. Filter combinations on best timeframe
        filter_results = self.test_filter_combinations(best_tf_code)
        
        # 3. Indicator parameter optimization
        indicator_results = self.test_indicator_combinations(best_tf_code)
        
        # Summary
        print("\n" + "="*80)
        print("📋 OPTIMIZATION SUMMARY")
        print("="*80)
        
        if tf_results:
            print(f"\n✅ Best Timeframe: {tf_results[0]['timeframe']} ({tf_results[0]['pnl_pct']:+.2f}%)")
        
        if filter_results:
            best_filter = filter_results[0]
            print(f"✅ Best Filters: {', '.join(best_filter['filter_names']) if best_filter['filter_names'] else 'None'} ({best_filter['pnl_pct']:+.2f}%)")
        
        if indicator_results:
            best_ind = indicator_results[0]
            print(f"✅ Best Indicators: CCI {best_ind['params']['cci']}, MACD {best_ind['params']['macd']}, RSI {best_ind['params']['rsi']} ({best_ind['pnl_pct']:+.2f}%)")
        
        return {
            'timeframes': tf_results,
            'filters': filter_results,
            'indicators': indicator_results
        }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Strategy Optimizer')
    parser.add_argument('--symbol', type=str, default='BTCUSDT')
    parser.add_argument('--days', type=int, default=365)
    parser.add_argument('--timeframes', action='store_true', help='Test timeframes only')
    parser.add_argument('--indicators', action='store_true', help='Test indicators only')
    parser.add_argument('--filters', action='store_true', help='Test filters only')
    
    args = parser.parse_args()
    
    optimizer = StrategyOptimizer(args.symbol, args.days)
    
    if args.timeframes:
        optimizer.test_timeframes()
    elif args.indicators:
        optimizer.test_indicator_combinations()
    elif args.filters:
        optimizer.test_filter_combinations()
    else:
        optimizer.run_full_optimization()
