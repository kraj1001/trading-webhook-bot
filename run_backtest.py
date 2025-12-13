#!/usr/bin/env python3
"""
Run Backtest
Main script to fetch data from Bybit, run the Gold Line strategy, and generate results.
"""

import argparse
import yaml
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent))

from data.bybit_connector import BybitConnector
from strategy.gold_line_strategy import GoldLineStrategy
from backtesting.backtest_engine import BacktestEngine


def load_config(config_path: str = 'config/strategy_params.yaml') -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Run Gold Line Strategy Backtest')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading pair symbol')
    parser.add_argument('--timeframe', type=str, default='15', help='Chart timeframe in minutes')
    parser.add_argument('--days', type=int, default=90, help='Number of days of historical data')
    parser.add_argument('--config', type=str, default='config/strategy_params.yaml', help='Config file path')
    parser.add_argument('--no-cache', action='store_true', help='Disable data caching')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Gold Line Strategy Backtest")
    print("=" * 60)
    print(f"Symbol: {args.symbol}")
    print(f"Timeframe: {args.timeframe} min")
    print(f"History: {args.days} days")
    print("=" * 60)
    
    # Load configuration
    config = load_config(args.config)
    
    # Fetch historical data
    print("\n📊 Fetching historical data from Bybit...")
    connector = BybitConnector(cache_dir=config.get('data', {}).get('cache_dir', 'data/cache'))
    
    df = connector.get_historical_data(
        symbol=args.symbol,
        interval=args.timeframe,
        days=args.days,
        use_cache=not args.no_cache
    )
    
    if df.empty:
        print("❌ Failed to fetch data. Please check your connection and symbol.")
        return
    
    print(f"✅ Loaded {len(df)} candles from {df.index[0]} to {df.index[-1]}")
    
    # Run strategy
    print("\n🎯 Running Gold Line strategy...")
    strategy = GoldLineStrategy(config)
    df_with_indicators, signals = strategy.run(df)
    
    signal_summary = strategy.get_signal_summary(signals)
    print(f"✅ Generated {signal_summary['total']} signals:")
    print(f"   - Buy signals: {signal_summary.get('buy_signals', 0)}")
    print(f"   - Sell signals: {signal_summary.get('sell_signals', 0)}")
    print(f"   - CCI Up alerts: {signal_summary.get('cci_up', 0)}")
    print(f"   - CCI Down alerts: {signal_summary.get('cci_down', 0)}")
    
    # Run backtest
    print("\n💰 Running backtest simulation...")
    backtester = BacktestEngine(config)
    result = backtester.run(df_with_indicators, signals, args.symbol, args.timeframe)
    
    # Print results
    print("\n" + "=" * 60)
    print("📈 BACKTEST RESULTS")
    print("=" * 60)
    print(f"Initial Capital:  ${result.initial_capital:,.2f}")
    print(f"Final Capital:    ${result.final_capital:,.2f}")
    print(f"Total PnL:        ${result.total_pnl:,.2f} ({result.total_pnl_pct:+.2f}%)")
    print("-" * 40)
    print(f"Total Trades:     {result.total_trades}")
    print(f"Winning Trades:   {result.winning_trades}")
    print(f"Losing Trades:    {result.losing_trades}")
    print(f"Win Rate:         {result.win_rate:.1f}%")
    print("-" * 40)
    print(f"Avg Win:          ${result.avg_win:,.2f}")
    print(f"Avg Loss:         ${result.avg_loss:,.2f}")
    print(f"Largest Win:      ${result.largest_win:,.2f}")
    print(f"Largest Loss:     ${result.largest_loss:,.2f}")
    print("-" * 40)
    print(f"Max Drawdown:     ${result.max_drawdown:,.2f} ({result.max_drawdown_pct:.2f}%)")
    print(f"Sharpe Ratio:     {result.sharpe_ratio:.2f}")
    print(f"Profit Factor:    {result.profit_factor:.2f}")
    print(f"Avg Trade Length: {result.avg_trade_duration:.1f} candles")
    print("=" * 60)
    
    # Save results
    print("\n💾 Saving results...")
    trades_file = backtester.save_results(result)
    print(f"✅ Trades saved for LLM analysis")
    
    print("\n🤖 To analyze with LLM, run:")
    print(f"   python analyze_results.py --trades {trades_file}")
    
    return result


if __name__ == '__main__':
    main()
