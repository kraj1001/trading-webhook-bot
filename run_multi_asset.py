#!/usr/bin/env python3
"""
Multi-Asset Extended Backtest
Tests strategy across multiple crypto assets with maximum available history.
Aggregates results for LLM fine-tuning.
"""

import yaml
import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from data.bybit_connector import BybitConnector
from strategy.gold_line_strategy import GoldLineStrategy
from backtesting.backtest_engine import BacktestEngine


def load_config(config_path: str = 'config/strategy_params.yaml') -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_multi_asset_test(assets: list, timeframe: str = '240', days: int = 3650):
    """
    Run backtests on multiple assets and aggregate results.
    
    Args:
        assets: List of trading pairs (e.g., ['BTCUSDT', 'ETHUSDT'])
        timeframe: Chart timeframe (default 240 = 4h)
        days: Days of history (default 3650 = 10 years, will fetch max available)
    """
    config = load_config()
    
    print("=" * 80)
    print(f"Multi-Asset Extended Backtest")
    print(f"Timeframe: {timeframe}min | Requested History: {days} days (~{days/365:.1f} years)")
    print(f"Assets: {', '.join(assets)}")
    print("=" * 80)
    
    all_results = []
    all_trades = []
    
    for symbol in assets:
        print(f"\n{'='*40}")
        print(f"📊 Testing {symbol}...")
        print(f"{'='*40}")
        
        try:
            # Fetch data (will get max available if less than requested)
            connector = BybitConnector(cache_dir='data/cache')
            df = connector.get_historical_data(
                symbol=symbol,
                interval=timeframe,
                days=days,
                use_cache=True
            )
            
            if df.empty or len(df) < 100:
                print(f"   ⚠️ Insufficient data for {symbol}")
                continue
            
            actual_days = (df.index[-1] - df.index[0]).days
            print(f"   📅 Data: {len(df)} candles ({actual_days} days)")
            print(f"   📅 From: {df.index[0]} to {df.index[-1]}")
            
            # Run strategy
            strategy = GoldLineStrategy(config)
            df_with_indicators, signals = strategy.run(df)
            
            # Run backtest
            backtester = BacktestEngine(config)
            result = backtester.run(df_with_indicators, signals, symbol, timeframe)
            
            # Save individual trades
            output_dir = Path('results')
            trades_file = backtester.save_results(result, str(output_dir))
            
            # Load trades for aggregation
            with open(trades_file, 'r') as f:
                trades = json.load(f)
                for t in trades:
                    t['symbol'] = symbol  # Add symbol to each trade
                all_trades.extend(trades)
            
            result_dict = {
                'symbol': symbol,
                'candles': len(df),
                'actual_days': actual_days,
                'trades': result.total_trades,
                'win_rate': result.win_rate,
                'pnl_pct': result.total_pnl_pct,
                'profit_factor': result.profit_factor,
                'max_dd': result.max_drawdown_pct,
                'sharpe': result.sharpe_ratio
            }
            all_results.append(result_dict)
            
            print(f"   ✅ {result.total_trades} trades | Win: {result.win_rate:.1f}% | PnL: {result.total_pnl_pct:+.2f}%")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print("\n" + "=" * 80)
    print("📈 MULTI-ASSET RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Symbol':<12} {'Candles':<10} {'Days':<8} {'Trades':<8} {'Win %':<8} {'PnL %':<10} {'PF':<6} {'Sharpe':<8}")
    print("-" * 80)
    
    total_trades = 0
    total_pnl = 0
    
    for r in all_results:
        print(f"{r['symbol']:<12} {r['candles']:<10} {r['actual_days']:<8} {r['trades']:<8} {r['win_rate']:.1f}%{'':<3} {r['pnl_pct']:+.2f}%{'':<4} {r['profit_factor']:.2f}{'':<3} {r['sharpe']:.2f}")
        total_trades += r['trades']
        total_pnl += r['pnl_pct']
    
    print("-" * 80)
    avg_win_rate = sum(r['win_rate'] for r in all_results) / len(all_results) if all_results else 0
    avg_pf = sum(r['profit_factor'] for r in all_results) / len(all_results) if all_results else 0
    print(f"{'TOTAL':<12} {'':<10} {'':<8} {total_trades:<8} {avg_win_rate:.1f}%{'':<3} {total_pnl/len(all_results):+.2f}%{'':<4} {avg_pf:.2f}")
    print("=" * 80)
    
    # Save aggregated trades for LLM training
    aggregated_file = Path('results') / 'all_trades_aggregated.json'
    with open(aggregated_file, 'w') as f:
        json.dump(all_trades, f, indent=2)
    
    print(f"\n💾 Aggregated {len(all_trades)} trades saved to: {aggregated_file}")
    
    # Generate training data
    print("\n📝 Generating LLM training data from all assets...")
    generate_training_data(all_trades)
    
    return all_results, all_trades


def generate_training_data(trades: list, output_file: str = 'results/training_data_all_assets.jsonl'):
    """Generate training data from all trades across assets"""
    
    training_examples = []
    
    for trade in trades:
        indicators = trade.get('indicators_at_entry', {})
        context = trade.get('market_context', {})
        
        example = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a crypto trading signal filter. Analyze the signal context and decide whether to TAKE or SKIP the trade."
                },
                {
                    "role": "user",
                    "content": f"""Symbol: {trade.get('symbol', 'UNKNOWN')}
Signal: {trade['direction']}
Entry Price: ${trade['entry_price']:.2f}
CCI: {indicators.get('cci', 0):.1f}
RSI: {indicators.get('rsi', 0):.1f}
MACD: {indicators.get('macd', 0):.2f}
Price vs Gold Line: {context.get('price_vs_gold_line', 'unknown')}
Volatility: {context.get('volatility', 0):.2f}%

Should I take this trade?"""
                },
                {
                    "role": "assistant",
                    "content": f"{'TAKE' if trade['result'] == 'WIN' else 'SKIP'} - Based on the indicators and {trade.get('symbol', 'asset')} behavior: {'Strong setup with favorable momentum.' if trade['result'] == 'WIN' else 'Weak setup, indicators suggest caution.'}"
                }
            ]
        }
        training_examples.append(example)
    
    # Write JSONL
    output_path = Path(output_file)
    with open(output_path, 'w') as f:
        for example in training_examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"✅ Generated {len(training_examples)} training examples to {output_path}")
    return str(output_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Multi-Asset Extended Backtest')
    parser.add_argument('--assets', type=str, nargs='+', 
                        default=['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT'],
                        help='Trading pairs to test')
    parser.add_argument('--timeframe', type=str, default='240', help='Timeframe')
    parser.add_argument('--days', type=int, default=3650, help='Days of history (max available)')
    
    args = parser.parse_args()
    run_multi_asset_test(args.assets, args.timeframe, args.days)
