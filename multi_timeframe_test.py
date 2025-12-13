"""
Multi-Timeframe Strategy Comparison
Tests top strategies across different timeframes.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import yaml
from data.bybit_connector import BybitConnector
from backtesting.backtest_engine import BacktestEngine

# Import all strategies
from strategy.volume_strategies import OBVStrategy, VolumeImbalanceStrategy
from strategy.editors_choice_strategies import (
    UTBotAlertStrategy, ChandelierExitStrategy, SSLChannelStrategy,
    CMFStrategy, AroonStrategy, VortexStrategy
)
from strategy.microstructure_strategies import (
    OrderFlowImbalanceStrategy, SmartMoneyStrategy, FractalStructureStrategy
)
from strategy.tradingview_strategies import (
    SuperTrendStrategy, ParabolicSARStrategy, StochasticStrategy
)
from strategy.extended_strategies import HullMAStrategy, MomentumStrategy


def get_top_strategies():
    """Return list of top-performing strategies"""
    return [
        ('OBV 10', OBVStrategy(10)),
        ('OBV 20', OBVStrategy(20)),
        ('CMF 10', CMFStrategy(10)),
        ('CMF 20', CMFStrategy(20)),
        ('UT Bot 1/10', UTBotAlertStrategy(1, 10)),
        ('UT Bot 2/10', UTBotAlertStrategy(2, 10)),
        ('SSL Channel 10', SSLChannelStrategy(10)),
        ('Chandelier 14', ChandelierExitStrategy(14, 2.0)),
        ('Aroon 14', AroonStrategy(14)),
        ('Vortex 14', VortexStrategy(14)),
        ('Smart Money 10', SmartMoneyStrategy(10, 50)),
        ('Order Flow 10', OrderFlowImbalanceStrategy(10, 1.5)),
        ('Fractal 5', FractalStructureStrategy(5, 3)),
        ('Hull MA', HullMAStrategy(9)),
        ('Parabolic SAR', ParabolicSARStrategy(0.02, 0.2)),
        ('SuperTrend', SuperTrendStrategy(10, 3.0)),
        ('Momentum 10', MomentumStrategy(10)),
        ('Stochastic', StochasticStrategy(14, 3, 80, 20)),
    ]


def test_strategy(strategy, df, config, name):
    """Test a single strategy and return results"""
    try:
        df_copy = df.copy()
        _, signals = strategy.run(df_copy)
        
        if not signals or len(signals) < 2:
            return None
        
        backtester = BacktestEngine(config)
        result = backtester.run(df_copy, signals, 'TEST', '0')
        
        return {
            'name': name,
            'trades': result.total_trades,
            'win_rate': result.win_rate,
            'pnl_pct': result.total_pnl_pct,
            'profit_factor': result.profit_factor,
        }
    except Exception as e:
        return None


def run_multi_timeframe_test(symbol='ETHUSDT', days=365):
    """Run all top strategies across multiple timeframes"""
    
    with open('config/strategy_params.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    connector = BybitConnector(cache_dir='data/cache')
    
    # Timeframes to test: 1h, 4h, 12h, 1D
    timeframes = [
        ('60', '1h'),
        ('240', '4h'),
        ('720', '12h'),
        ('D', '1D'),
    ]
    
    strategies = get_top_strategies()
    
    print("=" * 100)
    print(f"MULTI-TIMEFRAME STRATEGY COMPARISON - {symbol}")
    print(f"Testing {len(strategies)} strategies across {len(timeframes)} timeframes")
    print("=" * 100)
    
    # Store results by strategy
    all_results = {}
    
    for tf_code, tf_name in timeframes:
        print(f"\n{'='*40} {tf_name} {'='*40}")
        
        try:
            df = connector.get_historical_data(symbol, tf_code, days, use_cache=True)
            print(f"📊 Loaded {len(df)} candles for {tf_name}\n")
        except Exception as e:
            print(f"❌ Failed to load {tf_name} data: {e}")
            continue
        
        for name, strategy in strategies:
            result = test_strategy(strategy, df, config, name)
            
            if result:
                if name not in all_results:
                    all_results[name] = {}
                all_results[name][tf_name] = result
                print(f"  ✅ {name:<20} {result['trades']:>4} trades  {result['win_rate']:>5.1f}% win  {result['pnl_pct']:>+7.2f}%")
            else:
                print(f"  ⚪ {name:<20} No signals")
    
    # Summary table
    print("\n" + "=" * 100)
    print("📊 STRATEGY PERFORMANCE BY TIMEFRAME")
    print("=" * 100)
    print(f"{'Strategy':<22} {'1h':>10} {'4h':>10} {'12h':>10} {'1D':>10} {'Best TF':>10}")
    print("-" * 100)
    
    best_overall = None
    best_overall_pnl = -999
    
    for name in sorted(all_results.keys(), key=lambda x: max([r.get('pnl_pct', -999) for r in all_results[x].values()]), reverse=True):
        row = f"{name:<22}"
        best_tf = None
        best_pnl = -999
        
        for tf_name in ['1h', '4h', '12h', '1D']:
            if tf_name in all_results[name]:
                pnl = all_results[name][tf_name]['pnl_pct']
                row += f" {pnl:>+9.2f}%"
                if pnl > best_pnl:
                    best_pnl = pnl
                    best_tf = tf_name
            else:
                row += f" {'---':>10}"
        
        row += f" {best_tf:>10}" if best_tf else ""
        print(row)
        
        if best_pnl > best_overall_pnl:
            best_overall_pnl = best_pnl
            best_overall = (name, best_tf, best_pnl)
    
    print("=" * 100)
    if best_overall:
        print(f"\n🏆 BEST OVERALL: {best_overall[0]} on {best_overall[1]} = {best_overall[2]:+.2f}%")
    
    return all_results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Multi-Timeframe Strategy Test')
    parser.add_argument('--symbol', type=str, default='ETHUSDT')
    parser.add_argument('--days', type=int, default=365)
    
    args = parser.parse_args()
    run_multi_timeframe_test(args.symbol, args.days)
