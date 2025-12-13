"""
Bollinger Bands Breakout Strategy
Python port of the Pine Script "Demo GPT - Bollinger Bands" strategy.

Entry: Long when close > upper band
Exit: Close when close < lower band
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
from dataclasses import dataclass


@dataclass
class BBSignal:
    """Represents a trading signal"""
    timestamp: pd.Timestamp
    type: str  # 'BUY' or 'SELL'
    price: float
    indicators: Dict[str, float]


class BollingerBandsStrategy:
    """
    Bollinger Bands Breakout Strategy
    
    Entry: Long when close > upper band
    Exit: Close when close < lower band
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize strategy with configuration"""
        config = config or {}
        bb_config = config.get('bollinger', {})
        
        self.length = bb_config.get('length', 20)
        self.mult = bb_config.get('mult', 2.0)
        self.ma_type = bb_config.get('ma_type', 'SMA')
    
    def _calculate_ma(self, series: pd.Series, length: int, ma_type: str) -> pd.Series:
        """Calculate moving average based on type"""
        if ma_type == 'SMA':
            return series.rolling(window=length).mean()
        elif ma_type == 'EMA':
            return series.ewm(span=length, adjust=False).mean()
        elif ma_type == 'SMMA (RMA)':
            return series.ewm(alpha=1/length, adjust=False).mean()
        elif ma_type == 'WMA':
            weights = np.arange(1, length + 1)
            return series.rolling(window=length).apply(
                lambda x: np.dot(x, weights) / weights.sum(), raw=True
            )
        elif ma_type == 'VWMA':
            # Needs volume, fallback to SMA
            return series.rolling(window=length).mean()
        else:
            return series.rolling(window=length).mean()
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        df = df.copy()
        df.columns = df.columns.str.lower()
        
        # Calculate basis (middle band)
        df['bb_basis'] = self._calculate_ma(df['close'], self.length, self.ma_type)
        
        # Calculate standard deviation
        df['bb_dev'] = df['close'].rolling(window=self.length).std() * self.mult
        
        # Upper and lower bands
        df['bb_upper'] = df['bb_basis'] + df['bb_dev']
        df['bb_lower'] = df['bb_basis'] - df['bb_dev']
        
        # Band width (volatility indicator)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_basis'] * 100
        
        # Position relative to bands
        df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[BBSignal]]:
        """Generate buy/sell signals based on Bollinger Bands breakout"""
        df = self.calculate_indicators(df)
        
        signals = []
        df['buy_signal'] = False
        df['sell_signal'] = False
        
        in_position = False
        
        for i in range(1, len(df)):
            row = df.iloc[i]
            timestamp = df.index[i]
            
            # Skip if bands not calculated yet
            if pd.isna(row['bb_upper']) or pd.isna(row['bb_lower']):
                continue
            
            indicators = {
                'bb_upper': row['bb_upper'],
                'bb_lower': row['bb_lower'],
                'bb_basis': row['bb_basis'],
                'bb_width': row['bb_width'],
                'bb_pct': row['bb_pct']
            }
            
            # Long condition: close > upper band
            if row['close'] > row['bb_upper'] and not in_position:
                df.loc[df.index[i], 'buy_signal'] = True
                signals.append(BBSignal(
                    timestamp=timestamp,
                    type='BUY',
                    price=row['close'],
                    indicators=indicators
                ))
                in_position = True
            
            # Exit condition: close < lower band
            elif row['close'] < row['bb_lower'] and in_position:
                df.loc[df.index[i], 'sell_signal'] = True
                signals.append(BBSignal(
                    timestamp=timestamp,
                    type='SELL',
                    price=row['close'],
                    indicators=indicators
                ))
                in_position = False
        
        return df, signals
    
    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List]:
        """Run the strategy"""
        df, signals = self.generate_signals(df)
        
        # Convert to Signal format expected by backtester
        from strategy.gold_line_strategy import Signal
        
        converted_signals = []
        for s in signals:
            converted_signals.append(Signal(
                timestamp=s.timestamp,
                type=s.type,
                price=s.price,
                confidence=0.8,
                indicators=s.indicators,
                context=f"BB breakout: {'above upper' if s.type == 'BUY' else 'below lower'}"
            ))
        
        return df, converted_signals


def backtest_bollinger(symbol='BTCUSDT', timeframe='240', days=730, config=None):
    """Run Bollinger Bands backtest"""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    
    from data.bybit_connector import BybitConnector
    from backtesting.backtest_engine import BacktestEngine
    import yaml
    
    # Load config
    if config is None:
        with open('config/strategy_params.yaml', 'r') as f:
            config = yaml.safe_load(f)
    
    print("=" * 60)
    print("Bollinger Bands Breakout Strategy")
    print("=" * 60)
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}min")
    print(f"History: {days} days")
    print("=" * 60)
    
    # Fetch data
    connector = BybitConnector(cache_dir='data/cache')
    df = connector.get_historical_data(symbol, timeframe, days, use_cache=True)
    
    if df.empty:
        print("No data available")
        return None
    
    print(f"\n📊 Loaded {len(df)} candles")
    
    # Run strategy
    strategy = BollingerBandsStrategy(config)
    df_with_indicators, signals = strategy.run(df)
    
    buy_signals = len([s for s in signals if s.type == 'BUY'])
    sell_signals = len([s for s in signals if s.type == 'SELL'])
    print(f"✅ Generated {len(signals)} signals ({buy_signals} buys, {sell_signals} sells)")
    
    # Run backtest
    backtester = BacktestEngine(config)
    result = backtester.run(df_with_indicators, signals, symbol, timeframe)
    
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
    print(f"Profit Factor:    {result.profit_factor:.2f}")
    print(f"Sharpe Ratio:     {result.sharpe_ratio:.2f}")
    print(f"Max Drawdown:     {result.max_drawdown_pct:.2f}%")
    print("=" * 60)
    
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Bollinger Bands Backtest')
    parser.add_argument('--symbol', type=str, default='BTCUSDT')
    parser.add_argument('--timeframe', type=str, default='240')
    parser.add_argument('--days', type=int, default=730)
    parser.add_argument('--length', type=int, default=20, help='BB length')
    parser.add_argument('--mult', type=float, default=2.0, help='StdDev multiplier')
    
    args = parser.parse_args()
    
    config = {
        'bollinger': {
            'length': args.length,
            'mult': args.mult,
            'ma_type': 'SMA'
        },
        'trading': {
            'position_size_pct': 5.0,
            'stop_loss_pct': 1.0,
            'take_profit_pct': 10.0,
            'use_trailing_stop': True,
            'trailing_stop_pct': 0.5,
            'trailing_activation_pct': 0.5
        },
        'backtest': {
            'initial_capital': 10000,
            'commission_pct': 0.1,
            'slippage_pct': 0.05
        }
    }
    
    backtest_bollinger(args.symbol, args.timeframe, args.days, config)
