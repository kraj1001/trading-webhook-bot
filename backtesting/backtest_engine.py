"""
Backtesting Engine
Simulates trades based on strategy signals and calculates performance metrics.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
import json
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from strategy.gold_line_strategy import Signal


@dataclass
class Trade:
    """Represents a completed trade"""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str  # 'LONG' or 'SHORT'
    entry_price: float
    exit_price: float
    position_size: float
    pnl: float
    pnl_pct: float
    duration_candles: int
    exit_reason: str  # 'SIGNAL', 'STOP_LOSS', 'TAKE_PROFIT', 'END'
    entry_signal: Dict[str, Any]
    indicators_at_entry: Dict[str, float]
    indicators_at_exit: Dict[str, float]
    market_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestResult:
    """Results from a backtest run"""
    symbol: str
    timeframe: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    initial_capital: float
    final_capital: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_pnl_pct: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    profit_factor: float
    avg_trade_duration: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    trades: List[Trade]
    equity_curve: pd.Series


class BacktestEngine:
    """
    Backtesting engine for the Gold Line strategy.
    Simulates trades and tracks performance.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize backtest engine.
        
        Args:
            config: Configuration dict with trading parameters
        """
        trading_config = config.get('trading', {})
        backtest_config = config.get('backtest', {})
        
        self.initial_capital = backtest_config.get('initial_capital', 10000)
        self.commission_pct = backtest_config.get('commission_pct', 0.075) / 100
        self.slippage_pct = backtest_config.get('slippage_pct', 0.05) / 100
        self.position_size_pct = trading_config.get('position_size_pct', 2.0) / 100
        self.stop_loss_pct = trading_config.get('stop_loss_pct', 1.5) / 100
        self.take_profit_pct = trading_config.get('take_profit_pct', 3.0) / 100
        
        # Trailing stop settings
        self.use_trailing_stop = trading_config.get('use_trailing_stop', False)
        self.trailing_stop_pct = trading_config.get('trailing_stop_pct', 0.5) / 100
        self.trailing_activation_pct = trading_config.get('trailing_activation_pct', 0.5) / 100
    
    def run(
        self,
        df: pd.DataFrame,
        signals: List[Signal],
        symbol: str = 'BTCUSDT',
        timeframe: str = '15'
    ) -> BacktestResult:
        """
        Run backtest on historical data with generated signals.
        
        Args:
            df: DataFrame with OHLCV data and calculated indicators
            signals: List of Signal objects from strategy
            symbol: Trading pair symbol
            timeframe: Chart timeframe
        
        Returns:
            BacktestResult with performance metrics and trade list
        """
        capital = self.initial_capital
        equity_curve = [capital]
        trades: List[Trade] = []
        
        # Track current position
        in_position = False
        position_direction = None
        entry_price = 0
        entry_time = None
        entry_signal = None
        entry_indicators = {}
        position_size = 0
        highest_price = 0  # For trailing stop (longs)
        lowest_price = float('inf')  # For trailing stop (shorts)
        
        # Create a signal lookup by timestamp
        buy_signals = {s.timestamp: s for s in signals if s.type == 'BUY'}
        sell_signals = {s.timestamp: s for s in signals if s.type == 'SELL'}
        
        for i in range(len(df)):
            row = df.iloc[i]
            timestamp = df.index[i]
            current_price = row['close']
            high_price = row['high']
            low_price = row['low']
            
            # Check for exit conditions if in position
            if in_position:
                # Update trailing stop tracking
                highest_price = max(highest_price, high_price)
                lowest_price = min(lowest_price, low_price)
                
                pnl_pct = 0
                exit_reason = None
                exit_price_override = None
                
                if position_direction == 'LONG':
                    pnl_pct = (current_price - entry_price) / entry_price
                    
                    # Check trailing stop first (if enabled and activated)
                    if self.use_trailing_stop:
                        profit_from_entry = (highest_price - entry_price) / entry_price
                        if profit_from_entry >= self.trailing_activation_pct:
                            trailing_stop_price = highest_price * (1 - self.trailing_stop_pct)
                            if low_price <= trailing_stop_price:
                                exit_reason = 'TRAILING_STOP'
                                exit_price_override = trailing_stop_price
                    
                    # Check stop loss
                    if not exit_reason and pnl_pct <= -self.stop_loss_pct:
                        exit_reason = 'STOP_LOSS'
                    # Check take profit
                    elif not exit_reason and pnl_pct >= self.take_profit_pct:
                        exit_reason = 'TAKE_PROFIT'
                    # Check for opposite signal
                    elif not exit_reason and timestamp in sell_signals:
                        exit_reason = 'SIGNAL'
                
                else:  # SHORT
                    pnl_pct = (entry_price - current_price) / entry_price
                    
                    # Check trailing stop first (if enabled and activated)
                    if self.use_trailing_stop:
                        profit_from_entry = (entry_price - lowest_price) / entry_price
                        if profit_from_entry >= self.trailing_activation_pct:
                            trailing_stop_price = lowest_price * (1 + self.trailing_stop_pct)
                            if high_price >= trailing_stop_price:
                                exit_reason = 'TRAILING_STOP'
                                exit_price_override = trailing_stop_price
                    
                    if not exit_reason and pnl_pct <= -self.stop_loss_pct:
                        exit_reason = 'STOP_LOSS'
                    elif not exit_reason and pnl_pct >= self.take_profit_pct:
                        exit_reason = 'TAKE_PROFIT'
                    elif not exit_reason and timestamp in buy_signals:
                        exit_reason = 'SIGNAL'
                
                # Execute exit
                if exit_reason:
                    # Use trailing stop price if available, otherwise apply slippage
                    if exit_price_override:
                        exit_price = exit_price_override
                    else:
                        exit_price = current_price * (1 - self.slippage_pct if position_direction == 'LONG' else 1 + self.slippage_pct)
                    
                    # Calculate final PnL
                    if position_direction == 'LONG':
                        final_pnl_pct = (exit_price - entry_price) / entry_price
                    else:
                        final_pnl_pct = (entry_price - exit_price) / entry_price
                    
                    # Subtract commission
                    final_pnl_pct -= self.commission_pct * 2  # Entry + exit
                    
                    pnl = position_size * final_pnl_pct
                    capital += pnl
                    
                    # Get current indicators
                    exit_indicators = {
                        'cci': row.get('cci', 0),
                        'rsi': row.get('rsi', 0),
                        'macd': row.get('macd', 0),
                        'gold_line': row.get('gold_line', 0),
                        'slow_ema': row.get('slow_ema', 0)
                    }
                    
                    # Calculate duration
                    duration = i - df.index.get_loc(entry_time) if entry_time in df.index else 0
                    
                    # Record trade
                    trade = Trade(
                        entry_time=entry_time,
                        exit_time=timestamp,
                        direction=position_direction,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        position_size=position_size,
                        pnl=pnl,
                        pnl_pct=final_pnl_pct * 100,
                        duration_candles=duration,
                        exit_reason=exit_reason,
                        entry_signal=asdict(entry_signal) if entry_signal else {},
                        indicators_at_entry=entry_indicators,
                        indicators_at_exit=exit_indicators,
                        market_context={
                            'volatility': row.get('atr', 0) / current_price * 100,
                            'trend_strength': abs(row.get('macd_hist', 0)),
                            'price_vs_gold_line': 'above' if current_price > row.get('gold_line', current_price) else 'below'
                        }
                    )
                    trades.append(trade)
                    
                    in_position = False
                    position_direction = None
            
            # Check for entry signals if not in position
            if not in_position:
                if timestamp in buy_signals:
                    signal = buy_signals[timestamp]
                    
                    # Enter long
                    in_position = True
                    position_direction = 'LONG'
                    entry_price = current_price * (1 + self.slippage_pct)  # Slippage
                    entry_time = timestamp
                    entry_signal = signal
                    position_size = capital * self.position_size_pct
                    highest_price = current_price  # Reset for trailing stop
                    lowest_price = float('inf')
                    entry_indicators = {
                        'cci': row.get('cci', 0),
                        'rsi': row.get('rsi', 0),
                        'macd': row.get('macd', 0),
                        'gold_line': row.get('gold_line', 0),
                        'slow_ema': row.get('slow_ema', 0)
                    }
                
                elif timestamp in sell_signals:
                    signal = sell_signals[timestamp]
                    
                    # Enter short
                    in_position = True
                    position_direction = 'SHORT'
                    entry_price = current_price * (1 - self.slippage_pct)
                    entry_time = timestamp
                    entry_signal = signal
                    position_size = capital * self.position_size_pct
                    highest_price = 0  # Reset for trailing stop
                    lowest_price = current_price
                    entry_indicators = {
                        'cci': row.get('cci', 0),
                        'rsi': row.get('rsi', 0),
                        'macd': row.get('macd', 0),
                        'gold_line': row.get('gold_line', 0),
                        'slow_ema': row.get('slow_ema', 0)
                    }
            
            equity_curve.append(capital)
        
        # Close any open position at end
        if in_position:
            row = df.iloc[-1]
            current_price = row['close']
            exit_price = current_price
            
            if position_direction == 'LONG':
                final_pnl_pct = (exit_price - entry_price) / entry_price
            else:
                final_pnl_pct = (entry_price - exit_price) / entry_price
            
            final_pnl_pct -= self.commission_pct * 2
            pnl = position_size * final_pnl_pct
            capital += pnl
            
            trade = Trade(
                entry_time=entry_time,
                exit_time=df.index[-1],
                direction=position_direction,
                entry_price=entry_price,
                exit_price=exit_price,
                position_size=position_size,
                pnl=pnl,
                pnl_pct=final_pnl_pct * 100,
                duration_candles=len(df) - df.index.get_loc(entry_time),
                exit_reason='END',
                entry_signal=asdict(entry_signal) if entry_signal else {},
                indicators_at_entry=entry_indicators,
                indicators_at_exit={},
                market_context={}
            )
            trades.append(trade)
        
        # Calculate metrics
        return self._calculate_metrics(
            trades=trades,
            equity_curve=pd.Series(equity_curve),
            symbol=symbol,
            timeframe=timeframe,
            start_date=df.index[0],
            end_date=df.index[-1]
        )
    
    def _calculate_metrics(
        self,
        trades: List[Trade],
        equity_curve: pd.Series,
        symbol: str,
        timeframe: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp
    ) -> BacktestResult:
        """Calculate performance metrics from trades"""
        
        if not trades:
            return BacktestResult(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                initial_capital=self.initial_capital,
                final_capital=self.initial_capital,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0,
                total_pnl=0,
                total_pnl_pct=0,
                max_drawdown=0,
                max_drawdown_pct=0,
                sharpe_ratio=0,
                profit_factor=0,
                avg_trade_duration=0,
                avg_win=0,
                avg_loss=0,
                largest_win=0,
                largest_loss=0,
                trades=[],
                equity_curve=equity_curve
            )
        
        # Basic stats
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]
        
        total_wins = sum(t.pnl for t in winning_trades)
        total_losses = abs(sum(t.pnl for t in losing_trades))
        
        # Drawdown
        peak = equity_curve.expanding().max()
        drawdown = equity_curve - peak
        max_drawdown = drawdown.min()
        max_drawdown_pct = (max_drawdown / peak[drawdown.idxmin()]) * 100 if peak[drawdown.idxmin()] > 0 else 0
        
        # Sharpe Ratio (annualized, assuming 15min candles)
        returns = equity_curve.pct_change().dropna()
        if len(returns) > 1 and returns.std() > 0:
            # Annualize: 15 min = 35040 periods/year (24*4*365)
            periods_per_year = 35040 if timeframe == '15' else 525600 / int(timeframe)
            sharpe = (returns.mean() / returns.std()) * np.sqrt(periods_per_year)
        else:
            sharpe = 0
        
        return BacktestResult(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_capital=equity_curve.iloc[-1],
            total_trades=len(trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=len(winning_trades) / len(trades) * 100,
            total_pnl=sum(t.pnl for t in trades),
            total_pnl_pct=(equity_curve.iloc[-1] - self.initial_capital) / self.initial_capital * 100,
            max_drawdown=max_drawdown,
            max_drawdown_pct=abs(max_drawdown_pct),
            sharpe_ratio=sharpe,
            profit_factor=total_wins / total_losses if total_losses > 0 else float('inf'),
            avg_trade_duration=np.mean([t.duration_candles for t in trades]),
            avg_win=np.mean([t.pnl for t in winning_trades]) if winning_trades else 0,
            avg_loss=np.mean([t.pnl for t in losing_trades]) if losing_trades else 0,
            largest_win=max([t.pnl for t in trades]) if trades else 0,
            largest_loss=min([t.pnl for t in trades]) if trades else 0,
            trades=trades,
            equity_curve=equity_curve
        )
    
    def save_results(self, result: BacktestResult, output_dir: str = 'results'):
        """Save backtest results to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save summary
        summary = {
            'symbol': result.symbol,
            'timeframe': result.timeframe,
            'start_date': str(result.start_date),
            'end_date': str(result.end_date),
            'initial_capital': result.initial_capital,
            'final_capital': result.final_capital,
            'total_trades': result.total_trades,
            'winning_trades': result.winning_trades,
            'losing_trades': result.losing_trades,
            'win_rate': result.win_rate,
            'total_pnl': result.total_pnl,
            'total_pnl_pct': result.total_pnl_pct,
            'max_drawdown': result.max_drawdown,
            'max_drawdown_pct': result.max_drawdown_pct,
            'sharpe_ratio': result.sharpe_ratio,
            'profit_factor': result.profit_factor,
            'avg_trade_duration': result.avg_trade_duration,
            'avg_win': result.avg_win,
            'avg_loss': result.avg_loss,
            'largest_win': result.largest_win,
            'largest_loss': result.largest_loss
        }
        
        with open(output_path / f'summary_{timestamp}.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save trades for LLM analysis
        trades_data = []
        for trade in result.trades:
            trade_dict = {
                'entry_time': str(trade.entry_time),
                'exit_time': str(trade.exit_time),
                'direction': trade.direction,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'pnl': trade.pnl,
                'pnl_pct': trade.pnl_pct,
                'duration_candles': trade.duration_candles,
                'exit_reason': trade.exit_reason,
                'indicators_at_entry': trade.indicators_at_entry,
                'indicators_at_exit': trade.indicators_at_exit,
                'market_context': trade.market_context,
                'result': 'WIN' if trade.pnl > 0 else 'LOSS'
            }
            trades_data.append(trade_dict)
        
        with open(output_path / f'trades_{timestamp}.json', 'w') as f:
            json.dump(trades_data, f, indent=2)
        
        # Save equity curve
        result.equity_curve.to_csv(output_path / f'equity_{timestamp}.csv')
        
        print(f"Results saved to {output_path}")
        
        return output_path / f'trades_{timestamp}.json'
