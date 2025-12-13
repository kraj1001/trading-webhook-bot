#!/usr/bin/env python3
"""
Paper Trading Bot
Connects to Bybit Testnet for simulated trading with real market data.
Uses the winning MACD 8/17/9 strategy on 12-hour timeframe.
"""

import os
import time
import json
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('paper_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Current position state"""
    symbol: str
    direction: Optional[str] = None  # 'LONG', 'SHORT', or None
    entry_price: float = 0.0
    entry_time: Optional[datetime] = None
    quantity: float = 0.0
    unrealized_pnl: float = 0.0


@dataclass
class Trade:
    """Completed trade record"""
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    entry_time: str
    exit_time: str
    quantity: float
    pnl: float
    pnl_pct: float


class MACDStrategy:
    """MACD 8/17/9 Crossover Strategy - Our Best Performer"""
    
    def __init__(self, fast=8, slow=17, signal=9):
        self.fast = fast
        self.slow = slow
        self.signal_len = signal
    
    def calculate(self, df: pd.DataFrame) -> dict:
        """Calculate MACD indicators"""
        close = df['close']
        
        exp1 = close.ewm(span=self.fast, adjust=False).mean()
        exp2 = close.ewm(span=self.slow, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=self.signal_len, adjust=False).mean()
        histogram = macd - signal
        
        return {
            'macd': macd.iloc[-1],
            'signal': signal.iloc[-1],
            'histogram': histogram.iloc[-1],
            'prev_macd': macd.iloc[-2],
            'prev_signal': signal.iloc[-2]
        }
    
    def get_signal(self, indicators: dict) -> Optional[str]:
        """Get trading signal based on MACD crossover"""
        macd = indicators['macd']
        signal = indicators['signal']
        prev_macd = indicators['prev_macd']
        prev_signal = indicators['prev_signal']
        
        # Bullish crossover
        if prev_macd < prev_signal and macd >= signal:
            return 'BUY'
        
        # Bearish crossover
        if prev_macd > prev_signal and macd <= signal:
            return 'SELL'
        
        return None


class PaperTradingBot:
    """
    Paper Trading Bot using Bybit Testnet
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        
        # Trading settings
        self.symbol = self.config.get('symbol', 'BTCUSDT')
        self.timeframe = self.config.get('timeframe', '720')  # 12 hours
        self.initial_capital = self.config.get('initial_capital', 10000.0)
        self.position_size_pct = self.config.get('position_size_pct', 5.0)
        self.stop_loss_pct = self.config.get('stop_loss_pct', 1.0)
        self.take_profit_pct = self.config.get('take_profit_pct', 10.0)
        
        # State
        self.capital = self.initial_capital
        self.position = Position(symbol=self.symbol)
        self.trades: list[Trade] = []
        self.is_running = False
        
        # Strategy
        self.strategy = MACDStrategy(8, 17, 9)
        
        # Data connector
        from data.bybit_connector import BybitConnector
        self.connector = BybitConnector(cache_dir='data/cache')
        
        # Load previous state if exists
        self.state_file = Path('paper_trading_state.json')
        self.load_state()
        
        logger.info(f"Paper Trading Bot initialized")
        logger.info(f"Symbol: {self.symbol} | Timeframe: {self.timeframe}min")
        logger.info(f"Capital: ${self.capital:,.2f} | Position Size: {self.position_size_pct}%")
    
    def load_state(self):
        """Load previous state from file"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                self.capital = state.get('capital', self.initial_capital)
                self.trades = [Trade(**t) for t in state.get('trades', [])]
                
                pos = state.get('position', {})
                if pos.get('direction'):
                    self.position = Position(
                        symbol=self.symbol,
                        direction=pos['direction'],
                        entry_price=pos['entry_price'],
                        entry_time=datetime.fromisoformat(pos['entry_time']),
                        quantity=pos['quantity']
                    )
                
                logger.info(f"Loaded state: ${self.capital:,.2f}, {len(self.trades)} trades")
            except Exception as e:
                logger.warning(f"Could not load state: {e}")
    
    def save_state(self):
        """Save current state to file"""
        state = {
            'capital': self.capital,
            'trades': [asdict(t) for t in self.trades],
            'position': {
                'direction': self.position.direction,
                'entry_price': self.position.entry_price,
                'entry_time': self.position.entry_time.isoformat() if self.position.entry_time else None,
                'quantity': self.position.quantity
            },
            'last_update': datetime.now().isoformat()
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def get_current_price(self) -> float:
        """Get current price from Bybit"""
        df = self.connector.get_historical_data(
            symbol=self.symbol,
            interval='1',  # 1 minute for current price
            days=1,
            use_cache=False
        )
        return df['close'].iloc[-1] if not df.empty else 0
    
    def get_candles(self, lookback=50) -> pd.DataFrame:
        """Get recent candles for analysis"""
        df = self.connector.get_historical_data(
            symbol=self.symbol,
            interval=self.timeframe,
            days=int(lookback * int(self.timeframe) / 1440) + 10,
            use_cache=False
        )
        return df.tail(lookback) if not df.empty else pd.DataFrame()
    
    def calculate_position_size(self, price: float) -> float:
        """Calculate position size based on capital"""
        trade_value = self.capital * (self.position_size_pct / 100)
        return trade_value / price
    
    def open_position(self, direction: str, price: float):
        """Open a new position"""
        quantity = self.calculate_position_size(price)
        
        self.position = Position(
            symbol=self.symbol,
            direction=direction,
            entry_price=price,
            entry_time=datetime.now(),
            quantity=quantity
        )
        
        logger.info(f"📈 OPENED {direction} @ ${price:,.2f} | Qty: {quantity:.6f}")
        self.save_state()
    
    def close_position(self, price: float, reason: str = "Signal"):
        """Close current position"""
        if not self.position.direction:
            return
        
        # Calculate PnL
        if self.position.direction == 'LONG':
            pnl = (price - self.position.entry_price) * self.position.quantity
            pnl_pct = ((price / self.position.entry_price) - 1) * 100
        else:
            pnl = (self.position.entry_price - price) * self.position.quantity
            pnl_pct = ((self.position.entry_price / price) - 1) * 100
        
        # Record trade
        trade = Trade(
            symbol=self.symbol,
            direction=self.position.direction,
            entry_price=self.position.entry_price,
            exit_price=price,
            entry_time=self.position.entry_time.isoformat(),
            exit_time=datetime.now().isoformat(),
            quantity=self.position.quantity,
            pnl=pnl,
            pnl_pct=pnl_pct
        )
        self.trades.append(trade)
        
        # Update capital
        self.capital += pnl
        
        emoji = "✅" if pnl > 0 else "❌"
        logger.info(f"{emoji} CLOSED {self.position.direction} @ ${price:,.2f} | PnL: ${pnl:+,.2f} ({pnl_pct:+.2f}%) | Reason: {reason}")
        
        # Reset position
        self.position = Position(symbol=self.symbol)
        self.save_state()
    
    def check_stop_loss_take_profit(self, current_price: float) -> Optional[str]:
        """Check if stop loss or take profit is hit"""
        if not self.position.direction:
            return None
        
        if self.position.direction == 'LONG':
            pnl_pct = ((current_price / self.position.entry_price) - 1) * 100
        else:
            pnl_pct = ((self.position.entry_price / current_price) - 1) * 100
        
        if pnl_pct <= -self.stop_loss_pct:
            return 'STOP_LOSS'
        if pnl_pct >= self.take_profit_pct:
            return 'TAKE_PROFIT'
        
        return None
    
    def process_signal(self, signal: str, current_price: float):
        """Process trading signal"""
        if signal == 'BUY':
            if self.position.direction == 'SHORT':
                self.close_position(current_price, "MACD Bullish Crossover")
            if self.position.direction != 'LONG':
                self.open_position('LONG', current_price)
        
        elif signal == 'SELL':
            if self.position.direction == 'LONG':
                self.close_position(current_price, "MACD Bearish Crossover")
            if self.position.direction != 'SHORT':
                self.open_position('SHORT', current_price)
    
    def run_once(self):
        """Run one iteration of the trading loop"""
        try:
            # Get current data
            df = self.get_candles(50)
            if df.empty:
                logger.warning("No data available")
                return
            
            current_price = df['close'].iloc[-1]
            
            # Check stop loss / take profit
            sl_tp = self.check_stop_loss_take_profit(current_price)
            if sl_tp:
                self.close_position(current_price, sl_tp)
                return
            
            # Calculate indicators
            indicators = self.strategy.calculate(df)
            
            # Get signal
            signal = self.strategy.get_signal(indicators)
            
            # Log status
            pos_str = f"{self.position.direction} @ ${self.position.entry_price:,.2f}" if self.position.direction else "FLAT"
            logger.info(f"💹 {self.symbol}: ${current_price:,.2f} | MACD: {indicators['macd']:.2f} | Signal: {indicators['signal']:.2f} | Position: {pos_str}")
            
            # Process signal
            if signal:
                logger.info(f"🔔 Signal: {signal}")
                self.process_signal(signal, current_price)
            
        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
            import traceback
            traceback.print_exc()
    
    def get_stats(self) -> dict:
        """Get trading statistics"""
        if not self.trades:
            return {'message': 'No trades yet'}
        
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]
        
        total_pnl = sum(t.pnl for t in self.trades)
        win_rate = len(wins) / len(self.trades) * 100 if self.trades else 0
        
        return {
            'total_trades': len(self.trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'current_capital': self.capital,
            'roi': ((self.capital / self.initial_capital) - 1) * 100
        }
    
    def print_stats(self):
        """Print trading statistics"""
        stats = self.get_stats()
        
        print("\n" + "=" * 60)
        print("📊 PAPER TRADING STATISTICS")
        print("=" * 60)
        print(f"Initial Capital:  ${self.initial_capital:,.2f}")
        print(f"Current Capital:  ${self.capital:,.2f}")
        print(f"Total PnL:        ${stats.get('total_pnl', 0):+,.2f}")
        print(f"ROI:              {stats.get('roi', 0):+.2f}%")
        print("-" * 60)
        print(f"Total Trades:     {stats.get('total_trades', 0)}")
        print(f"Wins:             {stats.get('wins', 0)}")
        print(f"Losses:           {stats.get('losses', 0)}")
        print(f"Win Rate:         {stats.get('win_rate', 0):.1f}%")
        print("=" * 60)
        
        if self.position.direction:
            current_price = self.get_current_price()
            if self.position.direction == 'LONG':
                unrealized = ((current_price / self.position.entry_price) - 1) * 100
            else:
                unrealized = ((self.position.entry_price / current_price) - 1) * 100
            print(f"\n📍 Current Position: {self.position.direction}")
            print(f"   Entry: ${self.position.entry_price:,.2f}")
            print(f"   Current: ${current_price:,.2f}")
            print(f"   Unrealized PnL: {unrealized:+.2f}%")
    
    def run(self, interval_seconds: int = 300):
        """Run continuous paper trading"""
        self.is_running = True
        
        logger.info("=" * 60)
        logger.info("🚀 Starting Paper Trading Bot")
        logger.info(f"Strategy: MACD 8/17/9")
        logger.info(f"Timeframe: {self.timeframe} minutes")
        logger.info(f"Check interval: {interval_seconds} seconds")
        logger.info("=" * 60)
        
        try:
            while self.is_running:
                self.run_once()
                time.sleep(interval_seconds)
        except KeyboardInterrupt:
            logger.info("\n🛑 Paper Trading Bot stopped by user")
            self.print_stats()
        finally:
            self.save_state()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Paper Trading Bot')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading pair')
    parser.add_argument('--timeframe', type=str, default='720', help='Timeframe in minutes')
    parser.add_argument('--capital', type=float, default=10000, help='Initial capital')
    parser.add_argument('--interval', type=int, default=300, help='Check interval in seconds')
    parser.add_argument('--once', action='store_true', help='Run once and exit')
    parser.add_argument('--stats', action='store_true', help='Show stats and exit')
    
    args = parser.parse_args()
    
    config = {
        'symbol': args.symbol,
        'timeframe': args.timeframe,
        'initial_capital': args.capital,
        'position_size_pct': 5.0,
        'stop_loss_pct': 1.0,
        'take_profit_pct': 10.0
    }
    
    bot = PaperTradingBot(config)
    
    if args.stats:
        bot.print_stats()
    elif args.once:
        bot.run_once()
        bot.print_stats()
    else:
        bot.run(args.interval)


if __name__ == '__main__':
    main()
