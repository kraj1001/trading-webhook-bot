"""
Main Trading Bot
Runs strategies on schedule and manages positions
"""
import time
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, List

from .config import STRATEGIES, TRADING_MODE, INITIAL_CAPITAL, TELEGRAM_ALERTS
from .exchange import get_exchange, PaperExchange
from .database import TradeDatabase, TradeRecord
from .strategies import ScalpingHybridStrategy, LLMv4LowDDStrategy, LLMv3TightStrategy
from .llm_analysis import LLMAnalyzer
from .telegram_bot import get_telegram_bot

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class TradingBot:
    """Main trading bot orchestrator"""
    
    def __init__(self):
        try:
            self.exchange = get_exchange()
            logger.info("Successfully connected to exchange")
        except Exception as e:
            logger.error(f"Failed to connect to exchange: {e}")
            logger.warning("⚠️  Bot running in DASHBOARD-ONLY mode (No trading)")
            self.exchange = None
            
        self.db = TradeDatabase()
        self.llm = LLMAnalyzer()
        self.telegram = get_telegram_bot(self.db)
        self.strategies: Dict[str, object] = {}
        self.open_trades: Dict[str, int] = {}  # strategy -> trade_id
        
        # Initialize strategies
        if self.exchange:
            for config in STRATEGIES:
                if not config.enabled:
                    continue
                
                if "ScalpingHybrid" in config.name:
                    strategy = ScalpingHybridStrategy(config.symbol, config.timeframe)
                elif "v4" in config.name:
                    strategy = LLMv4LowDDStrategy(config.symbol, config.timeframe)
                elif "v3" in config.name:
                    strategy = LLMv3TightStrategy(config.symbol, config.timeframe)
                else:
                    continue
                
                self.strategies[config.name] = {
                    "strategy": strategy,
                    "config": config,
                    "capital": INITIAL_CAPITAL * (config.allocation_pct / 100)
                }
                logger.info(f"Initialized {config.name} on {config.symbol} {config.timeframe}")
    
    def fetch_candles(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Fetch and prepare candle data"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=100)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            return df
        except Exception as e:
            logger.error(f"Error fetching candles for {symbol}: {e}")
            return pd.DataFrame()

    def run_strategy(self, name: str, data: dict):
        """Execute a single strategy"""
        strategy = data['strategy']
        config = data['config']
        
        # 1. Fetch Data
        df = self.fetch_candles(config.symbol, config.timeframe)
        if df.empty:
            return

        # 2. Analyze
        if not df.empty:
            df = strategy.calculate_indicators(df)
            signal = strategy.check_entry(df)
        else:
            signal = None
        
        # 3. Log Status (for Dashboard)
        current_price = float(df['close'].iloc[-1]) if not df.empty else 0.0
        
        # Read indicators from the calculated DataFrame (convert to float for PostgreSQL)
        rsi = float(df['rsi'].iloc[-1]) if 'rsi' in df.columns else 0.0
        adx = float(df['adx'].iloc[-1]) if 'adx' in df.columns else 0.0
        
        # Determine status and message
        if signal:
            status = signal.action.upper()
            message = signal.reason
        else:
            status = "WAITING"
            message = f"RSI: {rsi:.1f} | ADX: {adx:.1f}"
        
        self.db.log_strategy_check(
            strategy=name,
            symbol=config.symbol,
            price=current_price,
            status=status,
            rsi=rsi,
            adx=adx,
            message=message
        )
        
        # 3.5 LLM Training Data Generation (Every 4 Hours)
        # Only run at 4-hour intervals to avoid Azure OpenAI rate limits
        current_minute = datetime.now().minute
        current_hour = datetime.now().hour
        
        # Run only at 00:XX, 04:XX, 08:XX, 12:XX, 16:XX, 20:XX (every 4 hours)
        if current_hour % 4 == 0 and current_minute < 5 and self.llm.enabled:
            try:
                # Get additional indicators for analysis
                ema_9 = float(df['ema_9'].iloc[-1]) if 'ema_9' in df.columns else 0.0
                ema_15 = float(df['ema_15'].iloc[-1]) if 'ema_15' in df.columns else 0.0
                ema_30 = float(df['ema_30'].iloc[-1]) if 'ema_30' in df.columns else 0.0
                macd = float(df['macd'].iloc[-1]) if 'macd' in df.columns else 0.0
                macd_signal = float(df['macd_signal'].iloc[-1]) if 'macd_signal' in df.columns else 0.0
                atr = float(df['atr'].iloc[-1]) if 'atr' in df.columns else 0.0
                
                analysis_data = {
                    'symbol': config.symbol,
                    'strategy': name,
                    'timeframe': config.timeframe,
                    'price': current_price,
                    'rsi': rsi,
                    'adx': adx,
                    'ema_9': ema_9,
                    'ema_15': ema_15,
                    'ema_30': ema_30,
                    'macd': macd,
                    'macd_signal': macd_signal,
                    'atr': atr,
                    'signal': signal.action if signal else None,
                    # For market regime detection
                    'high_10': float(df['high'].tail(10).max()) if not df.empty else 0.0,
                    'low_10': float(df['low'].tail(10).min()) if not df.empty else 0.0,
                    'price_change_pct': float((df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10] * 100) if len(df) >= 10 else 0.0,
                }
                
                # --- Entry/No-Signal Analysis (Every Hour) ---
                if signal:
                    logger.info(f"🤖 LLM Training: Analyzing ENTRY signal for {config.symbol}...")
                    result = self.llm.analyze_entry_signal(analysis_data)
                    analysis_type = "entry_signal"
                else:
                    logger.info(f"🤖 LLM Training: Analyzing NO-SIGNAL for {config.symbol}...")
                    result = self.llm.analyze_no_signal(analysis_data)
                    analysis_type = "no_signal"
                
                # Save to database
                self.db.save_training_data(
                    symbol=config.symbol,
                    strategy=name,
                    analysis_type=analysis_type,
                    signal_generated=bool(signal),
                    price=current_price,
                    rsi=rsi,
                    adx=adx,
                    ema_15=ema_15,
                    ema_30=ema_30,
                    macd=macd,
                    instruction=result.get('instruction', ''),
                    output=result.get('output', ''),
                    confidence=result.get('confidence', 'medium')
                )
                
                # --- Market Regime Detection (Every 4 Hours) ---
                if current_hour % 4 == 0:
                    logger.info(f"🤖 LLM Training: Analyzing MARKET REGIME for {config.symbol}...")
                    regime_result = self.llm.analyze_market_regime(analysis_data)
                    self.db.save_training_data(
                        symbol=config.symbol,
                        strategy=name,
                        analysis_type="market_regime",
                        signal_generated=False,
                        price=current_price,
                        rsi=rsi,
                        adx=adx,
                        ema_15=ema_15,
                        ema_30=ema_30,
                        macd=macd,
                        instruction=regime_result.get('instruction', ''),
                        output=regime_result.get('output', ''),
                        confidence=str(regime_result.get('confidence', 0.5))
                    )
                
                # --- Candlestick Pattern Recognition (Every 4 Hours) ---
                if current_hour % 4 == 0:
                    logger.info(f"🤖 LLM Training: Analyzing CANDLESTICK PATTERNS for {config.symbol}...")
                    candle_data = {
                        'symbol': config.symbol,
                        'timeframe': config.timeframe,
                        'candles': [
                            {'open': float(row['open']), 'high': float(row['high']), 
                             'low': float(row['low']), 'close': float(row['close'])}
                            for _, row in df.tail(10).iterrows()
                        ],
                        'rsi': rsi,
                        'near_support': rsi < 35,
                        'near_resistance': rsi > 65,
                        'volume_trend': 'high' if df['volume'].iloc[-1] > df['volume'].mean() else 'normal'
                    }
                    pattern_result = self.llm.analyze_candlestick_patterns(candle_data)
                    self.db.save_training_data(
                        symbol=config.symbol,
                        strategy=name,
                        analysis_type="candlestick_pattern",
                        signal_generated=False,
                        price=current_price,
                        rsi=rsi,
                        adx=adx,
                        ema_15=ema_15,
                        ema_30=ema_30,
                        macd=macd,
                        instruction=pattern_result.get('instruction', ''),
                        output=pattern_result.get('output', ''),
                        confidence=pattern_result.get('pattern_quality', 'medium')
                    )
                    
            except Exception as e:
                logger.error(f"LLM Training data generation failed: {e}")
        
        # 4. Execute Trades
        if signal and signal.action == "buy":
            self._open_trade(name, data['strategy'], config.symbol, "long", current_price, data['capital'])
        elif signal and signal.action == "sell":
            self._open_trade(name, data['strategy'], config.symbol, "short", current_price, data['capital'])
        elif signal and signal.action == "exit":
            if name in self.open_trades:
                self._close_trade(name, current_price, "Signal Exit")
                
    def _open_trade(self, strategy_name: str, strategy_obj, symbol: str, side: str, price: float, capital: float):
        """Open a new trade"""
        if strategy_name in self.open_trades:
            return  # Already has open trade
            
        qty = (capital * 0.98) / price  # 98% of allocation to account for fees
        
        # Map long/short to buy/sell for exchange API
        order_side = "buy" if side == "long" else "sell"
        
        try:
            order = self.exchange.create_order(symbol, order_side, qty)
            
            # Create database record
            record = TradeRecord(
                strategy=strategy_name,
                symbol=symbol,
                side=side,
                entry_time=datetime.now(),
                entry_price=order.price,
                quantity=order.quantity,
                is_paper=(TRADING_MODE == "paper"),
                is_open=True
            )
            
            trade_id = self.db.save_trade(record)
            self.open_trades[strategy_name] = trade_id
            
            # NOW set the strategy's position (only after successful order)
            strategy_obj.position = side
            strategy_obj.entry_price = order.price
            
            logger.info(f"✅ OPENED {side.upper()} | {strategy_name} | {symbol} @ {order.price}")
            
            # Send Telegram notification
            if TELEGRAM_ALERTS:
                self.telegram.notify_trade_opened(strategy_name, symbol, side, order.price)
        except Exception as e:
            logger.error(f"Failed to open trade: {e}")

    def _close_trade(self, strategy_name: str, exit_price: float, exit_reason: str):
        """Close an open trade"""
        trade_id = self.open_trades.pop(strategy_name)
        
        # Get the trade
        trades = self.db.get_open_trades(strategy_name)
        trade = next((t for t in trades if t.id == trade_id), None)
        
        if trade:
            # Calculate P&L
            if trade.side == "long":
                pnl_pct = (exit_price - trade.entry_price) / trade.entry_price * 100
            else:
                pnl_pct = (trade.entry_price - exit_price) / trade.entry_price * 100
            
            pnl_usd = trade.quantity * trade.entry_price * (pnl_pct / 100)
            
            self.db.close_trade(trade_id, exit_price, exit_reason, pnl_usd, pnl_pct)
            
            emoji = "✅" if pnl_pct > 0 else "❌"
            logger.info(f"{emoji} CLOSED | {strategy_name} | {exit_reason} | PnL: {pnl_pct:+.2f}% (${pnl_usd:+.2f})")
            
            # Send Telegram notification
            if TELEGRAM_ALERTS:
                self.telegram.notify_trade_closed(strategy_name, trade.symbol, pnl_pct, pnl_usd, exit_reason)
            
            # Trigger LLM Analysis
            if self.llm.enabled:
                logger.info(f"🤖 Analyzing trade #{trade_id} with GPT-4o...")
                try:
                    analysis = self.llm.analyze_trade({
                        "strategy": strategy_name,
                        "symbol": trade.symbol,
                        "side": trade.side,
                        "entry_time": trade.entry_time,
                        "exit_time": datetime.now(),
                        "entry_price": trade.entry_price,
                        "exit_price": exit_price,
                        "pnl_usd": round(pnl_usd, 2),
                        "pnl_pct": round(pnl_pct, 2),
                        "exit_reason": exit_reason,
                        "entry_rsi": trade.entry_rsi,
                        "entry_adx": trade.entry_adx,
                        "entry_macd": trade.entry_macd
                    })
                    self.db.update_trade_analysis(trade_id, analysis)
                    
                    # Also run enhanced post-trade review for training data (#5)
                    logger.info(f"🤖 Enhanced Post-Trade Review for #{trade_id}...")
                    trade_duration = datetime.now() - trade.entry_time if trade.entry_time else "Unknown"
                    enhanced_result = self.llm.enhanced_post_trade_review({
                        "trade_id": trade_id,
                        "strategy": strategy_name,
                        "symbol": trade.symbol,
                        "side": trade.side,
                        "entry_price": trade.entry_price,
                        "exit_price": exit_price,
                        "pnl_pct": round(pnl_pct, 2),
                        "pnl_usd": round(pnl_usd, 2),
                        "duration": str(trade_duration),
                        "exit_reason": exit_reason,
                        "entry_rsi": trade.entry_rsi,
                        "entry_adx": trade.entry_adx,
                        "entry_macd": trade.entry_macd,
                        "ema_alignment": "bullish" if trade.entry_ema15 and trade.entry_ema30 and trade.entry_ema15 > trade.entry_ema30 else "bearish"
                    })
                    
                    # Save training data
                    self.db.save_training_data(
                        symbol=trade.symbol,
                        strategy=strategy_name,
                        analysis_type="post_trade_review",
                        signal_generated=True,
                        price=exit_price,
                        rsi=trade.entry_rsi or 0,
                        adx=trade.entry_adx or 0,
                        ema_15=0,
                        ema_30=0,
                        macd=trade.entry_macd or 0,
                        instruction=enhanced_result.get('instruction', ''),
                        output=enhanced_result.get('output', ''),
                        confidence=str(enhanced_result.get('entry_quality', 5))
                    )
                except Exception as e:
                    logger.error(f"Analysis failed: {e}")

    
    def run_once(self):
        """Run all strategies once"""
        logger.info("=" * 60)
        logger.info(f"Running strategies... ({TRADING_MODE.upper()} mode)")
        
        for name, data in self.strategies.items():
            try:
                self.run_strategy(name, data)
            except Exception as e:
                logger.error(f"Error in {name}: {e}")
        
        logger.info("=" * 60)
    
    def run_loop(self, interval_seconds: int = 60):
        """Run bot in continuous loop"""
        logger.info(f"Starting trading bot in {TRADING_MODE.upper()} mode")
        logger.info(f"Check interval: {interval_seconds}s")
        
        while True:
            try:
                self.run_once()
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
            
            time.sleep(interval_seconds)


def main():
    """Entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Trading Bot")
    parser.add_argument("--mode", choices=["paper", "live", "testnet"], default="paper")
    parser.add_argument("--interval", type=int, default=3600, help="Check interval in seconds")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    args = parser.parse_args()
    
    # Override mode if specified
    import os
    os.environ["TRADING_MODE"] = args.mode
    
    bot = TradingBot()
    
    if args.once:
        bot.run_once()
    else:
        bot.run_loop(args.interval)


if __name__ == "__main__":
    main()
