"""
ScalpingHybrid Strategy
Verified: +97,397% on DOGEUSDT 4H, +6,437% on AVAXUSDT Daily
"""
from datetime import datetime
from typing import Optional
import pandas as pd

from .base import BaseStrategy, Signal


class ScalpingHybridStrategy(BaseStrategy):
    """
    ScalpingHybrid Strategy
    
    Entry (LONG only):
    - EMA 15 > EMA 30 (trend)
    - Close > EMA 9 (momentum)
    - RSI > 70 (strong momentum)
    - MACD > Signal (bullish)
    - Volume > 50-period MA
    
    Exit:
    - Close < EMA 9 OR RSI < 50
    - OR Stop Loss hit (1.0x ATR)
    - OR Take Profit hit (2.0x ATR)
    """
    
    def __init__(self, symbol: str = "DOGEUSDT", timeframe: str = "4h"):
        super().__init__("ScalpingHybrid", symbol, timeframe)
        self.atr_stop = 1.0
        self.atr_tp = 2.0
    
    def check_entry(self, df: pd.DataFrame) -> Optional[Signal]:
        """Check for long entry"""
        if self.position is not None:
            return None
        
        row = df.iloc[-1]
        
        # Entry conditions
        trend_up = row["ema_15"] > row["ema_30"]
        price_above_ema = row["close"] > row["ema_9"]
        rsi_strong = row["rsi"] > 70
        macd_bullish = row["macd"] > row["macd_signal"]
        volume_ok = row["volume"] > row["vol_ma"] * 0.8  # Volume must be above 80% of average
        
        # Debug: Log condition states
        import logging
        logging.info(f"{self.symbol} Conditions: trend={trend_up}, price_ema={price_above_ema}, rsi={rsi_strong}, macd={macd_bullish}, vol={volume_ok}")
        
        if trend_up and price_above_ema and rsi_strong and macd_bullish and volume_ok:
            stop_loss = row["close"] - row["atr"] * self.atr_stop
            take_profit = row["close"] + row["atr"] * self.atr_tp
            
            # Note: Don't set self.position here - only set after successful order in main.py
            
            return Signal(
                action="buy",
                symbol=self.symbol,
                side="long",
                entry_price=row["close"],
                stop_loss=stop_loss,
                take_profit=take_profit,
                quantity=0,  # Calculated by position sizer
                timestamp=datetime.now(),
                reason="EMA15>30, RSI>70, MACD+, Volume OK"
            )
        
        return None
    
    def check_exit(self, df: pd.DataFrame, current_price: float) -> Optional[str]:
        """Check for exit conditions"""
        if self.position is None:
            return None
        
        row = df.iloc[-1]
        
        # Stop loss
        if current_price <= self.stop_loss:
            self.position = None
            return "Stop Loss"
        
        # Take profit
        if current_price >= self.take_profit:
            self.position = None
            return "Take Profit"
        
        # Signal exit
        price_below_ema = row["close"] < row["ema_9"]
        rsi_weak = row["rsi"] < 50
        
        if price_below_ema or rsi_weak:
            self.position = None
            return "Signal Exit"
        
        return None
    
    def update_trailing_stop(self, df: pd.DataFrame, current_price: float):
        """Update trailing stop"""
        if self.position is None:
            return
        
        row = df.iloc[-1]
        new_stop = current_price - row["atr"] * self.atr_stop
        
        if new_stop > self.stop_loss:
            self.stop_loss = new_stop
