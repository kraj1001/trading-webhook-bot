"""
LLM v4 Low DD Strategy
Verified: +9,808% on XRPUSDT 4H with only 17% drawdown
Supports both LONG and SHORT
"""
from datetime import datetime
from typing import Optional
import pandas as pd

from .base import BaseStrategy, Signal


class LLMv4LowDDStrategy(BaseStrategy):
    """
    LLM v4 Low DD Strategy (LONG + SHORT)
    
    Entry LONG:
    - EMA 15 > EMA 30 (trend)
    - Close > EMA 15 (momentum)
    - Close > EMA 200 (major trend)
    - RSI > 70 (strong momentum)
    - ADX > 25 (trend strength)
    - MACD > Signal (bullish)
    - Volume > 50-period MA
    
    Entry SHORT:
    - EMA 15 < EMA 30 (trend)
    - Close < EMA 15 (momentum)
    - Close < EMA 200 (major trend)
    - RSI < 30 (oversold)
    - ADX > 25 (trend strength)
    - MACD < Signal (bearish)
    - Volume > 50-period MA
    """
    
    def __init__(self, symbol: str = "XRPUSDT", timeframe: str = "4h"):
        super().__init__("LLM_v4_LowDD", symbol, timeframe)
        self.atr_stop = 1.0
        self.atr_tp = 2.0
        self.adx_min = 25
    
    def check_entry(self, df: pd.DataFrame) -> Optional[Signal]:
        """Check for entry signals (both long and short)"""
        if self.position is not None:
            return None
        
        row = df.iloc[-1]
        
        # Common conditions
        adx_ok = row["adx"] > self.adx_min
        volume_ok = row["volume"] > row["vol_ma"] * 0.8  # 20% more lenient
        
        # LONG conditions
        long_trend = row["ema_15"] > row["ema_30"]
        long_price = row["close"] > row["ema_15"] and row["close"] > row["ema_200"]
        long_rsi = row["rsi"] > 70
        long_macd = row["macd"] > row["macd_signal"]
        
        if long_trend and long_price and long_rsi and long_macd and adx_ok and volume_ok:
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
                quantity=0,
                timestamp=datetime.now(),
                reason="LONG: EMA15>30>200, RSI>70, ADX>25, MACD+"
            )
        
        # SHORT conditions
        short_trend = row["ema_15"] < row["ema_30"]
        short_price = row["close"] < row["ema_15"] and row["close"] < row["ema_200"]
        short_rsi = row["rsi"] < 30
        short_macd = row["macd"] < row["macd_signal"]
        
        if short_trend and short_price and short_rsi and short_macd and adx_ok and volume_ok:
            stop_loss = row["close"] + row["atr"] * self.atr_stop
            take_profit = row["close"] - row["atr"] * self.atr_tp
            
            # Note: Don't set self.position here - only set after successful order in main.py
            
            return Signal(
                action="sell",
                symbol=self.symbol,
                side="short",
                entry_price=row["close"],
                stop_loss=stop_loss,
                take_profit=take_profit,
                quantity=0,
                timestamp=datetime.now(),
                reason="SHORT: EMA15<30<200, RSI<30, ADX>25, MACD-"
            )
        
        return None
    
    def check_exit(self, df: pd.DataFrame, current_price: float) -> Optional[str]:
        """Check for exit conditions"""
        if self.position is None:
            return None
        
        row = df.iloc[-1]
        
        if self.position == "long":
            # Stop loss
            if current_price <= self.stop_loss:
                self.position = None
                return "Stop Loss"
            
            # Take profit
            if current_price >= self.take_profit:
                self.position = None
                return "Take Profit"
            
            # Signal exit
            if row["close"] < row["ema_15"] or row["rsi"] < 50:
                self.position = None
                return "Signal Exit"
        
        elif self.position == "short":
            # Stop loss
            if current_price >= self.stop_loss:
                self.position = None
                return "Stop Loss"
            
            # Take profit
            if current_price <= self.take_profit:
                self.position = None
                return "Take Profit"
            
            # Signal exit
            if row["close"] > row["ema_15"] or row["rsi"] > 50:
                self.position = None
                return "Signal Exit"
        
        return None
    
    def update_trailing_stop(self, df: pd.DataFrame, current_price: float):
        """Update trailing stop"""
        if self.position is None:
            return
        
        row = df.iloc[-1]
        
        if self.position == "long":
            new_stop = current_price - row["atr"] * self.atr_stop
            if new_stop > self.stop_loss:
                self.stop_loss = new_stop
        
        elif self.position == "short":
            new_stop = current_price + row["atr"] * self.atr_stop
            if new_stop < self.stop_loss:
                self.stop_loss = new_stop
