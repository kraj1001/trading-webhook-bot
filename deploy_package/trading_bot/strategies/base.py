"""
Base Strategy Class
All strategies inherit from this
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Literal, List
import pandas as pd
import numpy as np


@dataclass
class Signal:
    """Trading signal from a strategy"""
    action: Literal["buy", "sell", "hold"]
    symbol: str
    side: Literal["long", "short"]
    entry_price: float
    stop_loss: float
    take_profit: float
    quantity: float
    timestamp: datetime
    reason: str


@dataclass  
class Trade:
    """A completed trade"""
    id: int
    strategy: str
    symbol: str
    side: Literal["long", "short"]
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    pnl_usd: Optional[float]
    pnl_pct: Optional[float]
    commission: float
    exit_reason: Optional[str]
    is_open: bool = True


class BaseStrategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self, name: str, symbol: str, timeframe: str):
        self.name = name
        self.symbol = symbol
        self.timeframe = timeframe
        self.position = None
        self.entry_price = 0
        self.stop_loss = 0
        self.take_profit = 0
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate common indicators"""
        # EMAs
        for period in [9, 15, 21, 30, 50, 100, 200]:
            df[f"ema_{period}"] = df["close"].ewm(span=period, adjust=False).mean()
        
        # RSI
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df["rsi"] = 100 - (100 / (1 + gain / loss))
        
        # MACD
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        
        # ATR
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs()
        ], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean()
        
        # ADX
        plus_dm = df["high"].diff()
        minus_dm = -df["low"].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        tr14 = tr.rolling(14).sum()
        plus_di = 100 * plus_dm.rolling(14).sum() / tr14
        minus_di = 100 * minus_dm.rolling(14).sum() / tr14
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        df["adx"] = dx.rolling(14).mean()
        
        # Volume MA
        df["vol_ma"] = df["volume"].rolling(50).mean()
        
        return df
    
    @abstractmethod
    def check_entry(self, df: pd.DataFrame) -> Optional[Signal]:
        """Check if entry conditions are met"""
        pass
    
    @abstractmethod
    def check_exit(self, df: pd.DataFrame, current_price: float) -> Optional[str]:
        """Check if exit conditions are met, returns exit reason"""
        pass
    
    def update_trailing_stop(self, df: pd.DataFrame, current_price: float):
        """Update trailing stop if applicable"""
        pass
