"""
Technical Indicators Module
Implements all indicators used in the Gold Line strategy.
These match the Pine Script implementations exactly.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


def ema(series: pd.Series, length: int) -> pd.Series:
    """Exponential Moving Average"""
    return series.ewm(span=length, adjust=False).mean()


def sma(series: pd.Series, length: int) -> pd.Series:
    """Simple Moving Average"""
    return series.rolling(window=length).mean()


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """
    Relative Strength Index
    RSI = 100 - (100 / (1 + RS))
    RS = Average Gain / Average Loss
    """
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi_values = 100 - (100 / (1 + rs))
    
    return rsi_values


def cci(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    """
    Commodity Channel Index
    CCI = (Typical Price - SMA) / (0.015 * Mean Deviation)
    """
    typical_price = (high + low + close) / 3
    sma_tp = sma(typical_price, length)
    
    # Mean deviation
    mean_dev = typical_price.rolling(window=length).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True
    )
    
    cci_values = (typical_price - sma_tp) / (0.015 * mean_dev)
    
    return cci_values


def macd(close: pd.Series, fast_length: int = 12, slow_length: int = 26, 
         signal_length: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Moving Average Convergence Divergence
    Returns: (MACD line, Signal line, Histogram)
    """
    fast_ema = ema(close, fast_length)
    slow_ema = ema(close, slow_length)
    
    macd_line = fast_ema - slow_ema
    signal_line = sma(macd_line, signal_length)  # Pine Script uses SMA for signal
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    """Average True Range"""
    high_low = high - low
    high_close = np.abs(high - close.shift(1))
    low_close = np.abs(low - close.shift(1))
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr_values = true_range.rolling(window=length).mean()
    
    return atr_values


def rising(series: pd.Series, length: int) -> pd.Series:
    """Check if series has been rising for 'length' periods"""
    result = pd.Series(True, index=series.index)
    for i in range(1, length + 1):
        result = result & (series > series.shift(i))
    return result


def falling(series: pd.Series, length: int) -> pd.Series:
    """Check if series has been falling for 'length' periods"""
    result = pd.Series(True, index=series.index)
    for i in range(1, length + 1):
        result = result & (series < series.shift(i))
    return result


def cross(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """Detect when series1 crosses series2 (either direction)"""
    above = series1 > series2
    below = series1 < series2
    
    cross_up = above & below.shift(1)
    cross_down = below & above.shift(1)
    
    return cross_up | cross_down


def crossover(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """Detect when series1 crosses above series2"""
    above = series1 > series2
    below_prev = series1.shift(1) <= series2.shift(1)
    return above & below_prev


def crossunder(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """Detect when series1 crosses below series2"""
    below = series1 < series2
    above_prev = series1.shift(1) >= series2.shift(1)
    return below & above_prev


def highest(series: pd.Series, length: int) -> pd.Series:
    """Highest value over length periods"""
    return series.rolling(window=length).max()


def lowest(series: pd.Series, length: int) -> pd.Series:
    """Lowest value over length periods"""
    return series.rolling(window=length).min()


def valuewhen(condition: pd.Series, source: pd.Series, occurrence: int = 0) -> pd.Series:
    """
    Returns the value of source when condition was true, for the Nth occurrence ago.
    This matches Pine Script's valuewhen function.
    """
    result = pd.Series(np.nan, index=source.index)
    last_value = np.nan
    
    for i in range(len(source)):
        if condition.iloc[i]:
            last_value = source.iloc[i]
        result.iloc[i] = last_value
    
    return result


def calculate_support_resistance(high: pd.Series, low: pd.Series, 
                                  close: pd.Series, length: int = 21) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Support and Resistance levels using valuewhen logic
    Matches the Pine Script [RS]Support and Resistance implementation
    """
    highest_high = highest(high, length)
    lowest_low = lowest(low, length)
    
    # Resistance: value when high >= highest high
    resistance_condition = high >= highest_high
    resistance = valuewhen(resistance_condition, high, 0)
    
    # Support: value when low <= lowest low
    support_condition = low <= lowest_low
    support = valuewhen(support_condition, low, 0)
    
    return support, resistance
