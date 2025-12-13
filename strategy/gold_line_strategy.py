"""
Gold Line Strategy Implementation
Complete Python port of the Pine Script "Price Action Channel" strategy.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from dataclasses import dataclass

from .indicators import (
    ema, sma, rsi, cci, macd, atr,
    rising, falling, cross, crossover, crossunder,
    highest, lowest, calculate_support_resistance
)


@dataclass
class Signal:
    """Represents a trading signal"""
    timestamp: pd.Timestamp
    type: str  # 'BUY', 'SELL', 'CCI_UP', 'CCI_DOWN', 'CROSS'
    price: float
    confidence: float
    indicators: Dict[str, float]
    context: str


class GoldLineStrategy:
    """
    Gold Line Price Action Strategy
    
    This is a trend-following price action system using:
    - Price Action Channel (Gold Line)
    - CCI for momentum
    - MACD for trend confirmation
    - RSI for overbought/oversold filtering
    - Support/Resistance levels
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize strategy with configuration"""
        self.config = config
        
        # CCI settings
        self.cci_length = config.get('cci', {}).get('length', 14)
        self.cci_upper = config.get('cci', {}).get('upper_level', 75)
        self.cci_lower = config.get('cci', {}).get('lower_level', -75)
        
        # MACD settings
        self.macd_fast = config.get('macd', {}).get('fast_length', 12)
        self.macd_slow = config.get('macd', {}).get('slow_length', 17)
        self.macd_signal = config.get('macd', {}).get('signal_length', 8)
        
        # RSI settings
        self.rsi_length = config.get('rsi', {}).get('length', 7)
        self.rsi_upper = config.get('rsi', {}).get('upper_limit', 70)
        self.rsi_lower = config.get('rsi', {}).get('lower_limit', 30)
        
        # Channel settings
        self.channel_low = config.get('channel', {}).get('low_length', 5)
        self.channel_high = config.get('channel', {}).get('high_length', 5)
        self.channel_median = config.get('channel', {}).get('median_length', 4)
        
        # Filters
        filters = config.get('filters', {})
        self.use_macd_filter = filters.get('use_macd_filter', True)
        self.use_rsi_filter = filters.get('use_rsi_filter', True)
        self.use_trend_filter = filters.get('use_trend_filter', True)
        self.use_macd_direction_filter = filters.get('use_macd_direction_filter', True)  # NEW: LLM recommended
        self.macd_direction_threshold = filters.get('macd_direction_threshold', -20)  # Only short when MACD < -20
        self.direction_candles = filters.get('direction_candles', 4)
        
        # Support/Resistance
        sr = config.get('support_resistance', {})
        self.sma_length = sr.get('sma_length', 8)
        self.sr_lookback = sr.get('lookback', 13)
        self.sr_length = sr.get('sr_length', 21)
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all indicators and add them to the dataframe.
        Expects df with columns: open, high, low, close, volume
        """
        df = df.copy()
        
        # Ensure proper column names (lowercase)
        df.columns = df.columns.str.lower()
        
        # HL2 (typical price for channel median)
        df['hl2'] = (df['high'] + df['low']) / 2
        
        # --- CCI ---
        df['cci'] = cci(df['high'], df['low'], df['close'], self.cci_length)
        df['cci_is_up'] = df['cci'] > self.cci_upper
        df['cci_is_down'] = df['cci'] < self.cci_lower
        
        # --- RSI ---
        df['rsi'] = rsi(df['close'], self.rsi_length)
        
        # --- MACD ---
        df['macd'], df['macd_signal_line'], df['macd_hist'] = macd(
            df['close'], self.macd_fast, self.macd_slow, self.macd_signal
        )
        
        # Slow EMA (trend line)
        df['slow_ema'] = ema(df['close'], self.macd_slow)
        
        # MACD output signal: 1 = bearish (red), -1 = bullish (green)
        df['macd_output'] = np.where(
            df['macd_signal_line'] > df['macd'],
            1,  # Red background (bearish)
            np.where(df['macd_signal_line'] < df['macd'], -1, 0)  # Green background (bullish)
        )
        
        # --- Price Action Channel (Gold Line) ---
        df['ema_low'] = ema(df['low'], self.channel_low)
        df['ema_high'] = ema(df['high'], self.channel_high)
        df['gold_line'] = ema(df['hl2'], self.channel_median)  # Median = Gold Line
        
        # --- ATR for label positioning ---
        df['atr'] = atr(df['high'], df['low'], df['close'], 30)
        
        # --- Trend direction ---
        df['slow_ema_rising'] = rising(df['slow_ema'], self.direction_candles)
        df['slow_ema_falling'] = falling(df['slow_ema'], self.direction_candles)
        
        # --- Cross detection ---
        df['slow_gold_cross'] = cross(df['slow_ema'], df['gold_line'])
        
        # --- Support/Resistance ---
        df['support'], df['resistance'] = calculate_support_resistance(
            df['high'], df['low'], df['close'], self.sr_length
        )
        
        # --- SMA for additional S/R ---
        df['sma_sr'] = sma(df['close'], self.sma_length)
        df['last_high'] = highest(df['close'], self.sr_lookback)
        df['last_low'] = lowest(df['close'], self.sr_lookback)
        
        return df
    
    def generate_cci_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate CCI-based signals (triangles from Pine Script)
        """
        df = df.copy()
        
        # Bullish candle condition
        bullish_candle = df['close'] > df['open']
        bearish_candle = df['close'] < df['open']
        
        # CCI Up Alert conditions
        cci_up_conditions = (
            df['cci_is_up'] &
            bullish_candle
        )
        
        # Apply filters
        if self.use_rsi_filter:
            cci_up_conditions = cci_up_conditions & (df['rsi'] > self.rsi_upper)
        
        if self.use_macd_filter:
            cci_up_conditions = cci_up_conditions & (df['macd_output'] < 0)  # Green background
        
        if self.use_trend_filter:
            cci_up_conditions = cci_up_conditions & (
                (df['gold_line'] > df['slow_ema']) & df['slow_ema_rising']
            )
        
        # CCI Down Alert conditions
        cci_dn_conditions = (
            df['cci_is_down'] &
            bearish_candle
        )
        
        if self.use_rsi_filter:
            cci_dn_conditions = cci_dn_conditions & (df['rsi'] < self.rsi_lower)
        
        if self.use_macd_filter:
            cci_dn_conditions = cci_dn_conditions & (df['macd_output'] > 0)  # Red background
        
        if self.use_trend_filter:
            cci_dn_conditions = cci_dn_conditions & (
                (df['gold_line'] < df['slow_ema']) & df['slow_ema_falling']
            )
        
        # Track consecutive signals
        df['cci_up_alert'] = 0
        df['cci_dn_alert'] = 0
        
        for i in range(1, len(df)):
            if cci_up_conditions.iloc[i]:
                if df['cci_up_alert'].iloc[i-1] == 0:
                    df.loc[df.index[i], 'cci_up_alert'] = 1
                else:
                    df.loc[df.index[i], 'cci_up_alert'] = df['cci_up_alert'].iloc[i-1] + 1
            
            if cci_dn_conditions.iloc[i]:
                if df['cci_dn_alert'].iloc[i-1] == 0:
                    df.loc[df.index[i], 'cci_dn_alert'] = 1
                else:
                    df.loc[df.index[i], 'cci_dn_alert'] = df['cci_dn_alert'].iloc[i-1] + 1
        
        return df
    
    def generate_price_action_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate Buy/Sell signals based on price action patterns
        (The label-based signals from Pine Script)
        """
        df = df.copy()
        
        # Buy signal conditions (engulfing / reversal patterns)
        buy_1 = (df['low'] < df['low'].shift(1)) & \
                (df['close'] > df['high'].shift(1)) & \
                (df['close'] > df['open'])
        
        buy_2 = (df['low'] < df['close'].shift(1)) & \
                (df['close'] > df['high'].shift(1)) & \
                (df['open'] < df['close']) & \
                (df['close'].shift(1) < df['open'].shift(1))
        
        buy_3 = (df['low'] < df['close'].shift(1)) & \
                (df['close'] > df['high'].shift(1))
        
        df['buy_raw'] = buy_1 | buy_2 | buy_3
        
        # Sell signal conditions
        sell_1 = (df['high'] > df['high'].shift(1)) & \
                 (df['close'] < df['low'].shift(1)) & \
                 (df['open'] > df['close'])
        
        sell_2 = (df['high'] > df['close'].shift(1)) & \
                 (df['close'] < df['low'].shift(1)) & \
                 (df['open'] > df['close']) & \
                 (df['close'].shift(1) > df['open'].shift(1))
        
        sell_3 = (df['high'] > df['close'].shift(1)) & \
                 (df['close'] < df['low'].shift(1))
        
        df['sell_raw'] = sell_1 | sell_2 | sell_3
        
        # Apply filter to prevent consecutive same-direction signals
        df['prev_signal'] = 0
        df['buy_final'] = False
        df['sell_final'] = False
        
        for i in range(1, len(df)):
            if df['buy_raw'].iloc[i]:
                df.loc[df.index[i], 'prev_signal'] = 1
            elif df['sell_raw'].iloc[i]:
                df.loc[df.index[i], 'prev_signal'] = -1
            else:
                df.loc[df.index[i], 'prev_signal'] = df['prev_signal'].iloc[i-1]
            
            # Final signals only when direction changes
            # NEW: Apply MACD direction filter (LLM recommended)
            macd_value = df['macd'].iloc[i]
            
            if df['buy_raw'].iloc[i] and df['prev_signal'].iloc[i-1] == -1:
                # MACD direction filter: only go LONG when MACD > 0
                if self.use_macd_direction_filter:
                    if macd_value > 0:  # MACD positive = bullish
                        df.loc[df.index[i], 'buy_final'] = True
                else:
                    df.loc[df.index[i], 'buy_final'] = True
            
            if df['sell_raw'].iloc[i] and df['prev_signal'].iloc[i-1] == 1:
                # MACD direction filter: only go SHORT when MACD < threshold
                if self.use_macd_direction_filter:
                    if macd_value < self.macd_direction_threshold:  # MACD negative = bearish
                        df.loc[df.index[i], 'sell_final'] = True
                else:
                    df.loc[df.index[i], 'sell_final'] = True
        
        return df
    
    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
        """
        Run the complete strategy on the dataframe.
        Returns the enriched dataframe and a list of signals.
        """
        # Calculate all indicators
        df = self.calculate_indicators(df)
        
        # Generate CCI signals
        df = self.generate_cci_signals(df)
        
        # Generate price action signals
        df = self.generate_price_action_signals(df)
        
        # Compile all signals into a list
        signals = []
        
        for i in range(len(df)):
            row = df.iloc[i]
            timestamp = df.index[i] if isinstance(df.index[i], pd.Timestamp) else pd.Timestamp(df.index[i])
            
            indicators = {
                'cci': row['cci'],
                'rsi': row['rsi'],
                'macd': row['macd'],
                'macd_signal': row['macd_signal_line'],
                'macd_hist': row['macd_hist'],
                'gold_line': row['gold_line'],
                'slow_ema': row['slow_ema'],
                'support': row['support'],
                'resistance': row['resistance']
            }
            
            # CCI Up Alert
            if row['cci_up_alert'] > 0:
                signals.append(Signal(
                    timestamp=timestamp,
                    type='CCI_UP',
                    price=row['close'],
                    confidence=min(row['cci_up_alert'] / 3, 1.0),
                    indicators=indicators.copy(),
                    context=f"CCI={row['cci']:.1f}, RSI={row['rsi']:.1f}, MACD Green"
                ))
            
            # CCI Down Alert
            if row['cci_dn_alert'] > 0:
                signals.append(Signal(
                    timestamp=timestamp,
                    type='CCI_DOWN',
                    price=row['close'],
                    confidence=min(row['cci_dn_alert'] / 3, 1.0),
                    indicators=indicators.copy(),
                    context=f"CCI={row['cci']:.1f}, RSI={row['rsi']:.1f}, MACD Red"
                ))
            
            # Buy Final
            if row['buy_final']:
                signals.append(Signal(
                    timestamp=timestamp,
                    type='BUY',
                    price=row['close'],
                    confidence=0.8,
                    indicators=indicators.copy(),
                    context="Price action reversal pattern - Buy"
                ))
            
            # Sell Final
            if row['sell_final']:
                signals.append(Signal(
                    timestamp=timestamp,
                    type='SELL',
                    price=row['close'],
                    confidence=0.8,
                    indicators=indicators.copy(),
                    context="Price action reversal pattern - Sell"
                ))
            
            # Cross signal
            if row['slow_gold_cross']:
                cross_type = 'CROSS_UP' if row['slow_ema'] > row['gold_line'] else 'CROSS_DOWN'
                signals.append(Signal(
                    timestamp=timestamp,
                    type=cross_type,
                    price=row['close'],
                    confidence=0.6,
                    indicators=indicators.copy(),
                    context=f"Slow EMA crossed Gold Line"
                ))
        
        return df, signals
    
    def get_signal_summary(self, signals: list) -> Dict[str, Any]:
        """Get summary statistics of generated signals"""
        if not signals:
            return {'total': 0}
        
        summary = {
            'total': len(signals),
            'buy_signals': len([s for s in signals if s.type == 'BUY']),
            'sell_signals': len([s for s in signals if s.type == 'SELL']),
            'cci_up': len([s for s in signals if s.type == 'CCI_UP']),
            'cci_down': len([s for s in signals if s.type == 'CCI_DOWN']),
            'crosses': len([s for s in signals if 'CROSS' in s.type]),
            'avg_confidence': np.mean([s.confidence for s in signals])
        }
        
        return summary
