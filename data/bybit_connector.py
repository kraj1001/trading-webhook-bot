"""
Bybit API Connector
Fetches historical OHLCV data using pybit library.
Uses public endpoints - no authentication required for historical data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import time
import json
from typing import Optional, List, Dict, Any

try:
    from pybit.unified_trading import HTTP
except ImportError:
    HTTP = None


class BybitConnector:
    """
    Connector for Bybit exchange API.
    Fetches historical kline (candlestick) data.
    """
    
    # Bybit API limits
    MAX_LIMIT = 1000  # Max candles per request
    RATE_LIMIT_DELAY = 0.1  # Seconds between requests
    
    # Supported intervals
    INTERVALS = {
        '1': '1',
        '3': '3',
        '5': '5',
        '15': '15',
        '30': '30',
        '60': '60',
        '120': '120',
        '240': '240',
        '360': '360',
        '720': '720',
        'D': 'D',
        'W': 'W',
        'M': 'M'
    }
    
    def __init__(self, testnet: bool = False, cache_dir: str = "data/cache"):
        """
        Initialize the Bybit connector.
        
        Args:
            testnet: Use testnet instead of mainnet
            cache_dir: Directory for caching data
        """
        if HTTP is None:
            raise ImportError("pybit not installed. Run: pip install pybit")
        
        self.client = HTTP(testnet=testnet)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_klines(
        self,
        symbol: str,
        interval: str = '15',
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch kline/candlestick data.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Kline interval ('1', '5', '15', '60', 'D', etc.)
            start_time: Start datetime
            end_time: End datetime
            limit: Number of candles (max 1000)
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        if interval not in self.INTERVALS:
            raise ValueError(f"Invalid interval. Supported: {list(self.INTERVALS.keys())}")
        
        params = {
            'category': 'linear',  # Linear perpetual (USDT)
            'symbol': symbol,
            'interval': interval,
            'limit': min(limit, self.MAX_LIMIT)
        }
        
        if start_time:
            params['start'] = int(start_time.timestamp() * 1000)
        
        if end_time:
            params['end'] = int(end_time.timestamp() * 1000)
        
        try:
            response = self.client.get_kline(**params)
            
            if response['retCode'] != 0:
                raise Exception(f"API Error: {response['retMsg']}")
            
            data = response['result']['list']
            
            if not data:
                return pd.DataFrame()
            
            # Bybit returns data in reverse order (newest first)
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
            ])
            
            # Convert types
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
                df[col] = df[col].astype(float)
            
            # Sort by timestamp (oldest first)
            df = df.sort_values('timestamp').reset_index(drop=True)
            df.set_index('timestamp', inplace=True)
            
            return df[['open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            print(f"Error fetching klines: {e}")
            return pd.DataFrame()
    
    def get_historical_data(
        self,
        symbol: str,
        interval: str = '15',
        days: int = 180,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch extended historical data by making multiple API calls.
        
        Args:
            symbol: Trading pair
            interval: Kline interval
            days: Number of days of history to fetch
            use_cache: Whether to use/update cache
        
        Returns:
            DataFrame with historical OHLCV data
        """
        cache_file = self.cache_dir / f"{symbol}_{interval}_{days}d.csv"
        
        # Check cache
        if use_cache and cache_file.exists():
            df = pd.read_csv(cache_file, parse_dates=['timestamp'], index_col='timestamp')
            cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            
            # If cache is less than 1 hour old, use it
            if cache_age < timedelta(hours=1):
                print(f"Using cached data: {cache_file}")
                return df
        
        print(f"Fetching {days} days of {symbol} {interval}m data from Bybit...")
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        # Calculate candles per request based on interval
        interval_minutes = int(interval) if interval.isdigit() else 1440  # Daily = 1440 mins
        candles_needed = (days * 24 * 60) // interval_minutes
        
        all_data = []
        current_end = end_time
        
        with_progress = True
        total_fetched = 0
        
        while current_end > start_time:
            df_chunk = self.get_klines(
                symbol=symbol,
                interval=interval,
                end_time=current_end,
                limit=self.MAX_LIMIT
            )
            
            if df_chunk.empty:
                break
            
            all_data.append(df_chunk)
            total_fetched += len(df_chunk)
            
            # Update current_end to fetch older data
            current_end = df_chunk.index[0] - timedelta(minutes=1)
            
            if with_progress:
                print(f"  Fetched {total_fetched} candles...")
            
            # Rate limiting
            time.sleep(self.RATE_LIMIT_DELAY)
            
            # Break if we've gone past start_time
            if df_chunk.index[0] <= start_time:
                break
        
        if not all_data:
            return pd.DataFrame()
        
        # Combine and deduplicate
        df = pd.concat(all_data)
        df = df[~df.index.duplicated(keep='first')]
        df = df.sort_index()
        
        # Filter to requested date range
        df = df[df.index >= start_time]
        
        # Save to cache
        if use_cache:
            df.to_csv(cache_file)
            print(f"Cached {len(df)} candles to {cache_file}")
        
        return df
    
    def get_available_symbols(self, quote_currency: str = 'USDT') -> List[str]:
        """Get list of available trading pairs"""
        try:
            response = self.client.get_instruments_info(category='linear')
            
            if response['retCode'] != 0:
                return []
            
            symbols = [
                item['symbol'] 
                for item in response['result']['list']
                if item['symbol'].endswith(quote_currency)
            ]
            
            return sorted(symbols)
            
        except Exception as e:
            print(f"Error fetching symbols: {e}")
            return []
    
    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get current ticker information"""
        try:
            response = self.client.get_tickers(
                category='linear',
                symbol=symbol
            )
            
            if response['retCode'] != 0:
                return {}
            
            if response['result']['list']:
                return response['result']['list'][0]
            
            return {}
            
        except Exception as e:
            print(f"Error fetching ticker: {e}")
            return {}


# Convenience function for quick data access
def fetch_bybit_data(
    symbol: str = 'BTCUSDT',
    interval: str = '15',
    days: int = 180
) -> pd.DataFrame:
    """
    Quick function to fetch Bybit historical data.
    
    Example:
        df = fetch_bybit_data('BTCUSDT', '15', 90)
    """
    connector = BybitConnector()
    return connector.get_historical_data(symbol, interval, days)
