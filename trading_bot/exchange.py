"""
Binance Exchange Wrapper
Supports Spot Testnet and Futures Testnet with direct API calls
"""
import os
import time
import hmac
import hashlib
import requests
import ccxt
from typing import Optional, Dict, Any, Literal
from dataclasses import dataclass
from datetime import datetime
import logging

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

# Trading Mode
TRADING_MODE = os.getenv("TRADING_MODE", "paper")

# API Keys
BINANCE_SPOT_API_KEY = os.getenv("BINANCE_SPOT_API_KEY", "")
BINANCE_SPOT_SECRET = os.getenv("BINANCE_SPOT_SECRET", "")
BINANCE_FUTURES_API_KEY = os.getenv("BINANCE_FUTURES_API_KEY", "")
BINANCE_FUTURES_SECRET = os.getenv("BINANCE_FUTURES_SECRET", "")

# Fees
SPOT_COMMISSION = 0.001
FUTURES_COMMISSION = 0.0004


@dataclass
class Order:
    """Represents an executed order"""
    id: str
    symbol: str
    side: Literal["buy", "sell"]
    type: str
    quantity: float
    price: float
    timestamp: datetime
    commission: float
    is_testnet: bool = True


class BinanceTestnetAPI:
    """Direct API wrapper for Binance Testnet (bypasses ccxt issues)"""
    
    def __init__(self):
        # Futures Testnet
        self.futures_url = "https://testnet.binancefuture.com"
        self.futures_key = BINANCE_FUTURES_API_KEY
        self.futures_secret = BINANCE_FUTURES_SECRET
        
        # Spot Testnet
        self.spot_url = "https://testnet.binance.vision"
        self.spot_key = BINANCE_SPOT_API_KEY
        self.spot_secret = BINANCE_SPOT_SECRET
        
        # For fetching real price data
        self.price_exchange = ccxt.binance()
        
        logger.info("✅ Binance Testnet API initialized")
    
    def _sign(self, params: dict, secret: str) -> str:
        """Create signature for authenticated requests"""
        query = "&".join([f"{k}={v}" for k, v in params.items()])
        return hmac.new(secret.encode(), query.encode(), hashlib.sha256).hexdigest()
    
    def _request(self, method: str, url: str, params: dict, key: str, secret: str):
        """Make authenticated request"""
        params["timestamp"] = int(time.time() * 1000)
        params["signature"] = self._sign(params, secret)
        headers = {"X-MBX-APIKEY": key}
        
        if method == "GET":
            resp = requests.get(url, params=params, headers=headers)
        else:
            resp = requests.post(url, params=params, headers=headers)
        
        if resp.status_code != 200:
            raise Exception(f"API Error: {resp.text}")
        
        return resp.json()
    
    def fetch_balance(self, market_type: str = "spot") -> Dict[str, float]:
        """Get account balance"""
        if market_type == "futures":
            url = f"{self.futures_url}/fapi/v2/balance"
            data = self._request("GET", url, {}, self.futures_key, self.futures_secret)
            return {item["asset"]: float(item["balance"]) for item in data if float(item["balance"]) > 0}
        else:
            url = f"{self.spot_url}/api/v3/account"
            data = self._request("GET", url, {}, self.spot_key, self.spot_secret)
            return {b["asset"]: float(b["free"]) for b in data["balances"] if float(b["free"]) > 0}
    
    def fetch_ticker(self, symbol: str, market_type: str = "spot") -> Dict[str, Any]:
        """Get current price (uses real Binance for accuracy)"""
        return self.price_exchange.fetch_ticker(symbol)
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = "4h", limit: int = 300, market_type: str = "spot"):
        """Get OHLCV data (uses real Binance for accuracy)"""
        return self.price_exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    
    def create_order(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
        quantity: float,
        market_type: str = "spot",
        price: Optional[float] = None
    ) -> Order:
        """Create order on testnet"""
        
        if market_type == "futures":
            url = f"{self.futures_url}/fapi/v1/order"
            key, secret = self.futures_key, self.futures_secret
        else:
            url = f"{self.spot_url}/api/v3/order"
            key, secret = self.spot_key, self.spot_secret
        
        # Get current price if not specified
        if price is None:
            ticker = self.fetch_ticker(symbol)
            price = ticker["last"]
        
        # Symbol-specific lot sizes (step size for quantity)
        # Binance Spot Testnet requires specific decimal places
        lot_sizes = {
            "DOGEUSDT": 0,    # Whole numbers only
            "XRPUSDT": 1,     # 1 decimal place
            "AVAXUSDT": 2,    # 2 decimal places
            "BTCUSDT": 5,     # 5 decimal places
            "ETHUSDT": 4,     # 4 decimal places
            "SOLUSDT": 2,     # 2 decimal places
        }
        clean_symbol = symbol.replace("/", "")
        decimals = lot_sizes.get(clean_symbol, 2)
        rounded_qty = round(quantity, decimals)
        
        # Ensure minimum quantity
        if rounded_qty <= 0:
            rounded_qty = 10 ** (-decimals)  # Minimum possible quantity
        
        params = {
            "symbol": clean_symbol,
            "side": side.upper(),
            "type": "MARKET",
            "quantity": rounded_qty,
        }
        
        try:
            data = self._request("POST", url, params, key, secret)
            
            order_id = data.get("orderId", str(time.time()))
            exec_price = float(data.get("avgPrice", price))
            exec_qty = float(data.get("executedQty", quantity))
            
            commission = exec_qty * exec_price * (FUTURES_COMMISSION if market_type == "futures" else SPOT_COMMISSION)
            
            logger.info(f"✅ TESTNET ORDER: {side.upper()} {exec_qty} {symbol} @ {exec_price}")
            
            return Order(
                id=str(order_id),
                symbol=symbol,
                side=side,
                type="market",
                quantity=exec_qty,
                price=exec_price,
                timestamp=datetime.now(),
                commission=commission,
                is_testnet=True
            )
        except Exception as e:
            logger.error(f"Order failed: {e}")
            raise


class PaperExchange:
    """Local simulated exchange (no Binance connection)"""
    
    def __init__(self, initial_balance: float = 15000):
        self.balance = {"USDT": initial_balance}
        self.order_id = 0
        self.price_exchange = ccxt.binance()
        logger.info(f"Paper Exchange initialized with ${initial_balance}")
    
    def fetch_balance(self, market_type: str = "spot") -> Dict[str, float]:
        return self.balance.copy()
    
    def fetch_ticker(self, symbol: str, market_type: str = "spot") -> Dict[str, Any]:
        return self.price_exchange.fetch_ticker(symbol)
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = "4h", limit: int = 300, market_type: str = "spot"):
        return self.price_exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    
    def create_order(
        self, 
        symbol: str, 
        side: Literal["buy", "sell"], 
        quantity: float,
        market_type: str = "spot",
        price: Optional[float] = None
    ) -> Order:
        self.order_id += 1
        
        if price is None:
            ticker = self.fetch_ticker(symbol)
            price = ticker["last"]
        
        commission = quantity * price * SPOT_COMMISSION
        
        logger.info(f"📝 Paper {side.upper()} {quantity} {symbol} @ {price}")
        
        return Order(
            id=f"paper_{self.order_id}",
            symbol=symbol,
            side=side,
            type="market",
            quantity=quantity,
            price=price,
            timestamp=datetime.now(),
            commission=commission,
            is_testnet=False
        )


def get_exchange():
    """Factory function to get appropriate exchange based on trading mode"""
    mode = os.getenv("TRADING_MODE", "paper")
    
    if mode == "testnet":
        return BinanceTestnetAPI()
    else:
        return PaperExchange()
