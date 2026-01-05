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

# API Base URLs (configurable via environment)
# Demo:       https://demo-api.binance.com
# Testnet:    https://testnet.binance.vision  
# Production: https://api.binance.com
BINANCE_SPOT_URL = os.getenv("BINANCE_SPOT_URL", "https://demo-api.binance.com")
# Futures Demo: https://demo-fapi.binance.com
BINANCE_FUTURES_URL = os.getenv("BINANCE_FUTURES_URL", "https://demo-fapi.binance.com")

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
    """Direct API wrapper for Binance (supports Demo, Testnet, and Production)"""
    
    def __init__(self):
        # Futures API
        self.futures_url = BINANCE_FUTURES_URL
        self.futures_key = BINANCE_FUTURES_API_KEY
        self.futures_secret = BINANCE_FUTURES_SECRET
        
        # Spot API (Demo, Testnet, or Production based on BINANCE_SPOT_URL)
        self.spot_url = BINANCE_SPOT_URL
        self.spot_key = BINANCE_SPOT_API_KEY
        self.spot_secret = BINANCE_SPOT_SECRET
        
        # For fetching real price data
        self.price_exchange = ccxt.binance()
        
        # Determine if using testnet/demo
        is_demo = "demo" in self.spot_url or "testnet" in self.spot_url
        
        logger.info(f"✅ Binance API initialized: {self.spot_url} ({'DEMO' if is_demo else 'PRODUCTION'})")
    
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
            
            # Futures Demo API returns executedQty=0 for NEW orders
            # Fall back to origQty and current price when not immediately filled
            exec_qty = float(data.get("executedQty", 0))
            exec_price = float(data.get("avgPrice", 0))
            
            if exec_qty == 0:
                # Order not yet filled (common on Futures Demo)
                exec_qty = float(data.get("origQty", quantity))
                exec_price = price  # Use the price we fetched earlier
                logger.warning(f"⚠️ Order NEW (not filled yet), using origQty={exec_qty}, price={exec_price}")
            
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
    
    def create_order_with_sl_tp(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
        quantity: float,
        stop_loss_price: float,
        take_profit_price: float,
        market_type: str = "futures"
    ) -> Order:
        """
        Create entry order with exchange-side stop-loss and take-profit.
        For Futures: Uses STOP_MARKET and TAKE_PROFIT_MARKET orders.
        """
        # First, place the entry order
        entry_order = self.create_order(symbol, side, quantity, market_type)
        
        if market_type != "futures":
            logger.warning("SL/TP orders only supported for futures. Entry placed without SL/TP.")
            return entry_order
        
        # Determine exit side (opposite of entry)
        exit_side = "SELL" if side == "buy" else "BUY"
        
        clean_symbol = symbol.replace("/", "")
        url = f"{self.futures_url}/fapi/v1/order"
        key, secret = self.futures_key, self.futures_secret
        
        # Get proper quantity decimals
        lot_sizes = {"DOGEUSDT": 0, "XRPUSDT": 1, "AVAXUSDT": 2, "BTCUSDT": 5, "ETHUSDT": 4, "SOLUSDT": 2}
        decimals = lot_sizes.get(clean_symbol, 2)
        rounded_qty = round(quantity, decimals)
        
        # Round prices to appropriate decimals
        price_decimals = 4 if "USDT" in symbol else 2
        sl_price = round(stop_loss_price, price_decimals)
        tp_price = round(take_profit_price, price_decimals)
        
        # Place Stop-Loss order (STOP_MARKET)
        try:
            sl_params = {
                "symbol": clean_symbol,
                "side": exit_side,
                "type": "STOP_MARKET",
                "quantity": rounded_qty,
                "stopPrice": sl_price,
                "reduceOnly": "true"
            }
            sl_data = self._request("POST", url, sl_params, key, secret)
            logger.info(f"📉 STOP-LOSS placed: {exit_side} {rounded_qty} {symbol} @ {sl_price}")
        except Exception as e:
            logger.error(f"Stop-loss order failed: {e}")
        
        # Place Take-Profit order (TAKE_PROFIT_MARKET)
        try:
            tp_params = {
                "symbol": clean_symbol,
                "side": exit_side,
                "type": "TAKE_PROFIT_MARKET",
                "quantity": rounded_qty,
                "stopPrice": tp_price,
                "reduceOnly": "true"
            }
            tp_data = self._request("POST", url, tp_params, key, secret)
            logger.info(f"📈 TAKE-PROFIT placed: {exit_side} {rounded_qty} {symbol} @ {tp_price}")
        except Exception as e:
            logger.error(f"Take-profit order failed: {e}")
        
        return entry_order

    def create_spot_order_with_oco(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
        quantity: float,
        stop_loss_price: float,
        take_profit_price: float
    ) -> Order:
        """
        Create Spot entry order then place OCO order for SL/TP protection.
        OCO = One-Cancels-the-Other: when SL or TP triggers, the other is canceled.
        """
        # First, place the entry order
        entry_order = self.create_order(symbol, side, quantity, market_type="spot")
        
        if entry_order.quantity <= 0:
            logger.error("Entry order failed, skipping OCO")
            return entry_order
        
        # For a BUY entry, we need to SELL at SL or TP
        # OCO order places both a limit sell (TP) and stop-limit sell (SL)
        clean_symbol = symbol.replace("/", "")
        url = f"{self.spot_url}/api/v3/order/oco"
        key, secret = self.spot_key, self.spot_secret
        
        # Get proper quantity decimals
        lot_sizes = {"DOGEUSDT": 0, "XRPUSDT": 1, "AVAXUSDT": 2, "BTCUSDT": 5, "ETHUSDT": 4, "SOLUSDT": 2}
        decimals = lot_sizes.get(clean_symbol, 2)
        rounded_qty = round(entry_order.quantity, decimals)
        
        # Get current market price for validation
        try:
            ticker = self.price_exchange.fetch_ticker(clean_symbol)
            current_price = ticker['last']
        except:
            current_price = entry_order.price if entry_order.price > 0 else stop_loss_price * 1.05
        
        # Round prices
        price_decimals = 5 if clean_symbol == "DOGEUSDT" else 4
        sl_price = round(stop_loss_price, price_decimals)
        tp_price = round(take_profit_price, price_decimals)
        
        # For SELL OCO: price (TP) must be > current_price, stopPrice (SL) must be < current_price
        # Validate price relationship
        if not (sl_price < current_price < tp_price):
            logger.error(f"OCO price validation failed: SL={sl_price} < current={current_price} < TP={tp_price}")
            logger.warning("⚠️ Entry placed but OCO skipped due to invalid price relationship")
            return entry_order
        
        # OCO needs a stop limit price slightly below stop price for sells
        sl_limit_price = round(sl_price * 0.995, price_decimals)  # 0.5% below stop
        
        try:
            oco_params = {
                "symbol": clean_symbol,
                "side": "SELL",  # Exit side (opposite of entry)
                "quantity": rounded_qty,
                "price": tp_price,  # Take-profit price (LIMIT order)
                "stopPrice": sl_price,  # Stop-loss trigger price
                "stopLimitPrice": sl_limit_price,  # Stop-loss limit price
                "stopLimitTimeInForce": "GTC"
            }
            logger.info(f"OCO params: qty={rounded_qty}, TP={tp_price}, SL_trigger={sl_price}, SL_limit={sl_limit_price}")
            oco_data = self._request("POST", url, oco_params, key, secret)
            
            logger.info(f"🎯 OCO ORDER placed for {symbol}:")
            logger.info(f"   📈 Take-Profit: SELL @ {tp_price}")
            logger.info(f"   📉 Stop-Loss: SELL @ {sl_price} (limit: {sl_limit_price})")
            
        except Exception as e:
            logger.error(f"OCO order failed: {e}")
            logger.warning("⚠️ Entry placed but no exchange-side SL/TP protection!")
            logger.info("Note: Binance Demo may not support OCO. Bot will monitor SL/TP internally.")
        
        return entry_order


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
