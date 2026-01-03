"""
Trading Bot Configuration
Supports dynamic loading from strategies.json
"""
import os
import json
from dataclasses import dataclass
from typing import Literal, List
from dotenv import load_dotenv

load_dotenv()

# Trading Mode
TRADING_MODE: Literal["paper", "live"] = os.getenv("TRADING_MODE", "paper")

# Binance API
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
BINANCE_TESTNET = os.getenv("BINANCE_TESTNET", "true").lower() == "true"

# Binance Futures Testnet (separate keys)
BINANCE_FUTURES_KEY = os.getenv("BINANCE_FUTURES_KEY", "")
BINANCE_FUTURES_SECRET = os.getenv("BINANCE_FUTURES_SECRET", "")

# Azure OpenAI
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY", "")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:////app/trades.db")

# Notifications
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_ALERTS = True  # Send trade alerts to Telegram


@dataclass
class StrategyConfig:
    """Configuration for a trading strategy"""
    name: str
    symbol: str
    timeframe: str
    market_type: Literal["spot", "futures"]
    enabled: bool = True
    allocation_pct: float = 20.0


# ─────────────────────────────────────────────────────────────────────────────
# Load configuration from strategies.json if it exists
# ─────────────────────────────────────────────────────────────────────────────

def load_config_from_json() -> dict:
    """Load configuration from strategies.json"""
    json_paths = [
        "/app/trading_bot/strategies.json",  # Docker container path
        os.path.join(os.path.dirname(__file__), "strategies.json"),  # Local path
    ]
    
    for path in json_paths:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
    
    return {}


# Load JSON config
_json_config = load_config_from_json()

# Initial Capital
INITIAL_CAPITAL = float(os.getenv(
    "INITIAL_CAPITAL", 
    str(_json_config.get("initial_capital", 100))
))

# Volume filter multiplier (0.8 = 20% more lenient than strict volume > MA)
VOLUME_FILTER_MULTIPLIER = float(_json_config.get("volume_filter_multiplier", 0.8))

# Check interval in seconds
CHECK_INTERVAL = int(_json_config.get("check_interval_seconds", 3600))

# Telegram alerts from config
if "telegram_alerts" in _json_config:
    TELEGRAM_ALERTS = _json_config["telegram_alerts"]


def _build_strategies() -> List[StrategyConfig]:
    """Build strategy list from JSON or defaults"""
    if "strategies" in _json_config:
        return [
            StrategyConfig(
                name=s["name"],
                symbol=s["symbol"],
                timeframe=s["timeframe"],
                market_type=s["market_type"],
                enabled=s.get("enabled", True),
                allocation_pct=s.get("allocation_pct", 20.0)
            )
            for s in _json_config["strategies"]
            if s.get("enabled", True)
        ]
    
    # Default strategies if no JSON
    return [
        StrategyConfig("ScalpingHybrid_DOGE", "DOGEUSDT", "4h", "spot", True, 20.0),
        StrategyConfig("LLM_v4_LowDD", "XRPUSDT", "4h", "spot", True, 25.0),
        StrategyConfig("LLM_v3_Tight", "XRPUSDT", "4h", "spot", True, 20.0),
        StrategyConfig("ScalpingHybrid_AVAX", "AVAXUSDT", "1d", "spot", True, 15.0),
    ]


STRATEGIES = _build_strategies()


# Risk Management
@dataclass
class RiskConfig:
    max_drawdown_pct: float = 35.0
    max_position_pct: float = 100.0
    daily_loss_limit_pct: float = 5.0
    stop_trading_on_daily_limit: bool = True


RISK = RiskConfig()

# Fees (Binance)
SPOT_COMMISSION = 0.001  # 0.1%
FUTURES_COMMISSION = 0.0004  # 0.04%
