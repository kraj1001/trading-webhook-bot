"""
Trading Bot Telegram Interface
Private bot for mobile access to trading status

Setup:
1. Create a bot with @BotFather on Telegram
2. Get your bot token
3. Get your chat ID (message the bot and check /api/telegram/get-chat-id)
4. Set environment variables:
   - TELEGRAM_BOT_TOKEN=your_bot_token
   - TELEGRAM_CHAT_ID=your_chat_id
"""
import os
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Optional
import requests

logger = logging.getLogger(__name__)

# Telegram Configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")  # Only respond to this chat ID

TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"


class TelegramBot:
    """Private Telegram bot for trading status"""
    
    def __init__(self, db=None):
        self.db = db
        self.enabled = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)
        self.authorized_chat_id = TELEGRAM_CHAT_ID
        
        if not self.enabled:
            logger.warning("Telegram bot disabled - missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")
    
    def _is_authorized(self, chat_id: str) -> bool:
        """Check if chat is authorized (only your private chat)"""
        return str(chat_id) == str(self.authorized_chat_id)
    
    def send_message(self, text: str, parse_mode: str = "Markdown", reply_markup: dict = None) -> bool:
        """Send message to your authorized chat only"""
        if not self.enabled:
            return False
        
        try:
            payload = {
                "chat_id": self.authorized_chat_id,
                "text": text,
                "parse_mode": parse_mode
            }
            if reply_markup:
                payload["reply_markup"] = reply_markup
            
            response = requests.post(
                f"{TELEGRAM_API}/sendMessage",
                json=payload,
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False
    
    def send_dashboard_link(self):
        """Send message with button to open dashboard"""
        message = """
📊 *TRADING BOT DASHBOARD*

Tap the button below to open the full dashboard with charts, trades, and analytics.
"""
        # Inline keyboard with URL button
        reply_markup = {
            "inline_keyboard": [[
                {
                    "text": "📈 Open Dashboard",
                    "url": "https://trading-bot.braveocean-cb90440a.australiaeast.azurecontainerapps.io/"
                }
            ], [
                {
                    "text": "📊 View Charts",
                    "url": "https://trading-bot.braveocean-cb90440a.australiaeast.azurecontainerapps.io/#chart"
                }
            ], [
                {
                    "text": "📋 Weekly Report",
                    "url": "https://trading-bot.braveocean-cb90440a.australiaeast.azurecontainerapps.io/api/weekly-report"
                }
            ]]
        }
        self.send_message(message, reply_markup=reply_markup)
    
    def notify_trade_opened(self, trade_id: int, strategy: str, symbol: str, side: str, 
                                price: float, quantity: float, market_type: str,
                                stop_loss: float = None, take_profit: float = None,
                                success: bool = True):
        """Notify when a trade is opened with full details"""
        logger.info(f"📱 Telegram: notify_trade_opened called for #{trade_id} {strategy} success={success}")
        
        # Strategy display name mapping
        display_names = {
            "ScalpingHybrid_DOGE": "DOGE Scalper 4H",
            "LLM_v4_LowDD": "Momentum Pro 4H",
            "LLM_v3_Tight": "Trend Hunter 4H", 
            "ScalpingHybrid_AVAX": "AVAX Swing 1D",
            "TEST_SHORT": "TEST SHORT",
            "TEST_SPOT_OCO": "TEST SPOT OCO"
        }
        display_strategy = display_names.get(strategy, strategy)
        
        # Convert any NumPy floats to Python floats for string formatting
        price = float(price) if price else 0.0
        quantity = float(quantity) if quantity else 0.0
        stop_loss = float(stop_loss) if stop_loss else None
        take_profit = float(take_profit) if take_profit else None
        
        if success:
            emoji = "🟢" if side == "long" else "🔴"
            status = "TRADE OPENED"
        else:
            emoji = "⚠️"
            status = "TRADE FAILED"
        
        # Format market type
        market_label = "📈 Futures" if market_type == "futures" else "📊 Spot"
        
        # Build SL/TP section
        sl_tp_info = ""
        if stop_loss and take_profit:
            protection = "OCO" if market_type == "spot" else "STOP/TP"
            sl_tp_info = f"""
🛡️ *Protection:* {protection}
📉 Stop-Loss: ${stop_loss:.4f}
📈 Take-Profit: ${take_profit:.4f}"""
        
        # Get Sydney time
        try:
            from datetime import timezone, timedelta
            sydney_tz = timezone(timedelta(hours=11))  # AEDT (UTC+11)
            sydney_time = datetime.now(sydney_tz).strftime('%H:%M:%S')
        except Exception:
            sydney_time = datetime.now().strftime('%H:%M:%S')
        
        message = f"""
{emoji} *{status}* (#{trade_id})

📋 Strategy: `{display_strategy}`
💰 Symbol: {symbol}
↕️ Side: *{side.upper()}*
{market_label}

💵 Entry: *${price:.4f}*
📦 Quantity: {quantity:.4f}{sl_tp_info}

⏰ Time: {sydney_time}
"""
        result = self.send_message(message)
        logger.info(f"📱 Telegram: notify_trade_opened result={result}")

    def notify_trade_closed(self, trade_id: int, strategy: str, symbol: str, side: str,
                            entry_price: float, exit_price: float,
                            pnl_pct: float, pnl_usd: float, reason: str):
        """Notify when a trade is closed with full details"""
        emoji = "✅" if pnl_pct > 0 else "❌"
        profit_emoji = "📈" if pnl_pct > 0 else "📉"
        
        # Strategy display name mapping
        display_names = {
            "ScalpingHybrid_DOGE": "DOGE Scalper 4H",
            "LLM_v4_LowDD": "Momentum Pro 4H",
            "LLM_v3_Tight": "Trend Hunter 4H", 
            "ScalpingHybrid_AVAX": "AVAX Swing 1D"
        }
        display_strategy = display_names.get(strategy, strategy)
        
        # Get Sydney time
        try:
            from datetime import timezone, timedelta
            sydney_tz = timezone(timedelta(hours=11))  # AEDT (UTC+11)
            sydney_time = datetime.now(sydney_tz).strftime('%H:%M:%S')
        except Exception:
            sydney_time = datetime.now().strftime('%H:%M:%S')
        
        message = f"""
{emoji} *TRADE CLOSED* (#{trade_id})

📋 Strategy: `{display_strategy}`
💰 Symbol: {symbol}
↕️ Side: {side.upper()}

💵 Entry: ${entry_price:.4f}
🏁 Exit: ${exit_price:.4f}

{profit_emoji} *P&L: {pnl_pct:+.2f}% (${pnl_usd:+.2f})*
📝 Reason: {reason}

⏰ Time: {sydney_time}
"""
        self.send_message(message)
    
    def send_status_update(self):
        """Send current strategy status"""
        if not self.db:
            return
        
        logs = self.db.get_latest_logs(limit=4)
        
        status_lines = ["📊 *STRATEGY STATUS*\n"]
        for log in logs:
            status_lines.append(
                f"*{log.strategy}*\n"
                f"  {log.symbol} @ ${log.price:.4f}\n"
                f"  RSI: {log.rsi:.1f} | ADX: {log.adx:.1f}\n"
                f"  Status: {log.status}\n"
            )
        
        self.send_message("\n".join(status_lines))
    
    def send_summary(self):
        """Send trading summary"""
        if not self.db:
            return
        
        trades = self.db.get_trades(limit=1000)
        closed = [t for t in trades if not t.is_open]
        total_pnl = sum(t.pnl_usd or 0 for t in closed)
        wins = len([t for t in closed if t.pnl_pct and t.pnl_pct > 0])
        
        message = f"""
📈 *TRADING SUMMARY*

Total Trades: {len(closed)}
Wins: {wins}
Win Rate: {(wins/len(closed)*100) if closed else 0:.1f}%
Total PnL: ${total_pnl:+.2f}
Current Equity: ${15000 + total_pnl:.2f}

_Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_
"""
        self.send_message(message)
    
    def process_command(self, chat_id: str, text: str) -> Optional[str]:
        """Process incoming commands (only from authorized chat)"""
        
        # Security check - only respond to your chat
        if not self._is_authorized(chat_id):
            logger.warning(f"Unauthorized access attempt from chat_id: {chat_id}")
            return None  # Silently ignore unauthorized requests
        
        command = text.lower().strip()
        
        if command in ["/start", "/help"]:
            return """
🤖 *Trading Bot Commands*

/dashboard - 📊 Open full dashboard
/status - Current strategy status
/summary - Trading performance summary
/trades - Recent trades
/balance - Account balance
/weekly - Weekly report

_Your private trading assistant_
"""
        
        elif command == "/dashboard":
            self.send_dashboard_link()
            return None
        
        elif command == "/status":
            self.send_status_update()
            return None
        
        elif command == "/summary":
            self.send_summary()
            return None
        
        elif command == "/trades":
            if self.db:
                trades = self.db.get_trades(limit=5)
                if not trades:
                    return "No trades yet."
                
                lines = ["📜 *RECENT TRADES*\n"]
                for t in trades:
                    emoji = "✅" if t.pnl_pct and t.pnl_pct > 0 else "❌" if t.pnl_pct else "⏳"
                    lines.append(
                        f"{emoji} {t.strategy}\n"
                        f"   {t.symbol} {t.side} @ ${t.entry_price:.4f}\n"
                        f"   PnL: {t.pnl_pct:+.2f}%\n" if t.pnl_pct else ""
                    )
                return "\n".join(lines)
        
        elif command == "/balance":
            return f"""
💰 *ACCOUNT BALANCE*

Starting: $15,000
Demo Mode: Binance Testnet

_Check /summary for current equity_
"""
        
        elif command == "/weekly":
            return "Generating weekly report... Check the dashboard for full report."
        
        return f"Unknown command: {command}\nType /help for available commands."


# Singleton instance
_bot_instance = None

def get_telegram_bot(db=None) -> TelegramBot:
    """Get or create Telegram bot instance"""
    global _bot_instance
    if _bot_instance is None:
        _bot_instance = TelegramBot(db)
    elif db and not _bot_instance.db:
        _bot_instance.db = db
    return _bot_instance
