# Trading Bot

Automated crypto trading system with 4 verified strategies.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your API keys

# Run in paper mode
python -m trading_bot.main --mode paper

# Run in live mode (after testing!)
python -m trading_bot.main --mode live
```

## Binance Setup Requirements

### 1. Create API Keys

1. Log into [Binance](https://www.binance.com)
2. Go to **API Management**
3. Create new API key with:
   - ✅ Enable Reading
   - ✅ Enable Spot Trading
   - ✅ Enable Futures Trading
   - ❌ Disable Withdrawals (safety)

### 2. IP Whitelist (Recommended)

Add your Azure server IP to the whitelist for security.

### 3. Futures Account

1. Enable Futures trading in Binance
2. Transfer USDT to Futures wallet
3. Set leverage to 1x (safer for paper testing)

## Environment Variables

```env
TRADING_MODE=paper          # paper or live
BINANCE_API_KEY=xxx
BINANCE_API_SECRET=xxx
BINANCE_TESTNET=true        # Use testnet for paper trading
AZURE_OPENAI_KEY=xxx        # For LLM analysis
TELEGRAM_BOT_TOKEN=xxx      # For notifications
TELEGRAM_CHAT_ID=xxx
DATABASE_URL=postgresql://...
```

## Strategies

| Strategy | Symbol | TF | Type |
|----------|--------|-----|------|
| ScalpingHybrid | DOGEUSDT | 4H | Spot Long |
| LLM v4 Low DD | XRPUSDT.P | 4H | Futures L+S |
| LLM v3 Tight | XRPUSDT.P | 4H | Futures L+S |
| ScalpingHybrid | AVAXUSDT | Daily | Spot Long |
