# Telegram Bot Setup Guide

Private mobile access to your trading bot via Telegram.

## Security

✅ **Your bot only responds to YOUR chat ID** - unauthorized users are silently ignored.

## Step 1: Create a Telegram Bot

1. Open Telegram and message **@BotFather**
2. Send `/newbot`
3. Follow the prompts to name your bot
4. Copy the **bot token** provided (looks like `123456789:ABCdefGHI...`)

## Step 2: Get Your Chat ID

1. Message your new bot (send any message)
2. Visit this URL in your browser (replace TOKEN with your bot token):
   ```
   https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
   ```
3. Look for `"chat":{"id":123456789}` - this number is your chat ID

## Step 3: Configure Azure Container App

Run these commands to add the environment variables:

```bash
az containerapp update \
  --name trading-bot \
  --resource-group trading-bot-rg \
  --set-env-vars \
    TELEGRAM_BOT_TOKEN=your_bot_token_here \
    TELEGRAM_CHAT_ID=your_chat_id_here
```

## Step 4: Set Webhook

Tell Telegram to send updates to your Azure bot:

```bash
curl "https://api.telegram.org/bot<YOUR_TOKEN>/setWebhook?url=https://trading-bot.braveocean-cb90440a.australiaeast.azurecontainerapps.io/api/telegram/webhook"
```

## Available Commands

| Command | Description |
|---------|-------------|
| `/status` | Current strategy status (RSI, ADX) |
| `/summary` | Trading performance summary |
| `/trades` | Recent trade list |
| `/balance` | Account balance info |
| `/weekly` | Weekly report |
| `/help` | Show all commands |

## Automatic Notifications

The bot will automatically notify you when:
- 🟢 Trade opened (entry)
- ✅ Trade closed with profit
- ❌ Trade closed with loss

## API Endpoints

- `POST /api/telegram/webhook` - Receives Telegram updates
- `GET /api/telegram/status` - Manually trigger status message
- `GET /api/telegram/summary` - Manually trigger summary message
- `GET /api/telegram/setup` - View setup status
