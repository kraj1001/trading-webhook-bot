# Binance Testnet Setup Guide

## What is Testnet?

Binance Testnet is a **sandbox environment** with:
- ✅ Free fake USDT (100,000+)
- ✅ Real API (same as live)
- ✅ Visual UI showing all trades
- ✅ No risk to real money

---

## Step 1: Access Testnet

### For Futures (XRP strategies):
1. Go to: **https://testnet.binancefuture.com**
2. Click "Log In with GitHub"
3. Authorize the GitHub connection

### For Spot (DOGE, AVAX strategies):
1. Go to: **https://testnet.binance.vision**
2. Click "Log In with GitHub"

---

## Step 2: Get Free Testnet USDT

### Futures Testnet:
- You automatically get **100,000 USDT** on login!
- Check your balance in the testnet dashboard

### Spot Testnet:
- Click "Generate" to get testnet coins

---

## Step 3: Create API Keys

### Futures Testnet:
1. Go to: https://testnet.binancefuture.com/en/futures/BTCUSDT
2. Click your profile icon (top right)
3. Click "API Management"
4. Click "Create API"
5. Copy the **API Key** and **Secret Key**

### Spot Testnet:
1. Go to: https://testnet.binance.vision
2. Click "Generate HMAC_SHA256 Key"
3. Copy the keys

---

## Step 4: Add Keys to Bot

Create `.env` file in `trading_bot/`:

```env
TRADING_MODE=testnet

# Futures Testnet Keys
BINANCE_FUTURES_API_KEY=your_futures_testnet_api_key
BINANCE_FUTURES_SECRET=your_futures_testnet_secret

# Spot Testnet Keys  
BINANCE_SPOT_API_KEY=your_spot_testnet_api_key
BINANCE_SPOT_SECRET=your_spot_testnet_secret

INITIAL_CAPITAL=15000
```

---

## Step 5: Run Bot

```bash
cd trading_bot
python -m trading_bot.main --mode testnet
```

---

## See Your Trades!

After the bot executes trades:

1. **Futures**: Go to https://testnet.binancefuture.com
   - Click "Orders" to see open orders
   - Click "Positions" to see active positions
   - Click "Trade History" for completed trades

2. **Spot**: Go to https://testnet.binance.vision
   - Check "Order History"
   - Check "Trade History"

---

## Testnet URLs

| Type | URL |
|------|-----|
| Futures UI | https://testnet.binancefuture.com |
| Futures API | https://testnet.binancefuture.com |
| Spot UI | https://testnet.binance.vision |
| Spot API | https://testnet.binance.vision |
