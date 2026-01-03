# Binance Account Setup Guide

## Step 1: Create Binance Account

1. Go to [Binance](https://www.binance.com/en/register)
2. Sign up with email
3. Complete KYC verification (required for API access)
   - Takes 1-2 days in Australia

## Step 2: Enable Futures Trading

1. Go to **Derivatives** → **USDⓈ-M Futures**
2. Click **Open Now**
3. Complete quiz (easy questions)
4. Start with **1x leverage** (safest)

## Step 3: Create API Keys

1. Go to **Profile** → **API Management**
2. Click **Create API**
3. Choose **System Generated**
4. Set permissions:
   - ✅ Enable Reading
   - ✅ Enable Spot & Margin Trading
   - ✅ Enable Futures
   - ❌ Enable Withdrawals (keep OFF for safety)
5. **Save the API Key and Secret** (shown only once!)

## Step 4: (Optional) IP Whitelist

For extra security, whitelist your Azure server IP:
1. Edit the API key
2. Add your server's IP address
3. This prevents anyone else from using your API key

## Step 5: Fund Your Account

For paper trading: No funds needed (uses testnet)
For live trading:
1. Deposit AUD via PayID or bank transfer
2. Buy USDT
3. Transfer some to Futures wallet

## API Key Example

```
API Key: abc123...xyz789
Secret:  def456...uvw012
```

Store these in your `.env` file:
```
BINANCE_API_KEY=abc123...xyz789
BINANCE_API_SECRET=def456...uvw012
```

## Testnet (Paper Trading)

For paper trading, you can also use Binance Testnet:
1. Go to [testnet.binancefuture.com](https://testnet.binancefuture.com)
2. Create separate testnet API keys
3. Get free testnet USDT (fake money)

Set in `.env`:
```
BINANCE_TESTNET=true
```
