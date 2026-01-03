# Azure Deployment Guide

## Prerequisites

1. **Azure CLI** installed: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli
2. **Docker** installed: https://docs.docker.com/get-docker/
3. **Azure Account** with active subscription

## Quick Deploy (5 minutes)

### Step 1: Login to Azure
```bash
az login
```

### Step 2: Run Deployment Script
```bash
cd trading_bot
chmod +x deploy-azure.sh
./deploy-azure.sh
```

### Step 3: Add API Key Secrets
After deployment, add your Binance API keys as secrets:

```bash
az containerapp secret set \
  --name trading-bot \
  --resource-group trading-bot-rg \
  --secrets \
    binance-futures-key=7d1ngilT0WjdaR7tW8A1ugOOEl9jBPNV5vxZ2kQxSYbcL3r5wBaTCPnSvHz6sH7k \
    binance-futures-secret=1urV98LXQW4KnkZfinoDlXGA0wuedhYvoRkSGGZAgVD7sAx7Pg91CE6U0ZDDNSRt \
    binance-spot-key=1QAUPhyeVCJjmblWUmkHiGaK0Af3TmHPzbSO5NIIqFhApj0iiOLjz6lRnCHJYhJ1 \
    binance-spot-secret=o4iRoI5n3MLZUdAHXiCBfinRURR6Jao7PviyCyRG4I1NTEzdZQoBvG7QU0xjzNhU
```

### Step 4: Restart to Apply Secrets
```bash
az containerapp revision restart \
  --name trading-bot \
  --resource-group trading-bot-rg
```

## View Dashboard

After deployment, your dashboard will be available at:
```
https://trading-bot.<random>.australiaeast.azurecontainerapps.io
```

## Monitor Logs

```bash
az containerapp logs show \
  --name trading-bot \
  --resource-group trading-bot-rg \
  --follow
```

## Estimated Costs

| Resource | Cost/Month |
|----------|------------|
| Container App (0.5 vCPU, 1GB) | ~$15 |
| Container Registry (Basic) | ~$5 |
| **Total** | **~$20/month** |

## Cleanup (if needed)

```bash
az group delete --name trading-bot-rg --yes
```
