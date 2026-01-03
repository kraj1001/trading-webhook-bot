#!/bin/bash
# Azure Container Apps Deployment Script
# Run this after setting up Azure CLI with: az login

# ====== CONFIGURATION ======
RESOURCE_GROUP="trading-bot-rg"
LOCATION="australiaeast"  # Choose closest region
CONTAINER_APP_NAME="trading-bot"
CONTAINER_ENV="trading-bot-env"
ACR_NAME="tradingbotacr$(date +%s)"  # Unique name

echo "🚀 Trading Bot Azure Deployment"
echo "================================"

# Step 1: Create Resource Group
echo "1. Creating Resource Group..."
az group create --name $RESOURCE_GROUP --location $LOCATION

# Step 2: Create Azure Container Registry
echo "2. Creating Container Registry..."
az acr create --resource-group $RESOURCE_GROUP --name $ACR_NAME --sku Basic --admin-enabled true

# Step 3: Login to ACR
echo "3. Logging into Container Registry..."
az acr login --name $ACR_NAME
ACR_SERVER=$(az acr show --name $ACR_NAME --query loginServer -o tsv)

# Step 4: Build and Push Docker Image
echo "4. Building and pushing Docker image..."
docker build -t $ACR_SERVER/trading-bot:latest .
docker push $ACR_SERVER/trading-bot:latest

# Step 5: Create Container Apps Environment
echo "5. Creating Container Apps Environment..."
az containerapp env create \
  --name $CONTAINER_ENV \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION

# Step 6: Get ACR credentials
ACR_USERNAME=$(az acr credential show --name $ACR_NAME --query username -o tsv)
ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query passwords[0].value -o tsv)

# Step 7: Deploy Container App
echo "6. Deploying Container App..."
az containerapp create \
  --name $CONTAINER_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --environment $CONTAINER_ENV \
  --image $ACR_SERVER/trading-bot:latest \
  --target-port 8000 \
  --ingress external \
  --cpu 0.5 \
  --memory 1.0Gi \
  --min-replicas 1 \
  --max-replicas 1 \
  --registry-server $ACR_SERVER \
  --registry-username $ACR_USERNAME \
  --registry-password $ACR_PASSWORD \
  --env-vars \
    "TRADING_MODE=testnet" \
    "BINANCE_FUTURES_API_KEY=secretref:binance-futures-key" \
    "BINANCE_FUTURES_SECRET=secretref:binance-futures-secret" \
    "BINANCE_SPOT_API_KEY=secretref:binance-spot-key" \
    "BINANCE_SPOT_SECRET=secretref:binance-spot-secret"

# Step 8: Get App URL
APP_URL=$(az containerapp show --name $CONTAINER_APP_NAME --resource-group $RESOURCE_GROUP --query properties.configuration.ingress.fqdn -o tsv)

echo ""
echo "================================"
echo "✅ Deployment Complete!"
echo "================================"
echo ""
echo "Dashboard URL: https://$APP_URL"
echo ""
echo "Next Steps:"
echo "1. Add secrets for API keys:"
echo "   az containerapp secret set --name $CONTAINER_APP_NAME --resource-group $RESOURCE_GROUP --secrets binance-futures-key=YOUR_KEY binance-futures-secret=YOUR_SECRET ..."
echo ""
echo "2. Monitor logs:"
echo "   az containerapp logs show --name $CONTAINER_APP_NAME --resource-group $RESOURCE_GROUP --follow"
