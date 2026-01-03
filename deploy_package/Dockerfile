FROM python:3.11-slim

ENV REVISION=v55

# Install dependencies (do this first for caching)
RUN pip install --no-cache-dir \
    ccxt \
    pandas \
    numpy \
    python-dotenv \
    sqlalchemy \
    psycopg2-binary \
    fastapi \
    uvicorn \
    requests

# Set working directory
WORKDIR /app

# Copy the trading_bot package
COPY trading_bot /app/trading_bot

# Set PYTHONPATH to /app so python can find the package
ENV PYTHONPATH=/app

# Expose dashboard port
EXPOSE 8000

# Run both bot and dashboard - 1 hour interval for production
CMD ["sh", "-c", "cd /app && python -m trading_bot.dashboard.api & python -m trading_bot.main --mode testnet --interval 3600"]
