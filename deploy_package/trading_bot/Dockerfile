FROM python:3.11-slim

WORKDIR /app

# Install dependencies
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

# Force rebuild when source changes
ARG CACHE_BUST=1
# Copy application into trading_bot package directory
COPY . ./trading_bot/

# Expose dashboard port
EXPOSE 8000

# Run both bot and dashboard
CMD ["sh", "-c", "python -m trading_bot.dashboard.api & python -m trading_bot.main --mode testnet"]
