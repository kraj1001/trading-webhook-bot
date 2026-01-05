#!/usr/bin/env python3
"""
Database Migration: Add new indicator columns to strategy_logs and trades tables
"""
from sqlalchemy import create_engine, text
import os

db_url = os.environ.get('DATABASE_URL', 'postgresql://tradingadmin:TradingBot2026!@tradingbotdb.postgres.database.azure.com/trades?sslmode=require')
engine = create_engine(db_url)

migrations = [
    # Strategy logs new columns
    'ALTER TABLE strategy_logs ADD COLUMN IF NOT EXISTS macd FLOAT',
    'ALTER TABLE strategy_logs ADD COLUMN IF NOT EXISTS macd_signal FLOAT',
    'ALTER TABLE strategy_logs ADD COLUMN IF NOT EXISTS ema_15 FLOAT',
    'ALTER TABLE strategy_logs ADD COLUMN IF NOT EXISTS ema_30 FLOAT',
    'ALTER TABLE strategy_logs ADD COLUMN IF NOT EXISTS ema_200 FLOAT',
    'ALTER TABLE strategy_logs ADD COLUMN IF NOT EXISTS volume FLOAT',
    'ALTER TABLE strategy_logs ADD COLUMN IF NOT EXISTS volume_ma FLOAT',
    'ALTER TABLE strategy_logs ADD COLUMN IF NOT EXISTS volume_pct FLOAT',
    'ALTER TABLE strategy_logs ADD COLUMN IF NOT EXISTS conditions_met VARCHAR(100)',
    # Trades new columns
    'ALTER TABLE trades ADD COLUMN IF NOT EXISTS market_type VARCHAR(10)',
    'ALTER TABLE trades ADD COLUMN IF NOT EXISTS stop_loss_price FLOAT',
    'ALTER TABLE trades ADD COLUMN IF NOT EXISTS take_profit_price FLOAT',
]

print(f"Connecting to: {db_url.split('@')[1].split('/')[0]}...")

with engine.connect() as conn:
    for sql in migrations:
        try:
            col_name = sql.split('ADD COLUMN IF NOT EXISTS ')[1].split()[0]
            conn.execute(text(sql))
            print(f'✓ Added {col_name}')
        except Exception as e:
            print(f'! Error: {e}')
    conn.commit()
    print('\n✅ Migration complete!')
