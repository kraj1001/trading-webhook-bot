"""
Trading Bot Dashboard API
FastAPI backend for trade analytics
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
from typing import List, Optional
import os
import sys

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_bot.database import TradeDatabase, TradeRecord

app = FastAPI(
    title="Trading Bot Dashboard",
    description="API for monitoring trading strategies",
    version="1.0.0"
)

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database
db = TradeDatabase()


from fastapi.responses import FileResponse

@app.get("/")
def root():
    """Serve the dashboard UI"""
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))


@app.get("/api/strategies")
def get_strategies():
    """Get list of all strategies with summary stats"""
    strategies = [
        {"name": "ScalpingHybrid_DOGE", "symbol": "DOGEUSDT", "timeframe": "4h", "market": "spot"},
        {"name": "LLM_v4_LowDD", "symbol": "XRPUSDT", "timeframe": "4h", "market": "futures"},
        {"name": "LLM_v3_Tight", "symbol": "XRPUSDT", "timeframe": "4h", "market": "futures"},
        {"name": "ScalpingHybrid_AVAX", "symbol": "AVAXUSDT", "timeframe": "1d", "market": "spot"},
    ]
    
    for strategy in strategies:
        stats = db.get_strategy_stats(strategy["name"])
        strategy["stats"] = stats
    
    return strategies


@app.get("/api/trades")
def get_trades(
    strategy: Optional[str] = None,
    limit: int = 100,
    days: Optional[int] = None
):
    """Get trade history"""
    start_date = None
    if days:
        start_date = datetime.now() - timedelta(days=days)
    
    trades = db.get_trades(strategy=strategy, start_date=start_date, limit=limit)
    
    return [
        {
            "id": t.id,
            "strategy": t.strategy,
            "symbol": t.symbol,
            "side": t.side,
            "entry_time": t.entry_time.isoformat() if t.entry_time else None,
            "exit_time": t.exit_time.isoformat() if t.exit_time else None,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "quantity": t.quantity,
            "pnl_usd": t.pnl_usd,
            "pnl_pct": t.pnl_pct,
            "exit_reason": t.exit_reason,
            "is_open": t.is_open,
        }
        for t in trades
    ]


@app.get("/api/stats/{strategy}")
def get_strategy_stats(strategy: str):
    """Get detailed stats for a strategy"""
    stats = db.get_strategy_stats(strategy)
    if not stats:
        raise HTTPException(status_code=404, detail="Strategy not found or no trades")
    return stats


@app.get("/api/logs")
def get_logs(limit: int = 50):
    """Get latest strategy check logs"""
    logs = db.get_latest_logs(limit=limit)
    return [
        {
            "id": l.id,
            "timestamp": l.timestamp.isoformat(),
            "strategy": l.strategy,
            "symbol": l.symbol,
            "status": l.status,
            "message": l.message,
            "price": l.price,
            "rsi": l.rsi,
            "adx": l.adx,
        }
        for l in logs
    ]


@app.get("/api/training-data")
def get_training_data(limit: int = 100):
    """Get LLM training data records"""
    records = db.get_training_data(limit=limit)
    return records


@app.get("/api/training-data/export")
def export_training_data(format: str = "jsonl"):
    """Export training data as JSONL for fine-tuning"""
    from fastapi.responses import PlainTextResponse
    import json
    
    records = db.get_training_data(limit=10000)
    
    if format == "jsonl":
        lines = []
        for r in records:
            line = json.dumps({
                "instruction": r.get("instruction", ""),
                "output": r.get("output", "")
            })
            lines.append(line)
        
        content = "\n".join(lines)
        return PlainTextResponse(
            content=content,
            media_type="application/jsonl",
            headers={"Content-Disposition": "attachment; filename=training_data.jsonl"}
        )
    else:
        return {"error": "Unsupported format. Use format=jsonl"}


# ─────────────────────────────────────────────────────────────────────────────
# Weekly Report Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/weekly-report")
def generate_weekly_report(days: int = 7):
    """Generate a GPT-4o powered weekly performance report"""
    from ..llm_analysis import LLMAnalyzer
    import json
    
    llm = LLMAnalyzer()
    
    if not llm.enabled:
        return {"error": "LLM not configured. Azure OpenAI credentials missing."}
    
    # Get trades from the past week
    start_date = datetime.now() - timedelta(days=days)
    trades = db.get_trades(start_date=start_date, limit=500)
    
    # Get strategy logs for context
    logs = db.get_latest_logs(limit=200)
    
    # Compile trade summary
    trades_data = []
    for t in trades:
        trades_data.append({
            "strategy": t.strategy,
            "symbol": t.symbol,
            "side": t.side,
            "pnl_pct": round(t.pnl_pct, 2) if t.pnl_pct else 0,
            "pnl_usd": round(t.pnl_usd, 2) if t.pnl_usd else 0,
            "entry_time": str(t.entry_time),
            "exit_time": str(t.exit_time) if t.exit_time else None,
            "entry_rsi": round(t.entry_rsi, 1) if t.entry_rsi else None,
            "entry_adx": round(t.entry_adx, 1) if t.entry_adx else None,
            "exit_reason": t.exit_reason
        })
    
    # Compile strategy check summary
    strategy_summary = {}
    for log in logs:
        if log.strategy not in strategy_summary:
            strategy_summary[log.strategy] = {
                "checks": 0,
                "signals": 0,
                "avg_rsi": [],
                "avg_adx": []
            }
        strategy_summary[log.strategy]["checks"] += 1
        if log.status != "WAITING":
            strategy_summary[log.strategy]["signals"] += 1
        if log.rsi:
            strategy_summary[log.strategy]["avg_rsi"].append(log.rsi)
        if log.adx:
            strategy_summary[log.strategy]["avg_adx"].append(log.adx)
    
    # Calculate averages
    for name, data in strategy_summary.items():
        data["avg_rsi"] = round(sum(data["avg_rsi"]) / len(data["avg_rsi"]), 1) if data["avg_rsi"] else 0
        data["avg_adx"] = round(sum(data["avg_adx"]) / len(data["avg_adx"]), 1) if data["avg_adx"] else 0
    
    # Generate LLM report
    if trades_data:
        report = llm.generate_weekly_report(trades_data)
    else:
        # If no trades, generate market condition report
        report = generate_no_trades_report(strategy_summary, days)
    
    return {
        "period": f"Last {days} days",
        "generated_at": datetime.now().isoformat(),
        "total_trades": len(trades_data),
        "trades": trades_data,
        "strategy_activity": strategy_summary,
        "llm_report": report
    }


def generate_no_trades_report(strategy_summary: dict, days: int) -> str:
    """Generate a report when no trades occurred"""
    report_lines = [
        f"# Weekly Trading Report ({days} Days)",
        "",
        "## Summary",
        "**No trades executed during this period.**",
        "",
        "## Market Conditions",
        "The market indicators did not meet our strict entry criteria:",
        ""
    ]
    
    for name, data in strategy_summary.items():
        report_lines.append(f"### {name}")
        report_lines.append(f"- Strategy checks: {data['checks']}")
        report_lines.append(f"- Average RSI: {data['avg_rsi']}")
        report_lines.append(f"- Average ADX: {data['avg_adx']}")
        
        # Explain why no trades
        if data['avg_rsi'] > 50:
            report_lines.append(f"- ⚠️ RSI averaging {data['avg_rsi']} (needs < 35 for entry)")
        if data['avg_adx'] < 25:
            report_lines.append(f"- ⚠️ ADX averaging {data['avg_adx']} (needs > 25 for trend confirmation)")
        report_lines.append("")
    
    report_lines.extend([
        "## Recommendation",
        "Market is in **overbought/ranging conditions**. Waiting for pullback with stronger trend signal.",
        "",
        "Entry will trigger when:",
        "- RSI drops below 30-35 (oversold)",
        "- ADX rises above 25-30 (strong trend)"
    ])
    
    return "\n".join(report_lines)


@app.get("/api/weekly-report/summary")
def get_weekly_summary(days: int = 7):
    """Get a quick summary without LLM analysis"""
    start_date = datetime.now() - timedelta(days=days)
    trades = db.get_trades(start_date=start_date, limit=500)
    
    # Calculate stats
    total_pnl = sum(t.pnl_usd or 0 for t in trades if not t.is_open)
    wins = [t for t in trades if t.pnl_pct and t.pnl_pct > 0]
    losses = [t for t in trades if t.pnl_pct and t.pnl_pct <= 0]
    
    by_strategy = {}
    for t in trades:
        if t.strategy not in by_strategy:
            by_strategy[t.strategy] = {"trades": 0, "pnl": 0, "wins": 0, "losses": 0}
        by_strategy[t.strategy]["trades"] += 1
        by_strategy[t.strategy]["pnl"] += t.pnl_usd or 0
        if t.pnl_pct and t.pnl_pct > 0:
            by_strategy[t.strategy]["wins"] += 1
        elif t.pnl_pct:
            by_strategy[t.strategy]["losses"] += 1
    
    return {
        "period": f"Last {days} days",
        "total_trades": len(trades),
        "total_pnl_usd": round(total_pnl, 2),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(trades) * 100, 1) if trades else 0,
        "by_strategy": by_strategy
    }


@app.get("/api/weekly-report/html")
def get_weekly_report_html(days: int = 7):
    """Get a human-readable HTML weekly report"""
    from fastapi.responses import HTMLResponse
    
    start_date = datetime.now() - timedelta(days=days)
    trades = db.get_trades(start_date=start_date, limit=500)
    
    # Calculate stats
    closed_trades = [t for t in trades if not t.is_open]
    total_pnl = sum(t.pnl_usd or 0 for t in closed_trades)
    wins = [t for t in closed_trades if t.pnl_pct and t.pnl_pct > 0]
    losses = [t for t in closed_trades if t.pnl_pct and t.pnl_pct <= 0]
    win_rate = (len(wins) / len(closed_trades) * 100) if closed_trades else 0
    
    # By strategy breakdown
    by_strategy = {}
    for t in closed_trades:
        if t.strategy not in by_strategy:
            by_strategy[t.strategy] = {"trades": 0, "pnl": 0, "wins": 0}
        by_strategy[t.strategy]["trades"] += 1
        by_strategy[t.strategy]["pnl"] += t.pnl_usd or 0
        if t.pnl_pct and t.pnl_pct > 0:
            by_strategy[t.strategy]["wins"] += 1
    
    # Generate HTML
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Weekly Trading Report</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
            color: #c9d1d9;
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{ max-width: 800px; margin: 0 auto; }}
        h1 {{ 
            color: #58a6ff; 
            font-size: 28px; 
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .subtitle {{ color: #8b949e; margin-bottom: 30px; }}
        .card {{
            background: rgba(22, 27, 34, 0.8);
            border: 1px solid #30363d;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        .card-title {{ 
            color: #58a6ff; 
            font-size: 16px; 
            font-weight: 600;
            margin-bottom: 15px;
            border-bottom: 1px solid #30363d;
            padding-bottom: 10px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
        }}
        .stat {{
            text-align: center;
            padding: 15px;
            background: rgba(13, 17, 23, 0.5);
            border-radius: 8px;
        }}
        .stat-value {{ 
            font-size: 24px; 
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .stat-label {{ color: #8b949e; font-size: 12px; text-transform: uppercase; }}
        .green {{ color: #3fb950; }}
        .red {{ color: #f85149; }}
        .blue {{ color: #58a6ff; }}
        .strategy-row {{
            display: flex;
            justify-content: space-between;
            padding: 12px 0;
            border-bottom: 1px solid #21262d;
        }}
        .strategy-row:last-child {{ border-bottom: none; }}
        .strategy-name {{ font-weight: 500; }}
        .trade-list {{ list-style: none; }}
        .trade-item {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #21262d;
            font-size: 14px;
        }}
        .trade-item:last-child {{ border-bottom: none; }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            color: #8b949e;
            font-size: 12px;
        }}
        .footer a {{ color: #58a6ff; text-decoration: none; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Weekly Trading Report</h1>
        <p class="subtitle">Period: {start_date.strftime('%Y-%m-%d')} to {datetime.now().strftime('%Y-%m-%d')} ({days} days)</p>
        
        <div class="card">
            <div class="card-title">📈 Performance Summary</div>
            <div class="stats-grid">
                <div class="stat">
                    <div class="stat-value blue">{len(closed_trades)}</div>
                    <div class="stat-label">Total Trades</div>
                </div>
                <div class="stat">
                    <div class="stat-value {'green' if total_pnl >= 0 else 'red'}">${total_pnl:+.2f}</div>
                    <div class="stat-label">Total P&L</div>
                </div>
                <div class="stat">
                    <div class="stat-value green">{len(wins)}</div>
                    <div class="stat-label">Wins</div>
                </div>
                <div class="stat">
                    <div class="stat-value red">{len(losses)}</div>
                    <div class="stat-label">Losses</div>
                </div>
                <div class="stat">
                    <div class="stat-value {'green' if win_rate >= 50 else 'red'}">{win_rate:.1f}%</div>
                    <div class="stat-label">Win Rate</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <div class="card-title">🎯 Strategy Breakdown</div>
            {"".join(f'''
            <div class="strategy-row">
                <span class="strategy-name">{name}</span>
                <span>
                    <span class="blue">{data["trades"]} trades</span> | 
                    <span class="{'green' if data['pnl'] >= 0 else 'red'}">${data['pnl']:+.2f}</span> |
                    <span class="green">{data['wins']} wins</span>
                </span>
            </div>
            ''' for name, data in by_strategy.items()) if by_strategy else '<p style="color: #8b949e;">No strategies traded this period.</p>'}
        </div>
        
        <div class="card">
            <div class="card-title">📜 Recent Trades</div>
            <ul class="trade-list">
            {"".join(f'''
            <li class="trade-item">
                <span>{t.strategy} | {t.symbol} {t.side}</span>
                <span class="{'green' if t.pnl_pct and t.pnl_pct > 0 else 'red'}">{t.pnl_pct:+.2f}% (${t.pnl_usd:+.2f})</span>
            </li>
            ''' for t in closed_trades[:10]) if closed_trades else '<li style="color: #8b949e; padding: 10px 0;">No closed trades this period.</li>'}
            </ul>
        </div>
        
        <div class="footer">
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
            <p><a href="/">← Back to Dashboard</a></p>
        </div>
    </div>
</body>
</html>
"""
    return HTMLResponse(content=html)


# ─────────────────────────────────────────────────────────────────────────────
# On-Demand LLM Analysis Endpoints
# ─────────────────────────────────────────────────────────────────────────────

from ..llm_analysis import LLMAnalyzer
from ..exchange import get_exchange

llm = LLMAnalyzer()

@app.get("/api/llm/market-regime")
def analyze_market_regime_now(symbol: str = "XRPUSDT", timeframe: str = "4h"):
    """On-demand Market Regime Detection using GPT-4o"""
    if not llm.enabled:
        return {"error": "LLM not configured. Missing Azure OpenAI credentials."}
    
    try:
        exchange = get_exchange()
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=100)
        
        import pandas as pd
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Calculate indicators
        df['rsi'] = calculate_rsi(df['close'])
        df['ema_9'] = df['close'].ewm(span=9).mean()
        df['ema_15'] = df['close'].ewm(span=15).mean()
        df['ema_30'] = df['close'].ewm(span=30).mean()
        df['adx'] = calculate_adx(df)
        df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['atr'] = calculate_atr(df)
        
        market_data = {
            'symbol': symbol,
            'timeframe': timeframe,
            'price': float(df['close'].iloc[-1]),
            'rsi': float(df['rsi'].iloc[-1]),
            'adx': float(df['adx'].iloc[-1]) if 'adx' in df.columns else 20.0,
            'ema_9': float(df['ema_9'].iloc[-1]),
            'ema_15': float(df['ema_15'].iloc[-1]),
            'ema_30': float(df['ema_30'].iloc[-1]),
            'macd': float(df['macd'].iloc[-1]),
            'macd_signal': float(df['macd_signal'].iloc[-1]),
            'atr': float(df['atr'].iloc[-1]) if 'atr' in df.columns else 0.0,
            'high_10': float(df['high'].tail(10).max()),
            'low_10': float(df['low'].tail(10).min()),
            'price_change_pct': float((df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10] * 100)
        }
        
        result = llm.analyze_market_regime(market_data)
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "analysis": result
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/llm/candlestick-patterns")
def analyze_patterns_now(symbol: str = "XRPUSDT", timeframe: str = "4h"):
    """On-demand Candlestick Pattern Recognition using GPT-4o"""
    if not llm.enabled:
        return {"error": "LLM not configured. Missing Azure OpenAI credentials."}
    
    try:
        exchange = get_exchange()
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=20)
        
        import pandas as pd
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['rsi'] = calculate_rsi(df['close'])
        
        candle_data = {
            'symbol': symbol,
            'timeframe': timeframe,
            'candles': [
                {'open': float(row['open']), 'high': float(row['high']), 
                 'low': float(row['low']), 'close': float(row['close'])}
                for _, row in df.tail(10).iterrows()
            ],
            'rsi': float(df['rsi'].iloc[-1]),
            'near_support': float(df['rsi'].iloc[-1]) < 35,
            'near_resistance': float(df['rsi'].iloc[-1]) > 65,
            'volume_trend': 'high' if df['volume'].iloc[-1] > df['volume'].mean() else 'normal'
        }
        
        result = llm.analyze_candlestick_patterns(candle_data)
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "analysis": result
        }
    except Exception as e:
        return {"error": str(e)}


def calculate_rsi(prices, period: int = 14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_adx(df, period: int = 14):
    """Calculate ADX indicator"""
    try:
        high = df['high']
        low = df['low']
        close = df['close']
        
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        
        tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (-minus_dm.rolling(window=period).mean() / atr)
        
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        adx = dx.rolling(window=period).mean()
        return adx.fillna(20)
    except:
        return pd.Series([20] * len(df))


def calculate_atr(df, period: int = 14):
    """Calculate ATR indicator"""
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(window=period).mean().fillna(0)


@app.get("/api/equity")
def get_equity_curve(strategy: Optional[str] = None):
    """Get equity curve data for charting"""
    trades = db.get_trades(strategy=strategy, limit=1000)
    
    # Sort by entry time
    trades = sorted([t for t in trades if t.exit_time], key=lambda x: x.exit_time)
    
    equity = 15000  # Starting capital
    curve = [{"date": trades[0].entry_time.isoformat() if trades else datetime.now().isoformat(), "equity": equity}]
    
    for trade in trades:
        if trade.pnl_usd:
            equity += trade.pnl_usd
            curve.append({
                "date": trade.exit_time.isoformat(),
                "equity": round(equity, 2),
                "trade_id": trade.id,
                "pnl": trade.pnl_usd
            })
    
    return curve


@app.get("/api/summary")
def get_summary():
    """Get overall portfolio summary"""
    strategies = ["ScalpingHybrid_DOGE", "LLM_v4_LowDD", "LLM_v3_Tight", "ScalpingHybrid_AVAX"]
    
    total_trades = 0
    total_pnl = 0
    wins = 0
    losses = 0
    
    for strategy in strategies:
        stats = db.get_strategy_stats(strategy)
        if stats:
            total_trades += stats.get("total_trades", 0)
            total_pnl += stats.get("total_pnl_usd", 0)
            wins += stats.get("wins", 0)
            losses += stats.get("losses", 0)
    
    # Calculate active equity
    starting_capital = 60000
    current_equity = starting_capital + total_pnl

    # Try using real exchange balance if possible
    try:
        from trading_bot.exchange import get_exchange
        exchange = get_exchange()
        if hasattr(exchange, "fetch_balance"):
            bals = exchange.fetch_balance()
            total_usdt = bals.get("USDT", 0)
            # Add other assets approx value (simplified)
            current_equity = total_usdt + total_pnl 
    except Exception as e:
        print(f"Balance fetch failed: {e}")

    return {
        "total_trades": total_trades,
        "total_pnl_usd": round(total_pnl, 2),
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / total_trades * 100, 1) if total_trades > 0 else 0,
        "starting_capital": starting_capital,
        "current_equity": round(current_equity, 2)
    }


@app.get("/api/balance")
def get_exchange_balance():
    """Get current testnet balance from exchange"""
    try:
        from trading_bot.exchange import get_exchange
        exchange = get_exchange()
        
        # Get spot balance
        spot_balance = exchange.fetch_balance()
        
        # Format response with significant balances
        result = {
            "spot": {},
            "spot_total_usdt": 0
        }
        
        for coin, amount in spot_balance.items():
            if amount > 0:
                result["spot"][coin] = round(amount, 6)
                if coin == "USDT":
                    result["spot_total_usdt"] = round(amount, 2)
        
        return result
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/tax")
def get_tax_estimate(rate: float = 0.30):
    """Estimate tax liability based on realized PnL"""
    limit = 5000  # Analyze last 5000 trades
    trades = db.get_trades(limit=limit)
    
    # Filter for closed trades this year (assuming simplified view for now)
    current_year = datetime.now().year
    
    realized_pnl = 0.0
    short_term_gains = 0.0
    
    for trade in trades:
        if not trade.exit_time or trade.exit_time.year != current_year:
            continue
            
        pnl = trade.pnl_usd or 0
        realized_pnl += pnl
        
        # Simplified: All crypto trades often treated as income or CGT
        # We'll just apply the flat rate to net profit
    
    taxable_income = max(0, realized_pnl)
    estimated_tax = taxable_income * rate
    
    return {
        "year": current_year,
        "realized_pnl": round(realized_pnl, 2),
        "tax_rate": rate,
        "estimated_tax": round(estimated_tax, 2),
        "net_profit_after_tax": round(realized_pnl - estimated_tax, 2)
    }


# ─────────────────────────────────────────────────────────────────────────────
# TradingView-Style Chart Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/ohlcv")
def get_ohlcv(symbol: str = "DOGEUSDT", timeframe: str = "4h", limit: int = 300):
    """
    Fetch OHLCV data with calculated indicators for TradingView-style chart.
    Returns candlestick data + EMA, RSI, MACD, ADX overlays.
    """
    import pandas as pd
    import numpy as np
    
    def ema(series, period):
        return series.ewm(span=period, adjust=False).mean()
    
    def rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def macd(series, fast=12, slow=26, signal=9):
        ema_fast = ema(series, fast)
        ema_slow = ema(series, slow)
        macd_line = ema_fast - ema_slow
        signal_line = ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        # +DM and -DM
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        # +DI and -DI
        plus_di = 100 * ema(plus_dm, period) / atr
        minus_di = 100 * ema(minus_dm, period) / atr
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx_val = ema(dx, period)
        return adx_val
    
    def atr_calc(high, low, close, period=14):
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    try:
        from trading_bot.exchange import get_exchange
        exchange = get_exchange()
        
        # Fetch OHLCV from Binance
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Calculate indicators using pure pandas
        df['ema_9'] = ema(df['close'], 9)
        df['ema_15'] = ema(df['close'], 15)
        df['ema_30'] = ema(df['close'], 30)
        df['rsi'] = rsi(df['close'], 14)
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = macd(df['close'], 12, 26, 9)
        
        # ADX
        df['adx'] = adx(df['high'], df['low'], df['close'], 14)
        
        # ATR
        df['atr'] = atr_calc(df['high'], df['low'], df['close'], 14)
        
        # Fill NaN with 0 for JSON serialization
        df = df.fillna(0)
        
        # Convert to list of dicts for JSON
        result = []
        for _, row in df.iterrows():
            result.append({
                "time": int(row['timestamp'].timestamp()),
                "open": round(float(row['open']), 6),
                "high": round(float(row['high']), 6),
                "low": round(float(row['low']), 6),
                "close": round(float(row['close']), 6),
                "volume": round(float(row['volume']), 2),
                "ema_9": round(float(row['ema_9']), 6),
                "ema_15": round(float(row['ema_15']), 6),
                "ema_30": round(float(row['ema_30']), 6),
                "rsi": round(float(row['rsi']), 2),
                "macd": round(float(row['macd']), 6),
                "macd_signal": round(float(row['macd_signal']), 6),
                "macd_hist": round(float(row['macd_hist']), 6),
                "adx": round(float(row['adx']), 2),
                "atr": round(float(row['atr']), 6),
            })
        
        return {"symbol": symbol, "timeframe": timeframe, "data": result}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch OHLCV: {str(e)}")


@app.get("/api/chart/trades")
def get_chart_trades(symbol: str = "DOGEUSDT", strategy: Optional[str] = None, limit: int = 100):
    """
    Fetch trades formatted for chart markers with entry/exit annotations.
    Returns trade markers with timestamps, prices, and reasons.
    """
    trades = db.get_trades(strategy=strategy, limit=limit)
    
    # Filter by symbol if specified
    if symbol:
        symbol_clean = symbol.replace("/", "")
        trades = [t for t in trades if t.symbol and symbol_clean in t.symbol.replace("/", "")]
    
    markers = []
    
    for trade in trades:
        # Entry marker
        if trade.entry_time and trade.entry_price:
            markers.append({
                "time": int(trade.entry_time.timestamp()),
                "price": float(trade.entry_price),
                "type": "entry",
                "side": trade.side or "long",
                "strategy": trade.strategy,
                "reason": getattr(trade, 'entry_reason', None) or f"{trade.strategy} Entry",
                "color": "#3fb950" if trade.side == "long" else "#58a6ff",  # green / blue
                "shape": "arrowUp" if trade.side == "long" else "arrowDown",
            })
        
        # Exit marker
        if trade.exit_time and trade.exit_price:
            markers.append({
                "time": int(trade.exit_time.timestamp()),
                "price": float(trade.exit_price),
                "type": "exit",
                "side": trade.side or "long",
                "strategy": trade.strategy,
                "reason": trade.exit_reason or "Exit",
                "pnl": trade.pnl_pct,
                "color": "#f85149" if trade.side == "long" else "#a371f7",  # red / purple
                "shape": "arrowDown" if trade.side == "long" else "arrowUp",
            })
    
    # Sort by time
    markers.sort(key=lambda x: x["time"])
    
    return {"symbol": symbol, "strategy": strategy, "markers": markers}


@app.get("/api/symbols")
def get_available_symbols():
    """Get list of available symbols for charting"""
    return [
        {"symbol": "DOGEUSDT", "name": "Dogecoin"},
        {"symbol": "XRPUSDT", "name": "Ripple"},
        {"symbol": "AVAXUSDT", "name": "Avalanche"},
        {"symbol": "BTCUSDT", "name": "Bitcoin"},
        {"symbol": "ETHUSDT", "name": "Ethereum"},
        {"symbol": "SOLUSDT", "name": "Solana"},
        {"symbol": "ADAUSDT", "name": "Cardano"},
        {"symbol": "BNBUSDT", "name": "BNB"},
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Telegram Bot Endpoints (Private Mobile Access)
# ─────────────────────────────────────────────────────────────────────────────

from ..telegram_bot import get_telegram_bot, TELEGRAM_CHAT_ID
from fastapi import Request

telegram_bot = get_telegram_bot(db)

@app.post("/api/telegram/webhook")
async def telegram_webhook(request: Request):
    """Receive Telegram updates via webhook (only responds to authorized chat)"""
    try:
        data = await request.json()
        
        if "message" in data:
            message = data["message"]
            chat_id = str(message.get("chat", {}).get("id", ""))
            text = message.get("text", "")
            
            # Process command (telegram_bot checks authorization internally)
            response = telegram_bot.process_command(chat_id, text)
            
            if response:
                telegram_bot.send_message(response)
        
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/api/telegram/status")
def send_telegram_status():
    """Manually trigger status update to Telegram"""
    if not telegram_bot.enabled:
        return {"error": "Telegram not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID."}
    
    telegram_bot.send_status_update()
    return {"success": True, "message": "Status sent to Telegram"}


@app.get("/api/telegram/summary")
def send_telegram_summary():
    """Manually trigger summary to Telegram"""
    if not telegram_bot.enabled:
        return {"error": "Telegram not configured."}
    
    telegram_bot.send_summary()
    return {"success": True, "message": "Summary sent to Telegram"}


@app.get("/api/telegram/setup")
def telegram_setup_info():
    """Get Telegram bot setup instructions"""
    return {
        "setup_steps": [
            "1. Message @BotFather on Telegram",
            "2. Send /newbot and follow instructions",
            "3. Copy the bot token provided",
            "4. Message your new bot to start a chat",
            "5. Get your chat ID from api.telegram.org/bot<TOKEN>/getUpdates",
            "6. Set environment variables in Azure Container Apps:",
            "   - TELEGRAM_BOT_TOKEN=your_token",
            "   - TELEGRAM_CHAT_ID=your_chat_id",
            "7. Set webhook URL: https://trading-bot.braveocean-cb90440a.australiaeast.azurecontainerapps.io/api/telegram/webhook"
        ],
        "bot_enabled": telegram_bot.enabled,
        "authorized_chat": TELEGRAM_CHAT_ID[:4] + "***" if TELEGRAM_CHAT_ID else "Not set"
    }


# ─────────────────────────────────────────────────────────────────────────────
# Admin Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/admin/close-all-trades")
def close_all_open_trades(reason: str = "Test cleanup"):
    """Close all open trades (admin function for cleaning up test trades)"""
    open_trades = db.get_open_trades()
    
    closed_count = 0
    for trade in open_trades:
        try:
            # Get current price for exit
            from trading_bot.exchange import get_exchange
            exchange = get_exchange()
            ticker = exchange.fetch_ticker(trade.symbol)
            exit_price = ticker["last"]
            
            # Calculate PnL
            if trade.side == "long":
                pnl_pct = ((exit_price - trade.entry_price) / trade.entry_price) * 100
            else:
                pnl_pct = ((trade.entry_price - exit_price) / trade.entry_price) * 100
            
            pnl_usd = trade.quantity * trade.entry_price * (pnl_pct / 100)
            
            # Close the trade
            db.close_trade(trade.id, exit_price, reason, pnl_usd, pnl_pct)
            closed_count += 1
        except Exception as e:
            print(f"Error closing trade {trade.id}: {e}")
    
    return {
        "success": True,
        "closed_count": closed_count,
        "reason": reason,
        "message": f"Closed {closed_count} test trades"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

