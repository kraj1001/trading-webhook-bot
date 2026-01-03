"""
Trading Bot MCP Server
Exposes trading bot functionality as MCP tools for AI agent interaction

Run with: python -m trading_bot.mcp_server
Configure in MCP settings to connect AI assistants to your live trading data.
"""
import json
import asyncio
from datetime import datetime, timedelta
from typing import Any
import os
import sys

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.models import InitializationOptions
from mcp.server import Server
import mcp.types as types
from mcp.server.stdio import stdio_server

from trading_bot.database import TradeDatabase
from trading_bot.config import STRATEGIES

# Initialize database
db = TradeDatabase()

# Create MCP server
server = Server("trading-bot")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    """List all available trading bot tools"""
    return [
        types.Tool(
            name="get_trading_summary",
            description="Get overall trading performance summary including total PnL, win rate, and equity",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        types.Tool(
            name="get_strategy_status",
            description="Get current status of all trading strategies including RSI, ADX, and signal status. Shows what each strategy is monitoring and current market conditions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "strategy": {
                        "type": "string",
                        "description": "Optional: Filter by specific strategy name (e.g., 'ScalpingHybrid_DOGE', 'LLM_v4_LowDD')"
                    }
                },
                "required": []
            }
        ),
        types.Tool(
            name="get_recent_trades",
            description="Get list of recent trades with entry/exit prices, PnL, and strategy details",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Number of trades to return (default: 20)"
                    },
                    "strategy": {
                        "type": "string",
                        "description": "Optional: Filter by strategy name"
                    }
                },
                "required": []
            }
        ),
        types.Tool(
            name="get_weekly_report",
            description="Generate a weekly performance report with trade analysis and strategy recommendations",
            inputSchema={
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "description": "Number of days to include in report (default: 7)"
                    }
                },
                "required": []
            }
        ),
        types.Tool(
            name="get_training_data_stats",
            description="Get statistics about LLM training data collected for fine-tuning the Momentum Expert model",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        types.Tool(
            name="get_market_regime",
            description="Analyze current market conditions (trending, ranging, volatile) for a symbol",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Trading pair (e.g., 'XRPUSDT', 'DOGEUSDT')"
                    }
                },
                "required": ["symbol"]
            }
        ),
        types.Tool(
            name="get_strategy_configuration",
            description="Get configuration details for all active trading strategies including allocation percentages",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    """Handle tool calls"""
    
    if name == "get_trading_summary":
        return await get_trading_summary()
    
    elif name == "get_strategy_status":
        strategy = arguments.get("strategy")
        return await get_strategy_status(strategy)
    
    elif name == "get_recent_trades":
        limit = arguments.get("limit", 20)
        strategy = arguments.get("strategy")
        return await get_recent_trades(limit, strategy)
    
    elif name == "get_weekly_report":
        days = arguments.get("days", 7)
        return await get_weekly_report(days)
    
    elif name == "get_training_data_stats":
        return await get_training_data_stats()
    
    elif name == "get_market_regime":
        symbol = arguments.get("symbol", "XRPUSDT")
        return await get_market_regime(symbol)
    
    elif name == "get_strategy_configuration":
        return await get_strategy_configuration()
    
    else:
        return [types.TextContent(type="text", text=f"Unknown tool: {name}")]


async def get_trading_summary() -> list[types.TextContent]:
    """Get overall trading summary"""
    trades = db.get_trades(limit=1000)
    closed_trades = [t for t in trades if not t.is_open]
    
    total_pnl = sum(t.pnl_usd or 0 for t in closed_trades)
    wins = [t for t in closed_trades if t.pnl_pct and t.pnl_pct > 0]
    losses = [t for t in closed_trades if t.pnl_pct and t.pnl_pct <= 0]
    
    # Get latest logs for current status
    logs = db.get_latest_logs(limit=4)
    
    summary = {
        "total_trades": len(closed_trades),
        "open_positions": len([t for t in trades if t.is_open]),
        "total_pnl_usd": round(total_pnl, 2),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(closed_trades) * 100, 1) if closed_trades else 0,
        "starting_capital": 15000,
        "current_equity": 15000 + total_pnl,
        "strategies_active": len(set(l.strategy for l in logs)),
        "last_check": logs[0].timestamp.isoformat() if logs else None
    }
    
    return [types.TextContent(type="text", text=json.dumps(summary, indent=2))]


async def get_strategy_status(strategy_filter: str = None) -> list[types.TextContent]:
    """Get current status of strategies"""
    logs = db.get_latest_logs(limit=50)
    
    # Group by strategy (get latest for each)
    latest_by_strategy = {}
    for log in logs:
        if strategy_filter and log.strategy != strategy_filter:
            continue
        if log.strategy not in latest_by_strategy:
            latest_by_strategy[log.strategy] = log
    
    statuses = []
    for strategy, log in latest_by_strategy.items():
        statuses.append({
            "strategy": strategy,
            "symbol": log.symbol,
            "status": log.status,
            "message": log.message,
            "price": log.price,
            "rsi": round(log.rsi, 1) if log.rsi else None,
            "adx": round(log.adx, 1) if log.adx else None,
            "last_check": log.timestamp.isoformat()
        })
    
    return [types.TextContent(type="text", text=json.dumps(statuses, indent=2))]


async def get_recent_trades(limit: int = 20, strategy: str = None) -> list[types.TextContent]:
    """Get recent trades"""
    trades = db.get_trades(strategy=strategy, limit=limit)
    
    trades_data = []
    for t in trades:
        trades_data.append({
            "id": t.id,
            "strategy": t.strategy,
            "symbol": t.symbol,
            "side": t.side,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "pnl_pct": round(t.pnl_pct, 2) if t.pnl_pct else None,
            "pnl_usd": round(t.pnl_usd, 2) if t.pnl_usd else None,
            "entry_time": str(t.entry_time),
            "exit_time": str(t.exit_time) if t.exit_time else None,
            "is_open": t.is_open,
            "exit_reason": t.exit_reason
        })
    
    return [types.TextContent(type="text", text=json.dumps(trades_data, indent=2))]


async def get_weekly_report(days: int = 7) -> list[types.TextContent]:
    """Generate weekly report"""
    start_date = datetime.now() - timedelta(days=days)
    trades = db.get_trades(start_date=start_date, limit=500)
    logs = db.get_latest_logs(limit=200)
    
    # Calculate stats
    closed_trades = [t for t in trades if not t.is_open]
    total_pnl = sum(t.pnl_usd or 0 for t in closed_trades)
    wins = [t for t in closed_trades if t.pnl_pct and t.pnl_pct > 0]
    
    # Strategy activity
    strategy_stats = {}
    for log in logs:
        if log.strategy not in strategy_stats:
            strategy_stats[log.strategy] = {"checks": 0, "signals": 0, "rsi_values": [], "adx_values": []}
        strategy_stats[log.strategy]["checks"] += 1
        if log.status != "WAITING":
            strategy_stats[log.strategy]["signals"] += 1
        if log.rsi:
            strategy_stats[log.strategy]["rsi_values"].append(log.rsi)
        if log.adx:
            strategy_stats[log.strategy]["adx_values"].append(log.adx)
    
    for name, data in strategy_stats.items():
        data["avg_rsi"] = round(sum(data["rsi_values"]) / len(data["rsi_values"]), 1) if data["rsi_values"] else 0
        data["avg_adx"] = round(sum(data["adx_values"]) / len(data["adx_values"]), 1) if data["adx_values"] else 0
        del data["rsi_values"]
        del data["adx_values"]
    
    report = {
        "period": f"Last {days} days",
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "total_trades": len(closed_trades),
            "total_pnl_usd": round(total_pnl, 2),
            "wins": len(wins),
            "win_rate": round(len(wins) / len(closed_trades) * 100, 1) if closed_trades else 0
        },
        "strategy_activity": strategy_stats,
        "market_conditions": "Overbought - RSI values consistently above 50, waiting for pullback" if all(s["avg_rsi"] > 50 for s in strategy_stats.values()) else "Mixed conditions"
    }
    
    return [types.TextContent(type="text", text=json.dumps(report, indent=2))]


async def get_training_data_stats() -> list[types.TextContent]:
    """Get training data statistics"""
    records = db.get_training_data(limit=10000)
    
    # Count by type
    by_type = {}
    for r in records:
        analysis_type = r.get("analysis_type", "unknown")
        by_type[analysis_type] = by_type.get(analysis_type, 0) + 1
    
    stats = {
        "total_records": len(records),
        "by_analysis_type": by_type,
        "ready_for_finetuning": len(records) >= 100,
        "recommendation": "Need at least 500 records for effective fine-tuning" if len(records) < 500 else "Sufficient data for fine-tuning",
        "export_url": "/api/training-data/export?format=jsonl"
    }
    
    return [types.TextContent(type="text", text=json.dumps(stats, indent=2))]


async def get_market_regime(symbol: str) -> list[types.TextContent]:
    """Get market regime analysis for a symbol"""
    logs = db.get_latest_logs(limit=100)
    
    # Filter by symbol
    symbol_logs = [l for l in logs if l.symbol == symbol]
    
    if not symbol_logs:
        return [types.TextContent(type="text", text=json.dumps({"error": f"No data for {symbol}"}))]
    
    latest = symbol_logs[0]
    avg_rsi = sum(l.rsi for l in symbol_logs if l.rsi) / len([l for l in symbol_logs if l.rsi]) if symbol_logs else 0
    avg_adx = sum(l.adx for l in symbol_logs if l.adx) / len([l for l in symbol_logs if l.adx]) if symbol_logs else 0
    
    # Determine regime
    if avg_rsi > 70:
        regime = "Overbought - potential reversal zone"
    elif avg_rsi < 30:
        regime = "Oversold - potential buying opportunity"
    elif avg_adx > 25:
        regime = "Trending - follow the momentum"
    else:
        regime = "Ranging - wait for breakout"
    
    analysis = {
        "symbol": symbol,
        "current_price": latest.price,
        "current_rsi": round(latest.rsi, 1) if latest.rsi else None,
        "current_adx": round(latest.adx, 1) if latest.adx else None,
        "avg_rsi": round(avg_rsi, 1),
        "avg_adx": round(avg_adx, 1),
        "regime": regime,
        "last_updated": latest.timestamp.isoformat()
    }
    
    return [types.TextContent(type="text", text=json.dumps(analysis, indent=2))]


async def get_strategy_configuration() -> list[types.TextContent]:
    """Get strategy configuration"""
    configs = []
    for config in STRATEGIES:
        configs.append({
            "name": config.name,
            "symbol": config.symbol,
            "timeframe": config.timeframe,
            "market_type": config.market_type,
            "allocation_pct": config.allocation_pct,
            "enabled": config.enabled
        })
    
    return [types.TextContent(type="text", text=json.dumps(configs, indent=2))]


async def main():
    """Run the MCP server"""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="trading-bot",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities={}
                )
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
