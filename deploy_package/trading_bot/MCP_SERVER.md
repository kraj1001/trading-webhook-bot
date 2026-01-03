# Trading Bot MCP Server

This MCP server exposes your trading bot as tools that AI assistants (Claude, Gemini, etc.) can use to query your trading data.

## Available Tools

| Tool | Description |
|------|-------------|
| `get_trading_summary` | Overall performance: PnL, win rate, equity |
| `get_strategy_status` | Current RSI, ADX, signals for each strategy |
| `get_recent_trades` | List of recent trades with P&L |
| `get_weekly_report` | GPT-4o powered weekly analysis |
| `get_training_data_stats` | LLM training data statistics |
| `get_market_regime` | Market condition analysis for a symbol |
| `get_strategy_configuration` | Strategy allocations and settings |

## Setup

### 1. Install MCP dependency
```bash
pip install mcp
```

### 2. Add to your MCP settings

Add this to `~/.gemini/mcp_settings.json`:

```json
{
  "mcpServers": {
    "trading-bot": {
      "command": "python",
      "args": ["-m", "trading_bot.mcp_server"],
      "cwd": "/Users/rajneeeshkambhatla/AntiGravity/TrainingFinancialLLM"
    }
  }
}
```

### 3. Restart your AI assistant

After adding the configuration, restart the assistant for the MCP server to be available.

## Example Queries

Once configured, you can ask your AI assistant:
- "What's my trading performance?"
- "Show me the status of all strategies"
- "Generate a weekly report"
- "What's the market regime for XRPUSDT?"
- "How much training data have we collected?"

## Running Manually (for testing)

```bash
cd /Users/rajneeeshkambhatla/AntiGravity/TrainingFinancialLLM
python -m trading_bot.mcp_server
```
