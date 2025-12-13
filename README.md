# Gold Line Crypto Trading LLM System

An intelligent crypto trading system that combines technical analysis with LLM-powered signal optimization.

## 🎯 Overview

This system converts the Pine Script "Gold Line" trading strategy to Python, integrates with Bybit for historical data, runs backtests, and uses LLM to analyze results and discover better trading patterns.

## 📁 Project Structure

```
TrainingFinancialLLM/
├── config/
│   └── strategy_params.yaml    # All configurable parameters
├── strategy/
│   ├── indicators.py           # Technical indicators (CCI, MACD, RSI, EMA)
│   └── gold_line_strategy.py   # Main strategy logic
├── data/
│   └── bybit_connector.py      # Bybit API integration
├── backtesting/
│   └── backtest_engine.py      # Trade simulation & metrics
├── llm/
│   └── analyzer.py             # LLM-powered analysis
├── results/                    # Backtest outputs
├── run_backtest.py             # Main backtest script
└── analyze_results.py          # LLM analysis script
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd TrainingFinancialLLM
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Alternative: Bybit MCP Server

There's also a Bybit MCP server ([sammcj/bybit-mcp](https://github.com/sammcj/bybit-mcp)) that provides an AI-native interface with additional features like:
- `get_ml_rsi` - ML-based RSI analysis
- `get_market_structure` - Market structure detection
- `get_order_blocks` - Institutional order zones

To use it, install via npm:
```bash
npm i -g pnpm && pnpm i
```

### 2. Configure (Optional)

Edit `config/strategy_params.yaml` to adjust:
- Indicator parameters (CCI, MACD, RSI lengths/thresholds)
- Trading settings (position size, stop loss, take profit)
- LLM provider (OpenAI or Anthropic)

### 3. Run Backtest

```bash
# Basic run with BTC
python run_backtest.py

# Custom symbol and timeframe
python run_backtest.py --symbol ETHUSDT --timeframe 15 --days 180
```

### 4. Set Up LLM (Choose One)

#### Option A: Ollama (FREE - Local)
```bash
# Install Ollama
brew install ollama

# Pull a model (choose one)
ollama pull llama3:8b      # Recommended, 8GB RAM
ollama pull mistral        # Alternative, lighter

# Run analysis (default config uses Ollama)
python analyze_results.py
```

#### Option B: OpenAI (Paid - Best Quality)

1. Go to [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Click "Create new secret key"
3. Copy the key and add to `.env`:
```bash
echo "OPENAI_API_KEY=sk-your-key-here" > .env
```
4. Update `config/strategy_params.yaml`:
```yaml
llm:
  provider: "openai"
  model: "gpt-4o-mini"  # Cheapest, good quality
```

#### Option C: Together.ai (FREE Credits)

1. Go to [api.together.xyz](https://api.together.xyz/settings/api-keys)
2. Sign up and get free $25 credits
3. Add to `.env`:
```bash
echo "TOGETHER_API_KEY=your-key-here" > .env
```
4. Update config:
```yaml
llm:
  provider: "together"
  model: "meta-llama/Llama-3-8b-chat-hf"
```

### 5. Run Analysis

```bash
# Run analysis
python analyze_results.py

# Generate training data for fine-tuning
python analyze_results.py --generate-training
```

## ⚙️ Configuration

All strategy parameters are in `config/strategy_params.yaml`:

```yaml
# CCI Settings
cci:
  length: 14
  upper_level: 75
  lower_level: -75

# MACD Settings
macd:
  fast_length: 12
  slow_length: 17
  signal_length: 8

# Filters - toggle on/off
filters:
  use_macd_filter: true
  use_rsi_filter: true
  use_trend_filter: true
```

## 📊 Output

The backtest generates:
- `results/summary_*.json` - Performance metrics
- `results/trades_*.json` - Detailed trade log for LLM analysis
- `results/equity_*.csv` - Equity curve data
- `results/analysis_*.md` - LLM analysis report

## 🔄 Workflow

```
1. Tweak parameters in YAML
         ↓
2. Run backtest → Get results
         ↓
3. LLM analyzes trades → Suggests improvements
         ↓
4. Update parameters → Repeat
```

## 📈 Strategy Components

| Component | Description |
|-----------|-------------|
| **Gold Line** | EMA of median price (HL2) - trend direction |
| **CCI** | Momentum oscillator for overbought/oversold |
| **MACD** | Trend confirmation filter |
| **RSI** | Additional momentum filter |
| **S/R Levels** | Support and resistance zones |

## License

MIT
