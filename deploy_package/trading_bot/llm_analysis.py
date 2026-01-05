"""
LLM Analysis Module
Uses Azure OpenAI to analyze trades and generate insights
"""
import os
import logging
import json
import requests
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime

from .config import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_KEY,
    AZURE_OPENAI_DEPLOYMENT,
    STRATEGIES
)

logger = logging.getLogger(__name__)

@dataclass
class TradeAnalysis:
    trade_id: int
    strategy: str
    symbol: str
    outcome: str  # WIN or LOSS
    pnl_usd: float
    analysis: str
    recommendation: str

class LLMAnalyzer:
    def __init__(self):
        self.endpoint = AZURE_OPENAI_ENDPOINT
        self.api_key = AZURE_OPENAI_KEY
        self.deployment = AZURE_OPENAI_DEPLOYMENT
        self.api_version = "2024-02-15-preview"
        
        # Check for local LLM (LM Studio) first, then Azure
        self.local_llm_url = os.getenv("LOCAL_LLM_URL", "http://localhost:1234/v1")
        self.use_local_llm = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
        
        if self.use_local_llm:
            logger.info(f"LLM Analyzer using LOCAL model at: {self.local_llm_url}")
            self.enabled = True
            self.mode = "local"
        elif self.endpoint and self.api_key:
            self.enabled = True
            self.mode = "azure"
            logger.info(f"LLM Analyzer using Azure OpenAI: {self.deployment}")
        else:
            logger.warning("No LLM credentials. Set USE_LOCAL_LLM=true for LM Studio or configure Azure.")
            self.enabled = False
            self.mode = None

    def _call_gpt4(self, system_prompt: str, user_prompt: str) -> str:
        """Call LLM - either local (LM Studio) or Azure OpenAI"""
        if not self.enabled:
            return "Analysis disabled (no credentials)"

        if self.mode == "local":
            return self._call_local_llm(system_prompt, user_prompt)
        else:
            return self._call_azure_openai(system_prompt, user_prompt)
    
    def _call_local_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call local LLM via LM Studio's OpenAI-compatible API"""
        url = f"{self.local_llm_url}/chat/completions"
        
        headers = {"Content-Type": "application/json"}
        
        payload = {
            "model": "local-model",  # LM Studio ignores this, uses loaded model
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 1000
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to LM Studio. Make sure it's running at " + self.local_llm_url)
            return "Error: LM Studio not running. Start LM Studio and load a model."
        except Exception as e:
            logger.error(f"Local LLM Error: {e}")
            return f"Error creating analysis: {str(e)}"
    
    def _call_azure_openai(self, system_prompt: str, user_prompt: str) -> str:
        """Call Azure OpenAI GPT-4"""
        url = f"{self.endpoint}/openai/deployments/{self.deployment}/chat/completions?api-version={self.api_version}"
        
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }
        
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 1000
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Azure OpenAI Error: {e}")
            return f"Error creating analysis: {str(e)}"

    def analyze_trade(self, trade_data: Dict) -> str:
        """Analyze a single completed trade"""
        strategy_name = trade_data.get("strategy")
        strategy_config = next((s for s in STRATEGIES if s.name == strategy_name), None)
        
        system_prompt = """You are a Senior Quantitative Analyst.
Your job is to analyze a completed trade and determine WHY it won or lost.
Focus on:
1. Market Context (Trend, Volatility)
2. Entry Timing (Was it too early/late?)
3. Exit Execution (Stop loss hit? Take profit?)
4. Strategy Rules (Did the strategy behave as expected?)

Provide a concise, bullet-point analysis. Be critical."""

        user_prompt = f"""
ANALYZE THIS TRADE:
Strategy: {strategy_name}
Symbol: {trade_data.get('symbol')}
Side: {trade_data.get('side')}
Entry Time: {trade_data.get('entry_time')}
Exit Time: {trade_data.get('exit_time')}
Entry Price: {trade_data.get('entry_price')}
Exit Price: {trade_data.get('exit_price')}
P&L: ${trade_data.get('pnl_usd')} ({trade_data.get('pnl_pct')}%)
Exit Reason: {trade_data.get('exit_reason')}

INDICATORS AT ENTRY:
RSI: {trade_data.get('entry_rsi')}
ADX: {trade_data.get('entry_adx')}
EMA15: {trade_data.get('entry_ema15')}
EMA30: {trade_data.get('entry_ema30')}

Was this a good trade setup? If it lost, what went wrong?
"""
        return self._call_gpt4(system_prompt, user_prompt)

    def generate_weekly_report(self, trades: List[Dict]) -> str:
        """Generate a weekly strategy performance report"""
        if not trades:
            return "No trades to analyze."
            
        system_prompt = """You are a Head of Trading Strategy.
Review the performance of these trading strategies over the last week.
Identify patterns in WINNING vs LOSING trades.
Suggest specific parameter adjustments to improve performance.
"""
        
        trades_summary = json.dumps(trades, default=str)
        
        user_prompt = f"""
WEEKLY PERFORMANCE REVIEW:
Total Trades: {len(trades)}

TRADE LOG:
{trades_summary}

TASKS:
1. Summarize performance by strategy
2. Identify 2-3 patterns in losing trades (e.g. "All loss trades happened when ADX < 20")
3. Recommend specific parameter changes (e.g. "Increase RSI threshold to 75")
"""
        return self._call_gpt4(system_prompt, user_prompt)

    def analyze_entry_signal(self, signal_data: Dict) -> Dict:
        """
        Analyze WHY a trading signal was generated.
        Returns structured data for LLM training.
        """
        system_prompt = """You are a Senior Quantitative Trader documenting trade decisions.
For each signal, explain:
1. What indicators triggered the entry
2. Why this is a high-probability setup
3. What risks exist
4. Your confidence level

Output as structured JSON with keys: analysis, risk_factors, confidence, training_output
The 'training_output' should be a concise 2-3 sentence explanation suitable for training a smaller LLM."""

        user_prompt = f"""
ENTRY SIGNAL GENERATED:
Symbol: {signal_data.get('symbol')}
Strategy: {signal_data.get('strategy')}
Signal: {signal_data.get('signal', 'BUY')}
Price: ${signal_data.get('price', 0):.4f}

INDICATORS:
- RSI: {signal_data.get('rsi', 0):.2f}
- ADX: {signal_data.get('adx', 0):.2f}
- EMA 15: {signal_data.get('ema_15', 0):.4f}
- EMA 30: {signal_data.get('ema_30', 0):.4f}
- MACD: {signal_data.get('macd', 0):.6f}

Explain WHY this signal was generated and create training data for a momentum expert LLM.
"""
        raw_response = self._call_gpt4(system_prompt, user_prompt)
        
        # Parse JSON response
        try:
            result = json.loads(raw_response)
        except json.JSONDecodeError:
            result = {
                "analysis": raw_response,
                "risk_factors": ["Unable to parse structured response"],
                "confidence": "medium",
                "training_output": raw_response[:500]
            }
        
        # Create training instruction/output pair
        instruction = f"Analyze these indicators for {signal_data.get('symbol')}: RSI={signal_data.get('rsi', 0):.1f}, ADX={signal_data.get('adx', 0):.1f}, MACD={signal_data.get('macd', 0):.6f}. Price at EMA15={signal_data.get('ema_15', 0):.4f}, EMA30={signal_data.get('ema_30', 0):.4f}. Should I enter a trade?"
        
        result['instruction'] = instruction
        result['output'] = result.get('training_output', result.get('analysis', ''))[:2000]
        result['signal_generated'] = True
        
        return result

    def analyze_no_signal(self, check_data: Dict) -> Dict:
        """
        Analyze WHY no trading signal was generated.
        Equally important for training - teaches model when NOT to trade.
        """
        system_prompt = """You are a Senior Quantitative Trader documenting why you chose NOT to trade.
Explain:
1. What indicators prevented the entry
2. What conditions are missing for a valid signal
3. What would need to change to trigger a trade

Output as structured JSON with keys: analysis, missing_conditions, what_would_trigger_trade, training_output
The 'training_output' should be a concise 2-3 sentence explanation suitable for training a smaller LLM."""

        user_prompt = f"""
NO SIGNAL GENERATED (opted not to trade):
Symbol: {check_data.get('symbol')}
Strategy: {check_data.get('strategy')}
Price: ${check_data.get('price', 0):.4f}

INDICATORS:
- RSI: {check_data.get('rsi', 0):.2f}
- ADX: {check_data.get('adx', 0):.2f}
- EMA 15: {check_data.get('ema_15', 0):.4f}
- EMA 30: {check_data.get('ema_30', 0):.4f}
- MACD: {check_data.get('macd', 0):.6f}

Explain WHY we chose NOT to trade and what would need to change.
"""
        raw_response = self._call_gpt4(system_prompt, user_prompt)
        
        # Parse JSON response
        try:
            result = json.loads(raw_response)
        except json.JSONDecodeError:
            result = {
                "analysis": raw_response,
                "missing_conditions": ["Unable to parse structured response"],
                "what_would_trigger_trade": "Unknown",
                "training_output": raw_response[:500]
            }
        
        # Create training instruction/output pair
        instruction = f"Should I enter a trade on {check_data.get('symbol')} with RSI={check_data.get('rsi', 0):.1f}, ADX={check_data.get('adx', 0):.1f}?"
        
        result['instruction'] = instruction
        result['output'] = result.get('training_output', result.get('analysis', ''))[:2000]
        result['signal_generated'] = False
        
        return result

    def export_training_data_jsonl(self, records: list, filepath: str) -> int:
        """Export training records to JSONL format for fine-tuning"""
        lines_written = 0
        with open(filepath, 'w') as f:
            for record in records:
                jsonl_line = json.dumps({
                    "instruction": record.get('instruction', ''),
                    "output": record.get('output', '')
                })
                f.write(jsonl_line + '\n')
                lines_written += 1
        logger.info(f"Exported {lines_written} training records to {filepath}")
        return lines_written

    def analyze_market_regime(self, market_data: Dict) -> Dict:
        """
        #1 Market Regime Detection
        Identify if market is trending, ranging, or volatile.
        """
        system_prompt = """You are a Market Analyst specialized in regime detection.
Analyze the market conditions and classify the current regime.

Your output must be valid JSON with these keys:
- regime: one of "trending_up", "trending_down", "ranging", "high_volatility", "breakout"
- confidence: float between 0 and 1
- reasoning: 2-3 sentence explanation
- key_indicators: list of 2-3 most important signals
- trading_bias: "bullish", "bearish", or "neutral"
- training_output: concise 1-2 sentence summary for training an LLM"""

        user_prompt = f"""
MARKET CONDITIONS FOR {market_data.get('symbol')}:
Timeframe: {market_data.get('timeframe', '4h')}

PRICE DATA (last 10 candles):
- Current Price: ${market_data.get('price', 0):.4f}
- 10-candle High: ${market_data.get('high_10', 0):.4f}
- 10-candle Low: ${market_data.get('low_10', 0):.4f}
- Price Change %: {market_data.get('price_change_pct', 0):.2f}%

TECHNICAL INDICATORS:
- RSI (14): {market_data.get('rsi', 0):.2f}
- ADX (14): {market_data.get('adx', 0):.2f}
- ATR (14): {market_data.get('atr', 0):.6f}
- EMA 9/15/30: {market_data.get('ema_9', 0):.4f} / {market_data.get('ema_15', 0):.4f} / {market_data.get('ema_30', 0):.4f}
- MACD: {market_data.get('macd', 0):.6f}
- MACD Signal: {market_data.get('macd_signal', 0):.6f}

Determine the current market regime and explain why.
"""
        raw_response = self._call_gpt4(system_prompt, user_prompt)
        
        try:
            result = json.loads(raw_response)
        except json.JSONDecodeError:
            result = {
                "regime": "unknown",
                "confidence": 0.5,
                "reasoning": raw_response[:500],
                "key_indicators": ["Unable to parse"],
                "trading_bias": "neutral",
                "training_output": raw_response[:200]
            }
        
        # Create training pair
        instruction = f"What is the market regime for {market_data.get('symbol')} with RSI={market_data.get('rsi', 0):.1f}, ADX={market_data.get('adx', 0):.1f}, price at {market_data.get('price', 0):.4f}?"
        result['instruction'] = instruction
        result['output'] = result.get('training_output', result.get('reasoning', ''))[:2000]
        result['analysis_type'] = 'market_regime'
        
        return result

    def analyze_candlestick_patterns(self, candle_data: Dict) -> Dict:
        """
        #4 Candlestick Pattern Recognition
        Identify chart patterns and candlestick formations.
        """
        system_prompt = """You are a Technical Analyst expert in candlestick patterns.
Analyze the recent price action and identify patterns.

Your output must be valid JSON with these keys:
- patterns_detected: list of pattern names (e.g., "bullish_engulfing", "doji", "hammer")
- pattern_quality: "high", "medium", or "low"
- formation_context: where the pattern appears (support, resistance, trend)
- confluence_factors: list of supporting technical factors
- recommended_action: trading recommendation based on patterns
- stop_loss_level: suggested stop loss placement
- training_output: concise 2-3 sentence summary for training"""

        # Format candle data
        candles_text = ""
        recent_candles = candle_data.get('candles', [])[-10:]
        for i, c in enumerate(recent_candles):
            candles_text += f"Candle {i+1}: O={c.get('open', 0):.4f}, H={c.get('high', 0):.4f}, L={c.get('low', 0):.4f}, C={c.get('close', 0):.4f}\n"

        user_prompt = f"""
CANDLESTICK ANALYSIS FOR {candle_data.get('symbol')}:
Timeframe: {candle_data.get('timeframe', '4h')}

LAST 10 CANDLES:
{candles_text}

CONTEXT:
- Current RSI: {candle_data.get('rsi', 0):.2f}
- Near Support: {candle_data.get('near_support', False)}
- Near Resistance: {candle_data.get('near_resistance', False)}
- Volume Trend: {candle_data.get('volume_trend', 'normal')}

Identify any candlestick patterns and their trading implications.
"""
        raw_response = self._call_gpt4(system_prompt, user_prompt)
        
        try:
            result = json.loads(raw_response)
        except json.JSONDecodeError:
            result = {
                "patterns_detected": [],
                "pattern_quality": "unknown",
                "formation_context": "Unable to determine",
                "confluence_factors": [],
                "recommended_action": raw_response[:300],
                "training_output": raw_response[:200]
            }
        
        # Create training pair
        patterns = result.get('patterns_detected', [])
        pattern_str = ", ".join(patterns) if patterns else "none"
        instruction = f"What candlestick patterns are present in {candle_data.get('symbol')} on {candle_data.get('timeframe', '4h')} timeframe?"
        result['instruction'] = instruction
        result['output'] = result.get('training_output', f"Detected patterns: {pattern_str}")[:2000]
        result['analysis_type'] = 'candlestick_pattern'
        
        return result

    def enhanced_post_trade_review(self, trade_data: Dict) -> Dict:
        """
        #5 Enhanced Post-Trade Review
        Detailed analysis of completed trade with lessons learned.
        """
        system_prompt = """You are a Trading Performance Coach analyzing a completed trade.
Provide an in-depth review focusing on what can be learned.

Your output must be valid JSON with these keys:
- result: "win" or "loss"
- entry_quality: 1-10 rating with explanation
- exit_quality: 1-10 rating with explanation
- timing_analysis: Was entry/exit too early, too late, or optimal?
- market_read: How well did the strategy read market conditions?
- risk_reward_actual: The actual R:R achieved vs planned
- key_lessons: List of 2-3 specific takeaways
- improvement_suggestions: What to do differently next time
- training_output: 2-3 sentence summary for training"""

        outcome = "WIN ✅" if trade_data.get('pnl_pct', 0) > 0 else "LOSS ❌"
        
        user_prompt = f"""
POST-TRADE ANALYSIS:
Trade ID: {trade_data.get('trade_id', 'N/A')}
Strategy: {trade_data.get('strategy')}
Symbol: {trade_data.get('symbol')}
Result: {outcome}

TRADE DETAILS:
- Side: {trade_data.get('side')}
- Entry Price: ${trade_data.get('entry_price', 0):.4f}
- Exit Price: ${trade_data.get('exit_price', 0):.4f}
- P&L: {trade_data.get('pnl_pct', 0):+.2f}% (${trade_data.get('pnl_usd', 0):+.2f})
- Duration: {trade_data.get('duration', 'Unknown')}
- Exit Reason: {trade_data.get('exit_reason', 'Unknown')}

INDICATORS AT ENTRY:
- RSI: {trade_data.get('entry_rsi', 'N/A')}
- ADX: {trade_data.get('entry_adx', 'N/A')}
- MACD: {trade_data.get('entry_macd', 'N/A')}
- EMA Alignment: {trade_data.get('ema_alignment', 'N/A')}

MARKET CONTEXT:
- Market Regime at Entry: {trade_data.get('entry_regime', 'Unknown')}
- Did conditions change during trade: {trade_data.get('conditions_changed', 'Unknown')}

Provide detailed analysis of what went right/wrong and lessons learned.
"""
        raw_response = self._call_gpt4(system_prompt, user_prompt)
        
        try:
            result = json.loads(raw_response)
        except json.JSONDecodeError:
            result = {
                "result": "win" if trade_data.get('pnl_pct', 0) > 0 else "loss",
                "entry_quality": 5,
                "exit_quality": 5,
                "timing_analysis": "Unable to parse detailed analysis",
                "key_lessons": [raw_response[:200]],
                "training_output": raw_response[:300]
            }
        
        # Create training pair
        pnl = trade_data.get('pnl_pct', 0)
        instruction = f"Review this {trade_data.get('symbol')} trade: {trade_data.get('side')} entry at ${trade_data.get('entry_price', 0):.4f}, exit at ${trade_data.get('exit_price', 0):.4f}, P&L {pnl:+.2f}%"
        result['instruction'] = instruction
        result['output'] = result.get('training_output', result.get('analysis', ''))[:2000]
        result['analysis_type'] = 'post_trade_review'
        
        return result
