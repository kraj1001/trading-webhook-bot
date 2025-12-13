"""
LLM Analyzer
Uses LLM to analyze backtest results and discover patterns for improved trading.
"""

import json
import os
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


@dataclass
class AnalysisResult:
    """Results from LLM analysis"""
    summary: str
    win_patterns: List[str]
    loss_patterns: List[str]
    recommendations: List[str]
    parameter_suggestions: Dict[str, Any]
    confidence: float


# Analysis prompts
SYSTEM_PROMPT = """You are an expert quantitative trading analyst specializing in crypto markets and technical analysis.
You are analyzing backtest results from a Gold Line price action strategy that uses:
- CCI (Commodity Channel Index) for momentum
- MACD for trend confirmation
- RSI for overbought/oversold conditions
- Price Action Channel (Gold Line) - EMA of median price
- Support/Resistance levels

Your task is to analyze the trade history and identify:
1. Patterns in winning trades - what market conditions led to success?
2. Patterns in losing trades - what conditions should be avoided?
3. Specific, actionable recommendations to improve the strategy
4. Suggested parameter adjustments based on the data

Be specific and quantitative in your analysis. Reference actual indicator values and patterns from the data."""

ANALYSIS_PROMPT = """Analyze the following backtest results and trade history:

## Performance Summary
{summary}

## Sample Trades (showing {num_trades} trades)
{trades_sample}

## Questions to Answer:

1. **Winning Trade Patterns**: What specific conditions (indicator values, market context) are present in winning trades?

2. **Losing Trade Patterns**: What conditions correlate with losses? What should be avoided?

3. **Filter Recommendations**: Based on the data, suggest new filters or conditions to add/modify:
   - Should CCI thresholds change?
   - Are there RSI ranges that work better?
   - Are certain times/conditions to avoid?

4. **Parameter Suggestions**: Provide specific parameter changes with rationale:
   - Current CCI: 14 length, 75/-75 levels
   - Current MACD: 12, 17, 8
   - Current RSI: 7 length, 70/30 levels

5. **Risk Management**: Any observations about position sizing, stop loss, or take profit levels?

Provide your analysis in a structured format with clear, actionable recommendations."""


class LLMAnalyzer:
    """
    LLM-powered analysis of backtest results.
    Discovers patterns and suggests strategy improvements.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the analyzer.
        
        Supported providers:
        - openai: OpenAI API (GPT-4, GPT-3.5)
        - anthropic: Anthropic API (Claude)
        - ollama: Local Ollama (free, requires Ollama installed)
        - together: Together.ai (free credits available)
        
        Args:
            config: Configuration with LLM settings
        """
        llm_config = config.get('llm', {})
        self.provider = llm_config.get('provider', 'openai')
        self.model = llm_config.get('model', 'gpt-4')
        self.temperature = llm_config.get('temperature', 0.3)
        self.max_tokens = llm_config.get('max_tokens', 2000)
        self.ollama_host = llm_config.get('ollama_host', 'http://localhost:11434')
        
        # Initialize client based on provider
        self.client = None
        api_key = None
        
        if self.provider == 'openai':
            api_key = os.getenv('OPENAI_API_KEY')
            if HAS_OPENAI and api_key:
                self.client = openai.OpenAI(api_key=api_key)
                
        elif self.provider == 'anthropic':
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if HAS_ANTHROPIC and api_key:
                self.client = anthropic.Anthropic(api_key=api_key)
                
        elif self.provider == 'ollama':
            # Ollama uses OpenAI-compatible API, no key needed
            if HAS_OPENAI:
                self.client = openai.OpenAI(
                    base_url=f"{self.ollama_host}/v1",
                    api_key="ollama"  # Ollama doesn't need a real key
                )
                
        elif self.provider == 'together':
            api_key = os.getenv('TOGETHER_API_KEY')
            if HAS_OPENAI and api_key:
                self.client = openai.OpenAI(
                    base_url="https://api.together.xyz/v1",
                    api_key=api_key
                )
    
    def analyze_trades(
        self,
        trades_file: str,
        summary: Optional[Dict[str, Any]] = None,
        max_trades_to_show: int = 50
    ) -> str:
        """
        Analyze trade history using LLM.
        
        Args:
            trades_file: Path to trades JSON file from backtest
            summary: Performance summary dict
            max_trades_to_show: Max trades to include in prompt
        
        Returns:
            Analysis text from LLM
        """
        # Load trades
        with open(trades_file, 'r') as f:
            trades = json.load(f)
        
        if not trades:
            return "No trades to analyze."
        
        # Prepare summary
        if summary is None:
            wins = [t for t in trades if t['result'] == 'WIN']
            losses = [t for t in trades if t['result'] == 'LOSS']
            summary = {
                'total_trades': len(trades),
                'wins': len(wins),
                'losses': len(losses),
                'win_rate': len(wins) / len(trades) * 100,
                'avg_win_pct': sum(t['pnl_pct'] for t in wins) / len(wins) if wins else 0,
                'avg_loss_pct': sum(t['pnl_pct'] for t in losses) / len(losses) if losses else 0
            }
        
        # Format trades for prompt
        trades_sample = self._format_trades_for_prompt(trades[:max_trades_to_show])
        
        # Build prompt
        prompt = ANALYSIS_PROMPT.format(
            summary=json.dumps(summary, indent=2),
            num_trades=min(len(trades), max_trades_to_show),
            trades_sample=trades_sample
        )
        
        # Call LLM
        return self._call_llm(prompt)
    
    def _format_trades_for_prompt(self, trades: List[Dict]) -> str:
        """Format trades into readable text for LLM"""
        lines = []
        
        for i, trade in enumerate(trades, 1):
            indicators = trade.get('indicators_at_entry', {})
            context = trade.get('market_context', {})
            
            line = f"""
Trade {i}: {trade['direction']} - {trade['result']}
- Entry: {trade['entry_time']} at ${trade['entry_price']:.2f}
- Exit: {trade['exit_time']} at ${trade['exit_price']:.2f} ({trade['exit_reason']})
- PnL: {trade['pnl_pct']:.2f}%
- Duration: {trade['duration_candles']} candles
- Entry Indicators: CCI={indicators.get('cci', 'N/A'):.1f}, RSI={indicators.get('rsi', 'N/A'):.1f}, MACD={indicators.get('macd', 'N/A'):.2f}
- Context: {context.get('price_vs_gold_line', 'N/A')}, Volatility={context.get('volatility', 0):.2f}%
"""
            lines.append(line)
        
        return '\n'.join(lines)
    
    def _call_llm(self, prompt: str) -> str:
        """Call the LLM API"""
        if self.client is None:
            return self._mock_analysis(prompt)
        
        try:
            if self.provider == 'openai':
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                return response.choices[0].message.content
            
            elif self.provider == 'anthropic':
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=SYSTEM_PROMPT,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.content[0].text
            
        except Exception as e:
            return f"LLM API Error: {e}\n\nFalling back to basic analysis...\n{self._basic_analysis(prompt)}"
    
    def _mock_analysis(self, prompt: str) -> str:
        """Provide mock analysis when no LLM API is available"""
        return """
## LLM Analysis (Mock Mode - No API Key Configured)

To enable real LLM analysis, set your API key in the .env file:
- OPENAI_API_KEY=your_key_here
- or ANTHROPIC_API_KEY=your_key_here

### Basic Pattern Analysis Available:

The system has collected detailed trade data including:
- Entry/exit indicators (CCI, RSI, MACD, Gold Line position)
- Market context (volatility, trend strength)
- Trade outcomes and durations

Once configured, the LLM will analyze this data to find:
1. Conditions that correlate with winning trades
2. Patterns that lead to losses
3. Specific parameter optimization suggestions
4. New filter recommendations

### Next Steps:
1. Add your LLM API key to .env
2. Run: python analyze_results.py
"""
    
    def _basic_analysis(self, prompt: str) -> str:
        """Basic rule-based analysis as fallback"""
        return """
### Basic Analysis (No LLM Available)

Review the trades manually to identify:
1. Were losses concentrated in low-volatility periods?
2. Did winning trades have stronger MACD histogram values?
3. What was the typical CCI value at entry for wins vs losses?

Consider experimenting with:
- Higher CCI threshold (80 instead of 75)
- Adding a volatility filter (ATR > 1%)
- Stricter trend confirmation (MACD histogram > threshold)
"""
    
    def generate_training_data(
        self,
        trades_file: str,
        output_file: str = 'training_data.jsonl'
    ) -> str:
        """
        Generate LLM training data from trade history.
        Creates examples for fine-tuning.
        
        Args:
            trades_file: Path to trades JSON file
            output_file: Output path for training data
        
        Returns:
            Path to generated training file
        """
        with open(trades_file, 'r') as f:
            trades = json.load(f)
        
        training_examples = []
        
        for trade in trades:
            # Create training example
            indicators = trade.get('indicators_at_entry', {})
            context = trade.get('market_context', {})
            
            example = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a crypto trading signal filter. Analyze the signal context and decide whether to TAKE or SKIP the trade."
                    },
                    {
                        "role": "user",
                        "content": f"""Signal: {trade['direction']}
Entry Price: ${trade['entry_price']:.2f}
CCI: {indicators.get('cci', 0):.1f}
RSI: {indicators.get('rsi', 0):.1f}
MACD: {indicators.get('macd', 0):.2f}
Price vs Gold Line: {context.get('price_vs_gold_line', 'unknown')}
Volatility: {context.get('volatility', 0):.2f}%

Should I take this trade?"""
                    },
                    {
                        "role": "assistant",
                        "content": f"{'TAKE' if trade['result'] == 'WIN' else 'SKIP'} - Based on the indicators: {self._generate_reasoning(trade)}"
                    }
                ]
            }
            training_examples.append(example)
        
        # Write JSONL format
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            for example in training_examples:
                f.write(json.dumps(example) + '\n')
        
        print(f"Generated {len(training_examples)} training examples to {output_path}")
        return str(output_path)
    
    def _generate_reasoning(self, trade: Dict) -> str:
        """Generate reasoning for trade decision"""
        indicators = trade.get('indicators_at_entry', {})
        
        if trade['result'] == 'WIN':
            return f"Strong momentum with CCI at {indicators.get('cci', 0):.1f} and favorable trend alignment."
        else:
            return f"Weak setup - indicators suggest caution. CCI: {indicators.get('cci', 0):.1f}, volatility may be too low."
